"""Rendering: tailored resume -> DOCX (docxtpl) -> PDF, plus page-fit optimization.

The pipeline is deliberately boring so the output stays ATS-safe:

    TailoredResume + MasterResume
        -> ResumeView      (plain structure; headers ALWAYS copied from the master resume)
        -> resume.docx     (template.docx filled with docxtpl; single column, no tables)
        -> resume.pdf      (LibreOffice headless; docx2pdf as a fallback)

If the PDF is over one page, `fit_to_page()` tries, in this order and
re-rendering after each step:

    1. ask Claude to tighten the longest bullets (no meaning change, re-validated)
    2. adjust spacing, then margins, then font size, within the ATS-safe limits
       in config.py
    3. remove the lowest-relevance content: bullets first, then projects

Every step is logged in the returned `FitResult`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import config
from src import textutil
from src.errors import RenderError
from src.models import MasterResume, TailoredBullet, TailoredEntry, TailoredResume
from src.relevance import Relevance

ENTRY_SEPARATOR = " — "       # "Org — Title"
CONTACT_SEPARATOR = " | "
BULLET_CHAR = "•"


# ---------------------------------------------------------------------------
# View: the exact content that goes on the page
# ---------------------------------------------------------------------------


@dataclass
class ViewEntry:
    left: str
    right: str = ""
    lines: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    source_id: str = ""
    start: str = ""
    end: str = ""


@dataclass
class ViewSection:
    name: str
    heading: str
    entries: list[ViewEntry] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)


@dataclass
class ResumeView:
    name: str
    contact_line: str
    contact: dict[str, str]
    sections: list[ViewSection] = field(default_factory=list)

    def atoms(self) -> list[str]:
        """Every piece of text on the page, in reading order."""
        out = [self.name, self.contact_line]
        for section in self.sections:
            out.append(section.heading)
            for entry in section.entries:
                out.append(entry.left)
                if entry.right:
                    out.append(entry.right)
                out.extend(entry.lines)
                out.extend(entry.bullets)
            out.extend(section.lines)
            out.extend(section.bullets)
        return [a for a in out if a.strip()]

    def plain_text(self) -> str:
        """The resume as plain text (what the judge reads and keywords are counted in)."""
        lines = [self.name, self.contact_line, ""]
        for section in self.sections:
            lines.append(section.heading)
            for entry in section.entries:
                lines.append(f"{entry.left}  {entry.right}".rstrip())
                lines.extend(entry.lines)
                lines.extend(f"- {b}" for b in entry.bullets)
            lines.extend(section.lines)
            lines.extend(f"- {b}" for b in section.bullets)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def body_text(self) -> str:
        """Everything except the name and contact line (for keyword counting)."""
        return "\n".join(self.atoms()[2:])

    def context(self) -> dict:
        """The docxtpl rendering context."""
        return {
            "name": self.name,
            "contact_line": self.contact_line,
            "sections": [
                {
                    "heading": s.heading,
                    "entries": [{"left": e.left, "right": e.right, "lines": e.lines, "bullets": e.bullets}
                                for e in s.entries],
                    "lines": s.lines,
                    "bullets": s.bullets,
                }
                for s in self.sections
            ],
        }

    def section(self, name: str) -> ViewSection | None:
        return next((s for s in self.sections if s.name == name), None)


def clean_bullet(text: str) -> str:
    """ASCII-safe punctuation and a closing period."""
    text = textutil.plain(text)
    return text if not text or text[-1] in ".!?" else text + "."


def _gpa(value: float) -> str:
    shown = f"{value:.1f}" if abs(value * 10 - round(value * 10)) < 1e-9 else f"{value:.2f}"
    return f"{config.GPA_LABEL}: {shown}/{config.GPA_SCALE:.1f}"


def _entry_view(entry: TailoredEntry, master: MasterResume) -> ViewEntry | None:
    """Header fields come from the master resume - never from the AI."""
    source = master.entry(entry.source_id)
    if source is None:
        return None
    left = textutil.plain(source.display_org)
    if source.display_title.strip():
        left += ENTRY_SEPARATOR + textutil.plain(source.display_title)
    if source.location.strip():
        left += f", {textutil.plain(source.location)}"
    return ViewEntry(
        left=left,
        right=textutil.format_date_range(source.dates.start, source.dates.end),
        bullets=[clean_bullet(b.text) for b in entry.bullets if b.text.strip()],
        source_id=source.id, start=source.dates.start, end=source.dates.end,
    )


def _chronological(entries: list[ViewEntry]) -> list[ViewEntry]:
    """Most recent first: by end date, then by start date."""
    return sorted(entries, key=lambda e: (textutil.parse_ym(e.end or e.start, end=True), textutil.parse_ym(e.start)),
                  reverse=True)


def build_view(resume: TailoredResume, master: MasterResume) -> ResumeView:
    contact = master.contact
    parts = [contact.location, contact.phone, contact.email, contact.linkedin, contact.github]
    view = ResumeView(
        name=textutil.plain(contact.name).upper(),
        contact_line=CONTACT_SEPARATOR.join(textutil.plain(p) for p in parts if p and p.strip()),
        contact={"name": contact.name, "email": contact.email, "phone": contact.phone,
                 "linkedin": contact.linkedin, "github": contact.github},
    )
    by_section: dict[str, list[ViewEntry]] = {"experience": [], "leadership": []}
    for entry in resume.experiences:
        source = master.entry(entry.source_id)
        built = _entry_view(entry, master)
        if source is None or built is None or not built.bullets:
            continue
        by_section.setdefault(source.section_name, []).append(built)
    projects = [v for v in (_entry_view(e, master) for e in resume.projects) if v is not None and v.bullets]

    for name in resume.section_order or [s for s in config.DEFAULT_SECTION_ORDER]:
        key = getattr(name, "value", name)
        section = ViewSection(name=key, heading=config.SECTION_HEADINGS[key])
        if key == "education":
            for edu in master.education:
                left = edu.school + (f", {edu.location}" if edu.location else "")
                lines = [edu.degree, *edu.notes]
                if edu.gpa is not None:
                    lines.append(_gpa(edu.gpa))
                honors = [h for h in edu.honors if h in resume.honors]
                if honors:
                    lines.append("Honors: " + ", ".join(honors))
                coursework = [c for c in edu.coursework if c in resume.coursework]
                if coursework:
                    lines.append("Relevant Coursework: " + ", ".join(coursework))
                section.entries.append(ViewEntry(left=textutil.plain(left), right=textutil.plain(edu.grad),
                                                 lines=[textutil.plain(line) for line in lines if line.strip()],
                                                 source_id=edu.id))
        elif key in ("experience", "leadership"):
            section.entries = _chronological(by_section.get(key, []))
        elif key == "projects":
            section.entries = _chronological(projects)
        elif key == "skills":
            for category, label in config.SKILL_CATEGORY_LABELS.items():
                items = [s for s in resume.skills if master.skills.category_of(s) == category]
                if items:
                    section.lines.append(f"{label}: {', '.join(items)}")
        elif key == "awards":
            section.bullets = [clean_bullet(a.text) for a in (master.award(i) for i in resume.award_ids) if a is not None]
        elif key == "certifications":
            section.lines = [textutil.plain(c) for c in master.certifications]
        if section.entries or section.lines or section.bullets:
            view.sections.append(section)
    return view


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


@dataclass
class LayoutParams:
    """Deviations from the template's own design, used only by page fitting."""

    margin_in: float | None = None      # None = keep the template's margins
    font_scale: float = 1.0
    spacing_scale: float = 1.0

    def describe(self, template_font_pt: float, template_margin_in: float) -> str:
        return (f'margins {self.margin_in if self.margin_in is not None else template_margin_in:.2f}", '
                f"body {template_font_pt * self.font_scale:.1f}pt, spacing x{self.spacing_scale:.1f}")

    def is_default(self) -> bool:
        return self.margin_in is None and self.font_scale == 1.0 and self.spacing_scale == 1.0


def template_info(template: Path | None = None) -> tuple[float, float]:
    """(margin in inches, body font size in points) of the template."""
    from docx import Document

    template = template or config.TEMPLATE_PATH
    if not template.exists():
        raise RenderError(f"Template not found: {template}", "Run: python scripts/build_template.py")
    doc = Document(str(template))
    section = doc.sections[0]
    margin = min(section.left_margin.inches, section.right_margin.inches,
                 section.top_margin.inches, section.bottom_margin.inches)
    size = doc.styles["Normal"].font.size
    return (round(margin, 3), size.pt if size is not None else config.DEFAULT_FONT_PT)


def _apply_layout(docx_path: Path, layout: LayoutParams) -> None:
    """Scale spacing / fonts and set margins on an already-rendered document."""
    from docx import Document
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.text import WD_TAB_ALIGNMENT
    from docx.shared import Inches, Pt

    doc = Document(str(docx_path))
    section = doc.sections[0]
    if layout.margin_in is not None:
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Inches(layout.margin_in))
    text_width = section.page_width.inches - section.left_margin.inches - section.right_margin.inches
    for style in doc.styles:
        if style.type != WD_STYLE_TYPE.PARAGRAPH:
            continue
        if layout.font_scale != 1.0 and style.font.size is not None:
            style.font.size = Pt(round(style.font.size.pt * layout.font_scale * 2) / 2)   # half-point steps
        fmt = style.paragraph_format
        if layout.spacing_scale != 1.0:
            if fmt.space_before is not None:
                fmt.space_before = Pt(fmt.space_before.pt * layout.spacing_scale)
            if fmt.space_after is not None:
                fmt.space_after = Pt(fmt.space_after.pt * layout.spacing_scale)
        # Keep right-aligned dates on the (possibly moved) right margin.
        rights = [t for t in fmt.tab_stops if t.alignment == WD_TAB_ALIGNMENT.RIGHT]
        if rights:
            for stop in rights:
                stop.position = Inches(text_width)
    doc.save(str(docx_path))


def render_docx(view: ResumeView, path: Path, layout: LayoutParams | None = None,
                template: Path | None = None) -> Path:
    from docxtpl import DocxTemplate

    template = template or config.TEMPLATE_PATH
    if not template.exists():
        raise RenderError(f"Template not found: {template}", "Run: python scripts/build_template.py")
    try:
        doc = DocxTemplate(str(template))
        doc.render(view.context(), autoescape=True)      # escapes "&" and "<" (Texas A&M)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(path))
    except Exception as exc:  # docxtpl / jinja raise many unrelated types
        raise RenderError(f"Could not fill the template: {exc}",
                          "If you edited template.docx, check that the {{ }} and {%p %} tags are intact.") from exc
    if layout is not None and not layout.is_default():
        _apply_layout(path, layout)
    return path


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def find_soffice() -> str | None:
    for candidate in config.SOFFICE_CANDIDATES:
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    return None


def _fontconfig_file() -> Path | None:
    """On macOS, headless LibreOffice only sees its bundled fonts unless told where
    the system fonts are; without this "Times New Roman" silently becomes Liberation Serif."""
    if sys.platform != "darwin":
        return None
    directory = config.CACHE_DIR / "fontconfig"
    directory.mkdir(parents=True, exist_ok=True)
    folders = [
        "/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts",
        str(Path.home() / "Library/Fonts"),
        "/Applications/Microsoft Word.app/Contents/Resources/DFonts",
        "/Applications/LibreOffice.app/Contents/Resources/fonts/truetype",
    ]
    dirs = "\n".join(f"  <dir>{f}</dir>" for f in folders if Path(f).exists())
    conf = directory / "fonts.conf"
    conf.write_text(
        '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig>\n'
        f"{dirs}\n  <cachedir>{directory / 'cache'}</cachedir>\n</fontconfig>\n",
        encoding="utf-8",
    )
    return conf


def docx_to_pdf(docx: Path, out_dir: Path | None = None) -> Path:
    """Convert with LibreOffice headless; fall back to docx2pdf (Microsoft Word)."""
    out_dir = out_dir or docx.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / (docx.stem + ".pdf")
    if pdf.exists():
        pdf.unlink()
    soffice = find_soffice()
    if soffice:
        profile = config.CACHE_DIR / "soffice-profile"      # private profile: never collides with an open LibreOffice
        profile.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        fontconfig = _fontconfig_file()
        if fontconfig is not None:
            env["FONTCONFIG_FILE"] = str(fontconfig)
        try:
            result = subprocess.run(
                [soffice, f"-env:UserInstallation={profile.resolve().as_uri()}", "--headless",
                 "--convert-to", "pdf", "--outdir", str(out_dir), str(docx)],
                capture_output=True, text=True, timeout=config.PDF_CONVERT_TIMEOUT_SECONDS, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RenderError("LibreOffice timed out converting the resume to PDF.") from exc
        if result.returncode != 0 or not pdf.exists():
            raise RenderError("LibreOffice could not convert the resume to PDF: "
                              + (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no output"))
        return pdf
    try:
        from docx2pdf import convert  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RenderError(
            "No PDF converter found.",
            "Install LibreOffice (macOS: brew install --cask libreoffice) or: pip install docx2pdf (needs Microsoft Word).",
        ) from exc
    try:
        convert(str(docx), str(pdf))
    except Exception as exc:
        raise RenderError(f"docx2pdf could not convert the resume: {exc}") from exc
    if not pdf.exists():
        raise RenderError("docx2pdf produced no PDF.")
    return pdf


def page_count(pdf: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


@dataclass
class Rendered:
    view: ResumeView
    docx: Path
    pdf: Path
    pages: int
    layout: LayoutParams


def render(resume: TailoredResume, master: MasterResume, out_dir: Path,
           layout: LayoutParams | None = None, basename: str = "resume") -> Rendered:
    """View -> DOCX -> PDF. Returns paths and the page count."""
    layout = layout or LayoutParams()
    view = build_view(resume, master)
    docx = render_docx(view, out_dir / f"{basename}.docx", layout)
    pdf = docx_to_pdf(docx, out_dir)
    return Rendered(view=view, docx=docx, pdf=pdf, pages=page_count(pdf), layout=layout)


# ---------------------------------------------------------------------------
# Page fit
# ---------------------------------------------------------------------------


@dataclass
class TightenItem:
    key: str                 # "<entry_id>#<bullet_index>"
    entry_id: str
    text: str
    fact_ids: list[str]


RenderFn = Callable[[TailoredResume, LayoutParams], int]
TightenFn = Callable[[list[TightenItem]], dict[str, str]]
AcceptFn = Callable[[TailoredEntry, TailoredBullet, str], bool]


@dataclass
class FitStep:
    kind: str                # render | tighten | spacing | margins | font | remove_bullet | remove_project | ...
    detail: str
    pages: int


@dataclass
class FitResult:
    pages: int
    fits: bool
    layout: LayoutParams
    steps: list[FitStep] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def log(self) -> list[str]:
        return [f"{s.kind}: {s.detail} -> {s.pages} page(s)" for s in self.steps]


def _descending(start: float, stop: float, step: float) -> list[float]:
    """start-step, start-2*step, ... down to (and including) stop."""
    values, current = [], start
    while current - step > stop + 1e-9:
        current = round(current - step, 4)
        values.append(current)
    if start > stop + 1e-9:
        values.append(stop)
    return values


def fit_to_page(resume: TailoredResume, master: MasterResume, render_fn: RenderFn, *, relevance: Relevance,
                tighten_fn: TightenFn | None = None, accept_fn: AcceptFn | None = None,
                template_margin_in: float | None = None, template_font_pt: float | None = None,
                max_pages: int | None = None) -> FitResult:
    """Make the resume fit, mutating `resume` (tightened text, removed content).

    `render_fn(resume, layout)` renders and returns the page count. Steps run
    strictly in the order the spec gives, re-rendering after each one, and stop
    as soon as the resume fits.
    """
    max_pages = max_pages or config.MAX_PAGES
    margin0 = template_margin_in if template_margin_in is not None else config.DEFAULT_MARGIN_IN
    font0 = template_font_pt if template_font_pt is not None else config.DEFAULT_FONT_PT
    layout = LayoutParams()
    result = FitResult(pages=0, fits=False, layout=layout)

    def attempt(kind: str, detail: str) -> bool:
        result.pages = render_fn(resume, layout)
        result.steps.append(FitStep(kind, detail, result.pages))
        result.fits = result.pages <= max_pages
        return result.fits

    def budget_left() -> bool:
        return len(result.steps) < config.PAGE_FIT_MAX_STEPS

    if attempt("render", "initial layout"):
        return result

    # Step 1: tighten the longest bullets (Claude), accepted only if re-validation is clean.
    if tighten_fn is not None:
        candidates = sorted(
            ((entry, index, bullet) for entry, index, bullet in resume.bullets()
             if len(bullet.text) >= config.TIGHTEN_MIN_CHARS),
            key=lambda item: -len(item[2].text),
        )[: config.TIGHTEN_BULLET_COUNT]
        if candidates:
            items = [TightenItem(f"{e.source_id}#{i}", e.source_id, b.text, list(b.fact_ids)) for e, i, b in candidates]
            tightened = tighten_fn(items) or {}
            accepted = 0
            for entry, index, bullet in candidates:
                new_text = textutil.plain(tightened.get(f"{entry.source_id}#{index}", ""))
                if not new_text or len(new_text) >= len(bullet.text) or len(new_text) < 0.4 * len(bullet.text):
                    continue
                if accept_fn is not None and not accept_fn(entry, bullet, new_text):
                    bullet.history.append("page-fit: a tightened version was rejected by validation")
                    continue
                bullet.history.append(f'page-fit: tightened from "{bullet.text}"')
                bullet.final_text = new_text
                accepted += 1
            if accepted and attempt("tighten", f"tightened {accepted} of {len(candidates)} long bullets"):
                return result
            if not accepted:
                result.steps.append(FitStep("tighten", "no tightened bullet passed validation", result.pages))

    # Step 2: spacing, then margins, then font - never past the ATS-safe limits.
    for scale in _descending(1.0, config.MIN_SPACING_SCALE, config.SPACING_STEP):
        if not budget_left():
            break
        layout.spacing_scale = scale
        if attempt("spacing", f"paragraph spacing x{scale:.1f}"):
            return result
    floor_margin = min(margin0, config.MIN_MARGIN_IN)
    for margin in _descending(margin0, floor_margin, config.MARGIN_STEP_IN):
        if not budget_left():
            break
        layout.margin_in = margin
        if attempt("margins", f'margins {margin:.2f}" (limit {config.MIN_MARGIN_IN:.2f}")'):
            return result
    floor_font = min(font0, float(config.MIN_FONT_PT))
    for size in _descending(font0, floor_font, config.FONT_STEP_PT):
        if not budget_left():
            break
        layout.font_scale = size / font0
        if attempt("font", f"body font {size:.1f}pt (limit {config.MIN_FONT_PT}pt)"):
            return result

    # Step 3: remove the lowest-relevance content, one item at a time.
    def entry_score(entry: TailoredEntry) -> float:
        source = master.entry(entry.source_id)
        return relevance.score_entry(source) if source is not None else 0.0

    while budget_left():
        removable = [(entry, index, bullet) for entry in resume.entries() for index, bullet in enumerate(entry.bullets)
                     if len(entry.bullets) > config.MIN_BULLETS_PER_ENTRY]
        if not removable:
            break
        entry, index, bullet = min(removable, key=lambda item: (relevance.score_text(item[2].text),
                                                                entry_score(item[0]), -item[1]))
        entry.bullets.pop(index)
        note = f'bullet from {entry.source_id}: "{bullet.text}"'
        result.removed.append(note)
        if attempt("remove_bullet", f"removed lowest-relevance {note}"):
            return result
    while budget_left() and resume.projects:
        project = min(resume.projects, key=entry_score)
        resume.projects.remove(project)
        note = f"project {project.source_id}"
        result.removed.append(note)
        if attempt("remove_project", f"removed lowest-relevance {note}"):
            return result
    # Beyond the spec's list, as a last resort: awards, then whole entries (never the last experience).
    while budget_left() and resume.award_ids:
        award = resume.award_ids.pop()
        result.removed.append(f"award {award}")
        if attempt("remove_award", f"removed award {award}"):
            return result
    while budget_left():
        sections = {e.source_id: getattr(master.entry(e.source_id), "section_name", "experience")
                    for e in resume.experiences}
        core = [e for e in resume.experiences if sections[e.source_id] == "experience"]
        extra = [e for e in resume.experiences if sections[e.source_id] != "experience"]
        pool = extra or (core if len(core) > 1 else [])
        if not pool:
            break
        victim = min(pool, key=entry_score)
        resume.experiences.remove(victim)
        result.removed.append(f"entry {victim.source_id}")
        if attempt("remove_entry", f"removed lowest-relevance entry {victim.source_id}"):
            return result
    return result


# ---------------------------------------------------------------------------
# Claude-backed tightening and its validation gate (used by loop.py)
# ---------------------------------------------------------------------------

TIGHTEN_USER = """The resume runs slightly over one page. Tighten the wording of the bullets below so each is shorter.

Keep the meaning exactly: the same facts, the same numbers written the same way, the same skills and tools, and the same ownership wording ("assisted", "helped", "led" stay as they are). Only remove filler and wordy phrasing. If a bullet cannot be shortened without changing what it claims, return it unchanged. Keep each `key` and its `fact_ids` as given.

{items}"""


def make_tighten_fn(llm, system: str) -> TightenFn:
    """Step 1 of page fitting: one Claude call that shortens the longest bullets."""
    import json

    from src.errors import LLMError
    from src.models import BulletFixBatch

    def tighten(items: list[TightenItem]) -> dict[str, str]:
        payload = [{"key": i.key, "text": i.text, "fact_ids": i.fact_ids} for i in items]
        try:
            batch = llm.parse(purpose="tighten", model=config.WRITE_MODEL, system=system,
                              user=TIGHTEN_USER.format(items=json.dumps(payload, indent=1, ensure_ascii=False)),
                              schema=BulletFixBatch)
        except LLMError:
            return {}
        return {fix.key: fix.rewritten_text for fix in batch.fixes}

    return tighten


def make_accept_fn(ctx) -> AcceptFn:
    """A tightened bullet is accepted only if it raises nothing the current text does not."""
    from src import validate
    from src.models import Severity

    def accept(entry: TailoredEntry, bullet: TailoredBullet, new_text: str) -> bool:
        before = {(v.code, v.token.lower()) for v in
                  validate.recheck(bullet.text, entry.source_id, bullet.fact_ids, bullet.source_bullet_id, ctx)}
        after = validate.recheck(new_text, entry.source_id, bullet.fact_ids, bullet.source_bullet_id, ctx)
        if any(v.severity == Severity.hard for v in after):
            return False
        return {(v.code, v.token.lower()) for v in after} <= before

    return accept
