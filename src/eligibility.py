"""Requirement-by-requirement eligibility check.

Stated requirements are compared **in code** against the master resume. Claude
is used only to interpret vague wording ("rising juniors"), and anything it
interpreted is labeled "interpreted" in the table.

The check flags; it never auto-rejects. The table is shown first, then I am
asked whether to continue. Batch mode continues and puts the flags at the top
of the report.
"""

from __future__ import annotations

import re
from datetime import date

import config
from src import textutil, ui
from src.analyze import quote_in_text
from src.errors import LLMError
from src.models import (
    ClassStanding,
    EligibilityReport,
    EligibilityRow,
    EligStatus,
    InterpretedRequirements,
    JobAnalysis,
    MasterResume,
    SkillKind,
    SkillsStatus,
    SkillStatus,
    WorkAuthKind,
)

_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august",
                "september", "october", "november", "december"]

INTERPRET_SYSTEM = """You interpret vaguely worded eligibility requirements from an internship posting.

You are given today's date and one or more sentences copied from the posting. Work out what they require:
- grad_earliest / grad_latest: the graduation window they imply, as "YYYY" or "YYYY-MM". Null when the sentences do not imply one.
- class_standing_allowed: the class standings a candidate must hold during the current school year (the school year that contains today's date). Example: for an internship next summer, "rising juniors" means students who will be juniors in the fall after the internship, so they are sophomores during the current school year. "Penultimate-year students" in a four-year program are juniors during the school year before their final year.
- work_authorization: citizen_required, us_person_required, no_sponsorship, sponsorship_available, or other - only when one of the sentences is about work authorization. Otherwise null.
- notes: one sentence explaining your reading.

Only interpret what the sentences support. If a sentence cannot be read with confidence, return null or an empty list for that part and say so in notes."""


# ---------------------------------------------------------------------------
# Facts about me, derived from the master resume
# ---------------------------------------------------------------------------


def candidate_grad(master: MasterResume) -> tuple[int, int] | None:
    """(year, month) I expect to graduate, from the first education entry."""
    for edu in master.education:
        if edu.class_year:
            month = config.DEFAULT_GRAD_MONTH
            for index, name in enumerate(_MONTH_NAMES):
                if re.search(rf"\b{name[:3]}", edu.grad.lower()):
                    month = index + 1
                    break
            return (edu.class_year, month)
    return None


def current_standing(class_year: int, today: date) -> ClassStanding:
    """My class standing during the school year that contains `today`."""
    year_end = today.year + 1 if today.month >= config.ACADEMIC_YEAR_ROLLOVER_MONTH else today.year
    years_left = class_year - year_end
    if years_left >= 3:
        return ClassStanding.freshman
    return {2: ClassStanding.sophomore, 1: ClassStanding.junior, 0: ClassStanding.senior}.get(
        years_left, ClassStanding.graduate
    )


def candidate_authorization(text: str) -> str:
    """"citizen" | "permanent_resident" | "visa" | "unknown" from the free-text field."""
    lowered = (text or "").lower()
    if not lowered.strip():
        return "unknown"
    if re.search(r"permanent resident|green card|\blpr\b", lowered):
        return "permanent_resident"
    if "citizen" in lowered and not re.search(r"\b(not|non)[- ]?(a )?(u\.?s\.? )?citizen", lowered):
        return "citizen"
    if re.search(r"\bf-?1\b|\bopt\b|\bcpt\b|\bh-?1b\b|\bj-?1\b|visa|international student", lowered):
        return "visa"
    return "unknown"


def _bound(value: str | None, *, end: bool) -> tuple[int, int] | None:
    if not value:
        return None
    parsed = textutil.parse_ym(str(value).strip(), end=end)
    return parsed if parsed[0] else None


def _fmt_ym(ym: tuple[int, int]) -> str:
    return f"{_MONTH_NAMES[ym[1] - 1][:3].title()} {ym[0]}"


def _worst(statuses: list[EligStatus]) -> EligStatus:
    for status in (EligStatus.fails, EligStatus.unknown, EligStatus.partial, EligStatus.meets):
        if status in statuses:
            return status
    return EligStatus.not_stated


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_grad_and_standing(analysis: JobAnalysis, master: MasterResume, today: date,
                            interpreted: InterpretedRequirements | None = None) -> EligibilityRow:
    grad_req, standing_req = analysis.requirements.grad_year_range, analysis.requirements.class_standing
    earliest, latest = grad_req.earliest, grad_req.latest
    allowed = list(standing_req.allowed)
    was_interpreted = False
    if interpreted is not None:
        if not (earliest or latest) and (interpreted.grad_earliest or interpreted.grad_latest):
            earliest, latest, was_interpreted = interpreted.grad_earliest, interpreted.grad_latest, True
        if not allowed and interpreted.class_standing_allowed:
            allowed, was_interpreted = list(interpreted.class_standing_allowed), True

    sources = [s for s in dict.fromkeys([grad_req.source, standing_req.source]) if s]
    row = EligibilityRow(requirement="Graduation year / class standing", status=EligStatus.not_stated,
                         evidence=" | ".join(sources), interpreted=was_interpreted)
    if not (earliest or latest or allowed):
        if sources:
            row.status = EligStatus.unknown
            row.detail = "the posting mentions this but it could not be compared automatically"
        return row

    grad = candidate_grad(master)
    if grad is None:
        row.status, row.detail = EligStatus.unknown, "class_year is missing from master_resume.yaml"
        return row

    statuses: list[EligStatus] = []
    details: list[str] = []
    low, high = _bound(earliest, end=False), _bound(latest, end=True)
    if low or high:
        ok = (low is None or grad >= low) and (high is None or grad <= high)
        window = " to ".join(str(v) for v in (earliest, latest) if v) if (earliest and latest) else (
            f"{earliest} or later" if earliest else f"{latest} or earlier")
        statuses.append(EligStatus.meets if ok else EligStatus.fails)
        details.append(f"I graduate {_fmt_ym(grad)}; posting wants {window}")
    if allowed:
        mine = current_standing(grad[0], today)
        statuses.append(EligStatus.meets if mine in allowed else EligStatus.fails)
        details.append(f"I am a {mine.value} this school year; posting wants {', '.join(s.value for s in allowed)}")
    row.status = _worst(statuses)
    row.detail = "; ".join(details)
    return row


def _field_families(field: str) -> set[str]:
    # A family matches when one of its fields appears as whole words in the text
    # ("B.S. in Computer Science" -> computing). The reverse is deliberately not
    # allowed: "Finance" must not match the quantitative family just because
    # "quantitative finance" is one of its members.
    wanted = textutil.normalize(field)
    return {
        family for family, members in config.FIELD_FAMILIES.items()
        if any(re.search(rf"(?<![a-z]){re.escape(member)}(?![a-z])", wanted) for member in members)
    }


def check_degree(analysis: JobAnalysis, master: MasterResume) -> EligibilityRow:
    req = analysis.requirements.degree_fields
    row = EligibilityRow(requirement="Degree field", status=EligStatus.not_stated, evidence=req.source or "")
    fields = [f.strip() for f in req.fields if f.strip()]
    if not fields:
        if req.source:
            row.status, row.detail = EligStatus.unknown, "no specific fields could be read from the posting"
        return row
    mine = [text for edu in master.education for text in (edu.major, edu.degree) if text.strip()]
    if not mine:
        row.status, row.detail = EligStatus.unknown, "no degree or major in master_resume.yaml"
        return row
    label = next((edu.major for edu in master.education if edu.major.strip()), mine[0])
    my_text = [textutil.normalize(m) for m in mine]
    my_families: set[str] = set().union(*[_field_families(m) for m in mine])

    for field in fields:                      # 1. the posting names my field
        wanted = textutil.normalize(field)
        if any(wanted == m or (len(wanted) > 4 and wanted in m) or (len(m) > 4 and m in wanted) for m in my_text):
            row.status, row.detail = EligStatus.meets, f'{label} matches "{field}"'
            return row
    for field in fields:                      # 2. umbrella terms ("STEM", "technical field")
        wanted = textutil.normalize(field)
        for umbrella, families in config.UMBRELLA_FIELDS.items():
            if re.search(rf"\b{umbrella}\b", wanted) and my_families & set(families):
                row.status, row.detail = EligStatus.meets, f'{label} counts as "{field}"'
                return row
    related = [f for f in fields if _field_families(f) & my_families]
    if related and req.related_fields_ok:     # 3. "or related field"
        row.status = EligStatus.meets
        row.detail = f"{label} is a related field to {related[0]} (posting accepts related fields)"
    elif related:
        row.status = EligStatus.partial
        row.detail = f"{label} is not listed; closest listed field is {related[0]}"
    elif req.related_fields_ok:
        row.status = EligStatus.unknown
        row.detail = f"{label} is not listed; posting accepts related fields ({', '.join(fields)})"
    else:
        row.status = EligStatus.fails
        row.detail = f"{label} is not among: {', '.join(fields)}"
    return row


def check_gpa(analysis: JobAnalysis, master: MasterResume) -> EligibilityRow:
    req = analysis.requirements.min_gpa
    row = EligibilityRow(requirement="GPA", status=EligStatus.not_stated, evidence=req.source or "")
    if req.value is None:
        if req.source:
            row.status, row.detail = EligStatus.unknown, "a GPA requirement is mentioned but no number was found"
        return row
    mine = next((edu.gpa for edu in master.education if edu.gpa is not None), None)
    if mine is None:
        row.status, row.detail = EligStatus.unknown, f"posting wants {req.value:.2f}+; no GPA in master_resume.yaml"
    elif mine + 1e-9 >= req.value:
        row.status, row.detail = EligStatus.meets, f"my GPA {mine:.2f} vs. minimum {req.value:.2f}"
    else:
        row.status, row.detail = EligStatus.fails, f"my GPA {mine:.2f} is below the minimum {req.value:.2f}"
    return row


def check_work_authorization(analysis: JobAnalysis, master: MasterResume,
                             interpreted: InterpretedRequirements | None = None) -> EligibilityRow:
    req = analysis.requirements.work_authorization
    kind, was_interpreted = req.kind, False
    if kind == WorkAuthKind.other and interpreted is not None and interpreted.work_authorization not in (None, WorkAuthKind.other):
        kind, was_interpreted = interpreted.work_authorization, True
    row = EligibilityRow(requirement="Work authorization", status=EligStatus.not_stated,
                         evidence=req.source or "", interpreted=was_interpreted)
    if kind == WorkAuthKind.not_stated:
        return row
    mine = candidate_authorization(master.work_authorization)
    described = master.work_authorization.strip() or "not set in master_resume.yaml"
    if kind == WorkAuthKind.sponsorship_available:
        row.status, row.detail = EligStatus.meets, "sponsorship is available"
    elif kind == WorkAuthKind.other:
        row.status, row.detail = EligStatus.unknown, f"unusual wording; mine: {described}"
    elif mine == "unknown":
        row.status, row.detail = EligStatus.unknown, f"mine: {described}"
    elif kind == WorkAuthKind.citizen_required:
        ok = mine == "citizen"
        row.status = EligStatus.meets if ok else EligStatus.fails
        row.detail = f"U.S. citizenship required; mine: {described}"
    elif kind == WorkAuthKind.us_person_required:
        ok = mine in ("citizen", "permanent_resident")
        row.status = EligStatus.meets if ok else EligStatus.fails
        row.detail = f"U.S. person (citizen or permanent resident) required; mine: {described}"
    elif kind == WorkAuthKind.no_sponsorship:
        if mine in ("citizen", "permanent_resident"):
            row.status, row.detail = EligStatus.meets, f"no sponsorship needed; mine: {described}"
        else:
            row.status, row.detail = EligStatus.unknown, f"posting does not sponsor; mine: {described}"
    return row


def assessable(name: str, kind: SkillKind, status: SkillsStatus) -> bool:
    """Soft skills are only assessed when I have explicitly listed them."""
    return kind != SkillKind.soft or status.status_of(name) is not None


def check_required_skills(analysis: JobAnalysis, status: SkillsStatus) -> tuple[EligibilityRow, list[str], list[str]]:
    row = EligibilityRow(requirement="Required skills", status=EligStatus.not_stated)
    skills = [s for s in analysis.required_skills if assessable(s.name, s.kind, status)]
    if not skills:
        return row, [], []
    have = [s.name for s in skills if status.status_of(s.name) == SkillStatus.verified]
    learning = [s.name for s in skills if status.status_of(s.name) == SkillStatus.learning]
    missing = [s.name for s in skills if s.name not in have and s.name not in learning]
    if not missing and not learning:
        row.status = EligStatus.meets
        row.detail = f"all {len(skills)} verified: {', '.join(have)}"
    else:
        row.status = EligStatus.partial if have else EligStatus.fails
        parts = [f"{len(have)} of {len(skills)} verified"]
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if learning:
            parts.append("learning: " + ", ".join(learning))
        row.detail = "; ".join(parts)
    return row, missing, learning


def check_prior_experience(analysis: JobAnalysis, master: MasterResume) -> EligibilityRow:
    req = analysis.requirements.prior_experience_required
    row = EligibilityRow(requirement="Prior internship/research experience",
                         status=EligStatus.not_stated, evidence=req.source or "")
    if not req.required:
        if req.source and req.required is False:
            row.detail = "not required"
        return row
    kind = (req.kind.value if req.kind else "any")
    wanted_types = config.PRIOR_EXPERIENCE_TYPES.get(kind, config.PRIOR_EXPERIENCE_TYPES["any"])
    matches = [e for e in master.experience if e.type.value in wanted_types]
    if matches:
        names = ", ".join(f"{e.title or e.type.value} at {e.org}" for e in matches[:3])
        row.status, row.detail = EligStatus.meets, f"{len(matches)} matching role(s): {names}"
    else:
        row.status, row.detail = EligStatus.fails, f"no {kind} experience in master_resume.yaml"
    return row


# ---------------------------------------------------------------------------
# Interpretation of vague wording (the only Claude call in this module)
# ---------------------------------------------------------------------------


def vague_sentences(analysis: JobAnalysis) -> list[str]:
    """Sentences the posting states but that carry no structured value yet."""
    req = analysis.requirements
    out: list[str] = []
    if req.grad_year_range.source and not (req.grad_year_range.earliest or req.grad_year_range.latest):
        out.append(req.grad_year_range.source)
    if req.class_standing.source and not req.class_standing.allowed:
        out.append(req.class_standing.source)
    if req.work_authorization.source and req.work_authorization.kind == WorkAuthKind.other:
        out.append(req.work_authorization.source)
    return list(dict.fromkeys(out))


def interpret_vague(llm, analysis: JobAnalysis, today: date) -> InterpretedRequirements | None:
    sentences = vague_sentences(analysis)
    if not sentences or llm is None:
        return None
    listing = "\n".join(f"- {s}" for s in sentences)
    try:
        return llm.parse(
            purpose="interpret",
            model=config.FAST_MODEL,
            system=INTERPRET_SYSTEM,
            user=f"Today's date: {today.isoformat()}\nRole: {analysis.role} at {analysis.company}\n\nSentences:\n{listing}",
            schema=InterpretedRequirements,
            cache_system=False,
        )
    except LLMError as exc:
        ui.warn(f"Could not interpret vague requirements ({exc.message}); marking them Unknown.")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_eligibility(analysis: JobAnalysis, master: MasterResume, status: SkillsStatus, *,
                      llm=None, posting_text: str = "", today: date | None = None) -> EligibilityReport:
    today = today or date.today()
    interpreted = interpret_vague(llm, analysis, today)
    skills_row, missing, learning = check_required_skills(analysis, status)
    rows = [
        check_grad_and_standing(analysis, master, today, interpreted),
        check_degree(analysis, master),
        check_gpa(analysis, master),
        check_work_authorization(analysis, master, interpreted),
        skills_row,
        check_prior_experience(analysis, master),
    ]
    for other in analysis.requirements.other:
        rows.append(EligibilityRow(
            requirement=f"Other: {other.text}", status=EligStatus.unknown,
            evidence=other.source or "", detail="not checked automatically - confirm this yourself",
        ))
    if posting_text:
        for row in rows:
            if row.evidence:
                quotes = [q for q in row.evidence.split(" | ") if q]
                row.quote_verified = all(quote_in_text(q, posting_text) for q in quotes)
    if interpreted is not None and interpreted.notes:
        for row in rows:
            if row.interpreted:
                row.detail = f"{row.detail} [interpreted: {interpreted.notes}]".strip()
    return EligibilityReport(rows=rows, missing_required_skills=missing, learning_required_skills=learning)


_STATUS_STYLE = {
    EligStatus.meets: "green", EligStatus.partial: "yellow", EligStatus.fails: "red",
    EligStatus.unknown: "yellow", EligStatus.not_stated: "dim",
}


def evidence_cell(row: EligibilityRow) -> str:
    """Plain-text evidence column: what was compared, then the quote."""
    parts = []
    if row.detail:
        parts.append(row.detail)
    if row.evidence:
        note = "" if row.quote_verified in (None, True) else " (quote not found verbatim in the posting)"
        parts.append(f'"{row.evidence}"{note}')
    return "\n".join(parts)


def show(report: EligibilityReport) -> None:
    rows = []
    for row in report.rows:
        label = row.status.value + (" (interpreted)" if row.interpreted else "")
        rows.append([row.requirement, f"[{_STATUS_STYLE[row.status]}]{label}[/]", evidence_cell(row)])
    ui.table("Eligibility", ["Requirement", "Status", "Evidence"], rows)


def confirm_continue(report: EligibilityReport, *, batch: bool = False) -> bool:
    """Flag, never auto-reject: show the table, then let me decide."""
    show(report)
    flags = report.flags()
    if batch:
        if flags:
            ui.warn(f"{len(flags)} eligibility flag(s); continuing (batch mode). See eligibility.md.")
        return True
    if flags:
        ui.warn(f"{len(flags)} eligibility flag(s) above.")
    return ui.ask_yes_no("Continue tailoring?", default=True)
