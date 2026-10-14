#!/usr/bin/env python3
"""Preview: render a resume straight from master_resume.yaml - no API key, no Claude.

    python scripts/preview.py                 # most recent content first
    python scripts/preview.py job.txt         # rank content against a saved job description

Useful for checking the template after editing it in Word, and for seeing how
your fact bank looks on the page before your first real run. It uses only your
approved bullets (or plain fact text), fits them to one page with the same
page-fit steps as a real run (minus the Claude tightening step), and runs the
ATS checks. Output goes to output/_preview/.

With a job file, relevance is a plain keyword match against the tool lexicon,
which is much cruder than a real run's analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import ats_check, render, store, tailor_resume, textutil, ui, validate  # noqa: E402
from src.errors import TailorError  # noqa: E402
from src.models import JobAnalysis, Requirements, WeightedSkill  # noqa: E402
from src.validate import ValidationContext  # noqa: E402


def crude_analysis(text: str, status) -> JobAnalysis:
    """Keyword-only stand-in for analyze.py: lexicon and tracked skills found in the posting."""
    vocabulary = [*config.TECH_LEXICON, *status.skills]
    found = textutil.find_terms(text, vocabulary)
    skills = [WeightedSkill(name=surface, weight=min(3, textutil.count_term(text, surface)), kind="tool")
              for surface in found.values()]
    return JobAnalysis(company="Preview", role="Preview", location="", job_type="other", required_skills=skills,
                       preferred_skills=[], tools=[], key_phrases=[], responsibilities=[],
                       keywords=list(found.values()), requirements=Requirements.empty())


def main(argv: list[str]) -> int:
    master, status = store.load_master_resume(), store.load_skills_status()
    analysis = None
    if len(argv) > 1:
        path = Path(argv[1])
        if not path.exists():
            raise TailorError(f"Job description file not found: {path}")
        analysis = crude_analysis(path.read_text(encoding="utf-8", errors="replace"), status)
        ui.info(f"Ranking against {len(analysis.keywords)} keyword(s) found in {path.name}: "
                + ", ".join(analysis.keywords[:12]))
    ctx = ValidationContext(master=master, status=status, analysis=analysis)
    outcome = validate.enforce(tailor_resume.baseline_draft(ctx), ctx)
    resume = outcome.resume
    out_dir = config.OUTPUT_DIR / "_preview"
    margin, font = render.template_info()
    fit = render.fit_to_page(resume, master, lambda r, layout: render.render(r, master, out_dir, layout).pages,
                             relevance=ctx.relevance, template_margin_in=margin, template_font_pt=font)
    final = render.render(resume, master, out_dir, fit.layout)
    report = ats_check.check_rendered(final.view, final.pdf, final.docx, analysis)
    ui.success(f"{final.pages} page(s), {sum(len(e.bullets) for e in resume.entries())} bullets, "
               f"layout {fit.layout.describe(font, margin)}")
    for note in fit.removed:
        ui.info(f"  removed to fit: {note}")
    ui.info(f"ATS checks: {report.score:.1f} / 10")
    for check in report.failures():
        ui.warn(f"{check.name}: {check.detail}")
    ui.info(f"Open: {final.pdf}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except TailorError as exc:
        ui.error(exc.message, exc.hint)
        sys.exit(1)
