"""Resume version archive: a snapshot of exactly what was produced for each posting.

Every run saves into its output folder:

    posting.txt                    the job posting text (the link is in archive.json)
    master_resume.snapshot.yaml    master_resume.yaml exactly as it was for this run
    resume.docx / resume.pdf       the final resume (written by render.py)
    <Name>_Resume.pdf              the same PDF under a filename fit to send
    tailored_resume.json           every bullet with its facts and approval state
    scores.json                    the full score breakdown and round history
    company_brief.json             the research brief with source tiers
    eligibility.json               the eligibility table
    usage.json                     token usage and estimated cost per Claude call
    archive.json                   an index of the above (company, role, link, date)

`python tailor.py --find "Company"` lists matching archives, so months later I
can pull up exactly what I sent.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

import config
from src import textutil, ui

if TYPE_CHECKING:
    from src.llm import UsageTracker
    from src.loop import RunState

INDEX_NAME = "archive.json"


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sendable_name(name: str) -> str:
    """"Deveshwar Vankeswaram" -> "Deveshwar_Vankeswaram_Resume.pdf"."""
    return f"{textutil.slug(name, 60) if name.strip() else 'Resume'}_Resume.pdf".replace("Resume_Resume", "Resume")


def write_archive(state: RunState, tracker: UsageTracker | None = None) -> Path:
    """Save the snapshot files and the index. Returns the path of archive.json."""
    assert state.resume is not None and state.final_quality is not None
    out = state.out_dir
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    (out / "posting.txt").write_text(state.job.text, encoding="utf-8")
    files["posting"] = "posting.txt"

    snapshot = out / "master_resume.snapshot.yaml"
    if config.MASTER_RESUME_PATH.exists():
        shutil.copy2(config.MASTER_RESUME_PATH, snapshot)
    else:
        snapshot.write_text(yaml.safe_dump(state.ctx.master.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
                            encoding="utf-8")
    files["master_resume_snapshot"] = snapshot.name

    if state.rendered is not None:
        files["resume_docx"], files["resume_pdf"] = state.rendered.docx.name, state.rendered.pdf.name
        named = out / sendable_name(state.ctx.master.contact.name)
        if state.rendered.pdf.exists() and named.name != state.rendered.pdf.name:
            shutil.copy2(state.rendered.pdf, named)
            files["resume_pdf_to_send"] = named.name

    _write_json(out / "tailored_resume.json", state.resume.model_dump(mode="json"))
    files["tailored_resume"] = "tailored_resume.json"

    scores = {
        "resume_quality": state.final_quality.model_dump(mode="json"),
        "target": state.options.target,
        "stop_note": state.stop_note,
        "rounds": [
            {"phase": phase, "round": record.round, "quality": record.quality.model_dump(mode="json"),
             "judgment": record.judgment.model_dump(mode="json"), "note": record.note}
            for phase, result in enumerate(state.loops, 1) for record in result.history
        ],
        "final_judgment": state.final_judgment.model_dump(mode="json") if state.final_judgment else None,
        "job_match": state.job_match.model_dump(mode="json") if state.job_match else None,
        "keyword_coverage": {
            "master": state.coverage_master.model_dump(mode="json") if state.coverage_master else None,
            "final": state.coverage_final.model_dump(mode="json") if state.coverage_final else None,
        },
        "ats": state.ats.model_dump(mode="json") if state.ats else None,
    }
    _write_json(out / "scores.json", scores)
    files["scores"] = "scores.json"

    _write_json(out / "company_brief.json", state.brief.model_dump(mode="json") if state.brief else None)
    files["company_brief"] = "company_brief.json"
    _write_json(out / "eligibility.json", state.eligibility.model_dump(mode="json"))
    files["eligibility"] = "eligibility.json"
    _write_json(out / "job_analysis.json", state.analysis.model_dump(mode="json"))
    files["job_analysis"] = "job_analysis.json"

    if tracker is not None:
        _write_json(out / "usage.json", {
            "total_cost_usd": round(tracker.total_cost(), 6),
            "calls": [call.model_dump(mode="json") for call in tracker.calls],
        })
        files["usage"] = "usage.json"

    index = {
        "company": state.analysis.company,
        "role": state.analysis.role,
        "url": state.job.url,
        "date": state.run_date.isoformat(),
        "mode": "strict" if state.options.strict else "normal",
        "resume_quality": state.final_quality.total,
        "job_match": state.job_match.score if state.job_match else None,
        "keyword_coverage": state.coverage_final.score if state.coverage_final else None,
        "pages": state.rendered.pages if state.rendered else None,
        "eligibility_flags": state.eligibility.flags(),
        "files": files,
    }
    path = out / INDEX_NAME
    _write_json(path, index)
    return path


# ---------------------------------------------------------------------------
# --find
# ---------------------------------------------------------------------------


def find(query: str, root: Path | None = None) -> list[dict]:
    """Archived submissions whose company or role matches `query` (newest first)."""
    root = root or config.OUTPUT_DIR
    wanted = textutil.squash(query)
    matches: list[dict] = []
    for index_path in sorted(root.glob(f"*/{INDEX_NAME}")) if root.exists() else []:
        try:
            entry = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        haystack = textutil.squash(f"{entry.get('company', '')} {entry.get('role', '')}")
        if not wanted or wanted in haystack:
            entry["folder"] = str(index_path.parent)
            matches.append(entry)
    return sorted(matches, key=lambda e: (e.get("date", ""), e.get("folder", "")), reverse=True)


def print_matches(query: str, root: Path | None = None) -> list[dict]:
    matches = find(query, root)
    if not matches:
        ui.info(f'No archived submissions match "{query}".')
        return matches
    rows = []
    for entry in matches:
        files = entry.get("files", {})
        resume = files.get("resume_pdf_to_send") or files.get("resume_pdf") or ""
        rows.append([entry.get("date", ""), entry.get("company", ""), entry.get("role", ""),
                     f"{entry.get('resume_quality', 0):.2f}", str(Path(entry["folder"]) / resume)])
    ui.table(f'Archived submissions matching "{query}"', ["Date", "Company", "Role", "Quality", "Resume I sent"], rows)
    ui.info("Each folder also holds the posting, the master resume snapshot, changes.md, and scores.md.")
    return matches
