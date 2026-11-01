"""Zero-intervention mode: `tailor_for_job()` for an app, and `python tailor.py --auto`.

An app that tailors automatically has no one to answer a prompt, so this mode:

* never prompts: no paste fallback (a blocked link is an error), the
  eligibility check continues and records its flags, and there is no interview;
* uses a rewrite only if the validator raised nothing at all. A rewrite whose
  only reason for review is that it is a substantial rewording is used
  (approval `accepted`); anything a check flagged reverts to the approved
  original wording, and a flagged cover-letter sentence is removed;
* always writes the cover letter, and report.json next to the resume.

The validator is the same one every run uses (src/validate.py), so the
anti-fabrication guarantee does not change: no employer, title, date, number,
skill, tool, scope, or ownership level the fact bank does not support can reach
the page. What --auto gives up is the human look at substantial rewordings,
so read the resume before you send it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from rich.console import Console

import config
from src import loop, startup, ui
from src.errors import TailorError

_URL_RE = re.compile(r"https?://\S+")


def _never_prompt(prompt: str = "") -> str:
    raise EOFError("--auto never prompts")


def default_out_dir() -> Path:
    """Where `--auto` writes when no --out is given: output/auto/<timestamp>/."""
    return config.OUTPUT_DIR / "auto" / datetime.now().strftime("%Y-%m-%d_%H%M%S")


# Live check (Aquatic SWE Intern, example data): rounds 5.60, 5.35, 5.96, 5.91, 5.88 - the best draft
# came in round 2 and three more rounds added a minute without beating it. Apps wait on this per
# application, so auto mode stops after two revision rounds unless told otherwise.
AUTO_MAX_ROUNDS = 2


def tailor_for_job(url_or_posting_text: str, out_dir: str | Path, master_path: str | Path | None = None,
                   skills_path: str | Path | None = None, data_dir: str | Path | None = None,
                   max_rounds: int = AUTO_MAX_ROUNDS) -> dict:
    """Tailor a resume and cover letter for one posting without asking anything.

    `url_or_posting_text` is either a link (fetched once) or the posting's text.
    Every file is written to `out_dir`. `master_path` / `skills_path` choose the
    fact bank and skills file, and `data_dir` the folder for postings/, cache/,
    backups/, and applications.csv (one row is appended per run); each defaults
    to this repo's own files.

    Returns the contents of report.json:
        ok, company, role, url, out_dir,
        files:  resume_pdf, resume_docx, cover_letter_txt, cover_letter_pdf, cover_letter_docx,
                cover_letter_md, scores, report_json, ... (absolute paths)
        scores: resume_quality, target, categories, job_match, keyword_coverage, ats, rounds
        bullets: total, reworded, verbatim, reverted
        eligibility_flags, stop_note, seconds, timings, ...
    On an expected failure (blocked link, usage limit, bad data file) it returns
    {"ok": False, "error": ..., "hint": ..., "out_dir": ..., "files": {"report_json": ...}}
    instead of raising. Progress goes to stderr.
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    text = (url_or_posting_text or "").strip()
    is_url = bool(_URL_RE.fullmatch(text))
    saved_paths = config.use_paths()                    # no arguments: only reads the current values
    saved_console = ui.console
    ui.console = Console(stderr=True, highlight=False)
    ui.set_input(_never_prompt, interactive=False)
    try:
        session = startup.run_startup(need_api=True, quiet=False,
                                      paths={"master": master_path, "skills": skills_path, "data_dir": data_dir})
        options = loop.RunOptions(auto=True, interview=False, cover_letter=True, research=True,
                                  max_rounds=max_rounds)
        loop.run_job(session, url=text if is_url else None, posting_text=None if is_url else text,
                     options=options, out_dir=out)
        return json.loads((out / "report.json").read_text(encoding="utf-8"))
    except TailorError as exc:
        ui.error(exc.message, exc.hint)
        report = out / "report.json"
        result = {"ok": False, "error": exc.message, "hint": exc.hint, "out_dir": str(out),
                  "files": {"report_json": str(report)}}
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    finally:
        config.restore_paths(saved_paths)
        ui.console = saved_console
        ui.set_input(None, interactive=None)
