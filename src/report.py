"""Reports written into the run's output folder, plus the applications log.

    eligibility.md   the requirement table (flags first)
    changes.md       per bullet: original vs. final, cited facts, change type,
                     approval decision, validation flags; the validation log;
                     page-fit steps and removals; interview facts
    scores.md        Resume Quality per round with category breakdowns, Job
                     Match, Keyword Coverage (master vs. final), ATS results,
                     Claude usage and cost
    gaps.md          real gaps, unmet requirements, open questions

and one row appended to data/applications.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import config
from src import eligibility
from src.models import Approval, EligStatus, FixType, Severity

if TYPE_CHECKING:
    from src.llm import UsageTracker
    from src.loop import RunState

CSV_FIELDS = ["date", "company", "role", "url", "quality", "job_match", "keyword_coverage",
              "eligibility_flags", "rounds", "output_folder", "status"]
INITIAL_STATUS = "drafted"      # I change it by hand to: applied / interview / offer / rejected


def cell(text: object) -> str:
    """Make a value safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", "<br>").strip()


def table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _header(state: RunState, title: str) -> str:
    link = f" - [posting]({state.job.url})" if state.job.url else ""
    return f"# {title}\n\n**{state.analysis.role}** at **{state.analysis.company}** - {state.run_date.isoformat()}{link}\n"


# ---------------------------------------------------------------------------
# eligibility.md
# ---------------------------------------------------------------------------


def eligibility_md(state: RunState) -> str:
    report = state.eligibility
    parts = [_header(state, "Eligibility")]
    flags = report.flags()
    if flags:
        parts.append("## Flags\n\n" + "\n".join(f"- **{flag}**" for flag in flags) + "\n")
        parts.append("_The tool flags; it never auto-rejects. Decide for yourself whether to apply._\n")
    else:
        parts.append("No eligibility concerns were found.\n")
    rows = [[r.requirement, r.status.value + (" (interpreted)" if r.interpreted else ""), eligibility.evidence_cell(r)]
            for r in report.rows]
    parts.append("## Requirements\n\n" + table(["Requirement", "Status", "Evidence"], rows) + "\n")
    if any(r.interpreted for r in report.rows):
        parts.append("_\"Interpreted\" means the posting's wording was vague and Claude read it; check the quote._\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# changes.md
# ---------------------------------------------------------------------------

_APPROVAL_LABEL = {
    Approval.auto: "auto-approved", Approval.pending: "PENDING", Approval.approved: "approved by me",
    Approval.edited: "edited by me", Approval.reverted: "reverted to verified wording",
    Approval.override: "kept by me although the facts do not support it",
}


def changes_md(state: RunState) -> str:
    assert state.resume is not None
    master = state.ctx.master
    parts = [_header(state, "Changes")]
    mode = "strict (approved wording unless I approve a rewrite)" if state.options.strict else "normal"
    parts.append(f"Mode: {mode}. Every bullet below cites the facts it is built from.\n")

    for entry in state.resume.entries():
        source = master.entry(entry.source_id)
        title = entry.source_id
        if source is not None:
            title = source.display_org + (f" - {source.display_title}" if source.display_title else "")
        parts.append(f"## {title}\n")
        if entry.replaced:
            parts.append(f"> **Replaced by the validator:** {entry.replaced}. Verified wording was used.\n")
        for number, bullet in enumerate(entry.bullets, 1):
            lines = [f"**{number}. {bullet.text}**", ""]
            if bullet.change_type.value != "unchanged" or bullet.approval != Approval.auto:
                lines.append(f"- Original ({bullet.original_kind.replace('_', ' ')}): {bullet.original_text}")
            if bullet.approval in (Approval.reverted,) and bullet.rewritten_text != bullet.text:
                lines.append(f"- Proposed rewrite (not used): {bullet.rewritten_text}")
            lines.append(f"- Change type: `{bullet.change_type.value}` - {_APPROVAL_LABEL[bullet.approval]}")
            lines.append(f"- Facts: {', '.join(f'`{f}`' for f in bullet.fact_ids) or 'none'}"
                         + (f" (from bullet `{bullet.source_bullet_id}`)" if bullet.source_bullet_id else ""))
            lines += [f"- Flag: {flag.message}" for flag in bullet.flags]
            lines += [f"- Note: {note}" for note in bullet.history]
            parts.append("\n".join(lines) + "\n")

    if state.decisions:
        rows = [[d.entry_id, d.action + (" (automatic)" if d.automatic else ""), d.original, d.proposed, d.final,
                 "; ".join(d.reasons), d.saved_as or ""] for d in state.decisions]
        parts.append("## Approval decisions\n\n" + table(
            ["Entry", "Decision", "Original", "Proposed", "Final", "Why it was flagged", "Saved as"], rows) + "\n")
    else:
        parts.append("## Approval decisions\n\nNo change needed approval.\n")

    hard = [v for v in state.violations if v.severity == Severity.hard]
    soft = [v for v in state.violations if v.severity == Severity.soft]
    parts.append("## Validation log\n")
    parts.append(f"{len(hard)} hard violation(s) were blocked and {len(soft)} soft flag(s) were raised "
                 f"across {sum(len(result.history) for result in state.loops) or 1} draft(s).\n")
    if state.violations:
        rows = [[v.code, v.severity.value, v.entry_id + (f" #{v.bullet_index + 1}" if v.bullet_index is not None else ""),
                 v.message, v.resolution] for v in state.violations]
        parts.append(table(["Check", "Severity", "Where", "What was wrong", "What was done"], rows) + "\n")

    if state.fit is not None:
        parts.append("## Page fit\n")
        parts.append("\n".join(f"{i}. {line}" for i, line in enumerate(state.fit.log(), 1)) + "\n")
        if state.fit.removed:
            parts.append("Removed to fit one page:\n\n" + "\n".join(f"- {item}" for item in state.fit.removed) + "\n")
        else:
            parts.append("Nothing was removed to fit the page.\n")

    if state.interview is not None and state.interview.asked:
        parts.append("## Interview\n")
        parts.append(f"{state.interview.asked} question(s) asked, {state.interview.skipped} skipped, "
                     f"{state.interview.rejected} rejected, {len(state.interview.new_fact_ids)} fact(s) added.\n")
        parts += [f"- {line}" for line in state.interview.log]
        if state.interview.backup:
            parts.append(f"\nBackup of the master resume before the write: `{state.interview.backup.name}`\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# scores.md
# ---------------------------------------------------------------------------


def scores_md(state: RunState, tracker: UsageTracker | None = None) -> str:
    assert state.final_quality and state.job_match and state.coverage_final and state.coverage_master
    quality = state.final_quality
    parts = [_header(state, "Scores")]
    reached = quality.total >= state.options.target
    parts.append(table(["Score", "Value", "What it measures"], [
        ["Resume Quality", f"{quality.total:.2f} / 10 ({'target reached' if reached else 'below target'}, target {state.options.target:.1f})",
         "How well this resume is written and chosen for this job. The only score that drives revisions."],
        ["Job Match", f"{state.job_match.score:.0f} / 100", "How well my verified skills and eligibility cover the job. Does not change with rewriting."],
        ["Keyword Coverage", f"{state.coverage_final.score:.0f}% (master resume: {state.coverage_master.score:.0f}%)",
         "Weighted share of the job's keywords present, synonym-aware."],
    ]) + "\n")
    parts.append(f"{state.stop_note}\n")
    parts += [f"- {note}" for note in state.notes]

    parts.append("\n## Resume Quality by round\n")
    weights = quality.weights
    head = ["Round", "Total"] + [f"{name} ({weights.get(name, 0):.0%})" for name in ("impact", "relevance", "clarity", "ats", "language")] + ["Fixes"]
    rows = []
    for phase, result in enumerate(state.loops, 1):
        for record in result.history:
            label = f"{record.round}" if len(state.loops) == 1 else f"{phase}.{record.round}"
            best = " (kept)" if result is state.loop and record.round == result.best_round else ""
            c = record.quality.categories
            fixes = record.judgment.fixes
            rows.append([label + best, f"{record.quality.total:.2f}"] + [f"{c.get(n, 0):.1f}" for n in ("impact", "relevance", "clarity", "ats", "language")]
                        + [f"{sum(f.type == FixType.REWRITE for f in fixes)} rewrite / {sum(f.type == FixType.ASK_USER for f in fixes)} ask / {sum(f.type == FixType.REAL_GAP for f in fixes)} gap"])
    c = quality.categories
    rows.append(["**final**", f"**{quality.total:.2f}**"] + [f"{c.get(n, 0):.1f}" for n in ("impact", "relevance", "clarity", "ats", "language")]
                + ["ATS scored on the real PDF"])
    parts.append(table(head, rows) + "\n")
    parts.append("_In-loop ATS uses the text-level checks; the final row uses every check on the rendered PDF and DOCX._\n")
    if state.final_judgment is not None:
        scores = state.final_judgment.category_scores
        parts.append("Judge's reasons (final):\n")
        parts += [f"- **{name}** {getattr(scores, name).score}/10 - {getattr(scores, name).justification}"
                  for name in config.JUDGED_CATEGORIES]

    match = state.job_match
    parts.append("\n## Job Match\n")
    parts.append(table(["Component", "Coverage", "Weight"], [
        [name.replace("_", " "), f"{value:.0%}", f"{config.JOB_MATCH_WEIGHTS[name]:.0%}"] for name, value in match.components.items()
    ]) + "\n")
    parts.append(f"- Verified: {', '.join(match.covered) or 'none'}")
    parts.append(f"- Learning (partial credit): {', '.join(match.partial) or 'none'}")
    parts.append(f"- Missing: {', '.join(match.missing) or 'none'}")

    parts.append("\n## Keyword Coverage\n")
    parts.append(table(["Text", "Coverage", "Missing keywords"], [
        ["Whole master resume", f"{state.coverage_master.score:.0f}%", ", ".join(state.coverage_master.missing) or "none"],
        ["Final resume", f"{state.coverage_final.score:.0f}%", ", ".join(state.coverage_final.missing) or "none"],
    ]) + "\n")
    parts.append("_Keywords missing from the whole master resume are skills or terms I have no verified fact for. "
                 "They are never added just to raise this number._\n")

    if state.ats is not None:
        parts.append(f"## ATS checks ({state.ats.score:.1f} / 10)\n")
        parts.append(table(["Check", "Result", "Detail"],
                           [[c.name.replace("_", " "), "pass" if c.passed else "FAIL", c.detail] for c in state.ats.checks]) + "\n")

    if tracker is not None and tracker.calls:
        tokens_in, tokens_out = tracker.total_tokens()
        parts.append("## Claude usage\n")
        rows = [[purpose, int(v["calls"]), f"{int(v['input']):,}", f"{int(v['output']):,}", f"${v['cost']:.4f}"]
                for purpose, v in tracker.by_purpose().items()]
        rows.append(["**total**", len(tracker.calls), f"{tokens_in:,}", f"{tokens_out:,}", f"**${tracker.total_cost():.4f}**"])
        parts.append(table(["Step", "Calls", "Input tokens", "Output tokens", "Cost (est.)"], rows) + "\n")
        billing = getattr(state, "billing", "api")
        if billing == "subscription":
            parts.append("_These calls ran through Claude Code on your Claude subscription, so nothing was billed per token; "
                         "the cost column is what the same calls would cost on the API._\n")
        elif billing == "local":
            parts.append("_These calls ran on a local model through Ollama and cost nothing._\n")
        else:
            parts.append("_Cost is estimated from the token counts and the prices in config.py._\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# gaps.md
# ---------------------------------------------------------------------------


def gaps_md(state: RunState) -> str:
    parts = [_header(state, "Gaps")]
    judgment = state.final_judgment or (state.loop.best_judgment if state.loop else None)
    real_gaps = judgment.fixes_of(FixType.REAL_GAP) if judgment else []
    questions = judgment.fixes_of(FixType.ASK_USER) if judgment else []

    parts.append("## Requirements I do not clearly meet\n")
    unmet = [r for r in state.eligibility.rows if r.status in (EligStatus.fails, EligStatus.partial, EligStatus.unknown)]
    if unmet:
        parts.append(table(["Requirement", "Status", "Detail"], [[r.requirement, r.status.value, r.detail] for r in unmet]) + "\n")
    else:
        parts.append("None found.\n")

    parts.append("## Real gaps the recruiter review found\n")
    parts.append("\n".join(f"- **{g.target}** - {g.detail}" for g in real_gaps) + "\n" if real_gaps else "None reported.\n")

    parts.append("## Skills this job wants that I cannot claim yet\n")
    match = state.job_match
    if match and (match.missing or match.partial):
        if match.missing:
            parts.append(f"- Missing: {', '.join(match.missing)}")
        if match.partial:
            parts.append(f"- Learning: {', '.join(match.partial)}")
        parts.append("\nAdd a skill to the resume only after there is a fact in master_resume.yaml that shows it "
                     "and it is `verified` in skills_status.yaml. `python tailor.py --insights` ranks which skills "
                     "would unlock the most postings.\n")
    else:
        parts.append("None.\n")

    parts.append("## Questions only I can answer\n")
    if questions:
        parts.append("\n".join(f"- {q.detail}" for q in questions) + "\n")
        parts.append("If the answer is yes, add it as a fact to master_resume.yaml (or run again without "
                     "`--no-interview` and answer when asked).\n")
    else:
        parts.append("None.\n")

    if state.final_quality and state.final_quality.total < state.options.target:
        parts.append("## Why the target was not reached\n")
        parts.append(f"{state.stop_note}\n")
        parts.append("The tool will not invent or inflate anything to close this gap. The honest ways to raise the "
                     "score are the items above.\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Writing files and the applications log
# ---------------------------------------------------------------------------


def write_reports(state: RunState, tracker: UsageTracker | None = None) -> list[Path]:
    files = {
        "eligibility.md": eligibility_md(state),
        "changes.md": changes_md(state),
        "scores.md": scores_md(state, tracker),
        "gaps.md": gaps_md(state),
    }
    written = []
    for name, text in files.items():
        path = state.out_dir / name
        path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        written.append(path)
    return written


def application_row(state: RunState) -> dict[str, str]:
    assert state.final_quality and state.job_match and state.coverage_final
    try:
        folder = str(state.out_dir.relative_to(config.ROOT))
    except ValueError:
        folder = str(state.out_dir)
    return {
        "date": state.run_date.isoformat(),
        "company": state.analysis.company,
        "role": state.analysis.role,
        "url": state.job.url,
        "quality": f"{state.final_quality.total:.2f}",
        "job_match": f"{state.job_match.score:.1f}",
        "keyword_coverage": f"{state.coverage_final.score:.1f}",
        "eligibility_flags": "; ".join(state.eligibility.flags()),
        "rounds": str(sum(len(result.history) - 1 for result in state.loops)),
        "output_folder": folder,
        "status": INITIAL_STATUS,
    }


def append_application(state: RunState, path: Path | None = None) -> Path:
    """Append one row to data/applications.csv (creating it with a header if needed)."""
    path = path or config.APPLICATIONS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(application_row(state))
    return path


def read_applications(path: Path | None = None) -> list[dict[str, str]]:
    path = path or config.APPLICATIONS_CSV
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
