"""Orchestrator: the whole run, and the revision loop at its center.

    startup checks
    job_text    = fetch
    analysis    = analyze(job_text)          -> saved to data/postings/
    eligibility = check(analysis, master)    -> table, "Continue tailoring?"
    brief       = research(company)          unless --no-research
    draft       = tailor(...) -> validate
    judgment    = judge(draft); quality = scoring.quality(judgment, ats)

    while quality < TARGET:
        stop when: no REWRITE fixes | MAX_ROUNDS reached | plateau
        draft = revise(draft, rewrite_fixes) -> validate
        judgment = judge(draft); quality = scoring.quality(...)
        keep the best-scoring draft

    approval -> render -> page-fit -> ats_check -> scores -> summary

Only Resume Quality drives the loop. Job Match and Keyword Coverage are
informational and never trigger a revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import config
from src import (
    analyze, approval, ats_check, eligibility, fetch, judge, render, research, revise, scoring,
    tailor_resume, textutil, ui,
)
from src.errors import TailorError, UserAbort
from src.fetch import JobText
from src.models import (
    AtsReport,
    CompanyBrief,
    EligibilityReport,
    Fix,
    FixType,
    JobAnalysis,
    JobMatch,
    Judgment,
    KeywordCoverage,
    LoopResult,
    Mode,
    QualityScore,
    RoundRecord,
    StopReason,
    TailoredResume,
    Violation,
)
from src.validate import ValidationContext, ValidationOutcome

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class RunOptions:
    strict: bool = False
    research: bool = True
    interview: bool = True
    max_rounds: int = config.MAX_ROUNDS
    target: float = config.TARGET_QUALITY
    batch: bool = False
    cover_letter: bool = False
    plateau_delta: float = config.PLATEAU_DELTA
    plateau_rounds: int = config.PLATEAU_ROUNDS


# ---------------------------------------------------------------------------
# The revision loop
# ---------------------------------------------------------------------------

JudgeFn = Callable[[TailoredResume], Judgment]
QualityFn = Callable[[TailoredResume, Judgment], QualityScore]
ReviseFn = Callable[[TailoredResume, list[Fix]], ValidationOutcome]
RoundHook = Callable[[RoundRecord], None]


def plateaued(history: list[RoundRecord], rounds: int, delta: float) -> bool:
    """True when each of the last `rounds` revision rounds gained less than `delta`."""
    if rounds <= 0 or len(history) <= rounds:
        return False
    recent = history[-(rounds + 1):]
    return all(recent[i + 1].quality.total - recent[i].quality.total < delta for i in range(rounds))


def run_revision_loop(first: ValidationOutcome, *, judge_fn: JudgeFn, quality_fn: QualityFn, revise_fn: ReviseFn,
                      options: RunOptions, on_round: RoundHook | None = None) -> LoopResult:
    """Judge, revise, and re-judge until a stop condition is met. Keeps the best draft.

    Stop conditions, checked in this order before every revision:
      1. target reached          quality >= options.target
      2. no REWRITE fixes        only questions for me or real gaps are left
      3. max rounds              options.max_rounds revisions have been made
      4. plateau                 the last `plateau_rounds` rounds each gained < `plateau_delta`
    """
    draft = first.resume
    violations: list[Violation] = list(first.violations)
    judgment = judge_fn(draft)
    record = RoundRecord(round=0, quality=quality_fn(draft, judgment), judgment=judgment, note="first draft")
    history = [record]
    best, best_round = draft, 0
    if on_round:
        on_round(record)

    rounds = 0
    while True:
        current = history[-1]
        if current.quality.total >= options.target:
            stop = StopReason.target_reached
            break
        fixes = revise.rewrite_fixes(current.judgment.fixes)
        if not fixes:
            stop = StopReason.no_rewrite_fixes
            break
        if rounds >= options.max_rounds:
            stop = StopReason.max_rounds
            break
        if plateaued(history, options.plateau_rounds, options.plateau_delta):
            stop = StopReason.plateau
            break
        outcome = revise_fn(draft, fixes)
        draft = outcome.resume
        violations.extend(outcome.violations)
        rounds += 1
        judgment = judge_fn(draft)
        record = RoundRecord(round=rounds, quality=quality_fn(draft, judgment), judgment=judgment,
                             note=f"applied {len(fixes)} rewrite fix(es)")
        history.append(record)
        if record.quality.total > history[best_round].quality.total:
            best, best_round = draft, rounds
        if on_round:
            on_round(record)
    return LoopResult(best=best, best_round=best_round, history=history, stop_reason=stop, violations=violations)


def describe_stop(result: LoopResult, options: RunOptions) -> str:
    """Why the loop stopped, in plain words (shown in the summary and scores.md)."""
    best = result.best_quality.total
    judgment = result.best_judgment
    if result.stop_reason == StopReason.target_reached:
        return f"Target reached: Resume Quality {best:.2f} >= {options.target:.1f}."
    questions, gaps = len(judgment.fixes_of(FixType.ASK_USER)), len(judgment.fixes_of(FixType.REAL_GAP))
    tail = f" Best draft: {best:.2f} of a {options.target:.1f} target (round {result.best_round})."
    if result.stop_reason == StopReason.no_rewrite_fixes:
        return (f"Stopped below target: rewording cannot raise the score further. What is left is "
                f"{questions} question(s) for me and {gaps} real gap(s)." + tail)
    if result.stop_reason == StopReason.max_rounds:
        return f"Stopped below target after the maximum of {options.max_rounds} revision round(s)." + tail
    return (f"Stopped below target: the last {options.plateau_rounds} rounds each gained less than "
            f"{options.plateau_delta} points, so more rewording is not helping." + tail)


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    """Everything one run produces; report.py and archive.py read from it."""

    options: RunOptions
    job: JobText
    analysis: JobAnalysis
    eligibility: EligibilityReport
    ctx: ValidationContext
    out_dir: Path
    run_date: date
    brief: CompanyBrief | None = None
    system: str = ""
    loop: LoopResult | None = None
    resume: TailoredResume | None = None
    violations: list[Violation] = field(default_factory=list)
    fit: render.FitResult | None = None
    rendered: render.Rendered | None = None
    ats: AtsReport | None = None
    final_judgment: Judgment | None = None
    final_quality: QualityScore | None = None
    job_match: JobMatch | None = None
    coverage_master: KeywordCoverage | None = None
    coverage_final: KeywordCoverage | None = None
    stop_note: str = ""
    notes: list[str] = field(default_factory=list)


def output_dir(analysis: JobAnalysis, run_date: date, root: Path | None = None) -> Path:
    """output/<Company>_<Role>_<YYYY-MM-DD>/ (numbered if it already exists)."""
    root = root or config.OUTPUT_DIR
    base = f"{textutil.slug(analysis.company, 30)}_{textutil.slug(analysis.role, 40)}_{run_date.isoformat()}"
    path, counter = root / base, 2
    while path.exists() and any(path.iterdir()):
        path = root / f"{base}_{counter}"
        counter += 1
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _show_round(record: RoundRecord) -> None:
    cats = record.quality.categories
    breakdown = "  ".join(f"{name} {cats[name]:.1f}" for name in ("impact", "relevance", "clarity", "language", "ats"))
    label = "first draft" if record.round == 0 else f"round {record.round}"
    ui.info(f"  [bold]{label}[/bold]: Resume Quality [bold]{record.quality.total:.2f}[/bold]  ({breakdown})  "
            f"{judge.summarize(record.judgment)}")


def step_loop(state: RunState, llm, first: ValidationOutcome) -> None:
    master, ctx = state.ctx.master, state.ctx

    def judge_fn(draft: TailoredResume) -> Judgment:
        text = render.build_view(draft, master).plain_text(redact_contact=True)
        with ui.console.status("Judging the draft (independent recruiter rubric)..."):
            return judge.judge_resume(llm, state.job.text, state.brief, text)

    def quality_fn(draft: TailoredResume, judgment: Judgment) -> QualityScore:
        return scoring.quality(judgment, ats_check.check_text(render.build_view(draft, master), state.analysis))

    def revise_fn(draft: TailoredResume, fixes: list[Fix]) -> ValidationOutcome:
        with ui.console.status(f"Revising: applying {len(fixes)} rewrite fix(es)..."):
            return revise.revise(llm, state.system, ctx, draft, fixes)

    state.loop = run_revision_loop(first, judge_fn=judge_fn, quality_fn=quality_fn, revise_fn=revise_fn,
                                   options=state.options, on_round=_show_round)
    state.resume = state.loop.best
    state.violations = list(state.loop.violations)
    state.stop_note = describe_stop(state.loop, state.options)
    ui.info(f"  {state.stop_note}")


def step_render(state: RunState, llm) -> None:
    """Render, fit to one page, and run the full ATS checks on the result."""
    assert state.resume is not None
    master = state.ctx.master
    margin, font = render.template_info()
    with ui.console.status("Rendering DOCX and PDF, fitting to one page..."):
        state.fit = render.fit_to_page(
            state.resume, master,
            lambda resume, layout: render.render(resume, master, state.out_dir, layout).pages,
            relevance=state.ctx.relevance,
            tighten_fn=render.make_tighten_fn(llm, state.system) if llm is not None and state.system else None,
            accept_fn=render.make_accept_fn(state.ctx),
            template_margin_in=margin, template_font_pt=font,
        )
        state.rendered = render.render(state.resume, master, state.out_dir, state.fit.layout)
        state.ats = ats_check.check_rendered(state.rendered.view, state.rendered.pdf, state.rendered.docx, state.analysis)
    if len(state.fit.steps) > 1:
        ui.info(f"  page fit: {len(state.fit.steps) - 1} step(s); layout {state.fit.layout.describe(font, margin)}")
        for note in state.fit.removed:
            ui.info(f"    removed {note}")
    if not state.fit.fits:
        state.notes.append(f"The resume could not be fitted to {config.MAX_PAGES} page ({state.fit.pages} pages).")
        ui.warn(state.notes[-1])
    for check in state.ats.failures():
        ui.warn(f"ATS check failed - {check.name}: {check.detail}")


def content_changed_since_loop(state: RunState, loop_text: str) -> bool:
    assert state.rendered is not None
    return state.rendered.view.plain_text(redact_contact=True) != loop_text


def step_scores(state: RunState, llm, loop_text: str) -> None:
    """Final Resume Quality (with the real ATS result), Job Match, Keyword Coverage."""
    assert state.loop is not None and state.rendered is not None and state.ats is not None
    judgment = state.loop.best_judgment
    if content_changed_since_loop(state, loop_text):
        # Approval or page fitting changed the text, so the final version is judged afresh.
        with ui.console.status("Judging the final resume..."):
            judgment = judge.judge_resume(llm, state.job.text, state.brief,
                                          state.rendered.view.plain_text(redact_contact=True))
    state.final_judgment = judgment
    state.final_quality = scoring.quality(judgment, state.ats)
    state.job_match = scoring.job_match(state.analysis, state.ctx.status, state.eligibility)
    state.coverage_master = scoring.keyword_coverage(state.analysis, scoring.master_text(state.ctx.master))
    state.coverage_final = scoring.keyword_coverage(state.analysis, state.rendered.view.body_text())


def print_summary(state: RunState, tracker=None) -> None:
    assert state.final_quality is not None and state.job_match is not None
    assert state.coverage_final is not None and state.coverage_master is not None and state.rendered is not None
    quality = state.final_quality
    met = quality.total >= state.options.target
    lines = [
        f"[bold]{state.analysis.role}[/bold] at [bold]{state.analysis.company}[/bold]",
        "",
        f"Resume Quality   [bold]{quality.total:.2f}[/bold] / 10   (target {state.options.target:.1f}: "
        f"{'[green]reached[/green]' if met else '[yellow]not reached[/yellow]'})",
        "                 " + "  ".join(f"{k} {v:.1f}" for k, v in quality.categories.items()),
        f"Job Match        [bold]{state.job_match.score:.0f}[/bold] / 100   (measures me, not the writing)",
        f"Keyword Coverage [bold]{state.coverage_final.score:.0f}%[/bold]   (whole master resume: {state.coverage_master.score:.0f}%)",
        f"ATS checks       {state.ats.score:.1f} / 10, {state.rendered.pages} page(s)" if state.ats else "",
        "",
        state.stop_note,
    ]
    lines += [f"[yellow]{note}[/yellow]" for note in state.notes]
    flags = state.eligibility.flags()
    if flags:
        lines.append(f"[yellow]Eligibility flags: {len(flags)}[/yellow] (see eligibility.md)")
    lines += ["", f"Output: {state.out_dir}"]
    if tracker is not None and tracker.calls:
        tokens_in, tokens_out = tracker.total_tokens()
        lines.append(f"Claude usage: {len(tracker.calls)} calls, {tokens_in:,} in / {tokens_out:,} out tokens, "
                     f"about ${tracker.total_cost():.3f}")
    ui.panel("Summary", "\n".join(line for line in lines if line is not None), style="green" if met else "yellow")


# ---------------------------------------------------------------------------
# One job, start to finish
# ---------------------------------------------------------------------------


def run_job(session, *, url: str | None = None, text_path: Path | None = None,
            options: RunOptions | None = None, today: date | None = None) -> RunState:
    """Tailor a resume for one posting. Raises `TailorError` on expected failures."""
    options = options or RunOptions()
    today = today or date.today()
    llm, master, status = session.llm, session.master, session.skills_status

    ui.step("Fetching the job posting")
    job = fetch.get_job_text(url, text_path, allow_paste=not options.batch)
    ui.info(f"  {job.words} words ({job.source})")

    ui.step("Analyzing the posting")
    with ui.console.status("Extracting requirements and keywords..."):
        analysis = analyze.analyze_job(llm, job, today)
    analyze.save_analysis(analysis)
    ui.info(f"  {analysis.role} at {analysis.company} ({analysis.job_type.value}); "
            f"{len(analysis.required_skills)} required and {len(analysis.preferred_skills)} preferred skills")
    for quote in analysis.unverified_quotes:
        ui.warn(f'A quoted requirement was not found word-for-word in the posting: "{quote[:90]}"')

    ui.step("Checking eligibility")
    report = eligibility.check_eligibility(analysis, master, status, llm=llm, posting_text=job.text, today=today)
    if not eligibility.confirm_continue(report, batch=options.batch):
        raise UserAbort("Stopped at the eligibility check - nothing was tailored.")

    ctx = ValidationContext(master=master, status=status, analysis=analysis,
                            mode=Mode.strict if options.strict else Mode.normal)
    state = RunState(options=options, job=job, analysis=analysis, eligibility=report, ctx=ctx,
                     out_dir=output_dir(analysis, today), run_date=today)

    if options.research:
        ui.step("Researching the company")
        with ui.console.status(f"Searching the web (up to {config.MAX_RESEARCH_SEARCHES} searches)..."):
            state.brief = research.research_company(llm, analysis, job, today)
        ui.info(f"  {research.summarize(state.brief)}")

    ui.step("Tailoring" + (" (strict: my approved wording)" if options.strict else ""))
    with ui.console.status("Writing the first draft..."):
        first, state.system = tailor_resume.tailor(llm, ctx, state.brief)
    ui.info(f"  {tailor_resume.summarize(first)}")

    ui.step(f"Judging and revising (target {options.target:.1f}, up to {options.max_rounds} rounds)")
    step_loop(state, llm, first)
    assert state.resume is not None
    loop_text = render.build_view(state.resume, master).plain_text(redact_contact=True)

    reverted = approval.revert_pending(state.resume)
    if reverted:
        ui.info(f"  {reverted} flagged change(s) reverted to verified wording")

    ui.step("Rendering")
    step_render(state, llm)

    ui.step("Scoring")
    step_scores(state, llm, loop_text)
    print_summary(state, getattr(llm, "tracker", None))
    return state


def run_batch(session, urls: list[str], options: RunOptions) -> int:
    raise TailorError("Batch mode is not built yet.")
