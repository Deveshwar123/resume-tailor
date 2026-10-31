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

    if below target and the judge has questions for me (and interviews are on):
        interview -> new facts (with my OK) -> re-tailor -> loop once more

    approval checkpoint -> render -> page-fit -> ats_check
    job match + keyword coverage -> report -> archive -> applications.csv -> summary

Only Resume Quality drives the loop. Job Match and Keyword Coverage are
informational and never trigger a revision.

Speed (a live run took 168-253 s): independent Claude calls overlap. Company
research starts as soon as the posting text is in, alongside the analysis and
eligibility calls, and the cover letter is written alongside the judge /
revise loop (config.PARALLEL_CALLS). In unattended runs (--batch, --auto)
approvals are settled inside the loop, so the judge scores exactly what will be
rendered and the final re-judge is needed only if page fitting changed text.
Every step's time is recorded in RunState.timings (scores.md, report.json).
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterator

import config
from src import (
    analyze, approval, archive, ats_check, cover_letter, eligibility, fetch, interview, judge, render, report,
    research, revise, scoring, tailor_resume, textutil, ui,
)
from src.llm import UsageTracker
from src.errors import FetchError, TailorError, UserAbort
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
    auto: bool = False                # --auto / tailor_for_job(): no prompts, validator-clean rewrites are used
    record: bool = True               # append a row to data/applications.csv

    @property
    def unattended(self) -> bool:
        """Nobody can be asked anything (batch or auto)."""
        return self.batch or self.auto


# ---------------------------------------------------------------------------
# The revision loop
# ---------------------------------------------------------------------------

JudgeFn = Callable[[TailoredResume], Judgment]
QualityFn = Callable[[TailoredResume, Judgment], QualityScore]
ReviseFn = Callable[[TailoredResume, list[Fix]], ValidationOutcome]
RoundHook = Callable[[RoundRecord], None]


def plateaued(history: list[RoundRecord], rounds: int, delta: float) -> bool:
    """True when each of the last `rounds` revision rounds failed to beat the best earlier score by `delta`.

    Gains are measured against the best draft so far, not the previous round: a
    drop followed by a recovery to the old level is not progress.
    """
    if rounds <= 0 or len(history) <= rounds:
        return False
    for i in range(len(history) - rounds, len(history)):
        best_before = max(record.quality.total for record in history[:i])
        if history[i].quality.total - best_before + 1e-9 >= delta:
            return False
    return True


def better(new: RoundRecord, old: RoundRecord) -> bool:
    """Does `new` beat `old`? Totals within TIE_EPSILON are a tie, won by higher Keyword Coverage."""
    gain = new.quality.total - old.quality.total
    if abs(gain) < config.TIE_EPSILON:
        return new.tiebreak > old.tiebreak + 1e-9
    return gain > 0


def run_revision_loop(first: ValidationOutcome, *, judge_fn: JudgeFn, quality_fn: QualityFn, revise_fn: ReviseFn,
                      options: RunOptions, on_round: RoundHook | None = None,
                      tiebreak_fn: Callable[[TailoredResume], float] | None = None) -> LoopResult:
    """Judge, revise, and re-judge until a stop condition is met. Keeps the best draft.

    Stop conditions, checked in this order before every revision:
      1. target reached          quality >= options.target
      2. no REWRITE fixes        only questions for me or real gaps are left
      3. max rounds              options.max_rounds revisions have been made
      4. plateau                 the last `plateau_rounds` rounds each failed to beat the best by `plateau_delta`

    `tiebreak_fn` (Keyword Coverage of a draft) decides between drafts with equal
    totals, so a revision that used more of the posting's language is kept
    instead of losing the tie to the first draft.
    """
    def record_for(round_number: int, draft: TailoredResume, judgment: Judgment, note: str) -> RoundRecord:
        return RoundRecord(round=round_number, quality=quality_fn(draft, judgment), judgment=judgment, note=note,
                           tiebreak=tiebreak_fn(draft) if tiebreak_fn else 0.0)

    draft = first.resume
    violations: list[Violation] = list(first.violations)
    judgment = judge_fn(draft)
    record = record_for(0, draft, judgment, "first draft")
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
        record = record_for(rounds, draft, judgment, f"applied {len(fixes)} rewrite fix(es)")
        history.append(record)
        if better(record, history[best_round]):
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
    return (f"Stopped below target: the last {options.plateau_rounds} rounds each failed to beat the best draft by "
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
    billing: str = "api"                      # api | subscription | local (how Claude calls are paid for)
    notes: list[str] = field(default_factory=list)
    loops: list[LoopResult] = field(default_factory=list)          # every loop run (before / after the interview)
    decisions: list[approval.Decision] = field(default_factory=list)
    interview: interview.InterviewResult | None = None
    cover_letter: cover_letter.CoverLetter | None = None
    ledger: scoring.BulletLedger = field(default_factory=scoring.BulletLedger)   # per-bullet judge scores
    timings: dict[str, float] = field(default_factory=dict)      # seconds per step
    seconds: float = 0.0                                         # the whole run
    files: dict[str, Path] = field(default_factory=dict)         # every file written, by key (archive.py)


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


@contextmanager
def timed(timings: dict[str, float], name: str) -> Iterator[None]:
    """Add the wall-clock time of the block to timings[name]."""
    started = time.monotonic()
    try:
        yield
    finally:
        timings[name] = round(timings.get(name, 0.0) + time.monotonic() - started, 1)


def settle_unattended(options: RunOptions, outcome: ValidationOutcome) -> ValidationOutcome:
    """Batch / --auto: resolve approvals on a fresh draft right away, so the judge scores the real text."""
    if options.unattended:
        approval.settle(outcome.resume, accept_clean=options.auto and not options.strict,
                        reason="--auto: flagged by the validator" if options.auto else "batch mode")
    return outcome


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
        view = render.build_view(draft, master)
        with ui.console.status("Judging the draft (independent recruiter rubric)..."):
            judgment, bullets = judge.judge_view(llm, state.job.text, state.brief, view)
        state.ledger.add(judgment, bullets)
        return judgment

    def quality_fn(draft: TailoredResume, judgment: Judgment) -> QualityScore:
        view = render.build_view(draft, master)
        means, scored = state.ledger.means(view.bullet_texts())
        return scoring.quality(judgment, ats_check.check_text(view, state.analysis), bullet_means=means,
                               bullets_scored=scored)

    def tiebreak_fn(draft: TailoredResume) -> float:
        return scoring.keyword_coverage(state.analysis, render.build_view(draft, master).body_text()).score

    def revise_fn(draft: TailoredResume, fixes: list[Fix]) -> ValidationOutcome:
        with ui.console.status(f"Revising: applying {len(fixes)} rewrite fix(es)..."):
            return settle_unattended(state.options, revise.revise(llm, state.system, ctx, draft, fixes))

    first = settle_unattended(state.options, first)
    state.loop = run_revision_loop(first, judge_fn=judge_fn, quality_fn=quality_fn, revise_fn=revise_fn,
                                   options=state.options, on_round=_show_round, tiebreak_fn=tiebreak_fn)
    state.loops.append(state.loop)
    state.resume = state.loop.best
    state.violations = list(state.loop.violations)
    state.stop_note = describe_stop(state.loop, state.options)
    ui.info(f"  {state.stop_note}")


def step_interview(state: RunState, llm, session, today: date) -> None:
    """Below target with open questions: ask me, add the facts I approve, re-tailor, loop once more."""
    options = state.options
    assert state.loop is not None
    if (state.loop.best_quality.total >= options.target or not options.interview or options.unattended
            or not ui.is_interactive()):
        return
    asks = interview.questions_from(state.loop.best_judgment.fixes)
    if not asks:
        return
    ui.step("Interview: the review raised questions only I can answer")
    state.interview = interview.run_interview(llm, session.master, session.skills_status, asks, today=today)
    if not state.interview.new_fact_ids:
        ui.info("  no new facts were added")
        return
    before = state.loop
    state.ctx = ValidationContext(master=session.master, status=session.skills_status, analysis=state.analysis,
                                  mode=state.ctx.mode, new_fact_ids=frozenset(state.interview.new_fact_ids))
    ui.step(f"Re-tailoring with {len(state.interview.new_fact_ids)} new fact(s), then one more loop")
    with ui.console.status("Writing a new draft..."):
        first, state.system = tailor_resume.tailor(llm, state.ctx, state.brief)
    step_loop(state, llm, first)
    if before.best_quality.total > state.loop.best_quality.total:
        # The new draft scored lower: keep the earlier, better one.
        state.loop, state.resume = before, before.best
        state.violations = list(before.violations)
        state.stop_note = describe_stop(before, options)
        state.notes.append("The draft written after the interview scored lower, so the earlier draft was kept.")


def step_approval(state: RunState) -> None:
    """The human checkpoint: nothing flagged is rendered without a decision."""
    assert state.resume is not None
    if state.options.unattended:
        settle_unattended(state.options, ValidationOutcome(resume=state.resume))    # anything still pending
        state.decisions = approval.settled_decisions(state.resume)
    else:
        state.decisions = approval.review(state.resume, state.ctx, batch=state.options.batch)
    ui.info(f"  {approval.summarize(state.decisions)}")
    approval.offer_to_save(state.ctx.master, state.decisions)


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
        state.ats = ats_check.check_rendered(state.rendered.view, state.rendered.pdf, state.rendered.docx, state.analysis,
                                             resolve_links=config.ATS_RESOLVE_LINKS)
    if len(state.fit.steps) > 1:
        ui.info(f"  page fit: {len(state.fit.steps) - 1} step(s); layout {state.fit.layout.describe(font, margin)}")
        for note in state.fit.removed:
            ui.info(f"    removed {note}")
    if not state.fit.fits:
        state.notes.append(f"The resume could not be fitted to {config.MAX_PAGES} page ({state.fit.pages} pages).")
        ui.warn(state.notes[-1])
    for check in state.ats.failures():
        ui.warn(f"ATS check failed - {check.name}: {check.detail}")


def step_cover_letter(state: RunState, llm, drafted: Future | None = None) -> None:
    """--cover-letter: same facts, same validator, same approval rules as the resume.

    `drafted` is the letter already being written in the background (started
    right after the tailor, alongside the revision loop).
    """
    sources = cover_letter.company_sources(state.job.text, state.brief)
    with ui.console.status("Drafting the cover letter..."):
        letter = drafted.result() if drafted is not None else cover_letter.write(
            llm, state.system, state.ctx, state.job.text, state.brief)
    cover_letter.review(letter, state.ctx, sources, batch=state.options.unattended)
    with ui.console.status("Rendering the cover letter..."):
        cover_letter.save(letter, state.ctx.master, state.out_dir, state.analysis.company, state.analysis.role, state.run_date)
    state.cover_letter = letter
    state.violations.extend(letter.violations)
    removed = sum(1 for sentence in letter.sentences if not sentence.kept)
    ui.info(f"  {letter.word_count()} words, {letter.pages} page(s); {removed} sentence(s) removed by validation or review")
    if letter.pages > 1:
        state.notes.append("The cover letter runs past one page; shorten it before sending.")


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
            judgment, bullets = judge.judge_view(llm, state.job.text, state.brief, state.rendered.view)
        state.ledger.add(judgment, bullets)
    state.final_judgment = judgment
    means, scored = state.ledger.means(state.rendered.view.bullet_texts())
    state.final_quality = scoring.quality(judgment, state.ats, bullet_means=means, bullets_scored=scored)
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
    stats = archive.bullet_stats(state)
    lines.append(f"Bullets          {stats['reworded']} of {stats['total']} reworded for this posting "
                 f"({stats['reverted']} rewrite(s) reverted to verified wording)")
    lines += ["", f"Output: {state.out_dir}", f"Run time: {state.seconds:.0f} s"]
    if tracker is not None and tracker.calls:
        tokens_in, tokens_out = tracker.total_tokens()
        usage = f"{len(tracker.calls)} calls, {tokens_in:,} in / {tokens_out:,} out tokens"
        billing = getattr(state, "billing", "api")
        if billing == "subscription":
            lines.append(f"Claude usage: {usage} - covered by your Claude subscription "
                         f"(API price would be about ${tracker.total_cost():.3f})")
        elif billing == "local":
            lines.append(f"Local model usage: {usage} - free")
        else:
            lines.append(f"Claude usage: {usage}, about ${tracker.total_cost():.3f}")
    ui.panel("Summary", "\n".join(line for line in lines if line is not None), style="green" if met else "yellow")


# ---------------------------------------------------------------------------
# One job, start to finish
# ---------------------------------------------------------------------------


def run_job(session, *, url: str | None = None, text_path: Path | None = None, posting_text: str | None = None,
            options: RunOptions | None = None, today: date | None = None, out_dir: Path | None = None) -> RunState:
    """Tailor a resume for one posting. Raises `TailorError` on expected failures.

    `posting_text` gives the posting directly (no fetch). `out_dir` writes every
    file there instead of output/<Company>_<Role>_<date>/.
    """
    options = options or RunOptions()
    today = today or date.today()
    llm, master, status = session.llm, session.master, session.skills_status
    tracker = getattr(llm, "tracker", None)
    first_call = len(tracker.calls) if tracker is not None else 0
    started = time.monotonic()
    timings: dict[str, float] = {}
    # Two workers: research overlaps the analysis, and the cover letter overlaps the revision loop.
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tailor") if config.PARALLEL_CALLS else None
    try:
        state = _run_job(session, llm, master, status, url, text_path, posting_text, options, today, out_dir,
                         timings, pool)
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    state.seconds = round(time.monotonic() - started, 1)

    with timed(state.timings, "reports"):
        ui.step("Writing reports and archiving this version")
        # Usage for THIS job only (in batch mode the tracker spans every job).
        usage = UsageTracker(calls=tracker.calls[first_call:]) if tracker is not None else None
        report.write_reports(state, usage)
        archive.write_archive(state, usage)
        if options.record:
            report.append_application(state)
    state.seconds = round(time.monotonic() - started, 1)
    archive.write_run_report(state, usage)          # report.json, with the final timings
    print_summary(state, usage)
    return state


def _run_job(session, llm, master, status, url, text_path, posting_text, options: RunOptions, today: date,
             out_dir: Path | None, timings: dict[str, float], pool: ThreadPoolExecutor | None) -> RunState:
    with timed(timings, "fetch"):
        ui.step("Fetching the job posting")
        if posting_text is not None:
            job = JobText(posting_text.strip(), url or "", "given")
            if job.words < 40:
                raise FetchError(f"The posting text has only {job.words} words.", "Pass the full job description.")
        else:
            job = fetch.get_job_text(url, text_path, allow_paste=not options.unattended)
        ui.info(f"  {job.words} words ({job.source})")

    # Company research needs only the posting, so it starts now and runs while the posting is analyzed.
    can_search = getattr(llm, "can_search", True)
    research_job: Future | None = None
    brief: CompanyBrief | None = None
    if options.research and can_search:
        brief = research.find_cached(job)
        if brief is None and pool is not None:
            research_job = pool.submit(research.collect, llm, job, today)

    with timed(timings, "analyze"):
        ui.step("Analyzing the posting")
        with ui.console.status("Extracting requirements and keywords..."):
            analysis = analyze.analyze_job(llm, job, today)
        analyze.save_analysis(analysis)
        ui.info(f"  {analysis.role} at {analysis.company} ({analysis.job_type.value}); "
                f"{len(analysis.required_skills)} required and {len(analysis.preferred_skills)} preferred skills")
        for quote in analysis.unverified_quotes:
            ui.warn(f'A quoted requirement was not found word-for-word in the posting: "{quote[:90]}"')

    with timed(timings, "eligibility"):
        ui.step("Checking eligibility")
        eligible = eligibility.check_eligibility(analysis, master, status, llm=llm, posting_text=job.text, today=today)
    if not eligibility.confirm_continue(eligible, batch=options.unattended):
        raise UserAbort("Stopped at the eligibility check - nothing was tailored.")

    ctx = ValidationContext(master=master, status=status, analysis=analysis,
                            mode=Mode.strict if options.strict else Mode.normal)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(options=options, job=job, analysis=analysis, eligibility=eligible, ctx=ctx,
                     out_dir=out_dir or output_dir(analysis, today), run_date=today,
                     billing=getattr(llm, "billing", "api"), timings=timings)

    with timed(timings, "research (waiting)"):
        if options.research and not can_search:
            ui.info("  Company research skipped: the local model cannot search the web.")
        elif options.research:
            ui.step("Researching the company")
            if brief is not None:
                state.brief = brief
                ui.info("  (cached)")
            elif research_job is not None:
                with ui.console.status("Finishing the company research..."):
                    collected = research_job.result()
                state.brief = research.read_cache(analysis.company) or research.finish(collected, analysis, job, today)
            else:
                with ui.console.status(f"Searching the web (up to {config.MAX_RESEARCH_SEARCHES} searches)..."):
                    state.brief = research.research_company(llm, analysis, job, today)
            ui.info(f"  {research.summarize(state.brief)}")

    with timed(timings, "tailor"):
        ui.step("Tailoring" + (" (strict: my approved wording)" if options.strict else ""))
        with ui.console.status("Writing the first draft..."):
            first, state.system = tailor_resume.tailor(llm, ctx, state.brief)
        ui.info(f"  {tailor_resume.summarize(first)}")

    # The letter depends on the fact bank, the posting, and the research, not on the resume
    # wording, so it is written while the resume is judged and revised.
    letter_job: Future | None = None
    if options.cover_letter and pool is not None:
        letter_job = pool.submit(cover_letter.write, llm, state.system, ctx, job.text, state.brief)

    with timed(timings, "judge and revise"):
        ui.step(f"Judging and revising (target {options.target:.1f}, up to {options.max_rounds} rounds)")
        step_loop(state, llm, first)
    system_before = state.system
    with timed(timings, "interview"):
        step_interview(state, llm, session, today)
    if state.system != system_before:
        letter_job = None                      # new interview facts: write the letter again from them
    assert state.resume is not None
    loop_text = render.build_view(state.resume, master).plain_text(redact_contact=True)

    with timed(timings, "approval"):
        ui.step("Approval checkpoint")
        step_approval(state)

    with timed(timings, "render"):
        ui.step("Rendering")
        step_render(state, llm)

    with timed(timings, "scores"):
        ui.step("Scoring")
        step_scores(state, llm, loop_text)

    if options.cover_letter:
        with timed(timings, "cover letter"):
            ui.step("Cover letter")
            step_cover_letter(state, llm, letter_job)
    return state


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def run_batch(session, urls: list[str], options: RunOptions, today: date | None = None) -> int:
    """Run many postings without prompts. One failure never stops the rest.

    In batch mode nothing can ask me anything: a login wall skips that link, the
    eligibility check continues (its flags lead eligibility.md), flagged rewrites
    revert to verified wording, and there is no interview.
    """
    from src import insights

    options.batch, options.interview = True, False
    results: list[tuple[str, RunState | None, str]] = []
    for number, url in enumerate(urls, 1):
        ui.console.rule(f"[{number}/{len(urls)}] {url}")
        try:
            results.append((url, run_job(session, url=url, options=options, today=today), ""))
        except TailorError as exc:
            ui.error(exc.message, exc.hint)
            results.append((url, None, exc.message))
        except Exception as exc:  # a bug in one posting must not cost me the rest of the batch
            log = config.OUTPUT_DIR / "batch_errors.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n=== {date.today().isoformat()} {url}\n{traceback.format_exc()}")
            ui.error(f"Unexpected error ({type(exc).__name__}: {exc}); details in {log}")
            results.append((url, None, f"unexpected {type(exc).__name__}"))

    rows = []
    for number, (url, state, problem) in enumerate(results, 1):
        if state is None or state.final_quality is None or state.job_match is None:
            rows.append([str(number), url[:48], "-", "-", "-", "-", f"[red]failed[/red]: {problem[:70]}"])
            continue
        flags = len(state.eligibility.flags())
        rows.append([str(number), state.analysis.company, state.analysis.role, f"{state.final_quality.total:.2f}",
                     f"{state.job_match.score:.0f}", str(flags) if flags else "-", str(state.out_dir.name)])
    ui.console.rule("Batch summary")
    ui.table("", ["#", "Company", "Role", "Quality", "Job Match", "Eligibility flags", "Output folder / result"], rows)
    done = sum(1 for _url, state, _problem in results if state is not None)
    tracker = getattr(session.llm, "tracker", None)
    cost = f", about ${tracker.total_cost():.2f} in Claude usage" if tracker is not None and tracker.calls else ""
    ui.info(f"{done} of {len(results)} posting(s) tailored{cost}.")
    if done:
        insights.generate(session.skills_status, session.master, today=today)
    return 0 if done == len(results) else 1
