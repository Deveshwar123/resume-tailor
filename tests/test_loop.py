"""Milestone 8: the revision loop and its stop conditions, plus one end-to-end run."""

from __future__ import annotations

from datetime import date

import pytest

import config
from src import loop, render, revise, tailor_resume
from src.loop import RunOptions, run_revision_loop
from src.models import (
    Judgment, QualityScore, RoundRecord, Severity, StopReason, TailoredResume, Violation,
)
from src.startup import Session
from src.validate import ValidationOutcome
from tests.conftest import SAMPLE_POSTING, ScriptedLLM, bd, ed, make_draft, make_extract, make_judgment

REWRITE = ("REWRITE", "bullet 1", "Lead with the result", 0.4)
ASK = ("ASK_USER", "Acme", "Did you measure the speed-up?", 0.5)
GAP = ("REAL_GAP", "skills", "No AWS experience", 0.6)


class Harness:
    """Scripted judge / quality / revise functions for run_revision_loop."""

    def __init__(self, totals: list[float], fixes: list | None = None, per_round_fixes: list[list] | None = None):
        self.totals = totals
        self.per_round_fixes = per_round_fixes
        self.fixes = [REWRITE] if fixes is None else fixes
        self.judged = 0
        self.revisions: list[list] = []

    def judge_fn(self, draft: TailoredResume) -> Judgment:
        fixes = self.per_round_fixes[self.judged] if self.per_round_fixes else self.fixes
        self.judged += 1
        return make_judgment(fixes=fixes)

    def quality_fn(self, draft: TailoredResume, judgment: Judgment) -> QualityScore:
        return QualityScore(total=self.totals[self.judged - 1])

    def revise_fn(self, draft: TailoredResume, fixes: list) -> ValidationOutcome:
        self.revisions.append(fixes)
        violation = Violation(code="B1", severity=Severity.hard, message=f"round {len(self.revisions)}")
        return ValidationOutcome(resume=TailoredResume(skills=[f"draft-{len(self.revisions)}"]), violations=[violation])

    def run(self, **options) -> loop.LoopResult:
        first = ValidationOutcome(resume=TailoredResume(skills=["draft-0"]))
        return run_revision_loop(first, judge_fn=self.judge_fn, quality_fn=self.quality_fn,
                                 revise_fn=self.revise_fn, options=RunOptions(**options))


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------


def test_stops_immediately_when_the_first_draft_meets_the_target():
    h = Harness([9.2])
    result = h.run()
    assert result.stop_reason == StopReason.target_reached
    assert h.revisions == [] and len(result.history) == 1 and result.best.skills == ["draft-0"]


def test_revises_until_the_target_is_reached():
    h = Harness([7.0, 8.0, 9.1, 9.9])
    result = h.run()
    assert result.stop_reason == StopReason.target_reached
    assert len(h.revisions) == 2 and result.best_round == 2 and result.best.skills == ["draft-2"]
    assert [r.quality.total for r in result.history] == [7.0, 8.0, 9.1]


def test_stops_when_only_questions_and_gaps_are_left():
    h = Harness([7.0], fixes=[ASK, GAP])
    result = h.run()
    assert result.stop_reason == StopReason.no_rewrite_fixes and h.revisions == []
    note = loop.describe_stop(result, RunOptions())
    assert "1 question(s) for me and 1 real gap(s)" in note and "7.00 of a 9.0 target" in note


def test_stops_at_max_rounds():
    h = Harness([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])       # always improving, never enough
    result = h.run(max_rounds=3)
    assert result.stop_reason == StopReason.max_rounds
    assert len(h.revisions) == 3 and len(result.history) == 4
    assert "maximum of 3 revision round(s)" in loop.describe_stop(result, RunOptions(max_rounds=3))


def test_zero_rounds_means_no_revision():
    h = Harness([6.0])
    result = h.run(max_rounds=0)
    assert result.stop_reason == StopReason.max_rounds and h.revisions == []


def test_stops_on_plateau():
    # +1.0, then two rounds that each beat the best by less than PLATEAU_DELTA (0.1).
    h = Harness([6.0, 7.0, 7.05, 7.08, 9.5])
    result = h.run()
    assert result.stop_reason == StopReason.plateau
    assert len(h.revisions) == 3                                 # the 9.5 round is never run
    assert "failed to beat the best draft by 0.1" in loop.describe_stop(result, RunOptions())


def test_a_drop_counts_toward_the_plateau():
    h = Harness([7.0, 6.5, 6.6, 9.9])
    result = h.run()
    assert result.stop_reason == StopReason.plateau and len(h.revisions) == 2


def test_one_slow_round_is_not_a_plateau():
    h = Harness([6.0, 6.1, 7.0, 7.1, 9.0])
    result = h.run()
    assert result.stop_reason == StopReason.target_reached and len(h.revisions) == 4


def test_plateau_settings_are_configurable():
    h = Harness([6.0, 6.2, 6.4, 6.6, 6.8])
    assert h.run(plateau_delta=0.1, max_rounds=4).stop_reason == StopReason.max_rounds
    h = Harness([6.0, 6.2, 6.4])
    assert h.run(plateau_rounds=1, plateau_delta=0.3).stop_reason == StopReason.plateau


def test_small_real_gains_keep_the_loop_going():
    # Per-bullet scoring moves the total by tenths; two +0.15 rounds are progress, not a plateau.
    h = Harness([6.0, 6.15, 6.3, 9.1])
    result = h.run()
    assert result.stop_reason == StopReason.target_reached and result.best_round == 3


def test_recovering_to_the_old_best_is_not_progress():
    # A drop and a recovery to the earlier level: neither round beat the best (7.0), so stop.
    h = Harness([7.0, 6.0, 7.0, 9.9])
    result = h.run()
    assert result.stop_reason == StopReason.plateau and len(h.revisions) == 2 and result.best_round == 0


def test_plateaued_helper():
    def history(*totals):
        return [RoundRecord(round=i, quality=QualityScore(total=t), judgment=make_judgment()) for i, t in enumerate(totals)]

    assert not loop.plateaued(history(7.0), 2, 0.3)
    assert not loop.plateaued(history(7.0, 7.1), 2, 0.3)             # not enough rounds yet
    assert loop.plateaued(history(7.0, 7.1, 7.2), 2, 0.3)
    assert not loop.plateaued(history(7.0, 7.1, 7.4), 2, 0.3)        # exactly delta is not "less than"
    assert loop.plateaued(history(7.0, 6.5, 6.9), 2, 0.3)            # measured against the best, not the last round


# ---------------------------------------------------------------------------
# Behavior around the conditions
# ---------------------------------------------------------------------------


def test_the_best_draft_is_kept_when_a_later_round_scores_lower():
    h = Harness([7.0, 8.5, 8.0, 8.1])
    result = h.run()
    assert result.best_round == 1 and result.best.skills == ["draft-1"]
    assert result.best_quality.total == 8.5 and result.stop_reason == StopReason.plateau


def test_a_tie_goes_to_the_draft_that_uses_more_of_the_postings_language():
    # Whole-number scores used to tie (5.80 -> 5.80 in a live run) and the first draft always won.
    h = Harness([7.0, 7.0, 6.9, 6.9])
    coverage = iter([40.0, 55.0, 70.0, 80.0])
    first = loop.ValidationOutcome(resume=TailoredResume(skills=["draft-0"]))
    result = run_revision_loop(first, judge_fn=h.judge_fn, quality_fn=h.quality_fn, revise_fn=h.revise_fn,
                               options=RunOptions(), tiebreak_fn=lambda _draft: next(coverage))
    assert result.best_round == 1 and result.best.skills == ["draft-1"]       # same total, more posting keywords
    assert [r.tiebreak for r in result.history] == [40.0, 55.0, 70.0]
    # Without a tiebreak function an exact tie keeps the earlier draft.
    assert Harness([7.0, 7.0, 6.9, 6.9]).run().best_round == 0


def test_only_rewrite_fixes_are_passed_to_the_reviser_and_violations_accumulate():
    h = Harness([7.0, 9.5], fixes=[REWRITE, ASK, GAP])
    result = h.run()
    assert [[f.type.value for f in fixes] for fixes in h.revisions] == [["REWRITE"]]
    assert [v.message for v in result.violations] == ["round 1"]


def test_target_is_checked_before_everything_else():
    h = Harness([9.5], fixes=[])                                   # at target AND no fixes
    assert h.run(max_rounds=0).stop_reason == StopReason.target_reached


def test_each_round_uses_the_fixes_from_its_own_judgment():
    second = ("REWRITE", "bullet 2", "Cut the filler", 0.3)
    h = Harness([7.0, 8.0, 9.5], per_round_fixes=[[REWRITE], [second], []])
    h.run()
    assert [fixes[0].detail for fixes in h.revisions] == ["Lead with the result", "Cut the filler"]


# ---------------------------------------------------------------------------
# revise.py
# ---------------------------------------------------------------------------


def test_revise_applies_rewrite_fixes_and_revalidates(ctx):
    current = tailor_resume.validate.enforce(
        make_draft(experiences=[ed("exp_acme", [bd("Assisted a 4-person team in developing a Python application.",
                                                   ["exp_acme_f01"], "exp_acme_b01", "unchanged")])]), ctx).resume
    inflated = make_draft(experiences=[ed("exp_acme", [bd("Led a 4-person team building a Python application",
                                                          ["exp_acme_f01"], "exp_acme_b01")])])
    llm = ScriptedLLM({"revise": [inflated], "fix": [{"fixes": []}]})
    system = tailor_resume.build_system(ctx)
    fixes = make_judgment(fixes=[REWRITE, ASK]).fixes
    outcome = revise.revise(llm, system, ctx, current, fixes)
    call = llm.calls[0]
    assert call["model"] == config.WRITE_MODEL and call["system"] == system
    assert "Lead with the result" in call["user"] and "Did you measure" not in call["user"]     # REWRITE only
    assert "exp_acme_b01" in call["user"]                                                       # current draft included
    # The revision inflated ownership: it is caught and the verified wording stays.
    assert outcome.resume.experiences[0].bullets[0].text == "Assisted a 4-person team in developing a Python application."
    assert any(v.code == "B3" for v in outcome.hard())


def test_revise_refuses_to_run_without_rewrite_fixes(ctx):
    with pytest.raises(ValueError):
        revise.revise(ScriptedLLM(), "system", ctx, TailoredResume(), make_judgment(fixes=[ASK, GAP]).fixes)


# ---------------------------------------------------------------------------
# End to end (Claude scripted, real rendering)
# ---------------------------------------------------------------------------

needs_soffice = pytest.mark.skipif(render.find_soffice() is None, reason="LibreOffice (soffice) is not installed")

FIRST = make_draft(
    experiences=[
        ed("exp_acme", [bd("Assisted a 4-person team in developing a Python application.", ["exp_acme_f01"], "exp_acme_b01", "unchanged"),
                        bd("Wrote SQL queries that reduced report generation time by 30%.", ["exp_acme_f02"], "exp_acme_b02", "unchanged")]),
        ed("exp_club", [bd("Led weekly workshops for 25 members on data analysis", ["exp_club_f01"], None, "unchanged")]),
    ],
    projects=[ed("proj_tracker", [bd("Built a budget tracking web app with JavaScript and React.", ["proj_tracker_f01"],
                                     "proj_tracker_b01", "unchanged")])],
    skills=["Python", "SQL", "data analysis", "React"], honors=["Dean's List"], coursework=["Databases"],
)
REVISED = make_draft(
    experiences=[
        ed("exp_acme", [bd("Wrote SQL queries that reduced report generation time by 30%.", ["exp_acme_f02"], "exp_acme_b02", "unchanged"),
                        bd("Assisted a 4-person team in developing a Python app.", ["exp_acme_f01"], "exp_acme_b01", "minor_wording"),
                        bd("Tested REST API endpoints with pytest", ["exp_acme_f03"], None, "unchanged")]),
        ed("exp_club", [bd("Led weekly workshops for 25 members on data analysis", ["exp_club_f01"], None, "unchanged")]),
    ],
    projects=[ed("proj_tracker", [bd("Built a budget tracking web app with JavaScript and React.", ["proj_tracker_f01"],
                                     "proj_tracker_b01", "unchanged")])],
    skills=["Python", "SQL", "data analysis", "pytest"], honors=["Dean's List"], coursework=["Databases"],
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for name, sub in (("OUTPUT_DIR", "output"), ("POSTINGS_DIR", "postings"), ("CACHE_DIR", "cache"),
                      ("BACKUP_DIR", "backups")):
        monkeypatch.setattr(config, name, tmp_path / sub)
    monkeypatch.setattr(config, "APPLICATIONS_CSV", tmp_path / "applications.csv")
    posting = tmp_path / "job.txt"
    posting.write_text(SAMPLE_POSTING)
    return tmp_path, posting


@needs_soffice
def test_run_job_end_to_end(master, status, workspace):
    tmp_path, posting = workspace
    from src.llm import SearchResult
    from src.models import BriefItemDraft, CompanyBriefDraft

    brief = CompanyBriefDraft(summary=[], values_language=[], recent_news=[], tech_stack=[],
                              intern_work=[BriefItemDraft(text="Interns build data pipelines in Python and SQL.",
                                                          source_url="posting")])
    llm = ScriptedLLM({
        "analyze": [make_extract()],
        "research": [SearchResult(text="notes", sources=[], searches=2)],
        "brief": [brief],
        "tailor": [FIRST],
        "judge": [make_judgment(7, 7, 8, 7, fixes=[REWRITE, ASK, GAP]), make_judgment(9, 9, 9, 9, fixes=[GAP]),
                  make_judgment(9, 9, 9, 9, fixes=[GAP])],
        "revise": [REVISED],
    })
    state = loop.run_job(Session(master=master, skills_status=status, llm=llm), text_path=posting,
                         options=RunOptions(), today=date(2026, 10, 13))
    assert llm.purposes()[:8] == ["analyze", "research", "brief", "tailor", "reword", "judge", "revise", "judge"]
    assert state.brief.intern_work[0].tier.value == "posting"
    tailor_call = next(c for c in llm.calls if c["purpose"] == "tailor")
    assert "Interns build data pipelines in Python and SQL." in tailor_call["system"]     # research shapes emphasis
    assert state.loop.stop_reason == StopReason.target_reached and state.loop.best_round == 1
    assert state.out_dir == tmp_path / "output" / "Acme_Corp_Data_Engineering_Intern_2026-10-13"
    assert (state.out_dir / "resume.pdf").exists() and (state.out_dir / "resume.docx").exists()
    assert state.rendered.pages == 1 and state.ats.final and state.ats.failures() == []
    assert state.final_quality.total == pytest.approx(9.15) and state.final_quality.ats_final
    assert state.job_match.score > 80 and state.coverage_final.score > 0
    assert len(list((tmp_path / "postings").glob("*.json"))) == 1          # saved for the dashboard
    text = state.rendered.view.plain_text()
    assert "Acme Corp — Software Engineering Intern, Austin, TX" in text and "Tested REST API endpoints with pytest." in text
    # The judge never saw my contact details.
    judge_calls = [c for c in llm.calls if c["purpose"] == "judge"]
    assert all(master.contact.email not in c["user"] and master.contact.phone not in c["user"] for c in judge_calls)


def test_declining_at_the_eligibility_prompt_stops_the_run(master, status, workspace):
    from src.errors import UserAbort
    from tests.conftest import answers

    _tmp, posting = workspace
    llm = ScriptedLLM({"analyze": [make_extract(requirements={"min_gpa": {"value": 4.0, "source": "x"}})]})
    answers("n")
    with pytest.raises(UserAbort):
        loop.run_job(Session(master=master, skills_status=status, llm=llm), text_path=posting, today=date(2026, 10, 13))
    assert llm.purposes() == ["analyze"]                                    # nothing was tailored



@needs_soffice
def test_no_research_flag_skips_the_web_search(master, status, workspace):
    _tmp, posting = workspace
    llm = ScriptedLLM({"analyze": [make_extract()], "tailor": [FIRST], "judge": [make_judgment(9, 9, 9, 9)]})
    state = loop.run_job(Session(master=master, skills_status=status, llm=llm), text_path=posting,
                         options=RunOptions(research=False), today=date(2026, 10, 13))
    assert "research" not in llm.purposes() and state.brief is None
    assert state.loop.stop_reason == StopReason.target_reached and state.rendered.pages == 1
