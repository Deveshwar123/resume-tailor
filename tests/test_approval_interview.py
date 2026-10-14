"""Milestone 10: the human approval checkpoint and the interview."""

from __future__ import annotations

from datetime import date

import pytest

import config
from src import approval, interview, loop, render, store, validate
from src.loop import RunOptions
from src.models import Approval, ChangeType, Mode, Ownership, ProposedFact, SkillStatus
from src.startup import Session
from src.validate import ValidationContext
from tests.conftest import SAMPLE_POSTING, ScriptedLLM, answers, bd, ed, make_draft, make_extract, make_judgment

ASSISTED = "Assisted a 4-person team in developing a Python application."
TODAY = date(2026, 10, 13)


@pytest.fixture
def files(tmp_path, monkeypatch, master, status):
    """Point the data files at a temp folder so nothing touches the real ones."""
    monkeypatch.setattr(config, "MASTER_RESUME_PATH", tmp_path / "master_resume.yaml")
    monkeypatch.setattr(config, "SKILLS_STATUS_PATH", tmp_path / "skills_status.yaml")
    monkeypatch.setattr(config, "BACKUP_DIR", tmp_path / "backups")
    (tmp_path / "master_resume.yaml").write_text("# my header\n\n")
    store.save_master_resume(master, backup=False)
    store.save_skills_status(status, backup=False)
    return tmp_path


def flagged(ctx, text="Assisted a 4-person team in developing a scalable Python application", facts=None,
            change="minor_wording"):
    """A resume with one verbatim bullet and one rewrite (labelled "minor" by the AI)."""
    draft = make_draft(experiences=[ed("exp_acme", [
        bd("Wrote SQL queries that reduced report generation time by 30%.", ["exp_acme_f02"], "exp_acme_b02", "unchanged"),
        bd(text, facts or ["exp_acme_f01"], "exp_acme_b01", change),
    ])])
    return validate.enforce(draft, ctx).resume


def target(resume):
    return resume.experiences[0].bullets[1]


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


def test_nothing_to_approve_asks_nothing(ctx):
    resume = validate.enforce(make_draft(experiences=[ed("exp_acme", [bd(ASSISTED, ["exp_acme_f01"], "exp_acme_b01", "unchanged")])]), ctx).resume
    answers()                                    # any prompt would raise EOFError -> default
    assert approval.review(resume, ctx) == [] and approval.summarize([]) == "no changes needed approval"


def test_approve_puts_the_rewrite_on_the_resume(ctx):
    resume = flagged(ctx)
    assert target(resume).approval == Approval.pending
    answers("a")
    decisions = approval.review(resume, ctx)
    assert [d.action for d in decisions] == ["approved"]
    assert target(resume).approval == Approval.approved and "scalable" in target(resume).text
    assert any("technical-complexity" in reason for reason in decisions[0].reasons)


def test_enter_reverts_to_the_verified_wording(ctx):
    resume = flagged(ctx)
    answers("")
    decisions = approval.review(resume, ctx)
    assert decisions[0].action == "reverted" and not decisions[0].automatic
    assert target(resume).text == ASSISTED and target(resume).approval == Approval.reverted


def test_edit_is_revalidated_before_it_is_accepted(ctx):
    resume = flagged(ctx)
    answers("e", "Assisted a 4-person team developing a Python application")
    decisions = approval.review(resume, ctx)
    assert decisions[0].action == "edited"
    assert target(resume).text == "Assisted a 4-person team developing a Python application"


def test_unsupported_edit_can_be_fixed_reverted_or_kept_on_the_record(ctx):
    resume = flagged(ctx)
    answers("e", "Led a 9-person team building a Python application", "e",
            "Assisted a 4-person team building a Python application")
    assert approval.review(resume, ctx)[0].action == "edited"

    resume = flagged(ctx)
    answers("e", "Led a 9-person team building a Python application", "r")
    assert approval.review(resume, ctx)[0].action == "reverted" and target(resume).text == ASSISTED

    resume = flagged(ctx)
    answers("e", "Led a 9-person team building a Python application", "k")
    decision = approval.review(resume, ctx)[0]
    assert decision.action == "override" and target(resume).approval == Approval.override
    assert approval.savable([decision]) == []            # an override is never saved as an approved bullet


def test_batch_mode_reverts_everything_without_asking(ctx):
    resume = flagged(ctx)
    answers("a")                                         # must not be consumed
    decisions = approval.review(resume, ctx, batch=True)
    assert decisions[0].action == "reverted" and decisions[0].automatic
    assert target(resume).text == ASSISTED
    assert "automatically" in approval.summarize(decisions)


def test_no_terminal_means_nothing_can_be_approved(ctx):
    resume = flagged(ctx)                                # conftest leaves prompts non-interactive
    assert approval.review(resume, ctx)[0].automatic and target(resume).text == ASSISTED


@pytest.mark.parametrize("text,reason", [
    ("Built a Python application with a 4-person team", "helper wording"),                              # ownership wording
    ("Assisted a 4-person team in developing a distributed Python application", "technical-complexity"),
    ("Assisted a team in developing a Python application", "drops the metric"),                         # metrics
    ("Assisted a 4-person team in developing a Python application, increasing adoption", "result wording"),
    ("Assisted a 4-person team in developing a company-wide Python application", "scope word"),         # scope
])
def test_changes_that_always_require_approval(ctx, text, reason):
    resume = flagged(ctx, text)
    assert target(resume).approval == Approval.pending
    assert any(reason in f.message for f in target(resume).flags), [f.message for f in target(resume).flags]


def test_minor_wording_is_auto_approved(ctx):
    resume = flagged(ctx, "Assisted a 4-person team in developing a Python app")
    assert target(resume).approval == Approval.auto and target(resume).change_type == ChangeType.minor_wording
    assert approval.review(resume, ctx) == []


def test_strict_mode_asks_about_every_rewrite(master, status, analysis):
    ctx = ValidationContext(master=master, status=status, analysis=analysis, mode=Mode.strict)
    resume = flagged(ctx, "Assisted a 4-person team in developing a Python app")
    assert target(resume).approval == Approval.pending and target(resume).text == ASSISTED
    answers("a")
    approval.review(resume, ctx)
    assert target(resume).text == "Assisted a 4-person team in developing a Python app"


def test_approved_rewrites_can_be_saved_as_approved_bullets(ctx, files):
    resume = flagged(ctx, "Assisted a 4-person team developing a Python application for internal reporting",
                     change="reworded")
    answers("a", "y")
    decisions = approval.review(resume, ctx)
    saved = approval.offer_to_save(ctx.master, decisions)
    assert saved == ["exp_acme_b03"] and decisions[0].saved_as == "exp_acme_b03"
    assert list((files / "backups").glob("master_resume.*.yaml"))            # backed up before writing
    reloaded = store.load_master_resume()
    bullet = next(b for b in reloaded.entry("exp_acme").bullets if b.id == "exp_acme_b03")
    assert bullet.approved and bullet.fact_ids == ["exp_acme_f01"]
    assert (files / "master_resume.yaml").read_text().startswith("# my header")
    # Next run: the same wording is trusted verbatim and needs no approval.
    again = validate.enforce(make_draft(experiences=[ed("exp_acme", [bd(bullet.text, ["exp_acme_f01"], "exp_acme_b03")])]),
                             ValidationContext(master=reloaded, status=ctx.status)).resume
    assert again.experiences[0].bullets[0].approval == Approval.auto
    assert approval.save_approved(ctx.master, decisions) == []               # not saved twice


def test_saving_is_opt_in(ctx, files):
    resume = flagged(ctx)
    answers("a", "")                                     # Enter = do not save
    assert approval.offer_to_save(ctx.master, approval.review(resume, ctx)) == []
    assert not (files / "backups").exists()


# ---------------------------------------------------------------------------
# Interview
# ---------------------------------------------------------------------------

ASK = ("ASK_USER", "Acme", "Did you measure how much faster the reports got?", 0.6)
ASK2 = ("ASK_USER", "skills", "Have you used any cloud platform?", 0.4)
ANSWER = "Yes, I also wrote a small Python script that loaded 3 CSV files into the database every night."


def proposal(**kw):
    base = dict(parent_id="exp_acme", kind="tool_use", ownership="contributor", metrics=["3"], skills=["Python"],
                text="Wrote a Python script that loaded 3 CSV files into the database every night")
    base.update(kw)
    return ProposedFact.model_validate(base)


def fixes(*items):
    return make_judgment(fixes=list(items)).fixes


def test_only_ask_user_fixes_become_questions():
    mixed = fixes(ASK, ("REWRITE", "b", "tighten", 0.2), ("REAL_GAP", "s", "no AWS", 0.3), *[ASK2] * 9)
    asks = interview.questions_from(mixed)
    assert len(asks) == interview.MAX_QUESTIONS and all(f.type.value == "ASK_USER" for f in asks)


def test_approved_answer_becomes_a_dated_interview_fact(master, status, files):
    llm = ScriptedLLM({"interview": [proposal()]})
    answers(ANSWER, "a", "")                             # answer, approve, file under the suggested entry
    result = interview.run_interview(llm, master, status, fixes(ASK), today=TODAY)
    assert result.new_fact_ids == ["exp_acme_f04"] and result.backup is not None and result.backup.exists()
    fact = store.load_master_resume().fact_index()["exp_acme_f04"][1]
    assert fact.source == "interview" and fact.added == "2026-10-13"
    assert fact.metrics == ["3"] and fact.ownership == Ownership.contributor and fact.skills == ["Python"]
    call = llm.calls[0]
    assert call["model"] == config.FAST_MODEL and ANSWER in call["user"] and "exp_acme: Acme Corp" in call["user"]


def test_questions_can_be_skipped_and_nothing_is_written(master, status, files):
    llm = ScriptedLLM()
    answers("", "")
    result = interview.run_interview(llm, master, status, fixes(ASK, ASK2), today=TODAY)
    assert result.asked == 2 and result.skipped == 2 and result.new_fact_ids == []
    assert llm.calls == [] and not (files / "backups").exists()


def test_rejected_fact_is_not_added(master, status, files):
    llm = ScriptedLLM({"interview": [proposal()]})
    answers(ANSWER, "r")
    result = interview.run_interview(llm, master, status, fixes(ASK), today=TODAY)
    assert result.rejected == 1 and result.new_fact_ids == []
    assert "exp_acme_f04" not in store.load_master_resume().fact_index()


def test_fact_can_be_edited_before_approval(master, status, files):
    llm = ScriptedLLM({"interview": [proposal()]})
    answers(ANSWER, "e", "Wrote a Python script that loaded CSV files nightly", "member", "a", "")
    interview.run_interview(llm, master, status, fixes(ASK), today=TODAY)
    fact = master.fact_index()["exp_acme_f04"][1]
    assert fact.text == "Wrote a Python script that loaded CSV files nightly"
    assert fact.ownership == Ownership.member and fact.metrics == []        # the edited text no longer has the "3"


def test_embellishments_are_called_out_against_my_own_answer():
    inflated = proposal(text="Led a team of 12 building a Python pipeline that loaded 3 CSV files", ownership="lead",
                        metrics=["12", "3"])
    warnings = interview.answer_warnings(inflated, ANSWER)
    assert any('"12" is not in your answer' in w for w in warnings)
    assert any('says "lead" but your answer did not' in w for w in warnings)
    assert any('ownership "lead" is higher' in w for w in warnings)
    assert interview.answer_warnings(proposal(), ANSWER) == []


def test_unknown_parent_is_chosen_from_a_list_or_rejected(master, status, files):
    llm = ScriptedLLM({"interview": [proposal(parent_id=None), proposal(parent_id="exp_nowhere")]})
    answers(ANSWER, "a", "2", ANSWER, "a", "")            # pick entry 2 (Data Club); then Enter = reject
    result = interview.run_interview(llm, master, status, fixes(ASK, ASK2), today=TODAY)
    assert result.new_fact_ids == ["exp_club_f03"] and result.rejected == 1


def test_unverified_skills_need_my_confirmation(master, status, files):
    llm = ScriptedLLM({"interview": [proposal(skills=["Python", "Docker", "Snowflake"],
                                              text="Wrote a Python script run in Docker that loaded 3 CSV files into Snowflake")]})
    answers(ANSWER, "a", "", "y", "n")                    # Docker: yes, I can back it up; Snowflake: no
    interview.run_interview(llm, master, status, fixes(ASK), today=TODAY)
    assert master.fact_index()["exp_acme_f04"][1].skills == ["Python", "Docker"]
    saved = store.load_skills_status()
    assert saved.status_of("Docker") == SkillStatus.verified and saved.status_of("Snowflake") is None


def test_unusable_answers_and_bad_metrics_are_dropped(master):
    assert interview.propose_fact(ScriptedLLM({"interview": [proposal(text="  ")]}), master, "q", "no") is None
    cleaned = interview.propose_fact(ScriptedLLM({"interview": [proposal(metrics=["3", "40%"])]}), master, "q", ANSWER)
    assert cleaned.metrics == ["3"]                       # "40%" is not in the fact text


def test_no_terminal_means_no_interview(master, status, files):
    from src import ui

    ui.set_input(None, interactive=False)
    result = interview.run_interview(ScriptedLLM(), master, status, fixes(ASK), today=TODAY)
    assert result.asked == 0


# ---------------------------------------------------------------------------
# End to end: below target -> interview -> re-tailor -> loop once more -> approval
# ---------------------------------------------------------------------------

needs_soffice = pytest.mark.skipif(render.find_soffice() is None, reason="LibreOffice (soffice) is not installed")

FIRST = make_draft(experiences=[ed("exp_acme", [bd(ASSISTED, ["exp_acme_f01"], "exp_acme_b01", "unchanged")])],
                   skills=["Python", "SQL"])
SECOND = make_draft(experiences=[ed("exp_acme", [
    bd(ASSISTED, ["exp_acme_f01"], "exp_acme_b01", "unchanged"),
    bd("Automated a nightly load of 3 CSV files into the database with a Python script", ["exp_acme_f04"]),
])], skills=["Python", "SQL"])


@needs_soffice
def test_interview_then_retailor_then_approval(master, status, files, tmp_path, monkeypatch):
    for name, sub in (("OUTPUT_DIR", "output"), ("POSTINGS_DIR", "postings"), ("CACHE_DIR", "cache")):
        monkeypatch.setattr(config, name, tmp_path / sub)
    posting = tmp_path / "job.txt"
    posting.write_text(SAMPLE_POSTING)
    llm = ScriptedLLM({
        "analyze": [make_extract()],
        "tailor": [FIRST, SECOND],
        "judge": [make_judgment(7, 7, 7, 7, fixes=[ASK]), make_judgment(9, 9, 9, 9), make_judgment(9, 9, 9, 9)],
        "interview": [proposal()],
    })
    # eligibility: continue | interview: answer, approve, file under Acme | approval: approve | save: no
    answers("", ANSWER, "a", "", "a", "")
    state = loop.run_job(Session(master=master, skills_status=status, llm=llm), text_path=posting,
                         options=RunOptions(research=False), today=TODAY)
    assert llm.purposes() == ["analyze", "tailor", "judge", "interview", "tailor", "judge"]
    assert state.interview.new_fact_ids == ["exp_acme_f04"] and len(state.loops) == 2
    assert state.ctx.new_fact_ids == frozenset({"exp_acme_f04"})
    second_prompt = [c for c in llm.calls if c["purpose"] == "tailor"][1]["system"]
    assert "exp_acme_f04" in second_prompt and '"new": true' in second_prompt
    # The bullet built from the new fact was held for approval, and I approved it.
    assert [d.action for d in state.decisions] == ["approved"]
    assert any("interview" in reason for reason in state.decisions[0].reasons)
    assert "Automated a nightly load of 3 CSV files" in state.rendered.view.plain_text()
    assert state.final_quality.total >= 9.0 and state.rendered.pages == 1
    assert "exp_acme_f04" in store.load_master_resume().fact_index()         # persisted, with a backup
    assert list((files / "backups").glob("master_resume.*.yaml"))


@needs_soffice
def test_no_interview_flag_never_asks(master, status, files, tmp_path, monkeypatch):
    for name, sub in (("OUTPUT_DIR", "output"), ("POSTINGS_DIR", "postings"), ("CACHE_DIR", "cache")):
        monkeypatch.setattr(config, name, tmp_path / sub)
    posting = tmp_path / "job.txt"
    posting.write_text(SAMPLE_POSTING)
    llm = ScriptedLLM({"analyze": [make_extract()], "tailor": [FIRST], "judge": [make_judgment(7, 7, 7, 7, fixes=[ASK])]})
    answers("")
    state = loop.run_job(Session(master=master, skills_status=status, llm=llm), text_path=posting,
                         options=RunOptions(research=False, interview=False), today=TODAY)
    assert "interview" not in llm.purposes() and state.interview is None
    assert state.final_quality.total < 9.0 and "question(s) for me" in state.stop_note
