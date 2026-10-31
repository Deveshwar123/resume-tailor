"""Milestone 5: resume tailoring (normal and --strict), with Claude scripted."""

from __future__ import annotations

import json

import config
from src import tailor_resume, validate
from src.errors import LLMError
from src.models import (
    Approval, BriefItem, BulletFix, BulletFixBatch, ChangeType, CompanyBrief, Mode, Tier,
)
from src.validate import ValidationContext
from tests.conftest import ScriptedLLM, bd, ed, make_draft

ASSISTED = "Assisted a 4-person team in developing a Python application."


def test_system_prompt_carries_rules_fact_bank_and_job(ctx):
    system = tailor_resume.build_system(ctx)
    assert "never" in system.lower() and "<fact_bank>" in system and "<job>" in system
    bank = json.loads(system.split("<fact_bank>\n")[1].split("\n</fact_bank>")[0])
    acme = next(e for e in bank["experiences"] if e["source_id"] == "exp_acme")
    assert acme["org"] == "Acme Corp" and acme["start"] == "2025-06"
    assert acme["facts"][0] == {"id": "exp_acme_f01", "ownership": "contributor", "kind": "responsibility",
                                "text": "Assisted a 4-person team in developing a Python application",
                                "metrics": ["4-person"], "skills": ["Python"]}
    assert acme["approved_bullets"][0]["id"] == "exp_acme_b01"
    # Role-level ownership is resolved per fact so Claude sees the real ceiling.
    lab = next(e for e in bank["experiences"] if e["source_id"] == "exp_lab")
    assert lab["facts"][0]["ownership"] == "member"
    assert json.loads(system.split("<job>\n")[1].split("\n</job>")[0])["company"] == "Acme Corp"


def test_system_prompt_is_stable_so_it_can_be_cached(ctx):
    assert tailor_resume.build_system(ctx) == tailor_resume.build_system(ctx)


def test_contact_details_are_never_sent_to_the_tailor(ctx):
    system = tailor_resume.build_system(ctx)
    assert ctx.master.contact.phone not in system and ctx.master.contact.email not in system


def test_only_verified_master_skills_are_offered(master_data, status, analysis):
    from src.models import MasterResume

    master_data["skills"]["tools"].append("PyTorch")            # in the master list but only "learning"
    ctx = ValidationContext(master=MasterResume.model_validate(master_data), status=status, analysis=analysis)
    allowed = tailor_resume.allowed_skills(ctx)
    assert "PyTorch" not in allowed and "Python" in allowed


def test_brief_drops_unverified_items_and_labels_weights(ctx):
    brief = CompanyBrief(
        company="Acme",
        tech_stack=[BriefItem(text="Acme runs Python services", tier=Tier.official),
                    BriefItem(text="Rumor: moving to Rust", tier=Tier.unverified)],
        recent_news=[BriefItem(text="Acme raised a Series C", tier=Tier.third_party)],
        intern_work=[BriefItem(text="Interns ship production code", tier=Tier.posting)],
    )
    system = tailor_resume.build_system(ctx, brief)
    research = json.loads(system.split("<company_research>\n")[1].split("\n</company_research>")[0])
    assert research["tech_stack"] == [{"text": "Acme runs Python services", "weight": "heavy"}]
    assert research["recent_news"][0]["weight"] == "light" and research["intern_work"][0]["weight"] == "heavy"
    assert "Rumor" not in system
    assert "<company_research>" not in tailor_resume.build_system(ctx, CompanyBrief())


def test_tailor_uses_the_write_model_and_validates_the_draft(ctx):
    draft = make_draft(
        experiences=[ed("exp_acme", [bd(ASSISTED, ["exp_acme_f01"], "exp_acme_b01", "unchanged"),
                                     bd("Wrote SQL queries that cut report generation time by 30%", ["exp_acme_f02"],
                                        "exp_acme_b02", "minor_wording")])],
        skills=["Python", "SQL", "Kubernetes"],
    )
    llm = ScriptedLLM({"tailor": [draft]})
    outcome, system = tailor_resume.tailor(llm, ctx)
    call = llm.calls[0]
    assert call["model"] == config.WRITE_MODEL and call["schema"].__name__ == "TailoredResumeDraft"
    assert call["system"] == system
    assert outcome.resume.skills == ["Python", "SQL"]            # Kubernetes is not mine to list
    # Nothing failed hard, so no retry call. One of the two bullets is verbatim (over the 25% share),
    # so the rewording pass runs once; the scripted pass finds nothing better.
    assert llm.purposes() == ["tailor", "reword"]


def test_inflated_bullet_triggers_one_fix_call_with_the_violation(ctx):
    draft = make_draft(experiences=[ed("exp_acme", [bd("Led development of a Python application", ["exp_acme_f01"])])])
    fix = BulletFixBatch(fixes=[BulletFix(key="exp_acme#0", fact_ids=["exp_acme_f01"],
                                          rewritten_text="Supported a 4-person team developing a Python application")])
    llm = ScriptedLLM({"tailor": [draft], "fix": [fix]})
    outcome, system = tailor_resume.tailor(llm, ctx)
    assert llm.purposes() == ["tailor", "fix"]
    fix_call = llm.calls[1]
    assert fix_call["system"] == system                          # same cached prefix
    assert "(lead level) but the cited facts support at most contributor" in fix_call["user"]
    assert "exp_acme#0" in fix_call["user"]
    assert "Assisted a 4-person team in developing a Python application" in fix_call["user"]   # the cited fact
    assert outcome.resume.experiences[0].bullets[0].text.startswith("Supported a 4-person team")


def test_fix_call_failure_falls_back_to_verified_wording(ctx):
    draft = make_draft(experiences=[ed("exp_acme", [bd("Led development of a Python application", ["exp_acme_f01"])])])

    def boom(_kwargs):
        raise LLMError("overloaded")

    llm = ScriptedLLM({"tailor": [draft], "fix": [boom]})
    outcome, _ = tailor_resume.tailor(llm, ctx)
    assert outcome.resume.experiences[0].bullets[0].text == ASSISTED


def test_strict_mode_changes_the_prompt_and_holds_rewrites_for_approval(master, status, analysis):
    ctx = ValidationContext(master=master, status=status, analysis=analysis, mode=Mode.strict)
    draft = make_draft(experiences=[ed("exp_acme", [
        bd(ASSISTED, ["exp_acme_f01"], "exp_acme_b01", "unchanged"),
        bd("Wrote SQL queries that cut report generation time by 30%", ["exp_acme_f02"], "exp_acme_b02", "minor_wording"),
    ])])
    llm = ScriptedLLM({"tailor": [draft]})
    outcome, system = tailor_resume.tailor(llm, ctx)
    assert "Mode: STRICT" in system and "Mode: normal" not in system
    kept, proposal = outcome.resume.experiences[0].bullets
    assert kept.approval == Approval.auto and kept.final_text == ASSISTED
    assert proposal.approval == Approval.pending
    assert proposal.final_text == "Wrote SQL queries that reduced report generation time by 30%."   # original stays
    assert "cut report generation" in proposal.rewritten_text                                        # proposal alongside


def test_empty_draft_falls_back_to_the_verified_baseline(ctx):
    llm = ScriptedLLM({"tailor": [make_draft()]})
    outcome, _ = tailor_resume.tailor(llm, ctx)
    assert outcome.resume.experiences and outcome.violations == []


def test_baseline_is_verified_wording_ranked_by_relevance(ctx):
    draft = tailor_resume.baseline_draft(ctx, max_bullets=3)
    outcome = validate.enforce(draft, ctx)
    assert outcome.violations == []
    bullets = [b for _e, _i, b in outcome.resume.bullets()]
    assert len(bullets) <= 3 and all(b.change_type == ChangeType.unchanged for b in bullets)
    # Acme (Python and SQL, the required skills) is the most relevant entry for this posting.
    assert draft.experiences[0].source_id == "exp_acme"
    assert draft.skills[:2] == ["Python", "SQL"]


# ---------------------------------------------------------------------------
# Tailoring rewrites: the prompt asks for it, and verbatim drafts get a second pass
# ---------------------------------------------------------------------------

VERBATIM = make_draft(
    experiences=[ed("exp_acme", [bd(ASSISTED, ["exp_acme_f01"], "exp_acme_b01", "unchanged"),
                                 bd("Wrote SQL queries that reduced report generation time by 30%.", ["exp_acme_f02"],
                                    "exp_acme_b02", "unchanged")])],
    projects=[ed("proj_tracker", [bd("Built a budget tracking web app with JavaScript and React.", ["proj_tracker_f01"],
                                     "proj_tracker_b01", "unchanged")])],
    skills=["Python", "SQL"],
)


def reword(*items):
    from src.models import RewordBatch

    return RewordBatch.model_validate({"rewrites": [
        {"key": k, "rewritten_text": t, "posting_terms": [], "reason": "fits the posting"} for k, t in items]})


def test_normal_mode_asks_for_a_rewrite_per_bullet_and_names_the_claim_words(ctx):
    system = tailor_resume.build_system(ctx)
    assert "A draft that repeats them word for word is not tailored" in system
    assert "Expect to rewrite most bullets" in system and "good starting points" not in system
    # The words the validator flags are spelled out, so rewrites avoid them instead of being reverted.
    assert "company-wide" in system and "scalable" in system and "resulting in" in system


def test_verbatim_bullets_get_one_reword_pass_judged_by_the_validator_and_the_posting(ctx):
    llm = ScriptedLLM({"tailor": [VERBATIM], "reword": [reword(
        ("exp_acme#0", "Led a 4-person team developing a Python application."),           # ownership inflated: hard
        ("exp_acme#1", "Wrote queries that reduced report generation time by 30%."),       # drops SQL: less posting language
        ("proj_tracker#0", "Built a JavaScript and React web app for budget tracking."),   # passes
        ("exp_zzz#9", "Built a rocket."),                                                  # not a bullet that was sent
    )]})
    outcome, system = tailor_resume.tailor(llm, ctx)
    call = llm.calls[1]
    assert llm.purposes() == ["tailor", "reword"] and call["system"] == system and call["model"] == config.WRITE_MODEL
    assert '"key": "exp_acme#0"' in call["user"] and '"ownership": "contributor"' in call["user"]
    acme, tracker = outcome.resume.experiences[0], outcome.resume.projects[0]
    assert [b.text for b in acme.bullets] == [ASSISTED, "Wrote SQL queries that reduced report generation time by 30%."]
    assert "used less of the posting's language" in acme.bullets[1].history[-1]
    kept = tracker.bullets[0]
    assert kept.rewritten_text == "Built a JavaScript and React web app for budget tracking."
    assert kept.change_type == ChangeType.reworded and kept.original_text.startswith("Built a budget tracking")
    assert kept.approval == Approval.pending and [f.code for f in kept.flags] == ["change"]   # approval still applies
    assert outcome.reworded == 1
    rejected = [v for v in outcome.violations if v.resolution.startswith("reword candidate rejected")]
    assert [v.code for v in rejected] == ["B3"]
    assert "3 bullets, 1 reworded (1 in the second pass)" in tailor_resume.summarize(outcome)


def test_no_reword_pass_when_the_draft_is_already_tailored(ctx):
    tailored = make_draft(experiences=[ed("exp_acme", [
        bd("Assisted a 4-person team in building a Python application for data work", ["exp_acme_f01"], "exp_acme_b01"),
        bd("Wrote SQL queries that cut report generation time by 30%", ["exp_acme_f02"], "exp_acme_b02", "minor_wording"),
        bd("Tested REST API endpoints in Python with pytest", ["exp_acme_f03", "exp_acme_f01"])])])
    llm = ScriptedLLM({"tailor": [tailored]})
    tailor_resume.tailor(llm, ctx)
    assert llm.purposes() == ["tailor"]


def test_strict_mode_never_runs_the_reword_pass(master, status, analysis):
    ctx = ValidationContext(master=master, status=status, analysis=analysis, mode=Mode.strict)
    llm = ScriptedLLM({"tailor": [VERBATIM]})
    tailor_resume.tailor(llm, ctx)
    assert llm.purposes() == ["tailor"]


def test_a_failed_reword_call_keeps_the_verified_wording(ctx):
    def boom(_kwargs):
        raise LLMError("overloaded")

    llm = ScriptedLLM({"tailor": [VERBATIM], "reword": [boom]})
    outcome, _ = tailor_resume.tailor(llm, ctx)
    assert outcome.reworded == 0 and outcome.resume.experiences[0].bullets[0].text == ASSISTED
