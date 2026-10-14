"""Milestone 4: evidence-based anti-fabrication (validate.py). No AI anywhere here."""

from __future__ import annotations

import pytest

from src import validate
from src.models import Approval, BulletFix, ChangeType, MasterResume, Mode, Severity
from src.validate import ValidationContext, enforce
from tests.conftest import bd, ed, make_analysis, make_draft

ASSISTED = "Assisted a 4-person team in developing a Python application."


def codes(outcome, severity=None):
    return [v.code for v in outcome.violations if severity is None or v.severity == severity]


def only_bullet(outcome):
    return outcome.resume.experiences[0].bullets[0]


def run(ctx, text, fact_ids, entry="exp_acme", source=None, change="reworded", retry_fn=None):
    return enforce(make_draft(experiences=[ed(entry, [bd(text, fact_ids, source, change)])]), ctx, retry_fn)


# ---------------------------------------------------------------------------
# B3 - ownership: "assisted" must never become "led"
# ---------------------------------------------------------------------------


def test_assisted_must_never_become_led(ctx):
    outcome = run(ctx, "Led development of a Python application", ["exp_acme_f01"])
    hard = [v for v in outcome.violations if v.severity == Severity.hard and v.code == "B3"]
    assert hard and "lead" in hard[0].message and "contributor" in hard[0].message
    # The inflated bullet never reaches the resume: it reverts to verified wording.
    bullet = only_bullet(outcome)
    assert bullet.text == ASSISTED and bullet.change_type == ChangeType.unchanged
    assert "Led" not in bullet.text and bullet.approval == Approval.auto


@pytest.mark.parametrize("text", [
    "Managed a 4-person team developing a Python application",
    "Spearheaded a 4-person Python application build",
    "Founded a 4-person team to develop a Python application",
    "Served as team lead for a 4-person Python application",
    "Directed a 4-person team in developing a Python application",
])
def test_other_ownership_inflations_fail(ctx, text):
    assert "B3" in codes(run(ctx, text, ["exp_acme_f01"]), Severity.hard)
    assert only_bullet(run(ctx, text, ["exp_acme_f01"])).text == ASSISTED


def test_ownership_at_or_below_the_fact_level_is_allowed(ctx):
    outcome = run(ctx, "Led weekly data analysis workshops for 25 members", ["exp_club_f01"], entry="exp_club")
    assert not codes(outcome, Severity.hard)
    assert only_bullet(outcome).text.startswith("Led weekly data analysis workshops")


def test_member_cannot_claim_contributor_verbs(ctx):
    outcome = run(ctx, "Contributed to weekly lab meetings and annotated 500 images", ["exp_lab_f01"], entry="exp_lab")
    assert "B3" in codes(outcome, Severity.hard)


def test_ambiguous_verbs_and_dropped_helper_wording_go_to_approval(ctx):
    outcome = run(ctx, "Built a Python application with a 4-person team", ["exp_acme_f01"])
    assert not codes(outcome, Severity.hard)
    bullet = only_bullet(outcome)
    assert bullet.approval == Approval.pending
    messages = " | ".join(f.message for f in bullet.flags)
    assert '"build"' in messages                      # ambiguous: solo "built" vs. as part of a team
    assert "drops the helper wording" in messages     # "assisted" disappeared


def test_combining_facts_with_different_ownership_is_flagged(master_data, status):
    master_data["experience"][1]["facts"][1]["ownership"] = "contributor"
    ctx = ValidationContext(master=MasterResume.model_validate(master_data), status=status)
    outcome = run(ctx, "Led weekly workshops for 25 members and a campus hackathon with 60+ participants",
                  ["exp_club_f01", "exp_club_f02"], entry="exp_club")
    assert any("different ownership levels" in f.message for f in only_bullet(outcome).flags)


# ---------------------------------------------------------------------------
# B1 - metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Wrote SQL queries that reduced report generation time by 45%",       # invented number
    "Wrote SQL queries that reduced report generation time by over 30%",  # inflated
    "Wrote SQL queries that reduced report generation time by 30%+",
    "Wrote 12 SQL queries that reduced report generation time by 30%",    # extra count
    "Wrote SQL queries that cut report generation time in half",
    "Wrote SQL queries that saved $30 in report generation",              # wrong unit
])
def test_invented_or_inflated_metrics_fail(ctx, text):
    outcome = run(ctx, text, ["exp_acme_f02"])
    assert "B1" in codes(outcome, Severity.hard)
    assert only_bullet(outcome).text == "Wrote SQL queries that reduced report generation time by 30%."


def test_supported_metrics_pass_in_any_spelling(ctx):
    assert not codes(run(ctx, "Reduced report generation time by 30 percent with SQL queries", ["exp_acme_f02"]), Severity.hard)
    assert not codes(run(ctx, "Supported a team of four developing a Python application", ["exp_acme_f01"]), Severity.hard)


def test_years_from_the_entry_dates_are_allowed(ctx):
    assert not codes(run(ctx, "Tested REST API endpoints with pytest in 2025", ["exp_acme_f03"]), Severity.hard)
    assert "B1" in codes(run(ctx, "Tested REST API endpoints with pytest in 2023", ["exp_acme_f03"]), Severity.hard)


def test_dropping_a_metric_needs_approval(ctx):
    outcome = run(ctx, "Assisted a team in developing a Python application", ["exp_acme_f01"])
    assert any("drops the metric" in f.message for f in only_bullet(outcome).flags)


# ---------------------------------------------------------------------------
# B2 - skills and tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,needle", [
    ("Assisted a 4-person team in developing a Python and PyTorch application", "learning"),   # not verified
    ("Assisted a 4-person team in developing a Python application on AWS", "missing"),
    ("Assisted a 4-person team in developing a Python application on Kubernetes", "no cited fact"),
    ("Assisted a 4-person team in developing a Python and SQL application", "no cited fact"),   # verified, wrong fact
])
def test_unverified_or_unsupported_skills_fail(ctx, text, needle):
    outcome = run(ctx, text, ["exp_acme_f01"])
    hard = [v for v in outcome.violations if v.code == "B2" and v.severity == Severity.hard]
    assert hard and needle in hard[0].message
    assert only_bullet(outcome).text == ASSISTED


def test_research_or_posting_tools_cannot_be_added(status, master):
    # The company uses Snowflake and the posting lists it - it still cannot appear.
    ctx = ValidationContext(master=master, status=status, analysis=make_analysis(tools=["Snowflake", "Python"]))
    outcome = run(ctx, "Wrote SQL queries in Snowflake that reduced report generation time by 30%", ["exp_acme_f02"])
    assert "B2" in codes(outcome, Severity.hard)


def test_posting_language_without_support_is_a_soft_flag(status, master):
    ctx = ValidationContext(master=master, status=status, analysis=make_analysis(
        required_skills=[{"name": "data pipelines", "weight": 3, "kind": "technical"},
                         {"name": "testing", "weight": 1, "kind": "technical"}]))
    outcome = run(ctx, "Assisted a 4-person team in developing Python data pipelines", ["exp_acme_f01"])
    assert not codes(outcome, Severity.hard)
    assert any("posting's term" in f.message and "data pipelines" in f.message for f in only_bullet(outcome).flags)
    # "testing" is loosely supported by the fact's own word "Tested", so it is not flagged.
    outcome = run(ctx, "Performed testing of REST API endpoints with pytest", ["exp_acme_f03"])
    assert not any("posting's term" in f.message for f in only_bullet(outcome).flags)


def test_new_names_are_flagged(ctx):
    outcome = run(ctx, "Assisted a 4-person team at Globex in developing a Python application", ["exp_acme_f01"])
    assert any('"Globex"' in f.message for f in only_bullet(outcome).flags)


def test_skills_tagged_on_a_fact_can_be_named(ctx):
    outcome = run(ctx, "Tested REST endpoints using pytest", ["exp_acme_f03"])
    assert not codes(outcome, Severity.hard)


# ---------------------------------------------------------------------------
# B4 - scope, and the approval triggers
# ---------------------------------------------------------------------------


def test_scope_inflation_is_flagged_for_approval(ctx):
    outcome = run(ctx, "Wrote SQL queries for all company-wide reports, reducing generation time by 30%", ["exp_acme_f02"])
    assert not codes(outcome, Severity.hard)
    bullet = only_bullet(outcome)
    flagged = {v.token for v in outcome.violations if v.code == "B4"}
    assert flagged == {"all", "company-wide"} and bullet.approval == Approval.pending


@pytest.mark.parametrize("text,code", [
    ("Assisted a 4-person team in developing a scalable Python application", "C1"),
    ("Assisted a 4-person team in developing a Python application, improving delivery speed", "C2"),
    ("Assisted a 4-person team in developing an end-to-end Python application", "B4"),
])
def test_complexity_results_and_scope_wording_need_approval(ctx, text, code):
    outcome = run(ctx, text, ["exp_acme_f01"])
    assert code in codes(outcome, Severity.soft) and only_bullet(outcome).approval == Approval.pending


# ---------------------------------------------------------------------------
# A - structure: fabricated ids, altered employers and dates
# ---------------------------------------------------------------------------


def test_altered_employer_is_removed_and_replaced_with_verified_content(ctx):
    draft = make_draft(experiences=[ed("exp_acme", [bd("Engineered a platform used by millions", ["exp_acme_f01"])],
                                       org="Acme Global Technologies Inc.")])
    outcome = enforce(draft, ctx)
    assert "A5" in codes(outcome, Severity.hard)
    entry = outcome.resume.experiences[0]
    assert entry.replaced and all(b.change_type == ChangeType.unchanged for b in entry.bullets)
    assert "millions" not in " ".join(b.text for b in entry.bullets)


@pytest.mark.parametrize("echo", [{"title": "Senior Software Engineer"}, {"start": "2023-06"}, {"end": "present"}])
def test_altered_title_or_dates_are_caught(ctx, echo):
    outcome = enforce(make_draft(experiences=[ed("exp_acme", [bd(ASSISTED, ["exp_acme_f01"])], **echo)]), ctx)
    assert "A5" in codes(outcome, Severity.hard)


def test_equivalent_date_spellings_are_not_alterations(ctx):
    outcome = enforce(make_draft(experiences=[ed("exp_acme", [bd(ASSISTED, ["exp_acme_f01"])],
                                                 start="June 2025", end="Aug 2025", org="ACME CORP")]), ctx)
    assert "A5" not in codes(outcome)


def test_nonexistent_experience_is_replaced_by_the_next_most_relevant(ctx):
    draft = make_draft(experiences=[
        ed("exp_club", [bd("Led weekly workshops for 25 members on data analysis", ["exp_club_f01"])]),
        ed("exp_megacorp", [bd("Ran the data platform", ["exp_megacorp_f01"])], org="MegaCorp", title="Staff Engineer"),
    ])
    outcome = enforce(draft, ctx)
    assert "A4" in codes(outcome, Severity.hard)
    ids = [e.source_id for e in outcome.resume.experiences]
    # Acme (Python + SQL, the posting's required skills) is the most relevant unselected entry.
    assert ids == ["exp_club", "exp_acme"]
    assert "MegaCorp" not in str(outcome.resume.model_dump())


def test_duplicate_selection_is_dropped(ctx):
    draft = make_draft(experiences=[ed("exp_acme", [bd(ASSISTED, ["exp_acme_f01"])]),
                                    ed("exp_acme", [bd(ASSISTED, ["exp_acme_f01"])])])
    outcome = enforce(draft, ctx)
    assert [e.source_id for e in outcome.resume.experiences] == ["exp_acme"]


def test_fabricated_and_foreign_fact_ids_are_stripped(ctx):
    outcome = run(ctx, "Assisted a 4-person team in developing a Python application for 25 members",
                  ["exp_acme_f01", "exp_acme_f99", "exp_club_f01"])
    found = codes(outcome, Severity.hard)
    assert "A1" in found and "A3" in found
    assert "B1" in found                       # "25" came from another entry's fact, so it is unsupported here
    assert only_bullet(outcome).text == ASSISTED


def test_bullet_with_no_valid_fact_is_dropped_and_entry_keeps_verified_wording(ctx):
    outcome = run(ctx, "Annotated 9,000 images and trained a vision model", ["exp_lab_f77"], entry="exp_lab")
    bullets = outcome.resume.experiences[0].bullets
    assert [b.text for b in bullets] == ["Participated in weekly lab meetings and annotated 500 images"]


def test_skills_list_must_be_verified_master_skills(ctx):
    outcome = enforce(make_draft(skills=["Python", "ML", "PyTorch", "Rust", "sql", "python3", "React"]), ctx)
    assert outcome.resume.skills == ["Python", "SQL", "React"]     # master spelling, de-duplicated
    assert codes(outcome, Severity.hard).count("B5") == 3           # ML and Rust not in list; PyTorch not verified


def test_honors_coursework_and_awards_must_exist(ctx):
    outcome = enforce(make_draft(honors=["Dean's List", "Rhodes Scholar"], coursework=["Databases", "Quantum Computing"],
                                 award_ids=["award_hack", "award_nobel"]), ctx)
    assert outcome.resume.honors == ["Dean's List"] and outcome.resume.coursework == ["Databases"]
    assert outcome.resume.award_ids == ["award_hack"] and codes(outcome).count("A6") == 3


def test_section_order_is_completed_and_empty_sections_dropped(ctx):
    draft = make_draft(experiences=[ed("exp_acme", [bd(ASSISTED, ["exp_acme_f01"])]),
                                    ed("exp_club", [bd("Led weekly workshops for 25 members on data analysis", ["exp_club_f01"])])],
                       skills=["Python"], order=["skills", "projects", "experience"])
    order = [s.value for s in enforce(draft, ctx).resume.section_order]
    assert order == ["skills", "experience", "education", "leadership"]     # no projects were selected


# ---------------------------------------------------------------------------
# C - outcomes: one retry, then revert or drop
# ---------------------------------------------------------------------------


def test_hard_violation_gets_exactly_one_retry_with_the_specific_violation(ctx):
    seen = []

    def retry(requests):
        seen.extend(requests)
        return [BulletFix(key=requests[0].key, rewritten_text="Supported a 4-person team developing a Python application",
                          fact_ids=["exp_acme_f01"])]

    outcome = run(ctx, "Led development of a Python application", ["exp_acme_f01"], retry_fn=retry)
    assert len(seen) == 1 and outcome.retried == 1
    assert any('"lead"' in message for message in seen[0].violations)
    bullet = only_bullet(outcome)
    assert bullet.text.startswith("Supported a 4-person team")
    assert "rewritten once after a validation failure" in bullet.history
    assert any(v.resolution == "fixed by the one allowed retry" for v in outcome.violations)


def test_failed_retry_reverts_to_verified_wording(ctx):
    calls = []

    def retry(requests):
        calls.append(requests)
        return [BulletFix(key=requests[0].key, rewritten_text="Headed development of a Python application",
                          fact_ids=["exp_acme_f01"])]

    outcome = run(ctx, "Led development of a Python application", ["exp_acme_f01"], retry_fn=retry)
    assert len(calls) == 1                                  # never a second retry
    assert only_bullet(outcome).text == ASSISTED
    assert any("retry still failed" in v.resolution for v in outcome.violations)


def test_revert_uses_plain_fact_text_when_no_approved_bullet_exists(ctx):
    outcome = run(ctx, "Managed a campus hackathon with 600 participants", ["exp_club_f02"], entry="exp_club")
    assert only_bullet(outcome).text == "Organized a campus hackathon with 60+ participants"
    assert only_bullet(outcome).original_kind == "fact"


# ---------------------------------------------------------------------------
# Change types and approval state
# ---------------------------------------------------------------------------


def test_verbatim_approved_bullet_passes_untouched(ctx):
    outcome = run(ctx, ASSISTED, ["exp_acme_f01"], source="exp_acme_b01", change="unchanged")
    bullet = only_bullet(outcome)
    assert outcome.violations == [] and bullet.approval == Approval.auto
    assert bullet.change_type == ChangeType.unchanged and bullet.original_kind == "approved_bullet"


def test_ai_cannot_hide_a_rewrite_by_labelling_it_minor(ctx):
    outcome = run(ctx, "Partnered with three engineers to ship an internal Python tool", ["exp_acme_f01"],
                  source="exp_acme_b01", change="minor_wording")
    assert only_bullet(outcome).change_type != ChangeType.minor_wording or only_bullet(outcome).approval == Approval.pending


def test_minor_wording_is_auto_approved_and_rewording_needs_approval(ctx):
    minor = only_bullet(run(ctx, "Assisted a 4-person team in developing a Python app", ["exp_acme_f01"],
                            source="exp_acme_b01", change="minor_wording"))
    assert minor.change_type == ChangeType.minor_wording and minor.approval == Approval.auto
    reworded = only_bullet(run(ctx, "Assisted a 4-person team developing a Python application for internal reporting needs",
                               ["exp_acme_f01"], source="exp_acme_b01"))
    assert reworded.approval == Approval.pending and any(f.code == "change" for f in reworded.flags)


def test_combining_facts_is_marked_combined_and_needs_approval(ctx):
    outcome = run(ctx, "Assisted a 4-person team in developing a Python application and tested REST API endpoints with pytest",
                  ["exp_acme_f01", "exp_acme_f03"])
    bullet = only_bullet(outcome)
    assert bullet.change_type == ChangeType.combined and bullet.approval == Approval.pending


def test_strict_mode_keeps_verified_wording_until_i_approve(master, status, analysis):
    ctx = ValidationContext(master=master, status=status, analysis=analysis, mode=Mode.strict)
    bullet = only_bullet(run(ctx, "Assisted a 4-person team in developing a Python app", ["exp_acme_f01"],
                             source="exp_acme_b01", change="minor_wording"))
    assert bullet.approval == Approval.pending and any(f.code == "strict" for f in bullet.flags)
    assert bullet.final_text == ASSISTED                      # the original stays on the resume
    assert bullet.rewritten_text.endswith("Python app")       # the proposal is kept alongside it


def test_bullets_from_new_interview_facts_need_approval(master, status, analysis):
    ctx = ValidationContext(master=master, status=status, analysis=analysis, new_fact_ids=frozenset({"exp_acme_f03"}))
    bullet = only_bullet(run(ctx, "Tested REST API endpoints using pytest", ["exp_acme_f03"], change="minor_wording"))
    assert bullet.approval == Approval.pending and any(f.code == "interview" for f in bullet.flags)


def test_recheck_validates_a_single_edited_bullet(ctx):
    assert validate.recheck(ASSISTED, "exp_acme", ["exp_acme_f01"], "exp_acme_b01", ctx) == []
    found = validate.recheck("Led a 9-person team building a Rust application", "exp_acme", ["exp_acme_f01"], None, ctx)
    assert {"B1", "B2", "B3"} <= {v.code for v in found if v.severity == Severity.hard}


# ---------------------------------------------------------------------------
# "manage" depends on what is managed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,people", [
    ("Managed a 4-person team developing a Python application", True),
    ("Managed a team of 5 engineers", True),
    ("Managed interns and volunteers", True),
    ("Managed and mentored three interns", True),          # no object of its own: the cautious reading
    ("Raised and managed team funds through grant writing", False),
    ("Managed student records for the club", False),
    ("Managed the engineering team's roadmap", False),
    ("Manage library materials and records", False),
])
def test_acts_on_people(text, people):
    from src import textutil

    assert textutil.acts_on_people(text, "manage") is people


def test_managing_things_goes_to_approval_but_managing_people_fails(ctx):
    people = run(ctx, "Managed a 4-person team developing a Python application", ["exp_acme_f01"])
    assert "B3" in codes(people, Severity.hard) and only_bullet(people).text == ASSISTED
    things = run(ctx, "Assisted a 4-person team and managed the Python application backlog", ["exp_acme_f01"])
    assert "B3" not in codes(things, Severity.hard)
    bullet = only_bullet(things)
    assert bullet.approval == Approval.pending and any('"manage" (lead level)' in f.message for f in bullet.flags)


def test_rewrites_name_the_words_they_add(ctx):
    reworded = only_bullet(run(ctx, "Assisted a 4-person team shipping a Python application for acceptance testing",
                               ["exp_acme_f01"], source="exp_acme_b01"))
    change = next(f for f in reworded.flags if f.code == "change")
    assert change.message == "reworded from the verified source wording; new words: shipping, acceptance, testing"
    # A minor change is auto-approved, and what it added is still recorded for changes.md.
    minor = only_bullet(run(ctx, "Assisted a 4-person team in developing a Python application with stakeholders",
                            ["exp_acme_f01"], source="exp_acme_b01", change="minor_wording"))
    assert minor.approval == Approval.auto
    assert minor.history == ["auto-approved minor wording change; new words: stakeholders"]


def test_new_words_compare_hyphenated_words_by_part():
    from src.validate import new_words

    assert new_words("Presented findings over two years", ["Presented findings across a two-year program"]) == []
    assert new_words("Taught block coding", ["Delivered block-code instruction"]) == ["taught"]
