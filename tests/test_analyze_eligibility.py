"""Milestone 3: job analysis clean-up and the eligibility check."""

from __future__ import annotations

from datetime import date

import pytest

import config
from src import analyze, eligibility
from src.fetch import JobText
from src.models import ClassStanding, EligStatus, InterpretedRequirements, MasterResume
from tests.conftest import SAMPLE_POSTING, ScriptedLLM, answers, make_analysis, make_extract

TODAY = date(2026, 10, 13)


def row(report, name):
    return next(r for r in report.rows if r.requirement.startswith(name))


# ---------------------------------------------------------------------------
# analyze.py
# ---------------------------------------------------------------------------


def test_analyze_calls_the_fast_model_with_structured_output():
    llm = ScriptedLLM({"analyze": [make_extract()]})
    result = analyze.analyze_job(llm, JobText(SAMPLE_POSTING, "https://careers.acme.com/jobs/123"), TODAY)
    call = llm.calls[0]
    assert call["model"] == config.FAST_MODEL and call["schema"].__name__ == "JobAnalysisExtract"
    assert SAMPLE_POSTING in call["user"]
    assert result.company == "Acme Corp" and result.url.endswith("/123")
    assert result.id.startswith("2026-10-13_Acme_Corp_Data_Engineering_Intern_")
    assert result.posting_words > 50 and result.unverified_quotes == []


def test_finalize_clamps_weights_and_dedupes_with_synonyms():
    result = make_analysis(
        required_skills=[{"name": "Python", "weight": 9, "kind": "language"},
                         {"name": "python3", "weight": 1, "kind": "language"},
                         {"name": "ML", "weight": 0, "kind": "technical"}],
        preferred_skills=[{"name": "machine learning", "weight": 2, "kind": "technical"},
                          {"name": "AWS", "weight": 2, "kind": "tool"}],
        keywords=["ML", "machine learning", "Postgres", "PostgreSQL", " SQL ", ""],
    )
    assert [(s.name, s.weight) for s in result.required_skills] == [("Python", 3), ("ML", 1)]
    assert [s.name for s in result.preferred_skills] == ["AWS"]        # ML is already required
    assert result.keywords == ["ML", "Postgres", "SQL"]


def test_quotes_not_found_in_the_posting_are_recorded():
    result = make_analysis(requirements={"min_gpa": {"value": 3.8, "source": "A 3.8 GPA is mandatory."}})
    assert result.unverified_quotes == ["A 3.8 GPA is mandatory."]


def test_postings_round_trip_to_disk(tmp_path, analysis):
    path = analyze.save_analysis(analysis, tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    loaded = analyze.load_postings(tmp_path)
    assert path.name == f"{analysis.id}.json" and len(loaded) == 1
    assert loaded[0].required_skills[0].name == "Python"


# ---------------------------------------------------------------------------
# eligibility.py - graduation year and class standing
# ---------------------------------------------------------------------------


def test_everything_met(master, status, analysis):
    report = eligibility.check_eligibility(analysis, master, status, posting_text=SAMPLE_POSTING, today=TODAY)
    assert row(report, "Graduation").status == EligStatus.meets
    assert "May 2028" in row(report, "Graduation").detail
    assert row(report, "Degree").status == EligStatus.meets
    assert row(report, "GPA").status == EligStatus.meets
    assert row(report, "Work authorization").status == EligStatus.meets
    assert row(report, "Required skills").status == EligStatus.meets
    assert all(r.quote_verified for r in report.rows if r.evidence)
    assert report.flags() == [] and not report.has_blocker()


@pytest.mark.parametrize("earliest,latest,expected", [
    ("2027-12", "2029-06", EligStatus.meets),
    ("2028", "2028", EligStatus.meets),
    ("2028-06", None, EligStatus.fails),        # I graduate May 2028, a month too early
    (None, "2027", EligStatus.fails),
    ("2029", "2030", EligStatus.fails),
])
def test_grad_year_window(master, status, earliest, latest, expected):
    a = make_analysis(requirements={"grad_year_range": {"earliest": earliest, "latest": latest, "source": "x"}})
    assert row(eligibility.check_eligibility(a, master, status, today=TODAY), "Graduation").status == expected


def test_class_standing_uses_the_current_school_year(master, status):
    assert eligibility.current_standing(2030, TODAY) == ClassStanding.freshman
    assert eligibility.current_standing(2028, TODAY) == ClassStanding.junior
    assert eligibility.current_standing(2028, date(2027, 9, 1)) == ClassStanding.senior
    a = make_analysis(requirements={"grad_year_range": {"earliest": None, "latest": None, "source": None},
                                    "class_standing": {"allowed": ["sophomore", "junior"], "source": "Sophomores and juniors only."}})
    assert row(eligibility.check_eligibility(a, master, status, today=TODAY), "Graduation").status == EligStatus.meets
    a = make_analysis(requirements={"grad_year_range": {"earliest": None, "latest": None, "source": None},
                                    "class_standing": {"allowed": ["freshman"], "source": "First-years only."}})
    result = row(eligibility.check_eligibility(a, master, status, today=TODAY), "Graduation")
    assert result.status == EligStatus.fails and "junior" in result.detail


def test_missing_class_year_is_unknown(master_data, status, analysis):
    master_data["education"][0]["class_year"] = None
    report = eligibility.check_eligibility(analysis, MasterResume.model_validate(master_data), status, today=TODAY)
    assert row(report, "Graduation").status == EligStatus.unknown


def test_unstated_requirements_are_not_flagged(master, status):
    a = make_analysis(requirements={
        "grad_year_range": {"earliest": None, "latest": None, "source": None},
        "degree_fields": {"fields": [], "related_fields_ok": False, "source": None},
        "min_gpa": {"value": None, "source": None},
        "work_authorization": {"kind": "not_stated", "source": None},
    }, required_skills=[])
    report = eligibility.check_eligibility(a, master, status, today=TODAY)
    assert {r.status for r in report.rows} == {EligStatus.not_stated}
    assert report.flags() == []


# ---------------------------------------------------------------------------
# Vague wording -> Claude interprets, result is labeled
# ---------------------------------------------------------------------------

VAGUE = {"grad_year_range": {"earliest": None, "latest": None, "source": None},
         "class_standing": {"allowed": [], "source": "Open to rising juniors."}}


def test_vague_wording_is_interpreted_and_labeled(master, status):
    llm = ScriptedLLM({"interpret": [InterpretedRequirements(
        grad_earliest=None, grad_latest=None, class_standing_allowed=[ClassStanding.sophomore],
        work_authorization=None, notes="Rising juniors are sophomores this school year.")]})
    report = eligibility.check_eligibility(make_analysis(requirements=VAGUE), master, status, llm=llm, today=TODAY)
    result = row(report, "Graduation")
    assert result.interpreted and result.status == EligStatus.fails       # sample person is a junior
    assert "interpreted" in result.detail
    assert llm.calls[0]["model"] == config.FAST_MODEL and "Open to rising juniors." in llm.calls[0]["user"]


def test_vague_wording_without_claude_is_unknown(master, status):
    result = row(eligibility.check_eligibility(make_analysis(requirements=VAGUE), master, status, today=TODAY), "Graduation")
    assert result.status == EligStatus.unknown and not result.interpreted


def test_clear_requirements_never_call_claude(master, status, analysis):
    llm = ScriptedLLM()
    eligibility.check_eligibility(analysis, master, status, llm=llm, today=TODAY)
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Degree, GPA, authorization, skills, experience
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fields,related,major,expected", [
    (["Computer Science"], False, "Computer Science", EligStatus.meets),
    (["Computer Science", "Data Science"], True, "Data Engineering", EligStatus.meets),     # related field
    (["Computer Science"], False, "Data Engineering", EligStatus.partial),                  # close, not listed
    (["Finance", "Accounting"], False, "Data Engineering", EligStatus.fails),
    (["STEM field"], False, "Data Engineering", EligStatus.meets),
    (["Engineering"], False, "Data Engineering", EligStatus.meets),
    (["Nursing"], True, "Data Engineering", EligStatus.unknown),
])
def test_degree_field(master_data, status, fields, related, major, expected):
    master_data["education"][0].update(major=major, degree="Bachelor of Science")
    a = make_analysis(requirements={"degree_fields": {"fields": fields, "related_fields_ok": related, "source": "x"}})
    report = eligibility.check_eligibility(a, MasterResume.model_validate(master_data), status, today=TODAY)
    assert row(report, "Degree").status == expected


@pytest.mark.parametrize("minimum,mine,expected", [
    (3.5, 3.6, EligStatus.meets), (3.6, 3.6, EligStatus.meets),
    (3.7, 3.6, EligStatus.fails), (3.0, None, EligStatus.unknown),
])
def test_gpa(master_data, status, minimum, mine, expected):
    master_data["education"][0]["gpa"] = mine
    a = make_analysis(requirements={"min_gpa": {"value": minimum, "source": "x"}})
    report = eligibility.check_eligibility(a, MasterResume.model_validate(master_data), status, today=TODAY)
    assert row(report, "GPA").status == expected


@pytest.mark.parametrize("mine,kind,expected", [
    ("US citizen", "citizen_required", EligStatus.meets),
    ("US permanent resident (green card); no visa sponsorship required", "citizen_required", EligStatus.fails),
    ("US permanent resident (green card); no visa sponsorship required", "us_person_required", EligStatus.meets),
    ("US permanent resident (green card); no visa sponsorship required", "no_sponsorship", EligStatus.meets),
    ("F-1 student visa", "us_person_required", EligStatus.fails),
    ("F-1 student visa", "no_sponsorship", EligStatus.unknown),
    ("F-1 student visa", "sponsorship_available", EligStatus.meets),
    ("", "citizen_required", EligStatus.unknown),
    ("US citizen", "other", EligStatus.unknown),
])
def test_work_authorization(master_data, status, mine, kind, expected):
    master_data["work_authorization"] = mine
    a = make_analysis(requirements={"work_authorization": {"kind": kind, "source": "x"}})
    report = eligibility.check_eligibility(a, MasterResume.model_validate(master_data), status, today=TODAY)
    assert row(report, "Work authorization").status == expected


def test_required_skills_partial_lists_missing_and_learning(master, status):
    a = make_analysis(required_skills=[
        {"name": "Python", "weight": 3, "kind": "language"}, {"name": "PyTorch", "weight": 2, "kind": "tool"},
        {"name": "Kubernetes", "weight": 1, "kind": "tool"}, {"name": "communication", "weight": 1, "kind": "soft"},
    ])
    report = eligibility.check_eligibility(a, master, status, today=TODAY)
    result = row(report, "Required skills")
    assert result.status == EligStatus.partial
    assert report.missing_required_skills == ["Kubernetes"]           # unlisted soft skill is not assessed
    assert report.learning_required_skills == ["PyTorch"]
    assert "missing: Kubernetes" in result.detail and "learning: PyTorch" in result.detail


def test_required_skills_none_verified_is_a_blocker(master, status):
    a = make_analysis(required_skills=[{"name": "Rust", "weight": 3, "kind": "language"}])
    report = eligibility.check_eligibility(a, master, status, today=TODAY)
    assert row(report, "Required skills").status == EligStatus.fails and report.has_blocker()


def test_prior_experience(master, master_data, status):
    need = {"prior_experience_required": {"required": True, "kind": "internship", "source": "x"}}
    assert row(eligibility.check_eligibility(make_analysis(requirements=need), master, status, today=TODAY),
               "Prior").status == EligStatus.meets
    master_data["experience"] = [e for e in master_data["experience"] if e["type"] != "internship"]
    stripped = MasterResume.model_validate(master_data)
    assert row(eligibility.check_eligibility(make_analysis(requirements=need), stripped, status, today=TODAY),
               "Prior").status == EligStatus.fails


def test_other_requirements_and_bad_quotes_are_surfaced(master, status):
    a = make_analysis(requirements={
        "other": [{"text": "Must be on site in Austin", "source": "This role is on site in Austin."}],
        "min_gpa": {"value": 3.5, "source": "GPA of 3.5 strictly required!"},
    })
    report = eligibility.check_eligibility(a, master, status, posting_text=SAMPLE_POSTING, today=TODAY)
    other = row(report, "Other: Must be on site")
    assert other.status == EligStatus.unknown and other.quote_verified is False
    assert row(report, "GPA").quote_verified is False
    assert "not found verbatim" in eligibility.evidence_cell(row(report, "GPA"))


# ---------------------------------------------------------------------------
# Flag, never auto-reject
# ---------------------------------------------------------------------------


def test_batch_mode_always_continues(master, status):
    a = make_analysis(requirements={"min_gpa": {"value": 4.0, "source": "x"}})
    report = eligibility.check_eligibility(a, master, status, today=TODAY)
    assert report.has_blocker() and eligibility.confirm_continue(report, batch=True) is True


def test_i_decide_whether_to_continue(master, status):
    a = make_analysis(requirements={"min_gpa": {"value": 4.0, "source": "x"}})
    report = eligibility.check_eligibility(a, master, status, today=TODAY)
    answers("n")
    assert eligibility.confirm_continue(report) is False
    answers("")                      # Enter accepts the default: continue
    assert eligibility.confirm_continue(report) is True
