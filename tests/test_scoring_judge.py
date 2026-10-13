"""Milestone 7: the independent judge and the three code-computed scores."""

from __future__ import annotations

from datetime import date

import pytest

import config
from src import eligibility, judge, scoring
from src.models import AtsCheck, AtsReport, BriefItem, CompanyBrief, FixType, Tier
from tests.conftest import SAMPLE_POSTING, ScriptedLLM, make_analysis, make_judgment

TODAY = date(2026, 10, 13)


def ats(score: float, final: bool = False) -> AtsReport:
    return AtsReport(checks=[AtsCheck(name="contact_info", passed=True)], score=score, final=final)


# ---------------------------------------------------------------------------
# Resume Quality: weighted math in Python
# ---------------------------------------------------------------------------


def test_quality_is_the_weighted_average():
    result = scoring.quality(make_judgment(impact=8, relevance=9, clarity=7, language=6), ats(10.0))
    # 0.25*8 + 0.25*9 + 0.20*7 + 0.15*10 (ATS, from code) + 0.15*6
    assert result.total == pytest.approx(8.05)
    assert result.categories == {"impact": 8.0, "relevance": 9.0, "clarity": 7.0, "language": 6.0, "ats": 10.0}
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_ats_score_comes_from_code_and_moves_the_total():
    judgment = make_judgment(9, 9, 9, 9)
    assert scoring.quality(judgment, ats(10.0)).total == pytest.approx(9.15)
    assert scoring.quality(judgment, ats(4.0)).total == pytest.approx(8.25)      # 6 ATS points * 0.15
    assert scoring.quality(judgment, ats(10.0, final=True)).ats_final is True


def test_out_of_range_judge_scores_are_clamped():
    result = scoring.quality(make_judgment(impact=14, relevance=0, clarity=10, language=-3), ats(25.0))
    assert result.categories == {"impact": 10.0, "relevance": 1.0, "clarity": 10.0, "language": 1.0, "ats": 10.0}


def test_a_perfect_resume_scores_ten_and_weights_are_configurable():
    assert scoring.quality(make_judgment(10, 10, 10, 10), ats(10.0)).total == 10.0
    only_impact = {"impact": 1.0, "relevance": 0.0, "clarity": 0.0, "ats": 0.0, "language": 0.0}
    assert scoring.quality(make_judgment(impact=3), ats(10.0), only_impact).total == 3.0


# ---------------------------------------------------------------------------
# Job Match: measures me, not the writing
# ---------------------------------------------------------------------------


def test_job_match_combines_skills_and_eligibility(master, status, analysis):
    report = eligibility.check_eligibility(analysis, master, status, today=TODAY)
    match = scoring.job_match(analysis, status, report)
    # Required: Python + SQL both verified (1.0). Preferred: AWS missing, Docker learning (0.5),
    # machine learning unlisted -> (0 + 0.5*1 + 0) / (2 + 1 + 1) = 0.125. Eligibility: all four stated rows met.
    assert match.components == {"required_skills": 1.0, "preferred_skills": 0.125, "eligibility": 1.0}
    assert match.score == pytest.approx(100 * (0.55 * 1.0 + 0.15 * 0.125 + 0.30 * 1.0), abs=0.05)
    assert match.covered == ["Python", "SQL"] and match.partial == ["Docker"]
    assert match.missing == ["AWS", "machine learning"]


def test_job_match_drops_when_i_fail_requirements(master, status):
    hard = make_analysis(required_skills=[{"name": "Rust", "weight": 3, "kind": "language"},
                                          {"name": "Python", "weight": 1, "kind": "language"}],
                         requirements={"min_gpa": {"value": 3.9, "source": "x"}})
    report = eligibility.check_eligibility(hard, master, status, today=TODAY)
    match = scoring.job_match(hard, status, report)
    assert match.components["required_skills"] == pytest.approx(0.25)
    assert match.components["eligibility"] == pytest.approx(0.75)      # GPA fails; grad, degree, authorization met


def test_job_match_does_not_depend_on_the_resume_text(master, status, analysis):
    # The function takes no resume at all: rewriting bullets cannot change it.
    report = eligibility.check_eligibility(analysis, master, status, today=TODAY)
    assert scoring.job_match(analysis, status, report) == scoring.job_match(analysis, status, report)
    import inspect
    assert "resume" not in inspect.signature(scoring.job_match).parameters


def test_job_match_ignores_unstated_requirements(status):
    bare = make_analysis(preferred_skills=[], requirements={
        "grad_year_range": {"earliest": None, "latest": None, "source": None},
        "degree_fields": {"fields": [], "related_fields_ok": False, "source": None},
        "min_gpa": {"value": None, "source": None},
        "work_authorization": {"kind": "not_stated", "source": None}})
    assert scoring.job_match(bare, status, None).components == {"required_skills": 1.0}
    assert scoring.job_match(bare, status, None).score == 100.0


# ---------------------------------------------------------------------------
# Keyword Coverage: weighted, synonym-aware
# ---------------------------------------------------------------------------


def test_keyword_weights_keep_the_highest_weight_per_term():
    a = make_analysis(
        required_skills=[{"name": "Python", "weight": 3, "kind": "language"}],
        preferred_skills=[{"name": "machine learning", "weight": 2, "kind": "technical"}],
        tools=["Python", "Postgres"], keywords=["ML", "data pipelines", "python3"])
    table = scoring.keyword_weights(a)
    assert table["python"] == ("Python", 3.0)                       # required beats the plain keyword
    assert table["machine learning"] == ("machine learning", 1.0)   # preferred 2 * 0.5 == plain 1.0
    assert set(table) == {"python", "machine learning", "postgresql", "data pipelines"}


def test_keyword_coverage_matches_synonyms():
    a = make_analysis(
        required_skills=[{"name": "Python", "weight": 3, "kind": "language"},
                         {"name": "PostgreSQL", "weight": 2, "kind": "tool"}],
        preferred_skills=[{"name": "machine learning", "weight": 2, "kind": "technical"}],
        tools=[], keywords=["data pipelines", "Kubernetes"])
    text = "Built ML models in Python; stored features in Postgres."
    coverage = scoring.keyword_coverage(a, text)
    assert coverage.matched == ["Python", "PostgreSQL", "machine learning"]     # ML and Postgres are synonyms
    assert coverage.missing == ["data pipelines", "Kubernetes"]
    # (3 + 2 + 1) matched of (3 + 2 + 1 + 1 + 1) total weight
    assert coverage.score == pytest.approx(75.0)


def test_keyword_coverage_for_master_and_empty_text(master, analysis):
    full = scoring.keyword_coverage(analysis, scoring.master_text(master))
    assert "Python" in full.matched and "SQL" in full.matched and "Docker" in full.missing
    assert scoring.keyword_coverage(analysis, "").score == 0.0
    assert "Assisted a 4-person team" in scoring.master_text(master)


# ---------------------------------------------------------------------------
# Judge: independent, category scores only
# ---------------------------------------------------------------------------

RESUME_TEXT = "TEST PERSON\nEXPERIENCE\n- Assisted a 4-person team in developing a Python application.\n"


def test_judge_sees_only_posting_trusted_brief_and_resume():
    brief = CompanyBrief(
        tech_stack=[BriefItem(text="Acme runs Python services", tier=Tier.official)],
        intern_work=[BriefItem(text="Interns ship production code", tier=Tier.posting)],
        recent_news=[BriefItem(text="Acme raised a Series C", tier=Tier.third_party)],
        summary=[BriefItem(text="Forum rumor about layoffs", tier=Tier.unverified)],
    )
    llm = ScriptedLLM({"judge": [make_judgment()]})
    judge.judge_resume(llm, SAMPLE_POSTING, brief, RESUME_TEXT)
    call = llm.calls[0]
    assert call["model"] == config.FAST_MODEL and call["schema"].__name__ == "Judgment"
    assert "Minimum GPA of 3.5." in call["system"]
    assert "Acme runs Python services" in call["system"] and "Interns ship production code" in call["system"]
    assert "Series C" not in call["system"] and "rumor" not in call["system"]      # tiers 3 and 4 never reach the judge
    assert call["user"] == f"<resume>\n{RESUME_TEXT.strip()}\n</resume>"            # the resume and nothing else


def test_judge_system_is_identical_every_round_so_it_caches():
    assert judge.build_system(SAMPLE_POSTING, None) == judge.build_system(SAMPLE_POSTING, None)
    assert "company_context" not in judge.build_system(SAMPLE_POSTING, CompanyBrief())


def test_judgment_has_no_total_and_is_cleaned():
    raw = make_judgment(impact=12, relevance=0, fixes=[
        ("REWRITE", "bullet 1", "Lead with the result", 0.2),
        ("ASK_USER", "Acme", "Did you measure the speed-up?", 0.9),
        ("REAL_GAP", "skills", "No AWS experience", 9.0),
        ("REWRITE", "bullet 2", "   ", 0.5),
    ])
    llm = ScriptedLLM({"judge": [raw]})
    result = judge.judge_resume(llm, SAMPLE_POSTING, None, RESUME_TEXT)
    assert result.category_scores.impact.score == 10 and result.category_scores.relevance.score == 1
    assert [f.type for f in result.fixes] == [FixType.REAL_GAP, FixType.ASK_USER, FixType.REWRITE]   # by expected gain
    assert result.fixes[0].expected_gain == 3.0                                    # clamped
    assert not hasattr(result, "total") and "total" not in type(result).model_fields
    assert judge.summarize(result) == "1 rewrite fix(es), 1 question(s), 1 real gap(s)"


def test_rubric_separates_writing_from_qualifications():
    system = judge.build_system(SAMPLE_POSTING, None)
    assert "REAL_GAP" in system and "do not let it lower Impact or Clarity" in system
    assert "strict technical recruiter" in system
