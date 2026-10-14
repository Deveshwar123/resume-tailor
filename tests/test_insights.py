"""Milestone 12: the application intelligence dashboard (all numbers computed in code)."""

from __future__ import annotations

import csv
import json
from datetime import date

import pytest

import config
from src import analyze, insights, report
from tests.conftest import make_analysis

TODAY = date(2026, 10, 13)


def posting(company, required, preferred=(), **requirements):
    """A saved analysis. Skills are (name, kind) or just a name (kind = tool)."""
    def skill(item):
        name, kind = item if isinstance(item, tuple) else (item, "tool")
        return {"name": name, "weight": 2, "kind": kind}

    a = make_analysis(company=company, role="Intern", required_skills=[skill(s) for s in required],
                      preferred_skills=[skill(s) for s in preferred], requirements=requirements or {})
    a.id = f"2026-10-13_{company}_Intern_{abs(hash(company)) % 16**8:08x}"
    return a


# Sample status: Python, SQL, JavaScript, React, pytest, Git, data analysis, REST are verified;
# PyTorch and Docker are learning; AWS and Tableau are missing; anything else is unlisted.
POSTINGS = [
    posting("A", ["Python", "SQL"]),                          # fully covered
    posting("B", ["Python", "AWS"]),                          # AWS is the only gap
    posting("C", ["SQL", "AWS"], ["Docker"]),                 # AWS is the only gap
    posting("D", ["Python", "PyTorch"]),                      # PyTorch (learning) is the only gap
    posting("E", ["AWS", "Docker", "Python"]),                # two gaps: AWS + Docker
    posting("F", ["AWS", "Docker", "Kubernetes"]),            # three gaps: unlocks nothing on its own
    posting("G", ["Tableau", ("communication", "soft")]),     # Tableau only; unlisted soft skill is not assessed
]


# ---------------------------------------------------------------------------
# Unlock analysis
# ---------------------------------------------------------------------------


def test_unlock_analysis_counts_sole_and_two_way_blockers(status):
    ranking, fully_covered = insights.unlock_analysis(POSTINGS, status)
    table = {u.name: (u.sole, u.within_two, u.required_in, u.status) for u in ranking}
    assert fully_covered == 1
    assert table["AWS"] == (2, 3, 4, "missing")          # sole in B and C; one of two in E; also in F
    assert table["PyTorch"] == (1, 1, 1, "learning")
    assert table["Tableau"] == (1, 1, 1, "missing")
    assert table["Docker"] == (0, 1, 2, "learning")      # never the only gap
    assert table["Kubernetes"] == (0, 0, 1, "missing")   # only appears with two other gaps
    assert "communication" not in table and "Python" not in table


def test_top_three_skills_are_ranked_by_postings_unlocked(status):
    ins = insights.compute(POSTINGS, [], status, today=TODAY)
    top = ins.top_unlocks(3)
    assert [u.name for u in top] == ["AWS", "PyTorch", "Tableau"]
    assert top[0].postings == ["Intern at B", "Intern at C"]
    # Kubernetes unlocks nothing within two skills, so it can never be a "top" unlock.
    assert "Kubernetes" not in [u.name for u in ins.top_unlocks(10)]


def test_verifying_a_skill_changes_the_analysis(status):
    from src.models import SkillStatus

    status.skills["AWS"] = SkillStatus.verified
    ranking, fully_covered = insights.unlock_analysis(POSTINGS, status)
    assert fully_covered == 3                               # B and C are now covered too
    table = {u.name: (u.sole, u.within_two) for u in ranking}
    # E now has Docker as its only gap; F is down to two gaps (Docker + Kubernetes).
    assert "AWS" not in table and table["Docker"] == (1, 2) and table["Kubernetes"] == (0, 1)


def test_unmet_required_uses_synonyms(status):
    p = posting("Syn", ["python3", "Structured Query Language", "Amazon Web Services"])
    assert list(insights.unmet_required(p, status).values()) == ["Amazon Web Services"]


# ---------------------------------------------------------------------------
# Skill demand, patterns, trends, funnel
# ---------------------------------------------------------------------------


def test_skill_demand_table(status):
    demand = {d.name: d for d in insights.skill_demand(POSTINGS, status)}
    assert (demand["AWS"].required, demand["AWS"].preferred, demand["AWS"].status) == (4, 0, "missing")
    assert (demand["Docker"].required, demand["Docker"].preferred, demand["Docker"].status) == (2, 1, "learning")
    assert (demand["Python"].required, demand["Python"].status, demand["Python"].listed) == (4, "verified", True)
    assert demand["Kubernetes"].status == "missing" and demand["Kubernetes"].listed is False
    ordered = [d.name for d in insights.skill_demand(POSTINGS, status)]
    assert ordered[:2] == ["AWS", "Python"]                 # most demanded first


def test_reruns_of_the_same_posting_are_counted_once(status):
    again = POSTINGS[1].model_copy(update={"analyzed_on": "2026-10-20"})
    ins = insights.compute([*POSTINGS, again], [], status, today=TODAY)
    assert ins.postings == len(POSTINGS)


def test_eligibility_patterns(master):
    postings = [
        posting("A", ["Python"], class_standing={"allowed": ["junior", "senior"], "source": "x"},
                min_gpa={"value": 3.5, "source": "x"}, work_authorization={"kind": "citizen_required", "source": "x"}),
        posting("B", ["Python"], class_standing={"allowed": ["freshman", "sophomore"], "source": "x"},
                min_gpa={"value": 3.0, "source": "x"}, work_authorization={"kind": "no_sponsorship", "source": "x"},
                grad_year_range={"earliest": "2029", "latest": "2030", "source": "x"}),
        posting("C", ["Python"], class_standing={"allowed": [], "source": None}, min_gpa={"value": None, "source": None},
                work_authorization={"kind": "not_stated", "source": None},
                prior_experience_required={"required": True, "kind": "internship", "source": "x"}),
        posting("D", ["Python"], class_standing={"allowed": ["junior"], "source": "x"},
                min_gpa={"value": None, "source": None}, work_authorization={"kind": "us_person_required", "source": "x"}),
    ]
    patterns = {p.label: p for p in insights.eligibility_patterns(postings, master)}
    assert patterns["require junior standing or above"].pct == 50.0
    assert patterns["are explicitly open to freshmen"].count == 1
    assert patterns["state a minimum GPA"].count == 2 and patterns["state a minimum GPA"].note == "median minimum 3.25"
    assert patterns["require U.S. citizenship"].pct == 25.0 and patterns["do not sponsor visas"].count == 1
    assert patterns["require a U.S. person (citizen or permanent resident)"].count == 1
    assert patterns["require prior internship, research, or work experience"].count == 1
    # Sample person graduates May 2028: the default window (Dec 2027 - Jun 2029) includes them; B's (2029-2030) does not.
    assert patterns["have a graduation window that excludes mine (2028)"].count == 1


APPLICATIONS = [
    {"date": "2026-09-05", "quality": "8.00", "job_match": "70.0", "keyword_coverage": "50.0", "status": "applied"},
    {"date": "2026-09-20", "quality": "9.00", "job_match": "80.0", "keyword_coverage": "60.0", "status": "interview"},
    {"date": "2026-10-02", "quality": "9.20", "job_match": "90.0", "keyword_coverage": "70.0", "status": "offer"},
    {"date": "2026-10-09", "quality": "8.80", "job_match": "60.0", "keyword_coverage": "", "status": "rejected"},
    {"date": "2026-10-11", "quality": "bad", "job_match": "50.0", "keyword_coverage": "40.0", "status": ""},
]


def test_score_trends_by_month():
    periods, averages = insights.score_trends(APPLICATIONS)
    assert [(p.month, p.count) for p in periods] == [("2026-09", 2), ("2026-10", 3)]
    assert periods[0].quality == 8.5 and periods[0].job_match == 75.0
    assert periods[1].quality == 9.0                      # the unreadable value is skipped, not counted as zero
    assert periods[1].keyword_coverage == 55.0            # blank skipped
    assert averages == {"quality": 8.75, "job_match": 70.0, "keyword_coverage": 55.0}


def test_outcome_funnel_counts_stage_reached():
    statuses, funnel = insights.outcome_funnel(APPLICATIONS)
    assert statuses == {"applied": 1, "interview": 1, "offer": 1, "rejected": 1, "drafted": 1}
    assert funnel == [("drafted", 5), ("applied", 4), ("interview", 2), ("offer", 1)]


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(config, "POSTINGS_DIR", tmp_path / "postings")
    monkeypatch.setattr(config, "APPLICATIONS_CSV", tmp_path / "applications.csv")
    for p in POSTINGS:
        analyze.save_analysis(p, tmp_path / "postings")
    with (tmp_path / "applications.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report.CSV_FIELDS)
        writer.writeheader()
        for row in APPLICATIONS:
            writer.writerow({**{k: "" for k in report.CSV_FIELDS}, **row, "company": "X", "role": "Intern"})
    return tmp_path


def test_generate_writes_html_markdown_and_json(workspace, status, master):
    html_path, md_path = insights.generate(status, master, today=TODAY, quiet=True)
    md = md_path.read_text()
    assert "1. **AWS** (missing) - the only missing required skill in 2 posting(s); one of at most two in 3" in md
    assert "| AWS | 4 | 0 | 4 | Missing |" in md and "| Kubernetes | 1 | 0 | 1 | Missing (not listed) |" in md
    assert "% of postings state a minimum GPA" in md and "| 2026-09 | 2 | 8.50 | 75.0 | 55.0% |" in md
    assert "| Interview | 2 |" in md
    page = html_path.read_text()
    assert page.startswith("<!doctype html>") and "<svg" in page and "Skills that would unlock the most postings" in page
    assert "http://" not in page and "https://" not in page          # fully static: no network, no CDN
    assert "prefers-color-scheme:dark" in page and 'id="tip"' in page  # dark mode and a hover layer
    assert page.count("<table>") >= 4                                  # every chart has a table view
    data = json.loads((workspace / "output" / "insights.json").read_text())
    assert data["postings"] == 7 and data["unlocks"][0]["name"] == "AWS"


def test_names_from_postings_are_escaped_in_the_page(status):
    hostile = posting("Evil", ['<script>alert(1)</script>', "Python"])
    page = insights.render_html(insights.compute([hostile], [], status, today=TODAY))
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_empty_state_is_helpful(tmp_path, monkeypatch, status):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(config, "POSTINGS_DIR", tmp_path / "postings")
    monkeypatch.setattr(config, "APPLICATIONS_CSV", tmp_path / "applications.csv")
    html_path, md_path = insights.generate(status, today=TODAY, quiet=True)
    assert "No postings have been analyzed yet" in md_path.read_text()
    assert "No postings have been analyzed yet" in html_path.read_text()


def test_funnel_prompts_me_to_fill_in_statuses(status):
    drafted = [{"date": "2026-10-01", "quality": "9", "job_match": "80", "keyword_coverage": "60", "status": "drafted"}]
    md = insights.render_markdown(insights.compute(POSTINGS, drafted, status, today=TODAY))
    assert "Every row is still `drafted`" in md
