"""Milestone 11: reports, the resume version archive, application tracking, cost."""

from __future__ import annotations

import csv
import json
from datetime import date
from types import SimpleNamespace

import pytest

import config
from src import archive, loop, render, report
from src.llm import UsageTracker
from src.loop import RunOptions
from src.startup import Session
from tests.conftest import SAMPLE_POSTING, ScriptedLLM, bd, ed, make_draft, make_extract, make_judgment

needs_soffice = pytest.mark.skipif(render.find_soffice() is None, reason="LibreOffice (soffice) is not installed")
TODAY = date(2026, 10, 13)

FIRST = make_draft(
    experiences=[
        ed("exp_acme", [bd("Assisted a 4-person team in developing a Python application.", ["exp_acme_f01"], "exp_acme_b01", "unchanged"),
                        bd("Led SQL work that reduced report generation time by 45%", ["exp_acme_f02"], "exp_acme_b02")]),
        ed("exp_megacorp", [bd("Ran the data platform", ["exp_megacorp_f01"])], org="MegaCorp", title="Staff Engineer"),
    ],
    skills=["Python", "SQL", "Kubernetes"],
)
REVISED = make_draft(
    experiences=[
        ed("exp_acme", [bd("Assisted a 4-person team in developing a Python application.", ["exp_acme_f01"], "exp_acme_b01", "unchanged"),
                        bd("Wrote SQL queries for all company-wide reports, reducing generation time by 30%", ["exp_acme_f02"], "exp_acme_b02")]),
        ed("exp_club", [bd("Led weekly workshops for 25 members on data analysis", ["exp_club_f01"], None, "unchanged")]),
    ],
    skills=["Python", "SQL", "data analysis"],
)
GAP = ("REAL_GAP", "Preferred qualifications", "No AWS or Docker experience is shown", 0.5)
ASK = ("ASK_USER", "Acme", "Did you measure how much faster the reports got?", 0.4)
REWRITE = ("REWRITE", "bullet 2", "Lead with the result", 0.3)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One full (scripted) run shared by the tests below; they only read its output."""
    import copy

    from src import ui
    from src.models import MasterResume, SkillsStatus
    from tests.conftest import SAMPLE_MASTER, SAMPLE_STATUS

    tmp_path = tmp_path_factory.mktemp("run")
    master = MasterResume.model_validate(copy.deepcopy(SAMPLE_MASTER))
    status = SkillsStatus.model_validate({"skills": dict(SAMPLE_STATUS)})
    ui.set_input(None, interactive=False)
    monkeypatch = pytest.MonkeyPatch()
    for name, sub in (("OUTPUT_DIR", "output"), ("POSTINGS_DIR", "postings"), ("CACHE_DIR", "cache"), ("BACKUP_DIR", "backups")):
        monkeypatch.setattr(config, name, tmp_path / sub)
    monkeypatch.setattr(config, "APPLICATIONS_CSV", tmp_path / "applications.csv")
    monkeypatch.setattr(config, "MASTER_RESUME_PATH", tmp_path / "master_resume.yaml")
    (tmp_path / "master_resume.yaml").write_text("# snapshot me\ncontact: {name: Test Person}\n")
    posting = tmp_path / "job.txt"
    posting.write_text(SAMPLE_POSTING)
    llm = ScriptedLLM({
        "analyze": [make_extract(requirements={"min_gpa": {"value": 3.9, "source": "Minimum GPA of 3.5."}})],
        "tailor": [FIRST],
        "fix": [{"fixes": []}, {"fixes": []}],
        "judge": [make_judgment(7, 8, 7, 7, fixes=[REWRITE, ASK, GAP]), make_judgment(8, 8, 8, 8, fixes=[ASK, GAP]),
                  make_judgment(8, 8, 8, 7, fixes=[ASK, GAP])],
        "revise": [REVISED],
    }, usage_per_call=(1000, 100))          # every scripted call "uses" 1,000 input and 100 output tokens
    # Usage from before this job (an earlier posting in a batch) must not be billed to it.
    llm.tracker.record("tailor", "claude-sonnet-5-5", SimpleNamespace(input_tokens=500_000, output_tokens=50_000,
                       cache_read_input_tokens=0, cache_creation_input_tokens=0, server_tool_use=None))
    state = loop.run_job(Session(master=master, skills_status=status, llm=llm), text_path=posting,
                         options=RunOptions(research=False, interview=False, batch=True), today=TODAY)
    yield state, tmp_path
    monkeypatch.undo()


@needs_soffice
def test_all_reports_are_written(run):
    state, _ = run
    for name in ("eligibility.md", "changes.md", "scores.md", "gaps.md"):
        assert (state.out_dir / name).read_text().startswith("# ")


@needs_soffice
def test_eligibility_report_puts_flags_first(run):
    text = (run[0].out_dir / "eligibility.md").read_text()
    assert text.index("## Flags") < text.index("## Requirements")
    assert "GPA: Doesn't meet" in text and "never auto-rejects" in text
    assert '| GPA | Doesn\'t meet | my GPA 3.60 is below the minimum 3.90<br>"Minimum GPA of 3.5." |' in text


@needs_soffice
def test_changes_report_shows_every_decision(run):
    state, _ = run
    text = (state.out_dir / "changes.md").read_text()
    # Per bullet: original vs. final, facts, change type, approval decision, flags.
    assert "**2. Wrote SQL queries that reduced report generation time by 30%.**" in text
    assert "Proposed rewrite (not used): Wrote SQL queries for all company-wide reports" in text
    assert "`exp_acme_f02`" in text and "reverted to verified wording" in text
    assert 'adds the scope word "company-wide"' in text
    # Approval decisions (automatic, because this was a batch run).
    assert "## Approval decisions" in text and "reverted (automatic)" in text
    # Validation log: the invented metric, the inflated ownership, the fake employer, the unverified skill.
    assert "## Validation log" in text
    for needle in ('states "45%"', '"lead" (lead level)', 'selected experience "exp_megacorp"', 'skill "Kubernetes"'):
        assert needle in text, needle
    assert "MegaCorp" not in (state.out_dir / "resume.docx").name and "Ran the data platform" not in state.rendered.view.plain_text()
    assert "## Page fit" in text and "render: initial layout -> 1 page(s)" in text


@needs_soffice
def test_scores_report_has_rounds_three_scores_ats_and_cost(run):
    state, _ = run
    text = (state.out_dir / "scores.md").read_text()
    assert "## Resume Quality by round" in text and "| 0 |" in text and "| 1 (kept) |" in text and "| **final** |" in text
    assert "impact (25%)" in text and "ats (15%)" in text
    assert "below target" in text and "question(s) for me" in text          # why it stopped
    assert "## Job Match" in text and "required skills | 100%" in text and "- Missing: AWS, machine learning" in text
    assert "## Keyword Coverage" in text and "Whole master resume" in text and "Final resume" in text
    assert "## ATS checks" in text and "| page count | pass | 1 page(s) |" in text
    # 4 fast-model calls (analyze, fix, 2 judge) and 3 write-model calls (tailor, reword, revise), this job only.
    # Batch mode settles approvals inside the loop, so no final re-judge is needed.
    assert "## Claude usage" in text and "| tailor | 1 | 1,000 | 100 | $0.0030 |" in text
    assert "| judge | 2 | 2,000 | 200 | $0.0003 |" in text and "| fix | 1 | 1,000 | 100 | $0.0001 |" in text
    assert "| **total** | 7 | 7,000 | 700 | **$0.0096** |" in text


@needs_soffice
def test_gaps_report_lists_real_gaps_and_open_questions(run):
    text = (run[0].out_dir / "gaps.md").read_text()
    assert "No AWS or Docker experience is shown" in text
    assert "| GPA | Doesn't meet |" in text
    assert "- Missing: AWS, machine learning" in text and "- Learning: Docker" in text
    assert "Did you measure how much faster the reports got?" in text
    assert "## Why the target was not reached" in text and "will not invent or inflate" in text


@needs_soffice
def test_application_is_logged_to_csv(run):
    state, tmp_path = run
    rows = list(csv.DictReader((tmp_path / "applications.csv").open()))
    assert list(rows[0]) == report.CSV_FIELDS
    row = rows[0]
    assert row["date"] == "2026-10-13" and row["company"] == "Acme Corp" and row["role"] == "Data Engineering Intern"
    assert row["quality"] == f"{state.final_quality.total:.2f}" and row["rounds"] == "1" and row["status"] == "drafted"
    assert "GPA: Doesn't meet" in row["eligibility_flags"] and row["output_folder"].endswith("Acme_Corp_Data_Engineering_Intern_2026-10-13")
    log = tmp_path / "applications.csv"
    report.append_application(state, log)                    # a second run appends; the header is not repeated
    assert len(report.read_applications(log)) == 2


@needs_soffice
def test_archive_snapshots_exactly_what_was_produced(run):
    state, _ = run
    out = state.out_dir
    index = json.loads((out / "archive.json").read_text())
    assert index["company"] == "Acme Corp" and index["date"] == "2026-10-13" and index["pages"] == 1
    for key in ("posting", "master_resume_snapshot", "resume_docx", "resume_pdf", "resume_pdf_to_send",
                "tailored_resume", "scores", "company_brief", "eligibility", "job_analysis", "usage"):
        assert (out / index["files"][key]).exists(), key
    assert (out / "posting.txt").read_text() == SAMPLE_POSTING.strip()
    assert (out / "master_resume.snapshot.yaml").read_text().startswith("# snapshot me")     # the file as it was
    assert (out / "Test_Person_Resume.pdf").read_bytes() == (out / "resume.pdf").read_bytes()
    tailored = json.loads((out / "tailored_resume.json").read_text())
    assert tailored["experiences"][0]["bullets"][0]["fact_ids"] == ["exp_acme_f01"]
    scores = json.loads((out / "scores.json").read_text())
    assert len(scores["rounds"]) == 2 and scores["job_match"]["score"] == state.job_match.score
    usage = json.loads((out / "usage.json").read_text())
    assert len(usage["calls"]) == 7 and usage["total_cost_usd"] == pytest.approx(0.0096, abs=1e-5)


@needs_soffice
def test_find_lists_matching_archives(run):
    state, tmp_path = run
    root = tmp_path / "output"
    assert [m["company"] for m in archive.find("acme", root)] == ["Acme Corp"]
    assert archive.find("ACME corp", root)[0]["folder"] == str(state.out_dir)
    assert archive.find("data engineering", root)              # the role matches too
    assert archive.find("globex", root) == [] and archive.print_matches("globex", root) == []
    assert len(archive.print_matches("Acme", root)) == 1


# ---------------------------------------------------------------------------
# Small pieces that need no rendering
# ---------------------------------------------------------------------------


def test_table_cells_are_escaped():
    text = report.table(["A", "B"], [["x | y", "line1\nline2"]])
    assert text.splitlines()[2] == "| x \\| y | line1<br>line2 |"


def test_find_sorts_newest_first_and_survives_bad_files(tmp_path):
    for folder, payload in (("A_2026-09-01", {"company": "Acme", "role": "Intern", "date": "2026-09-01"}),
                            ("A_2026-10-01", {"company": "Acme", "role": "Analyst", "date": "2026-10-01"}),
                            ("B_2026-10-02", {"company": "Globex", "role": "Intern", "date": "2026-10-02"})):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "archive.json").write_text(json.dumps(payload))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "archive.json").write_text("{nope")
    assert [m["date"] for m in archive.find("acme", tmp_path)] == ["2026-10-01", "2026-09-01"]
    assert len(archive.find("", tmp_path)) == 3
    assert archive.find("acme", tmp_path / "missing") == []


def test_sendable_filename():
    assert archive.sendable_name("Deveshwar Vankeswaram") == "Deveshwar_Vankeswaram_Resume.pdf"
    assert archive.sendable_name("") == "Resume.pdf"


def test_usage_tracker_totals_by_step():
    tracker = UsageTracker()
    usage = SimpleNamespace(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0, server_tool_use=SimpleNamespace(web_search_requests=2))
    tracker.record("research", "claude-sonnet-5-5", usage)
    tracker.record("research", "claude-sonnet-5-5", usage)
    row = tracker.by_purpose()["research"]
    assert row["calls"] == 2 and row["input"] == 2000 and row["output"] == 200
    assert tracker.total_cost() == pytest.approx(2 * ((1000 * 2 + 100 * 10) / 1e6 + 2 * 0.01))
