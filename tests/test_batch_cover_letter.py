"""Milestone 13: batch processing and the optional cover letter."""

from __future__ import annotations

import csv
import json
from datetime import date

import pytest

import config
from src import cover_letter, fetch, loop, render, ui
from src.cover_letter import CoverLetterDraft, SentenceFix, SentenceFixBatch
from src.errors import FetchError
from src.fetch import JobText
from src.loop import RunOptions
from src.models import BriefItem, CompanyBrief, Severity, Tier
from src.startup import Session
from tests.conftest import SAMPLE_POSTING, ScriptedLLM, answers, bd, ed, make_draft, make_extract, make_judgment

needs_soffice = pytest.mark.skipif(render.find_soffice() is None, reason="LibreOffice (soffice) is not installed")
TODAY = date(2026, 10, 13)

DRAFT = make_draft(
    experiences=[ed("exp_acme", [
        bd("Assisted a 4-person team in developing a Python application.", ["exp_acme_f01"], "exp_acme_b01", "unchanged"),
        bd("Wrote SQL queries that reduced report generation time by 30%.", ["exp_acme_f02"], "exp_acme_b02", "unchanged"),
    ])],
    skills=["Python", "SQL"],
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for name, sub in (("OUTPUT_DIR", "output"), ("POSTINGS_DIR", "postings"), ("CACHE_DIR", "cache"), ("BACKUP_DIR", "backups")):
        monkeypatch.setattr(config, name, tmp_path / sub)
    monkeypatch.setattr(config, "APPLICATIONS_CSV", tmp_path / "applications.csv")
    return tmp_path


# ---------------------------------------------------------------------------
# Cover letter: the same evidence rules as a resume bullet
# ---------------------------------------------------------------------------


def letter(*sentences, paragraphs=None):
    """sentences: (text, about, fact_ids) tuples, all in one paragraph unless `paragraphs` is given."""
    groups = paragraphs or [list(sentences)]
    return CoverLetterDraft.model_validate({
        "greeting": "Dear Hiring Manager,", "closing": "Sincerely,",
        "paragraphs": [{"sentences": [{"text": t, "about": a, "fact_ids": f} for t, a, f in group]} for group in groups],
    })


SOURCES = [SAMPLE_POSTING]


def check(ctx, text, about="me", facts=()):
    return cover_letter.check_sentence(text, about, list(facts), ctx, SOURCES)


def test_supported_sentences_pass(ctx):
    assert check(ctx, "I am writing to apply for the Data Engineering Intern role at Acme Corp.", "other") == []
    assert check(ctx, "At Acme Corp, I assisted a 4-person team in developing a Python application.", facts=["exp_acme_f01"]) == []
    assert not [v for v in check(ctx, "I wrote SQL queries that reduced report generation time by 30%.", facts=["exp_acme_f02"])
                if v.severity == Severity.hard]
    # Facts from different experiences may sit in one sentence.
    assert not [v for v in check(ctx, "I wrote SQL queries at Acme Corp and built a budget tracking web app with JavaScript and React.",
                                 facts=["exp_acme_f02", "proj_tracker_f01"]) if v.severity == Severity.hard]


def test_education_is_accepted_as_evidence(ctx):
    assert check(ctx, "I am a Computer Science student at State University with a 3.6 GPA.") == []
    assert any(v.code == "B1" for v in check(ctx, "I am a Computer Science student with a 3.9 GPA."))


@pytest.mark.parametrize("text,facts,code", [
    ("My SQL work cut reporting costs by 45%.", ["exp_acme_f02"], "B1"),                 # invented metric
    ("I led a 4-person team building a Python application.", ["exp_acme_f01"], "B3"),    # ownership
    ("I am proficient in Kubernetes and Python.", ["exp_acme_f01"], "B2"),               # unsupported skill
    ("I have used PyTorch for two years.", [], "B2"),                                    # no fact at all
    ("I built data pipelines at Acme.", ["exp_acme_f99"], "A1"),                         # fabricated fact id
])
def test_unsupported_claims_about_me_fail(ctx, text, facts, code):
    hard = {v.code for v in check(ctx, text, facts=facts) if v.severity == Severity.hard}
    assert code in hard


def test_company_statements_must_be_grounded(ctx):
    assert check(ctx, "Interns at Acme build data pipelines in Python and SQL.", "company") == []
    flagged = check(ctx, "Acme is the largest quantum logistics bank in Europe.", "company")
    assert [v.code for v in flagged] == ["L1"] and flagged[0].severity == Severity.soft
    brief = CompanyBrief(recent_news=[BriefItem(text="Acme opened a quantum logistics lab in Europe.", tier=Tier.official)],
                         summary=[BriefItem(text="Rumor: Acme is closing.", tier=Tier.unverified)])
    sources = cover_letter.company_sources(SAMPLE_POSTING, brief)
    assert "Rumor: Acme is closing." not in sources
    assert cover_letter.check_sentence("Acme opened a quantum logistics lab in Europe.", "company", [], ctx, sources) == []
    # A "company" sentence that is really about me is checked as a claim about me.
    assert any(v.code == "B2" for v in check(ctx, "My Kubernetes experience fits Acme well.", "company"))


def test_hard_failures_get_one_retry_then_the_sentence_is_removed(ctx):
    draft = letter(
        ("At Acme Corp, I assisted a 4-person team in developing a Python application.", "me", ["exp_acme_f01"]),
        ("My SQL work cut reporting costs by 45%.", "me", ["exp_acme_f02"]),
        ("I led the team's Python work.", "me", ["exp_acme_f01"]),
    )
    asked = []

    def retry(requests):
        asked.extend(requests)
        return [SentenceFix(key=requests[0]["key"], text="I wrote SQL queries that reduced report generation time by 30%.",
                            fact_ids=["exp_acme_f02"]),
                SentenceFix(key=requests[1]["key"], text="I managed the team's Python work.", fact_ids=["exp_acme_f01"])]

    result = cover_letter.validate_letter(draft, ctx, SOURCES, retry)
    assert len(asked) == 2 and "45%" in asked[0]["problems"][0]
    kept = [s.text for s in result.sentences if s.kept]
    assert kept == ["At Acme Corp, I assisted a 4-person team in developing a Python application.",
                    "I wrote SQL queries that reduced report generation time by 30%."]
    removed = [s for s in result.sentences if not s.kept]
    assert len(removed) == 1 and "removed" in removed[0].notes[0]
    assert "led" not in " ".join(result.paragraphs()) and "45%" not in " ".join(result.paragraphs())
    assert {v.resolution for v in result.violations if v.severity == Severity.hard} == {
        "fixed by the one allowed retry", "sent back to Claude once; sentence removed"}


def test_flagged_sentences_are_removed_when_no_one_can_approve(ctx):
    draft = letter(("Acme is the largest quantum logistics bank in Europe.", "company", []),
                   ("At Acme Corp, I assisted a 4-person team in developing a Python application.", "me", ["exp_acme_f01"]))
    result = cover_letter.validate_letter(draft, ctx, SOURCES)
    assert len(result.pending()) == 1
    cover_letter.review(result, ctx, SOURCES, batch=True)
    assert [s.kept for s in result.sentences] == [False, True] and result.pending() == []


def test_i_can_approve_edit_or_remove_flagged_sentences(ctx):
    def fresh():
        return cover_letter.validate_letter(letter(
            ("Acme is the largest quantum logistics bank in Europe.", "company", []),
            ("At Acme Corp, I assisted a 4-person team in developing a scalable Python application.", "me", ["exp_acme_f01"]),
        ), ctx, SOURCES)

    result = fresh()
    answers("a", "e", "At Acme Corp, I assisted a 4-person team in developing a Python application.")
    cover_letter.review(result, ctx, SOURCES)
    assert [s.decision for s in result.sentences] == ["approved", "edited"]
    assert "scalable" not in " ".join(result.paragraphs())

    result = fresh()
    answers("", "e", "At Acme Corp, I led a 40-person team.")       # Enter removes; an unsupported edit is refused
    cover_letter.review(result, ctx, SOURCES)
    assert [s.decision for s in result.sentences] == ["removed", "removed"]


def test_markdown_shows_where_every_sentence_comes_from(ctx, master):
    result = cover_letter.validate_letter(letter(paragraphs=[
        [("I am writing to apply for the Data Engineering Intern role at Acme Corp.", "other", [])],
        [("At Acme Corp, I assisted a 4-person team in developing a Python application.", "me", ["exp_acme_f01"])],
    ]), ctx, SOURCES)
    text = cover_letter.to_markdown(result, master, "Acme Corp", "Data Engineering Intern", TODAY)
    assert "October 13, 2026" in text and "Dear Hiring Manager," in text and "Sincerely,\nTest Person" in text
    assert "| At Acme Corp, I assisted a 4-person team in developing a Python application. | me | exp_acme_f01 | auto |" in text
    assert len(result.paragraphs()) == 2 and result.word_count() == 27


@needs_soffice
def test_cover_letter_end_to_end(master, status, workspace):
    posting = workspace / "job.txt"
    posting.write_text(SAMPLE_POSTING)
    draft = letter(paragraphs=[
        [("I am writing to apply for the Data Engineering Intern role at Acme Corp.", "other", []),
         ("Interns at Acme build data pipelines in Python and SQL.", "company", [])],
        [("At Acme Corp, I assisted a 4-person team in developing a Python application.", "me", ["exp_acme_f01"]),
         ("I also wrote SQL queries that reduced report generation time by 30%.", "me", ["exp_acme_f02"]),
         ("I am an expert in Kubernetes.", "me", [])],
        [("I would welcome the chance to talk about the role.", "other", [])],
    ])
    llm = ScriptedLLM({"analyze": [make_extract()], "tailor": [DRAFT], "judge": [make_judgment(9, 9, 9, 9)],
                       "cover_letter": [draft], "fix": [SentenceFixBatch(fixes=[])]})
    state = loop.run_job(Session(master=master, skills_status=status, llm=llm), text_path=posting,
                         options=RunOptions(research=False, cover_letter=True, batch=True), today=TODAY)
    call = next(c for c in llm.calls if c["purpose"] == "cover_letter")
    assert call["system"] == state.system and call["model"] == config.WRITE_MODEL       # same cached fact bank
    out = state.out_dir
    assert (out / "cover_letter.pdf").exists() and (out / "cover_letter.docx").exists()
    text = (out / "cover_letter.md").read_text()
    assert "reduced report generation time by 30%" in text and "Dear Hiring Manager," in text
    assert "| I am an expert in Kubernetes. | me | - | removed |" in text                # listed, but not in the letter
    assert "Kubernetes" not in " ".join(state.cover_letter.paragraphs()) and state.cover_letter.pages == 1
    index = json.loads((out / "archive.json").read_text())
    assert index["files"]["cover_letter_pdf"] == "cover_letter.pdf"
    assert any("Kubernetes" in v.message for v in state.violations)                      # logged with the resume's violations


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@needs_soffice
def test_batch_runs_every_link_without_prompts(master, status, workspace, monkeypatch):
    def fake_get(url, text_path=None, *, allow_paste=True):
        assert allow_paste is False                                   # batch mode never offers the paste prompt
        if "linkedin" in url:
            raise FetchError("www.linkedin.com blocked the request (HTTP 999); it likely needs a login.")
        return JobText(SAMPLE_POSTING + f"\nReference: {url}\n", url, "fetched")

    monkeypatch.setattr(fetch, "get_job_text", fake_get)

    def no_prompt(_prompt=""):
        raise AssertionError("batch mode must never prompt")

    ui.set_input(no_prompt, interactive=True)                         # a terminal is attached, and still nothing is asked
    flagged = make_draft(experiences=[ed("exp_acme", [
        bd("Assisted a 4-person team in developing a scalable Python application", ["exp_acme_f01"], "exp_acme_b01")])],
        skills=["Python"])
    llm = ScriptedLLM({
        "analyze": [make_extract(company="Acme Corp", requirements={"min_gpa": {"value": 3.9, "source": "x"}}),
                    make_extract(company="Globex")],
        "tailor": [flagged, DRAFT],
        "judge": [make_judgment(7, 7, 7, 7, fixes=[("ASK_USER", "Acme", "Any metrics?", 0.5)]), make_judgment(7, 7, 7, 7),
                  make_judgment(9, 9, 9, 9)],
    }, usage_per_call=(1000, 100))
    urls = ["https://careers.acme.com/jobs/1", "https://www.linkedin.com/jobs/view/2", "https://globex.com/careers/3"]
    code = loop.run_batch(Session(master=master, skills_status=status, llm=llm), urls, RunOptions(research=False), today=TODAY)

    assert code == 1                                                  # one link failed; the others still ran
    assert "interview" not in llm.purposes()                          # questions are never asked in batch mode
    rows = list(csv.DictReader((workspace / "applications.csv").open()))
    assert [r["company"] for r in rows] == ["Acme Corp", "Globex"]
    acme, globex = workspace / "output" / rows[0]["output_folder"], workspace / "output" / rows[1]["output_folder"]
    acme = workspace / "output" / "Acme_Corp_Data_Engineering_Intern_2026-10-13"
    globex = workspace / "output" / "Globex_Data_Engineering_Intern_2026-10-13"
    # Eligibility flags lead the report, and the run continued anyway.
    assert (acme / "eligibility.md").read_text().index("## Flags") < (acme / "eligibility.md").read_text().index("## Requirements")
    # The flagged rewrite could not be approved, so the verified wording was used.
    assert "scalable" not in (acme / "resume.docx").read_bytes().decode("latin-1")
    assert "reverted (automatic)" in (acme / "changes.md").read_text()
    # Each job reports only its own Claude usage.
    assert len(json.loads((acme / "usage.json").read_text())["calls"]) == 4       # analyze, tailor, judge x2
    assert len(json.loads((globex / "usage.json").read_text())["calls"]) == 3     # analyze, tailor, judge
    # The dashboard is refreshed at the end.
    assert (workspace / "output" / "insights.html").exists()
    assert "2 analyzed posting(s)" in (workspace / "output" / "insights.md").read_text()


def test_batch_with_only_failures_returns_an_error_code(master, status, workspace, monkeypatch):
    def always_blocked(url, text_path=None, *, allow_paste=True):
        raise FetchError("blocked")

    monkeypatch.setattr(fetch, "get_job_text", always_blocked)
    code = loop.run_batch(Session(master=master, skills_status=status, llm=ScriptedLLM()), ["https://a.example/1"],
                          RunOptions(research=False), today=TODAY)
    assert code == 1 and not (workspace / "output" / "insights.html").exists()
