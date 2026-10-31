"""Speed: independent calls overlap, cached model checks, no redundant re-judge, timings recorded."""

from __future__ import annotations

import json
import threading
from datetime import date

import pytest

import config
from src import loop, render, research, startup
from src.loop import RunOptions
from src.models import BriefItem, CompanyBrief, Tier
from src.startup import Session
from tests.conftest import SAMPLE_POSTING, ScriptedLLM, bd, ed, make_draft, make_extract, make_judgment

needs_soffice = pytest.mark.skipif(render.find_soffice() is None, reason="LibreOffice (soffice) is not installed")
TODAY = date(2026, 10, 13)
DRAFT = make_draft(experiences=[ed("exp_acme", [
    bd("Assisted a 4-person team in building a Python application for data work", ["exp_acme_f01"], "exp_acme_b01"),
    bd("Wrote SQL queries that cut report generation time by 30%", ["exp_acme_f02"], "exp_acme_b02", "minor_wording"),
])], skills=["Python", "SQL"])


def waits_for(event: threading.Event, result, seconds: float = 10.0):
    """A scripted response that only returns once `event` is set, i.e. once another call has started."""
    def respond(_kwargs):
        assert event.wait(seconds), "the other call never started, so the two did not overlap"
        return result
    return respond


def starts(event: threading.Event, result):
    def respond(_kwargs):
        event.set()
        return result
    return respond


@pytest.fixture(autouse=True)
def own_cache(tmp_path, monkeypatch):
    """A private cache: these tests depend on whether a research brief is cached."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")


@pytest.fixture
def posting(tmp_path):
    path = tmp_path / "job.txt"
    path.write_text(SAMPLE_POSTING)
    return path


@needs_soffice
def test_research_overlaps_the_analysis_and_the_letter_overlaps_the_loop(master, status, posting):
    from src.cover_letter import CoverLetterDraft
    from src.llm import SearchResult
    from src.models import CompanyBriefDraft

    research_started, letter_started = threading.Event(), threading.Event()
    letter = CoverLetterDraft.model_validate({"greeting": "Dear Hiring Manager,", "closing": "Sincerely,", "paragraphs": [
        {"sentences": [{"text": "I am applying for the Data Engineering Intern role at Acme Corp.", "about": "other",
                        "fact_ids": []}]}]})
    empty_brief = CompanyBriefDraft(summary=[], tech_stack=[], values_language=[], intern_work=[], recent_news=[])
    llm = ScriptedLLM({
        # The analysis only answers once research has started: they run at the same time.
        "analyze": [waits_for(research_started, make_extract())],
        "research": [starts(research_started, SearchResult(text="notes", sources=[]))],
        "brief": [empty_brief],
        "tailor": [DRAFT],
        # The judge only answers once the cover letter has started: the letter is written during the loop.
        "judge": [waits_for(letter_started, make_judgment(9, 9, 9, 9))],
        "cover_letter": [starts(letter_started, letter)],
    })
    state = loop.run_job(Session(master=master, skills_status=status, llm=llm), text_path=posting,
                         options=RunOptions(cover_letter=True, batch=True), today=TODAY)
    assert state.cover_letter is not None and state.cover_letter.pages == 1
    assert state.brief is not None and state.brief.company == "Acme Corp"
    # Every step's time is recorded and reported.
    for step in ("fetch", "analyze", "tailor", "judge and revise", "render", "cover letter", "reports"):
        assert step in state.timings
    report = json.loads((state.out_dir / "report.json").read_text())
    assert report["seconds"] == state.seconds and report["timings"]["tailor"] >= 0
    usage = json.loads((state.out_dir / "usage.json").read_text())
    assert all("seconds" in call for call in usage["calls"])


def test_with_parallel_calls_off_the_run_is_sequential(master, status, posting, monkeypatch):
    monkeypatch.setattr(config, "PARALLEL_CALLS", False)
    llm = ScriptedLLM({"analyze": [make_extract()], "tailor": [DRAFT], "judge": [make_judgment(9, 9, 9, 9)]})
    llm.queue("research")                          # nothing queued: research must not run before the analysis
    with pytest.raises(AssertionError, match="research"):
        loop.run_job(Session(master=master, skills_status=status, llm=llm), text_path=posting,
                     options=RunOptions(batch=True), today=TODAY)
    assert llm.purposes()[:2] == ["analyze", "research"]


def test_a_cached_brief_is_found_from_the_posting_before_the_analysis():
    from src.fetch import JobText

    research.write_cache(CompanyBrief(company="Acme Corp", researched_on="2026-10-13",
                                      tech_stack=[BriefItem(text="Acme runs Python services", tier=Tier.official)]))
    found = research.find_cached(JobText(SAMPLE_POSTING, "https://careers.acme.com/jobs/123"))
    assert found is not None and found.company == "Acme Corp"
    assert research.find_cached(JobText("Globex is hiring a data intern. " * 20, "https://globex.com/jobs/1")) is None
    # A job-board link that carries the company's name also counts.
    assert research.find_cached(JobText("We are hiring. " * 30, "https://boards.greenhouse.io/acmecorp/jobs/9")) is not None


class CountingLLM:
    def __init__(self):
        self.verified = 0

    def verify_models(self, models):
        self.verified += 1

    def verify_access(self, model):
        pass


def test_model_checks_are_remembered_for_a_day(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(config, "MASTER_RESUME_PATH", config.ROOT / "master_resume.example.yaml")
    monkeypatch.setattr(config, "SKILLS_STATUS_PATH", config.ROOT / "skills_status.example.yaml")
    fake = CountingLLM()
    monkeypatch.setattr(startup, "make_llm", lambda env: fake)
    startup.run_startup(quiet=True)
    startup.run_startup(quiet=True)
    assert fake.verified == 1                                      # the second run trusts the recent check
    startup.run_startup(quiet=True, test_call=True)
    assert fake.verified == 2                                      # --check always checks afresh
    monkeypatch.setattr(config, "WRITE_MODEL", "some-other-model")
    startup.run_startup(quiet=True)
    assert fake.verified == 3                                      # different models: check again


def test_light_steps_use_the_fast_model_and_the_judge_runs_at_medium_effort():
    from src import tailor_resume
    import inspect

    assert config.EFFORT["judge"] == "medium"
    assert "config.FAST_MODEL" in inspect.getsource(tailor_resume.make_retry_fn)
    assert "config.FAST_MODEL" in inspect.getsource(render.make_tighten_fn)
