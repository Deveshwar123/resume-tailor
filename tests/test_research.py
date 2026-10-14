"""Milestone 9: company research with source tiers assigned in code."""

from __future__ import annotations

import json
import time
from datetime import date

import pytest

import config
from src import research
from src.errors import LLMError
from src.fetch import JobText
from src.llm import SearchResult
from src.models import BriefItemDraft, CompanyBriefDraft, Tier
from tests.conftest import SAMPLE_POSTING, ScriptedLLM, make_analysis

JOB_URL = "https://careers.acme.com/jobs/123"
TODAY = date(2026, 10, 13)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url,company,job_url,expected", [
    ("https://www.boeing.com/company", "Boeing", "", Tier.official),
    ("https://jobs.boeing.com/intern", "The Boeing Company", "", Tier.official),
    ("https://www.metacareers.com/life", "Meta", "", Tier.official),              # name + "careers"
    ("https://engineering.atspotify.com/x", "Spotify", "https://engineering.atspotify.com/jobs/1", Tier.official),
    ("https://www.gs.com/careers", "Goldman Sachs", "", Tier.official),            # config override
    ("https://www.jpmorgan.com/technology", "JPMorgan Chase & Co.", "", Tier.official),
    ("https://www.twosigma.com/careers", "Two Sigma", "", Tier.official),
    ("https://appleinsider.com/article", "Apple", "", Tier.unverified),            # contains the name, not the company
    ("https://www.reuters.com/technology/acme", "Acme Corp", "", Tier.third_party),
    ("https://techcrunch.com/2026/acme", "Acme Corp", "", Tier.third_party),
    ("https://www.reddit.com/r/cscareerquestions/acme", "Acme Corp", "", Tier.unverified),
    ("https://www.glassdoor.com/Reviews/Acme", "Acme Corp", "", Tier.unverified),
    ("https://boards.greenhouse.io/acme/jobs/9", "Acme Corp", "", Tier.unverified),  # a job board is never "official"
    ("https://en.wikipedia.org/wiki/Acme", "Acme Corp", "", Tier.unverified),
])
def test_tier_for_url(url, company, job_url, expected):
    assert research.tier_for_url(url, company, job_url) == expected


def test_the_job_link_itself_is_the_posting_tier():
    assert research.tier_for_url("https://careers.acme.com/jobs/123/", "Acme Corp", JOB_URL) == Tier.posting
    assert research.tier_for_url("https://boards.greenhouse.io/acme/jobs/9", "Acme",
                                 "https://boards.greenhouse.io/acme/jobs/9") == Tier.posting
    # Other pages on the careers site I linked to are official.
    assert research.tier_for_url("https://careers.acme.com/teams/data", "Acme Corp", JOB_URL) == Tier.official


def test_domain_helpers():
    assert research.domain_label("jobs.boeing.com") == "boeing"
    assert research.domain_label("careers.bbc.co.uk") == "bbc"
    assert research.company_slug("JPMorgan Chase & Co.") == "jpmorganchase"
    assert research.company_tokens("The Goldman Sachs Group, Inc.") == ["goldman", "sachs"]


def test_items_are_tagged_from_their_source_not_by_the_model():
    returned = {"https://careers.acme.com/teams/data": "Data team", "https://www.reuters.com/acme": "Reuters"}

    def tag(text, source):
        return research.assign_tier(BriefItemDraft(text=text, source_url=source), "Acme Corp", JOB_URL,
                                    returned, SAMPLE_POSTING)

    assert tag("The data team builds pipelines in Python.", "https://careers.acme.com/teams/data/").tier == Tier.official
    assert tag("Acme raised new funding.", "https://www.reuters.com/acme").tier == Tier.third_party
    # A source the search tool never returned is not trusted, even on the company's own domain.
    assert tag("Acme uses Snowflake.", "https://www.acme.com/secret-stack").tier == Tier.unverified
    assert tag("Acme is great.", None).tier == Tier.unverified
    # "posting" items must actually reflect the posting text.
    assert tag("Interns build data pipelines in Python and SQL.", "posting").tier == Tier.posting
    assert tag("Interns travel to Tokyo for quantum hardware training.", "posting").tier == Tier.unverified


# ---------------------------------------------------------------------------
# research_company
# ---------------------------------------------------------------------------

FOUND = SearchResult(
    text="- Acme's data team builds pipelines in Python (https://careers.acme.com/teams/data)\n"
         "- Acme raised a Series C (https://www.reuters.com/acme)\n- Interns say the work is fun (https://www.reddit.com/r/acme)",
    sources=[("https://careers.acme.com/teams/data", "Data team"), ("https://www.reuters.com/acme", "Reuters"),
             ("https://www.reddit.com/r/acme", "Reddit")],
    searches=3,
)
DRAFT = CompanyBriefDraft(
    summary=[BriefItemDraft(text="Acme builds logistics software.", source_url="https://careers.acme.com/teams/data")],
    tech_stack=[BriefItemDraft(text="The data team builds pipelines in Python.", source_url="https://careers.acme.com/teams/data"),
                BriefItemDraft(text="Acme is moving to Rust.", source_url="https://www.reddit.com/r/acme")],
    values_language=[],
    intern_work=[BriefItemDraft(text="Interns build data pipelines in Python and SQL.", source_url="posting")],
    recent_news=[BriefItemDraft(text="Acme raised a Series C.", source_url="https://www.reuters.com/acme")],
)


def run(llm=None, **kw):
    llm = llm or ScriptedLLM({"research": [FOUND], "brief": [DRAFT]})
    return llm, research.research_company(llm, make_analysis(), JobText(SAMPLE_POSTING, JOB_URL), TODAY, **kw)


def test_research_builds_a_tiered_brief():
    llm, brief = run()
    assert brief.company == "Acme Corp" and brief.researched_on == "2026-10-13"
    assert brief.summary[0].tier == Tier.official
    assert [i.tier for i in brief.tech_stack] == [Tier.official, Tier.unverified]
    assert brief.intern_work[0].tier == Tier.posting and brief.recent_news[0].tier == Tier.third_party
    assert [(s.url, s.tier) for s in brief.sources] == [
        ("https://careers.acme.com/teams/data", Tier.official), ("https://www.reuters.com/acme", Tier.third_party),
        ("https://www.reddit.com/r/acme", Tier.unverified)]
    assert research.summarize(brief) == "2 official, 1 from the posting, 1 news, 1 unverified (ignored); 3 source(s)"


def test_research_calls_use_the_write_model_with_capped_search_then_structure():
    llm, _ = run()
    search, structure = llm.calls
    assert search["purpose"] == "research" and search["model"] == config.WRITE_MODEL
    assert "2026-10-13" in search["system"] and f"at most {config.MAX_RESEARCH_SEARCHES} searches" in search["system"]
    assert "Acme Corp" in search["user"] and JOB_URL in search["user"]
    assert structure["purpose"] == "brief" and structure["schema"].__name__ == "CompanyBriefDraft"
    assert "1. https://careers.acme.com/teams/data" in structure["user"] and "Minimum GPA of 3.5." in structure["user"]


def test_emphasis_filters_by_tier():
    _, brief = run()
    for_tailor = brief.items(tuple(config.TAILOR_TIER_WEIGHTS))
    for_judge = brief.items(config.JUDGE_TIERS)
    assert "Rust" not in json.dumps({k: [i.text for i in v] for k, v in for_tailor.items()})   # unverified is ignored
    assert "recent_news" in for_tailor and "recent_news" not in for_judge                       # news: tailor only


def test_research_is_cached_for_24_hours():
    run()
    llm = ScriptedLLM()                                    # no responses queued: any call would fail
    _, cached = run(llm)
    assert llm.calls == [] and cached.summary[0].text == "Acme builds logistics software."
    path = research._cache_path("Acme Corp")
    payload = json.loads(path.read_text())
    payload["saved_at"] = time.time() - (config.CACHE_TTL_HOURS + 1) * 3600
    path.write_text(json.dumps(payload))
    llm, _ = run()
    assert llm.purposes() == ["research", "brief"]         # expired: researched again


def test_research_failure_never_stops_the_run():
    def boom(_kwargs):
        raise LLMError("web search unavailable")

    _, brief = run(ScriptedLLM({"research": [boom]}))
    assert brief.is_empty() and brief.company == "Acme Corp"
    assert research.summarize(brief) == "no usable findings"
