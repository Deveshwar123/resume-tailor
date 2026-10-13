"""Milestone 2: job posting extraction, caching, wall detection, paste fallback."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
import requests

import config
from src import fetch
from src.errors import FetchError
from tests.conftest import answers

PARAGRAPH = (
    "We are hiring a Data Engineering Intern for Summer 2027 to build data pipelines in Python and SQL. "
    "You will work with analysts and engineers to clean datasets, automate reports, and test new tooling. "
)
TOPICS = ["ingestion", "warehousing", "orchestration", "testing", "monitoring", "modeling", "reporting",
          "governance", "security", "documentation", "automation", "visualization", "forecasting", "tooling"]
# trafilatura drops repeated paragraphs, so every paragraph is distinct.
BODY = "".join(
    f"<p>Area {i + 1} is {topic}. In this part of the internship you will own a small {topic} task, pair with "
    f"a mentor each week, write Python and SQL for the {topic} backlog, and present what you learned about "
    f"{topic} to the wider data team at the end of the sprint.</p>"
    for i, topic in enumerate(TOPICS)
)
POSTING_HTML = (
    "<html><head><title>Data Engineering Intern</title></head><body><nav>Home Careers Sign in</nav>"
    f"<main><h1>Data Engineering Intern</h1><p>{PARAGRAPH}</p>{BODY}</main>"
    "<footer>Copyright Acme</footer></body></html>"
)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")


def _serve(monkeypatch, body: str = POSTING_HTML, status: int = 200, counter: list | None = None):
    def fake_get(url, **kwargs):
        if counter is not None:
            counter.append((url, kwargs))
        return SimpleNamespace(status_code=status, text=body)

    monkeypatch.setattr(fetch.requests, "get", fake_get)


def test_fetch_extracts_main_text_with_one_browser_like_request(monkeypatch):
    calls: list = []
    _serve(monkeypatch, counter=calls)
    text = fetch.fetch_job("https://careers.acme.com/jobs/123")
    assert "Data Engineering Intern" in text and "data pipelines in Python and SQL" in text
    assert "Copyright Acme" not in text
    assert len(calls) == 1                                       # one request, no crawling
    assert "Mozilla/5.0" in calls[0][1]["headers"]["User-Agent"]


def test_fetch_is_cached_for_24_hours(monkeypatch):
    calls: list = []
    _serve(monkeypatch, counter=calls)
    url = "https://careers.acme.com/jobs/123"
    first = fetch.fetch_job(url)
    assert fetch.fetch_job(url) == first and len(calls) == 1     # second call served from cache
    # Age the cache entry past the TTL: the next call fetches again.
    path = fetch._cache_path(url)
    payload = json.loads(path.read_text())
    payload["fetched_at"] = time.time() - (config.CACHE_TTL_HOURS + 1) * 3600
    path.write_text(json.dumps(payload))
    fetch.fetch_job(url)
    assert len(calls) == 2


def test_short_page_is_treated_as_a_wall(monkeypatch):
    _serve(monkeypatch, "<html><body><main><p>Sign in to view this job.</p></main></body></html>")
    with pytest.raises(FetchError, match="login or captcha wall|words were extracted"):
        fetch.fetch_job("https://www.linkedin.com/jobs/view/1")


def test_wall_reason_ignores_markers_in_full_postings():
    full = "Please log in to continue applying later. " + PARAGRAPH * 30
    assert fetch.wall_reason(full) is None
    assert "captcha" in fetch.wall_reason("Complete the captcha to continue " * 5)
    assert "only" in fetch.wall_reason("Great internship. Apply now.", "https://acme.wd1.myworkdayjobs.com/x")


@pytest.mark.parametrize("status,message", [(403, "blocked the request"), (404, "not found"), (500, "HTTP 500")])
def test_http_errors_are_clean(monkeypatch, status, message):
    _serve(monkeypatch, status=status)
    with pytest.raises(FetchError, match=message):
        fetch.fetch_job("https://careers.acme.com/jobs/123")


def test_network_failure_is_clean(monkeypatch):
    def boom(url, **kwargs):
        raise requests.ConnectionError("dns")

    monkeypatch.setattr(fetch.requests, "get", boom)
    with pytest.raises(FetchError, match="Could not fetch"):
        fetch.fetch_job("https://careers.acme.com/jobs/123")


def test_invalid_url_is_rejected():
    with pytest.raises(FetchError, match="not a valid"):
        fetch.fetch_job("careers.acme.com/jobs/123")


def test_json_ld_job_posting_is_used_when_the_page_is_javascript_only(monkeypatch):
    description = f"<p>{PARAGRAPH}</p>{BODY}<ul><li>Python</li><li>SQL</li></ul>"
    ld = json.dumps({"@context": "https://schema.org", "@type": "JobPosting", "title": "Data Engineering Intern",
                     "hiringOrganization": {"@type": "Organization", "name": "Acme"},
                     "jobLocation": {"@type": "Place", "address": {"addressLocality": "Austin", "addressRegion": "TX"}},
                     "description": description})
    html = f'<html><head><script type="application/ld+json">{ld}</script></head><body><div id="root"></div></body></html>'
    _serve(monkeypatch, html)
    text = fetch.fetch_job("https://acme.wd1.myworkdayjobs.com/job/1")
    assert text.startswith("Data Engineering Intern")
    assert "Company: Acme" in text and "Location: Austin, TX" in text and "- Python" in text


def test_text_file_fallback(tmp_path):
    path = tmp_path / "job.txt"
    path.write_text("Quant intern. " + PARAGRAPH * 3)
    job = fetch.get_job_text(text_path=path)
    assert job.source == "file" and job.words > 20
    path.write_text("   ")
    with pytest.raises(FetchError, match="empty"):
        fetch.get_job_text(text_path=path)


def test_paste_fallback_when_the_page_is_a_wall(monkeypatch):
    _serve(monkeypatch, status=403)
    pasted = [f"Line {i}: " + PARAGRAPH for i in range(4)]
    answers(*pasted, "")
    job = fetch.get_job_text("https://www.linkedin.com/jobs/view/1")
    assert job.source == "pasted" and "Line 3" in job.text
    # The paste is cached, so a re-run within 24h does not prompt again.
    assert fetch.get_job_text("https://www.linkedin.com/jobs/view/1").source == "cache"


def test_batch_mode_never_prompts(monkeypatch):
    _serve(monkeypatch, status=403)
    answers("this should not be read")
    with pytest.raises(FetchError) as info:
        fetch.get_job_text("https://www.linkedin.com/jobs/view/1", allow_paste=False)
    assert "--text" in (info.value.hint or "")


def test_empty_paste_is_an_error(monkeypatch):
    _serve(monkeypatch, status=403)
    answers("")
    with pytest.raises(FetchError, match="No usable job description"):
        fetch.get_job_text("https://www.linkedin.com/jobs/view/1")
