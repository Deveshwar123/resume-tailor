"""Job posting extraction: link -> clean job text.

Rules from the spec:

* exactly ONE request for the URL I give - never crawl, never follow links
* a normal browser User-Agent, main text extracted with trafilatura
* results cached by URL for 24 hours
* if the page is short or looks like a login / captcha wall (LinkedIn, Workday
  and Handshake often are), ask me to paste the description, or use --text
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura

import config
from src import textutil, ui
from src.errors import FetchError


@dataclass
class JobText:
    text: str
    url: str = ""
    source: str = "fetched"      # fetched | cache | file | pasted

    @property
    def words(self) -> int:
        return textutil.word_count(self.text)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:20]
    return config.CACHE_DIR / "fetch" / f"{digest}.json"


def read_cache(url: str, ttl_hours: float | None = None) -> str | None:
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ttl = (config.CACHE_TTL_HOURS if ttl_hours is None else ttl_hours) * 3600
    if time.time() - float(payload.get("fetched_at", 0)) > ttl:
        return None
    return payload.get("text") or None


def write_cache(url: str, text: str) -> None:
    path = _cache_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "fetched_at": time.time(), "text": text}), encoding="utf-8")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_LD_JSON_RE = re.compile(
    r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL
)


def _html_to_text(fragment: str) -> str:
    """Flatten an HTML fragment (a JSON-LD description) to readable text."""
    fragment = html_lib.unescape(fragment or "")
    fragment = re.sub(r"(?i)<\s*(br|/p|/div|/li|/h[1-6]|/tr)\s*/?>", "\n", fragment)
    fragment = re.sub(r"(?i)<\s*li[^>]*>", "- ", fragment)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    fragment = html_lib.unescape(fragment)
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in fragment.splitlines()]
    return "\n".join(line for line in lines if line)


def _find_job_postings(node: object) -> list[dict]:
    found: list[dict] = []
    if isinstance(node, dict):
        kind = node.get("@type")
        kinds = kind if isinstance(kind, list) else [kind]
        if "JobPosting" in kinds:
            found.append(node)
        for value in node.values():
            found.extend(_find_job_postings(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_find_job_postings(value))
    return found


def extract_json_ld(html: str) -> str:
    """Job text from a schema.org JobPosting block embedded in the same page.

    Many applicant-tracking systems render the description with JavaScript but
    still embed it as JSON-LD, so this recovers postings trafilatura cannot see.
    """
    best = ""
    for raw in _LD_JSON_RE.findall(html or ""):
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        for posting in _find_job_postings(data):
            parts: list[str] = []
            if posting.get("title"):
                parts.append(str(posting["title"]))
            org = posting.get("hiringOrganization")
            if isinstance(org, dict) and org.get("name"):
                parts.append(f"Company: {org['name']}")
            location = posting.get("jobLocation")
            locations = location if isinstance(location, list) else [location]
            places = []
            for item in locations:
                address = item.get("address") if isinstance(item, dict) else None
                if isinstance(address, dict):
                    place = ", ".join(str(address[k]) for k in ("addressLocality", "addressRegion") if address.get(k))
                    if place:
                        places.append(place)
            if places:
                parts.append("Location: " + "; ".join(dict.fromkeys(places)))
            for key in ("description", "responsibilities", "qualifications", "skills",
                        "educationRequirements", "experienceRequirements"):
                value = posting.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(_html_to_text(value))
            text = "\n".join(parts).strip()
            if len(text) > len(best):
                best = text
    return best


def extract_text(html: str) -> str:
    """Main text of a job page: the longer of trafilatura's result and JSON-LD."""
    main = trafilatura.extract(
        html or "", include_comments=False, include_tables=True, favor_recall=True
    ) or ""
    structured = extract_json_ld(html or "")
    return (structured if textutil.word_count(structured) > textutil.word_count(main) else main).strip()


def wall_reason(text: str, url: str = "") -> str | None:
    """Why this text is unusable as a posting, or None if it looks fine."""
    count = textutil.word_count(text)
    lowered = (text or "").lower()
    # A full-length posting can mention "sign in" in passing; only short pages are walls.
    if count < max(config.MIN_JOB_WORDS * 2, 600):
        for marker in config.WALL_MARKERS:
            if marker in lowered:
                return f'the page looks like a login or captcha wall ("{marker}")'
    if count < config.MIN_JOB_WORDS:
        host = urlparse(url).netloc.lower()
        hint = ""
        if any(walled in host for walled in config.WALLED_HOSTS):
            hint = f"; {host} usually needs a login or JavaScript"
        return f"only {count} words were extracted (need about {config.MIN_JOB_WORDS}){hint}"
    return None


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _validate_url(url: str) -> str:
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise FetchError(f'"{url}" is not a valid http(s) link.', "Pass a full link, or use --text job.txt.")
    return url


def fetch_job(url: str, *, use_cache: bool = True) -> str:
    """Fetch one job posting and return its clean text.

    Raises `FetchError` when the request fails or the page is a login wall /
    too short; callers that can prompt offer the paste fallback instead.
    """
    url = _validate_url(url)
    if use_cache:
        cached = read_cache(url)
        if cached:
            return cached
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": config.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=config.FETCH_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
    except requests.Timeout as exc:
        raise FetchError(f"The request to {urlparse(url).netloc} timed out.") from exc
    except requests.RequestException as exc:
        raise FetchError(f"Could not fetch the posting: {exc.__class__.__name__}.") from exc
    if response.status_code in (401, 403, 429, 999):
        raise FetchError(
            f"{urlparse(url).netloc} blocked the request (HTTP {response.status_code}); it likely needs a login."
        )
    if response.status_code == 404:
        raise FetchError("The posting was not found (HTTP 404). It may have been taken down.")
    if response.status_code >= 400:
        raise FetchError(f"The site returned HTTP {response.status_code}.")
    text = extract_text(response.text)
    reason = wall_reason(text, url)
    if reason:
        raise FetchError(f"Could not read the posting: {reason}.")
    write_cache(url, text)
    return text


def read_text_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        raise FetchError(f"Could not read {path}: {exc.strerror or exc}.") from exc
    if not text:
        raise FetchError(f"{path} is empty.")
    return text


def paste_fallback(reason: str) -> str:
    """Ask me to paste the job description. Returns "" if nothing was pasted."""
    ui.warn(reason)
    return ui.read_multiline(
        "Paste the job description below, then press Enter on a blank line "
        "(or run again with --text job.txt):"
    )


def get_job_text(url: str | None = None, text_path: Path | None = None, *, allow_paste: bool = True) -> JobText:
    """Resolve the posting text from --text, the cache, the web, or a paste.

    `allow_paste=False` (batch mode) turns a wall into a `FetchError` instead of
    a prompt, so an unattended run moves on to the next link.
    """
    if text_path is not None:
        return JobText(read_text_file(text_path), url or "", "file")
    if not url:
        raise FetchError("No job link or --text file was given.")
    url = _validate_url(url)
    cached = read_cache(url)
    if cached:
        return JobText(cached, url, "cache")
    try:
        return JobText(fetch_job(url, use_cache=False), url, "fetched")
    except FetchError as exc:
        if not allow_paste or not ui.is_interactive():
            raise FetchError(exc.message, "Save the description to a file and run: python tailor.py --text job.txt") from exc
        pasted = paste_fallback(exc.message)
        if textutil.word_count(pasted) < 40:
            raise FetchError(
                "No usable job description was pasted.",
                "Save the description to a file and run: python tailor.py --text job.txt",
            ) from exc
        write_cache(url, pasted)       # so a re-run within 24h does not ask again
        return JobText(pasted, url, "pasted")
