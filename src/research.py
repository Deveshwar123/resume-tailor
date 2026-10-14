"""Company research (Sonnet + Anthropic's server-side web search) with source tiers.

Two calls:

1. Research: Sonnet searches the web (capped at MAX_RESEARCH_SEARCHES) and
   writes short notes, each with the URL it came from.
2. Structuring: the notes become a `CompanyBriefDraft` - one sourced sentence
   per item.

The tier of every item is then assigned **in code** from its source, never by
the model:

    official     the company's own domain (site, careers pages, engineering blog)
    posting      the job description itself
    third_party  reputable news or reporting (allowlist in config.py)
    unverified   forums, aggregators, anything else, and anything whose source
                 the search tool did not actually return

The tailor weights official and posting items heavily, third-party items
lightly, and never sees unverified ones. Research only changes what gets
emphasized; validate.py makes sure it can never add a claim, skill, or tool.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import config
from src import textutil, ui
from src.errors import LLMError
from src.fetch import JobText
from src.models import (
    BriefItem,
    BriefItemDraft,
    BriefSource,
    CompanyBrief,
    CompanyBriefDraft,
    JobAnalysis,
    Tier,
)

RESEARCH_SYSTEM = """You research a company for a student who is tailoring an internship resume. Today's date is {today}. Your training data ends before today, so search rather than rely on memory for anything that may have changed.

Find, in this order of priority:
1. What the company does and what the team or division in the posting works on.
2. The technologies, tools, and methods its engineers or analysts say they use.
3. The words the company uses about its values and how it works.
4. What interns in this kind of role actually do there.
5. Notable news from the last year that a candidate should know.

Prefer the company's own website, careers pages, and engineering or research blog. Use reputable news for recent events. Forums and review aggregators are low-trust: use them only if nothing better exists, and say so.

You have at most {max_searches} searches, so make each one count. Report only what a source states; do not guess or generalize from the industry. If you find nothing reliable for a topic, say that nothing was found.

Write plain notes: one finding per line, each ending with the exact URL it came from in parentheses."""

RESEARCH_USER = """Company: {company}
Role: {role}
Location: {location}
Job posting link: {url}"""

BRIEF_SYSTEM = """You turn research notes about a company into a structured brief for a resume-tailoring tool.

Rules
- One short, factual sentence per item. Keep only what the notes or the job posting actually state.
- `source_url` for each item must be one of: a URL copied exactly from the numbered source list; the word "posting" when the item comes from the job posting text; or null when the notes give no source.
- Do not merge findings from different sources into one item.
- Leave a field empty when the notes have nothing for it.

Fields
- summary: what the company and this team do.
- tech_stack: technologies, tools, and methods the company says it uses.
- values_language: the company's own words about its values and ways of working.
- intern_work: what interns in this kind of role do.
- recent_news: notable developments from the last year."""

BRIEF_USER = """<sources>
{sources}
</sources>

<research_notes>
{notes}
</research_notes>

<job_posting>
{posting}
</job_posting>"""

_COMPANY_SUFFIXES = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp", "corporation", "co", "company",
    "plc", "group", "holdings", "the", "and", "of",
}
# Words a company's own domain commonly adds to its name (metacareers.com, aboutamazon.com).
_DOMAIN_AFFIXES = {
    "careers", "career", "jobs", "job", "tech", "technology", "engineering", "eng", "blog", "labs", "lab",
    "ai", "research", "dev", "developers", "hq", "inc", "group", "corp", "about", "life", "team", "global",
    "news", "newsroom", "cloud", "docs",
}
_TWO_LEVEL_SUFFIXES = {"co.uk", "com.au", "co.in", "co.jp", "com.br", "co.nz", "com.sg", "co.kr", "com.mx", "ac.uk"}


# ---------------------------------------------------------------------------
# Domains and tiers (pure code)
# ---------------------------------------------------------------------------


def host_of(url: str) -> str:
    candidate = url if "://" in url else f"https://{url}"
    host = (urlparse(candidate).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def domain_label(host: str) -> str:
    """The registrable name of a host: "jobs.boeing.com" -> "boeing"."""
    parts = host.split(".")
    if len(parts) < 2:
        return host
    if ".".join(parts[-2:]) in _TWO_LEVEL_SUFFIXES and len(parts) >= 3:
        return parts[-3]
    return parts[-2]


def registrable(host: str) -> str:
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_LEVEL_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def company_tokens(company: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", textutil.normalize(company).replace("&", " and "))
    return [w for w in words if w not in _COMPANY_SUFFIXES]


def company_slug(company: str) -> str:
    return "".join(company_tokens(company))


def _is_host_in(host: str, domains) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def is_official_host(host: str, company: str, job_url: str = "") -> bool:
    """Is `host` the company's own site?

    True when the registrable name is the company name (optionally with a word
    such as "careers" or "blog"), when it is one of the name's own words, when
    it is the host of the job link I gave (unless that is a job board), or when
    it is listed in OFFICIAL_DOMAIN_OVERRIDES.
    """
    if not host or _is_host_in(host, config.JOB_BOARD_HOSTS) or _is_host_in(host, config.REPUTABLE_NEWS_DOMAINS):
        return False
    slug, tokens = company_slug(company), company_tokens(company)
    if _is_host_in(host, config.OFFICIAL_DOMAIN_OVERRIDES.get(slug, [])):
        return True
    job_host = host_of(job_url) if job_url else ""
    if job_host and registrable(job_host) == registrable(host) and not _is_host_in(job_host, config.JOB_BOARD_HOSTS):
        return True
    label = domain_label(host).replace("-", "")
    if not slug or not label:
        return False
    if label == slug or (len(label) >= 4 and label in tokens):
        return True
    for affix in _DOMAIN_AFFIXES:
        if label in (slug + affix, affix + slug):
            return True
    return False


def _same_page(a: str, b: str) -> bool:
    def norm(url: str) -> str:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        return f"{(parsed.hostname or '').lower().removeprefix('www.')}{parsed.path.rstrip('/')}"

    return bool(a and b) and norm(a) == norm(b)


def tier_for_url(url: str, company: str, job_url: str = "") -> Tier:
    """Tier of a source the search tool returned."""
    host = host_of(url)
    if job_url and _same_page(url, job_url):
        return Tier.posting
    if is_official_host(host, company, job_url):
        return Tier.official
    if _is_host_in(host, config.REPUTABLE_NEWS_DOMAINS):
        return Tier.third_party
    return Tier.unverified


def _grounded_in_posting(text: str, posting: str) -> bool:
    """A "posting" item must actually reflect the posting text (most of its content words appear there)."""
    wanted = textutil.content_words(text)
    if not wanted:
        return False
    have = set(textutil.words(posting))
    stems = {w[:6] for w in have}
    hits = sum(1 for w in wanted if w in have or w[:6] in stems)
    return hits / len(wanted) >= 0.6


def assign_tier(item: BriefItemDraft, company: str, job_url: str, returned_urls: dict[str, str],
                posting_text: str) -> BriefItem:
    """Tag one item. A source the search tool never returned cannot be trusted."""
    source = (item.source_url or "").strip()
    text = textutil.plain(item.text)
    if not source:
        return BriefItem(text=text, source_url=None, tier=Tier.unverified)
    if source.lower() == "posting":
        tier = Tier.posting if _grounded_in_posting(text, posting_text) else Tier.unverified
        return BriefItem(text=text, source_url="posting", tier=tier)
    match = next((url for url in returned_urls if _same_page(url, source)), None)
    if match is None:
        return BriefItem(text=text, source_url=source, tier=Tier.unverified)
    return BriefItem(text=text, source_url=match, tier=tier_for_url(match, company, job_url))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_path(company: str) -> Path:
    return config.CACHE_DIR / "research" / f"{company_slug(company) or 'unknown'}.json"


def read_cache(company: str) -> CompanyBrief | None:
    path = _cache_path(company)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(payload.get("saved_at", 0)) > config.CACHE_TTL_HOURS * 3600:
            return None
        return CompanyBrief.model_validate(payload["brief"])
    except (OSError, ValueError, KeyError):
        return None


def write_cache(brief: CompanyBrief) -> None:
    path = _cache_path(brief.company)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"saved_at": time.time(), "brief": brief.model_dump(mode="json")}), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_brief(draft: CompanyBriefDraft, analysis: JobAnalysis, job: JobText,
                returned: list[tuple[str, str]], today: date) -> CompanyBrief:
    """Assign tiers in code and assemble the brief."""
    returned_urls = dict(returned)
    brief = CompanyBrief(company=analysis.company, researched_on=today.isoformat())
    for name in CompanyBrief.FIELDS:
        seen: set[str] = set()
        for item in getattr(draft, name):
            if not item.text.strip() or textutil.squash(item.text) in seen:
                continue
            seen.add(textutil.squash(item.text))
            getattr(brief, name).append(assign_tier(item, analysis.company, job.url, returned_urls, job.text))
    brief.sources = [BriefSource(url=url, title=title, tier=tier_for_url(url, analysis.company, job.url))
                     for url, title in returned]
    return brief


def research_company(llm, analysis: JobAnalysis, job: JobText, today: date | None = None,
                     use_cache: bool = True) -> CompanyBrief:
    """Research the company. Never fatal: on failure the run continues without a brief."""
    today = today or date.today()
    if use_cache:
        cached = read_cache(analysis.company)
        if cached is not None:
            return cached
    try:
        found = llm.search(
            purpose="research", model=config.WRITE_MODEL,
            system=RESEARCH_SYSTEM.format(today=today.isoformat(), max_searches=config.MAX_RESEARCH_SEARCHES),
            user=RESEARCH_USER.format(company=analysis.company, role=analysis.role,
                                      location=analysis.location or "not stated", url=job.url or "not given"),
            max_searches=config.MAX_RESEARCH_SEARCHES,
        )
        if found.errors:
            ui.warn("Web search reported: " + ", ".join(found.errors))
        sources = "\n".join(f"{i}. {url}  ({title})" for i, (url, title) in enumerate(found.sources, 1)) or "(none)"
        draft = llm.parse(
            purpose="brief", model=config.FAST_MODEL, system=BRIEF_SYSTEM,
            user=BRIEF_USER.format(sources=sources, notes=found.text or "(no notes)", posting=job.text),
            schema=CompanyBriefDraft, cache_system=False,
        )
    except LLMError as exc:
        ui.warn(f"Company research failed ({exc.message}); continuing without it.")
        return CompanyBrief(company=analysis.company, researched_on=today.isoformat())
    brief = build_brief(draft, analysis, job, found.sources, today)
    write_cache(brief)
    return brief


def tier_counts(brief: CompanyBrief) -> dict[str, int]:
    counts = {tier.value: 0 for tier in Tier}
    for name in CompanyBrief.FIELDS:
        for item in getattr(brief, name):
            counts[item.tier.value] += 1
    return counts


def summarize(brief: CompanyBrief) -> str:
    counts = tier_counts(brief)
    if not sum(counts.values()):
        return "no usable findings"
    return (f"{counts['official']} official, {counts['posting']} from the posting, {counts['third_party']} news, "
            f"{counts['unverified']} unverified (ignored); {len(brief.sources)} source(s)")
