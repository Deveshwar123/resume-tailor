"""Job analysis: posting text -> structured requirements and keywords.

One Haiku call with structured output, then deterministic clean-up in code:
weights are clamped, keywords are de-duplicated with the synonym map, and every
"exact source sentence" Claude returned is checked against the posting text.
Each analysis is saved to data/postings/ so insights.py can read it later.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import config
from src import textutil
from src.fetch import JobText
from src.models import JobAnalysis, JobAnalysisExtract, WeightedSkill

SYSTEM = """You extract structured data from a job posting for a resume-tailoring tool.

Accuracy rules:
- Extract only what the posting says. Never infer a requirement it does not state.
- Every `source` field must be the exact sentence (or bullet) copied character-for-character from the posting. Use null when the posting says nothing about that requirement.
- Fill a structured value (dates, standings, GPA, authorization kind) only when the posting states it explicitly. When the wording needs interpretation ("rising juniors", "penultimate-year students", "returning to school after the internship"), leave the structured value empty or null and still copy the sentence into `source`. Code interprets it later and labels it.

Field guide:
- company, role, location: as written. Use "" when the posting does not say.
- job_type: software, data, finance, research, or other - the closest fit for the role.
- required_skills: skills the posting requires ("must", "required", "minimum qualifications", or listed without a qualifier). preferred_skills: "preferred", "nice to have", "bonus", "a plus".
  - weight: 3 = central to the role (in the title, repeated, or listed first); 2 = clearly important; 1 = mentioned once.
  - kind: language (programming or query languages), tool (named products, platforms, libraries), technical (technical concepts such as machine learning or ETL), domain (domain knowledge such as accounting or equity research), soft (communication, teamwork, leadership).
  - Use the shortest standard name for each skill ("Python", "SQL", "machine learning"), one skill per item.
- tools: every named technology, product, or platform in the posting.
- key_phrases: 5-12 distinctive phrases the posting uses to describe the work, copied verbatim.
- responsibilities: what the intern will do, one short item each.
- keywords: a flat list of the terms an applicant tracking system would match for this role (skills, tools, methods, domain terms). No duplicates, no full sentences.
- requirements:
  - grad_year_range: earliest/latest graduation as "YYYY" or "YYYY-MM" when the posting gives dates or class years.
  - class_standing.allowed: the standings the candidate must hold during the school year in which they apply, only when named directly (freshman, sophomore, junior, senior, graduate). "First-year" is freshman; "second-year" is sophomore.
  - degree_fields.fields: majors or fields of study named. related_fields_ok is true when the posting adds "or related field" or similar.
  - min_gpa.value: the minimum GPA on a 4.0 scale, when stated.
  - work_authorization.kind: citizen_required (U.S. citizenship required), us_person_required (citizen or permanent resident, "U.S. person", export-control wording), no_sponsorship (must be authorized to work without sponsorship; sponsorship not offered), sponsorship_available, other (stated but none of these), not_stated.
  - prior_experience_required: required is true only when prior internship, research, or work experience is a stated requirement; kind is internship, research, work, or any.
  - other: any other hard requirement (location, availability dates, clearance, enrollment status), each with its source sentence."""


def _posting_id(company: str, role: str, text: str, today: date) -> str:
    digest = hashlib.sha256(textutil.squash(text).encode("utf-8")).hexdigest()[:8]
    return f"{today.isoformat()}_{textutil.slug(company, 30)}_{textutil.slug(role, 40)}_{digest}"


def _clean_skills(skills: list[WeightedSkill]) -> list[WeightedSkill]:
    """Clamp weights to 1-3 and drop duplicates (keeping the highest weight)."""
    best: dict[str, WeightedSkill] = {}
    for skill in skills:
        name = skill.name.strip()
        if not name:
            continue
        cleaned = WeightedSkill(name=name, weight=min(3, max(1, int(skill.weight))), kind=skill.kind)
        key = textutil.canon(name)
        if key not in best or cleaned.weight > best[key].weight:
            best[key] = cleaned
    return list(best.values())


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for item in items:
        item = item.strip()
        if item:
            seen.setdefault(textutil.canon(item), item)
    return list(seen.values())


def _quotes(extract: JobAnalysisExtract) -> list[str]:
    req = extract.requirements
    quotes = [
        req.grad_year_range.source, req.class_standing.source, req.degree_fields.source,
        req.min_gpa.source, req.work_authorization.source, req.prior_experience_required.source,
        *[item.source for item in req.other],
    ]
    return [q for q in quotes if q and q.strip()]


def quote_in_text(quote: str, text: str) -> bool:
    """True when `quote` appears in `text`, ignoring spacing, case, and punctuation."""
    squashed = textutil.squash(quote)
    return bool(squashed) and squashed in textutil.squash(text)


def finalize(extract: JobAnalysisExtract, job: JobText, today: date | None = None) -> JobAnalysis:
    """Deterministic clean-up of Claude's extraction."""
    today = today or date.today()
    required = _clean_skills(extract.required_skills)
    required_keys = {textutil.canon(s.name) for s in required}
    preferred = [s for s in _clean_skills(extract.preferred_skills) if textutil.canon(s.name) not in required_keys]
    analysis = JobAnalysis(
        **extract.model_dump(exclude={"required_skills", "preferred_skills", "tools", "key_phrases",
                                      "responsibilities", "keywords", "company", "role", "location"}),
        company=extract.company.strip() or "Unknown company",
        role=extract.role.strip() or "Unknown role",
        location=extract.location.strip(),
        required_skills=required,
        preferred_skills=preferred,
        tools=_dedupe(extract.tools),
        key_phrases=_dedupe(extract.key_phrases),
        responsibilities=[r.strip() for r in extract.responsibilities if r.strip()],
        keywords=_dedupe(extract.keywords),
        url=job.url,
        analyzed_on=today.isoformat(),
        posting_words=job.words,
    )
    analysis.id = _posting_id(analysis.company, analysis.role, job.text, today)
    analysis.unverified_quotes = [q for q in _quotes(extract) if not quote_in_text(q, job.text)]
    return analysis


def analyze_job(llm, job: JobText, today: date | None = None) -> JobAnalysis:
    """Run the analysis call and return the cleaned `JobAnalysis`."""
    today = today or date.today()
    extract = llm.parse(
        purpose="analyze",
        model=config.FAST_MODEL,
        system=SYSTEM,
        user=f"Today's date: {today.isoformat()}\n\nJob posting:\n<posting>\n{job.text}\n</posting>",
        schema=JobAnalysisExtract,
        cache_system=False,
    )
    return finalize(extract, job, today)


# ---------------------------------------------------------------------------
# Persistence (read by insights.py)
# ---------------------------------------------------------------------------


def save_analysis(analysis: JobAnalysis, directory: Path | None = None) -> Path:
    directory = directory or config.POSTINGS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{analysis.id}.json"
    path.write_text(json.dumps(analysis.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def load_postings(directory: Path | None = None) -> list[JobAnalysis]:
    """Every saved analysis, oldest first. Unreadable files are skipped."""
    directory = directory or config.POSTINGS_DIR
    postings: list[JobAnalysis] = []
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        try:
            postings.append(JobAnalysis.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return postings
