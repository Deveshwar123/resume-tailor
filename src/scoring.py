"""All score totals, computed in Python. Claude never produces a total.

Three separate scores:

1. Resume Quality (0-10, the only score with a target): the weighted average of
   the judge's four category scores plus the ATS score from ats_check.py.
2. Job Match (0-100, informational): how well my verified skills and my
   eligibility cover the job's weighted requirements. It measures *me*, not the
   writing, so it does not change when the resume is rewritten.
3. Keyword Coverage (0-100, informational): the weighted share of the job's
   keywords present in a text, synonym-aware (ML = machine learning). Reported
   for the whole master resume and for the final resume.
"""

from __future__ import annotations

import config
from src import textutil
from src.eligibility import assessable
from src.models import (
    AtsReport,
    EligibilityReport,
    EligStatus,
    JobAnalysis,
    JobMatch,
    Judgment,
    KeywordCoverage,
    MasterResume,
    QualityScore,
    SkillsStatus,
    SkillStatus,
    WeightedSkill,
)

# ---------------------------------------------------------------------------
# 1. Resume Quality
# ---------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def quality(judgment: Judgment, ats: AtsReport, weights: dict[str, float] | None = None) -> QualityScore:
    """Weighted average of the judged categories and the code-computed ATS score."""
    weights = dict(weights or config.QUALITY_WEIGHTS)
    scores = judgment.category_scores
    categories = {name: clamp(float(getattr(scores, name).score), 1.0, 10.0) for name in config.JUDGED_CATEGORIES}
    categories["ats"] = clamp(float(ats.score), 0.0, 10.0)
    total_weight = sum(weights[name] for name in categories)
    total = sum(weights[name] * categories[name] for name in categories) / total_weight
    return QualityScore(total=round(total, 2), categories=categories, weights=weights, ats_final=ats.final)


# ---------------------------------------------------------------------------
# 2. Job Match
# ---------------------------------------------------------------------------

_ELIGIBILITY_CREDIT = {
    EligStatus.meets: 1.0,
    EligStatus.partial: 0.5,
    EligStatus.unknown: config.UNKNOWN_ELIGIBILITY_CREDIT,
    EligStatus.fails: 0.0,
}


def _skill_component(skills: list[WeightedSkill], status: SkillsStatus,
                     covered: list[str], partial: list[str], missing: list[str]) -> float | None:
    """Weighted share of skills I have (verified = 1, learning = partial credit)."""
    total = earned = 0.0
    for skill in skills:
        if not assessable(skill.name, skill.kind, status):
            continue
        total += skill.weight
        mine = status.status_of(skill.name)
        if mine == SkillStatus.verified:
            earned += skill.weight
            covered.append(skill.name)
        elif mine == SkillStatus.learning:
            earned += skill.weight * config.LEARNING_SKILL_CREDIT
            partial.append(skill.name)
        else:
            missing.append(skill.name)
    return earned / total if total else None


def job_match(analysis: JobAnalysis, status: SkillsStatus, eligibility: EligibilityReport | None = None) -> JobMatch:
    """How well I cover this job. Independent of how the resume is written."""
    covered: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    components: dict[str, float] = {}
    required = _skill_component(analysis.required_skills, status, covered, partial, missing)
    if required is not None:
        components["required_skills"] = required
    preferred = _skill_component(analysis.preferred_skills, status, covered, partial, missing)
    if preferred is not None:
        components["preferred_skills"] = preferred
    if eligibility is not None:
        # The skills row is already counted above; "Other" rows are never auto-checked.
        rows = [r for r in eligibility.rows
                if r.status != EligStatus.not_stated and r.requirement != "Required skills"
                and not r.requirement.startswith("Other:")]
        if rows:
            components["eligibility"] = sum(_ELIGIBILITY_CREDIT[r.status] for r in rows) / len(rows)
    weight = sum(config.JOB_MATCH_WEIGHTS[name] for name in components)
    score = 100.0 * sum(config.JOB_MATCH_WEIGHTS[name] * value for name, value in components.items()) / weight if weight else 0.0
    return JobMatch(score=round(score, 1), components={k: round(v, 3) for k, v in components.items()},
                    covered=covered, partial=partial, missing=missing)


# ---------------------------------------------------------------------------
# 3. Keyword Coverage
# ---------------------------------------------------------------------------


def keyword_weights(analysis: JobAnalysis) -> dict[str, tuple[str, float]]:
    """{canonical keyword: (display, weight)}; a term keeps its highest weight."""
    table: dict[str, tuple[str, float]] = {}

    def add(term: str, weight: float) -> None:
        term = term.strip()
        key = textutil.canon(term)
        if key and weight > table.get(key, ("", 0.0))[1]:
            table[key] = (table.get(key, (term, 0.0))[0] or term, weight)

    for skill in analysis.required_skills:
        add(skill.name, float(skill.weight))
    for skill in analysis.preferred_skills:
        add(skill.name, skill.weight * config.PREFERRED_KEYWORD_FACTOR)
    for term in [*analysis.tools, *analysis.keywords]:
        add(term, config.PLAIN_KEYWORD_WEIGHT)
    return table


def keyword_coverage(analysis: JobAnalysis, text: str) -> KeywordCoverage:
    """Weighted percentage of the job's keywords that appear in `text`."""
    table = keyword_weights(analysis)
    matched: list[str] = []
    missing: list[str] = []
    earned = total = 0.0
    for _key, (display, weight) in table.items():
        total += weight
        if textutil.contains_term(text, display):
            earned += weight
            matched.append(display)
        else:
            missing.append(display)
    return KeywordCoverage(score=round(100.0 * earned / total, 1) if total else 0.0, matched=matched, missing=missing)


def master_text(master: MasterResume) -> str:
    """Everything in the master resume, as text (the baseline for Keyword Coverage)."""
    parts: list[str] = []
    for edu in master.education:
        parts += [edu.school, edu.degree, edu.major, *edu.honors, *edu.coursework, *edu.notes]
    for entry in master.entries():
        parts += [entry.display_org, entry.display_title, *entry.skills]
        parts += [fact.text for fact in entry.facts]
        parts += [bullet.text for bullet in entry.bullets]
        parts += [skill for fact in entry.facts for skill in fact.skills]
    parts += master.skills.all() + master.certifications + [a.text for a in master.awards]
    return "\n".join(p for p in parts if p and p.strip())
