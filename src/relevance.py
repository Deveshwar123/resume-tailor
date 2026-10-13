"""Deterministic relevance scoring of master-resume content against a job.

Used wherever code (not Claude) has to rank verified content:

* validate.py - picking the "next-most-relevant verified content" when an AI
  selection is invalid
* render.py - choosing the lowest-relevance bullet / project to remove during
  page fitting
* tailor_resume.py - ordering the fact bank shown to Claude

The score is a weighted count of job terms (skills, tools, keywords) that a
piece of text mentions, using the same synonym-aware matcher as everything else.
"""

from __future__ import annotations

from functools import lru_cache

from src import textutil
from src.models import Entry, Fact, JobAnalysis

# How much each kind of job term counts.
REQUIRED_FACTOR = 2.0
PREFERRED_FACTOR = 1.0
TOOL_WEIGHT = 1.5
KEYWORD_WEIGHT = 1.0
JOB_TYPE_TAG_BONUS = 2.0


class Relevance:
    """Scores text, facts, and entries for one job. With no job, everything scores 0."""

    def __init__(self, analysis: JobAnalysis | None = None) -> None:
        self.analysis = analysis
        self.terms: dict[str, tuple[str, float]] = {}     # canonical -> (display, weight)
        if analysis is None:
            return
        for skill in analysis.required_skills:
            self._add(skill.name, skill.weight * REQUIRED_FACTOR)
        for skill in analysis.preferred_skills:
            self._add(skill.name, skill.weight * PREFERRED_FACTOR)
        for tool in analysis.tools:
            self._add(tool, TOOL_WEIGHT)
        for keyword in analysis.keywords:
            self._add(keyword, KEYWORD_WEIGHT)

    def _add(self, term: str, weight: float) -> None:
        key = textutil.canon(term)
        if key and weight > self.terms.get(key, ("", 0.0))[1]:
            self.terms[key] = (term.strip(), weight)

    @lru_cache(maxsize=4096)
    def score_text(self, text: str) -> float:
        if not self.terms or not text:
            return 0.0
        found = textutil.find_terms(text, [display for display, _ in self.terms.values()])
        return sum(self.terms[key][1] for key in found if key in self.terms)

    def score_fact(self, fact: Fact) -> float:
        return self.score_text(fact.text + " | " + ", ".join(fact.skills))

    def score_entry(self, entry: Entry) -> float:
        text = " | ".join([
            entry.display_title, entry.display_org,
            *[fact.text for fact in entry.facts],
            ", ".join(entry.skills), ", ".join(entry.tags),
        ])
        score = self.score_text(text)
        if self.analysis is not None and self.analysis.job_type.value in {t.lower() for t in entry.tags}:
            score += JOB_TYPE_TAG_BONUS
        # Recency breaks ties: a current or recent role edges out an old one.
        year = textutil.parse_ym(entry.dates.end, end=True)[0] or textutil.parse_ym(entry.dates.start)[0]
        recency = min(year, 2100) / 100000.0 if year else 0.0
        return score + recency

    def rank_entries(self, entries: list[Entry]) -> list[Entry]:
        """Most relevant first; stable for equal scores."""
        return sorted(entries, key=lambda e: -self.score_entry(e))

    def rank_facts(self, facts: list[Fact]) -> list[Fact]:
        return sorted(facts, key=lambda f: -self.score_fact(f))
