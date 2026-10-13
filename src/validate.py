"""Deterministic, evidence-based anti-fabrication. No AI in this module.

Run on every draft and every revision. Each bullet is checked against the
facts it cites; nothing the AI wrote is trusted on its own say-so.

A. Structural checks
   A1  a cited fact_id does not exist
   A2  a source_bullet_id does not exist (or belongs to another entry)
   A3  a cited fact belongs to a different experience / project
   A4  the selected experience / project does not exist (or is duplicated)
   A5  the AI altered an employer, title, or date
   A6  an honor, coursework item, or award is not in the master resume
   Employer, title, dates, and location are ALWAYS copied from the master
   resume by code. An invalid selection (A4, A5) is removed and replaced with
   the next-most-relevant verified content.

B. Evidence checks per bullet
   B1  every number / % / $ / count must appear in a cited fact         (hard)
   B2  every named skill or tool must be supported by a cited fact and
       be `verified` in skills_status.yaml                              (hard)
       a posting term or new name the cited facts never mention         (soft)
   B3  ownership verbs above the cited facts' ownership level           (hard)
       ambiguous verbs, changed ownership wording, dropped "assisted"   (soft)
   B4  scope words that are not in the cited facts                      (soft)
   B5  the skills list must be a subset of my verified skills           (hard)

C. Approval triggers (soft): added technical-complexity wording (C1), added
   result wording (C2), a dropped metric (C3).

Outcomes: hard violations get ONE retry (the caller passes `retry_fn`, which
asks Claude to fix the specific violation). If the retry still fails, the
bullet reverts to an approved bullet or the plain fact text, or is dropped.
Soft flags send the bullet to approval.py. Everything is returned as
`Violation` records so report.py can log it in changes.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

import config
from src import textutil
from src.models import (
    Approval,
    Bullet,
    BulletDraft,
    BulletFix,
    BulletFixRequest,
    ChangeType,
    Entry,
    EntryDraft,
    Fact,
    Flag,
    JobAnalysis,
    MasterResume,
    Mode,
    SectionName,
    Severity,
    SkillKind,
    SkillsStatus,
    SkillStatus,
    TailoredBullet,
    TailoredEntry,
    TailoredResume,
    TailoredResumeDraft,
    Violation,
)
from src.relevance import Relevance

RetryFn = Callable[[list[BulletFixRequest]], list[BulletFix]]

_CHANGE_RANK = {ChangeType.unchanged: 0, ChangeType.minor_wording: 1, ChangeType.reworded: 2, ChangeType.combined: 3}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ValidationContext:
    """Everything the validator needs to know for one run."""

    master: MasterResume
    status: SkillsStatus
    analysis: JobAnalysis | None = None
    mode: Mode = Mode.normal
    new_fact_ids: frozenset[str] = frozenset()      # facts added by the interview this run

    def __post_init__(self) -> None:
        self.facts = self.master.fact_index()
        self.bullets = self.master.bullet_index()
        self.relevance = Relevance(self.analysis)
        # "Named" skills: tools and skills I track. Unsupported use is a HARD violation.
        named = [
            *config.TECH_LEXICON, *self.status.skills, *self.master.skills.all(),
            *[s for e in self.master.entries() for s in e.skills],
            *[s for e in self.master.entries() for f in e.facts for s in f.skills],
        ]
        self.named_terms = _dedupe_terms(named)
        named_keys = {textutil.canon(t) for t in self.named_terms}
        # Posting-only terms may be ordinary words ("testing"), so they are SOFT.
        posting: list[str] = []
        if self.analysis is not None:
            posting = [*self.analysis.tools,
                       *[s.name for s in self.analysis.all_skills() if s.kind != SkillKind.soft]]
        self.posting_terms = [t for t in _dedupe_terms(posting) if textutil.canon(t) not in named_keys]


def _dedupe_terms(terms: Iterable[str]) -> list[str]:
    seen: dict[str, str] = {}
    for term in terms:
        term = (term or "").strip()
        if term:
            seen.setdefault(textutil.canon(term), term)
    return list(seen.values())


@dataclass
class Evidence:
    """The verified material a bullet is allowed to draw on."""

    entry: Entry
    facts: list[Fact]
    source_bullet: Bullet | None
    texts: list[str]                 # fact texts, metric strings, approved source wording
    skill_tags: set[str]             # canonical skills tagged on the cited facts
    max_tier: int
    min_tier: int

    @property
    def joined(self) -> str:
        return " | ".join(self.texts)

    @property
    def header(self) -> list[str]:
        return [self.entry.display_org, self.entry.display_title, self.entry.location]


def gather_evidence(entry: Entry, facts: list[Fact], source_bullet: Bullet | None) -> Evidence:
    texts = [f.text for f in facts] + [m for f in facts for m in f.metrics]
    if source_bullet is not None and source_bullet.approved:
        texts.append(source_bullet.text)        # wording I already signed off
    tiers = [config.OWNERSHIP_TIERS[MasterResume.effective_ownership(entry, f).value] for f in facts]
    return Evidence(
        entry=entry, facts=facts, source_bullet=source_bullet, texts=texts,
        skill_tags={textutil.canon(s) for f in facts for s in f.skills},
        max_tier=max(tiers, default=config.OWNERSHIP_TIERS[entry.ownership.value]),
        min_tier=min(tiers, default=config.OWNERSHIP_TIERS[entry.ownership.value]),
    )


# ---------------------------------------------------------------------------
# B / C: evidence checks for one bullet
# ---------------------------------------------------------------------------

_STEM_SUFFIXES = ("ization", "isation", "ations", "ation", "ments", "ment", "ings", "ing",
                  "ies", "ied", "ers", "er", "ed", "es", "s", "e")


def _stem(word: str) -> str:
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _stems_match(a: str, b: str) -> bool:
    if a == b:
        return True
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= 5 and common >= 0.7 * min(len(a), len(b))


def loosely_supported(term: str, evidence_text: str) -> bool:
    """True when every content word of `term` has a same-stem word in the evidence."""
    wanted = [_stem(w) for w in textutil.content_words(term)]
    have = [_stem(w) for w in textutil.words(evidence_text)]
    return bool(wanted) and all(any(_stems_match(w, h) for h in have) for w in wanted)


def is_verbatim(text: str, ev: Evidence) -> bool:
    """The bullet is wording I already verified, so it needs no evidence check."""
    if ev.source_bullet is not None and ev.source_bullet.approved and textutil.same_text(text, ev.source_bullet.text):
        return True
    if any(b.approved and textutil.same_text(text, b.text) for b in ev.entry.bullets):
        return True
    if any(textutil.same_text(text, f.text) for f in ev.facts):
        return True
    return len(ev.facts) > 1 and textutil.same_text(text, _join_facts(ev.facts))


def _join_facts(facts: list[Fact]) -> str:
    return "; ".join(f.text.rstrip(". ") for f in facts)


def check_bullet(text: str, ev: Evidence, ctx: ValidationContext,
                 entry_id: str = "", index: int | None = None) -> list[Violation]:
    """All B and C checks for one bullet. Pure: returns violations, changes nothing."""
    if is_verbatim(text, ev):
        return []
    out: list[Violation] = []

    def add(code: str, severity: Severity, message: str, token: str = "") -> None:
        out.append(Violation(code=code, severity=severity, entry_id=entry_id or ev.entry.id,
                             bullet_index=index, message=message, token=token))

    evidence_text = ev.joined
    years = textutil.years_in(ev.entry.dates.start, ev.entry.dates.end)

    # B1 - metrics: every number must come from a cited fact (or the entry's own dates).
    for atom in textutil.unsupported_atoms(text, [*ev.texts, *years]):
        claim = f'"{atom.raw}{"+" if atom.more and "+" not in atom.raw else ""}"'
        add("B1", Severity.hard, f"states {claim}, which is not in the cited facts", atom.raw)

    # B2 - named skills and tools.
    judged_surfaces: list[str] = []
    for canonical, surface in textutil.find_terms(text, ctx.named_terms).items():
        judged_surfaces.append(surface)
        status = ctx.status.status_of(canonical)
        in_tags = canonical in ev.skill_tags
        in_text = textutil.contains_term(evidence_text, canonical)
        if status in (SkillStatus.learning, SkillStatus.missing):
            add("B2", Severity.hard, f'names "{surface}", which is marked {status.value} in skills_status.yaml', surface)
        elif not in_tags and not in_text:
            add("B2", Severity.hard, f'names "{surface}", which no cited fact supports', surface)
        elif in_tags and not in_text and status is None:
            add("B2", Severity.hard,
                f'names "{surface}", which is tagged on the cited fact but is not verified in skills_status.yaml', surface)
    for canonical, surface in textutil.find_terms(text, ctx.posting_terms).items():
        judged_surfaces.append(surface)
        supported = (canonical in ev.skill_tags or textutil.contains_term(evidence_text, canonical)
                     or loosely_supported(canonical, evidence_text))
        if not supported:
            add("B2", Severity.soft, f'uses the posting\'s term "{surface}", which the cited facts do not mention', surface)
    judged = textutil.squash(" ".join(judged_surfaces))
    for token in textutil.new_proper_nouns(text, [*ev.texts, *ev.header]):
        if textutil.squash(token) not in judged:
            add("B2", Severity.soft, f'introduces the name "{token}", which is not in the cited facts', token)

    # B3 - ownership.
    evidence_verbs = textutil.ownership_terms(evidence_text)
    bullet_verbs = textutil.ownership_terms(text)
    for lemma, tier in bullet_verbs.items():
        word = lemma.removeprefix("noun:")
        if lemma not in evidence_verbs:
            if tier > ev.max_tier:
                severity = Severity.soft if textutil.is_ambiguous_ownership(lemma) else Severity.hard
                add("B3", severity,
                    f'uses "{word}" ({textutil.tier_name(tier)} level) but the cited facts support at most '
                    f"{textutil.tier_name(ev.max_tier)}", word)
            else:
                add("B3", Severity.soft, f'changes the ownership wording (adds "{word}")', word)
        elif tier > ev.min_tier and ev.max_tier > ev.min_tier:
            add("B3", Severity.soft,
                f'combines facts with different ownership levels; check that "{word}" only describes the '
                f"{textutil.tier_name(tier)}-level part", word)
    dropped = textutil.downgrade_qualifiers(evidence_text)
    if dropped and not textutil.downgrade_qualifiers(text):
        add("B3", Severity.soft, f'drops the helper wording ("{", ".join(sorted(dropped))}") from the cited facts',
            sorted(dropped)[0])

    # B4 - scope inflation.
    for word in sorted(textutil.phrases_in(text, config.SCOPE_WORDS) - textutil.phrases_in(evidence_text, config.SCOPE_WORDS)):
        add("B4", Severity.soft, f'adds the scope word "{word}", which is not in the cited facts', word)

    # C1 - technical complexity.
    def singular(found: set[str]) -> set[str]:
        return {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in found}

    for word in sorted(singular(textutil.phrases_in(text, config.COMPLEXITY_WORDS))
                       - singular(textutil.phrases_in(evidence_text, config.COMPLEXITY_WORDS))):
        add("C1", Severity.soft, f'adds technical-complexity wording ("{word}")', word)

    # C2 - results wording.
    new_results = (textutil.stems_in(text, config.RESULT_WORDS) - textutil.stems_in(evidence_text, config.RESULT_WORDS)) | (
        textutil.phrases_in(text, config.RESULT_PHRASES) - textutil.phrases_in(evidence_text, config.RESULT_PHRASES))
    for word in sorted(new_results):
        add("C2", Severity.soft, f'adds result wording ("{word}")', word)

    # C3 - a metric the cited fact carries was dropped.
    present = {a.key() for a in textutil.numeric_atoms(text)}
    for fact in ev.facts:
        for metric in fact.metrics:
            atoms = textutil.numeric_atoms(metric)
            if atoms and not any(a.key() in present for a in atoms):
                add("C3", Severity.soft, f'drops the metric "{metric}" from {fact.id}', metric)
    return out


# ---------------------------------------------------------------------------
# A: structure
# ---------------------------------------------------------------------------

_MONTH_LOOKUP = {m.lower(): i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _norm_date(value: str) -> str:
    """Normalize an echoed date so "Oct 2024", "October 2024" and "2024-10" compare equal."""
    text = (value or "").strip().lower()
    if text in ("present", "current", "now", "ongoing"):
        return "present"
    m = re.match(r"^([a-z]{3})[a-z]*\.?\s+(\d{4})$", text)
    if m and m.group(1) in _MONTH_LOOKUP:
        return f"{m.group(2)}-{_MONTH_LOOKUP[m.group(1)]:02d}"
    return text


def header_mismatch(draft: EntryDraft, entry: Entry) -> str | None:
    """Which echoed header field differs from the master resume, if any.

    An empty echo asserts nothing and is ignored; a non-empty value that differs
    means the AI altered an employer, title, or date.
    """
    pairs = [
        ("employer", draft.org, entry.display_org, textutil.squash),
        ("title", draft.title, entry.display_title, textutil.squash),
        ("start date", draft.start, entry.dates.start, _norm_date),
        ("end date", draft.end, entry.dates.end, _norm_date),
    ]
    for label, echoed, real, norm in pairs:
        if (echoed or "").strip() and norm(echoed) != norm(real):
            return f'{label}: AI wrote "{echoed.strip()}", master resume says "{real}"'
    return None


def _verified_bullet(bullet: Bullet | None, fact: Fact | None, note: str) -> TailoredBullet:
    """A bullet made of verified wording only (an approved bullet or a plain fact)."""
    if bullet is not None:
        text, fact_ids, source, kind = bullet.text, list(bullet.fact_ids), bullet.id, "approved_bullet"
    else:
        assert fact is not None
        text, fact_ids, source, kind = fact.text, [fact.id], None, "fact"
    return TailoredBullet(
        rewritten_text=text, fact_ids=fact_ids, source_bullet_id=source, change_type=ChangeType.unchanged,
        rationale=note, original_text=text, original_kind=kind, approval=Approval.auto,
        final_text=text, history=[note],
    )


def verified_entry(entry: Entry, ctx: ValidationContext, note: str, limit: int | None = None) -> TailoredEntry:
    """An entry rendered purely from verified wording, most relevant bullets first."""
    limit = limit or config.REPLACEMENT_MAX_BULLETS
    approved = sorted((b for b in entry.bullets if b.approved), key=lambda b: -ctx.relevance.score_text(b.text))
    bullets = [_verified_bullet(b, None, note) for b in approved[:limit]]
    if not bullets:
        bullets = [_verified_bullet(None, f, note) for f in ctx.relevance.rank_facts(entry.facts)[:limit]]
    return TailoredEntry(source_id=entry.id, kind=entry.kind, bullets=bullets, replaced=note)


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


@dataclass
class ValidationOutcome:
    resume: TailoredResume
    violations: list[Violation] = field(default_factory=list)
    retried: int = 0

    def hard(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == Severity.hard]

    def soft(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == Severity.soft]


@dataclass
class _Work:
    """One bullet moving through enforcement."""

    entry: Entry
    index: int
    text: str
    ai_change: ChangeType
    rationale: str
    facts: list[Fact]
    source: Bullet | None
    structural: list[Violation]          # A1-A3 found while resolving ids
    evidence: list[Violation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reverted: TailoredBullet | None = None
    dropped: bool = False

    @property
    def key(self) -> str:
        return f"{self.entry.id}#{self.index}"

    def hard(self) -> list[Violation]:
        return [v for v in self.evidence if v.severity == Severity.hard]


def _resolve_ids(entry: Entry, fact_ids: list[str], source_bullet_id: str | None, index: int,
                 ctx: ValidationContext) -> tuple[list[Fact], Bullet | None, list[Violation]]:
    """Keep only fact ids / source bullet ids that exist and belong to this entry."""
    found: list[Violation] = []
    facts: list[Fact] = []
    for fact_id in dict.fromkeys(fact_ids or []):
        hit = ctx.facts.get(fact_id)
        if hit is None:
            found.append(Violation(code="A1", severity=Severity.hard, entry_id=entry.id, bullet_index=index,
                                   message=f'cites fact "{fact_id}", which does not exist', token=fact_id,
                                   resolution="id removed"))
        elif hit[0].id != entry.id:
            found.append(Violation(code="A3", severity=Severity.hard, entry_id=entry.id, bullet_index=index,
                                   message=f'cites fact "{fact_id}", which belongs to {hit[0].id}, not this entry',
                                   token=fact_id, resolution="id removed"))
        else:
            facts.append(hit[1])
    source: Bullet | None = None
    if source_bullet_id:
        hit_b = ctx.bullets.get(source_bullet_id)
        if hit_b is None or hit_b[0].id != entry.id:
            found.append(Violation(code="A2", severity=Severity.hard, entry_id=entry.id, bullet_index=index,
                                   message=f'names source bullet "{source_bullet_id}", which is not a bullet of this entry',
                                   token=source_bullet_id, resolution="id cleared"))
        else:
            source = hit_b[1]
    return facts, source, found


def original_for(facts: list[Fact], source: Bullet | None) -> tuple[str, str]:
    """(verified source wording, kind) that a rewrite is compared against."""
    if source is not None:
        return source.text, ("approved_bullet" if source.approved else "draft_bullet")
    if len(facts) == 1:
        return facts[0].text, "fact"
    return _join_facts(facts), "facts"


def classify_change(text: str, original: str, facts: list[Fact], source: Bullet | None,
                    ai_label: ChangeType) -> ChangeType:
    """The change type as measured by code; the AI's own label can only raise it."""
    if textutil.same_text(text, original):
        return ChangeType.unchanged
    covered_by_source = source is not None and {f.id for f in facts} <= set(source.fact_ids)
    if len(facts) > 1 and not covered_by_source:
        measured = ChangeType.combined
    elif textutil.similarity(text, original) >= config.MINOR_WORDING_SIMILARITY:
        measured = ChangeType.minor_wording
    else:
        measured = ChangeType.reworded
    return measured if _CHANGE_RANK[measured] >= _CHANGE_RANK[ai_label] else ai_label


def build_bullet(text: str, facts: list[Fact], source: Bullet | None, ai_change: ChangeType, rationale: str,
                 soft: list[Violation], ctx: ValidationContext, notes: list[str] | None = None) -> TailoredBullet:
    """Assemble the final bullet record: change type, approval flags, rendered text."""
    original, kind = original_for(facts, source)
    change = classify_change(text, original, facts, source, ai_change)
    flags = [Flag(code=v.code, message=v.message) for v in soft]
    if change in (ChangeType.reworded, ChangeType.combined):
        flags.append(Flag(code="change", message=f"{change.value} from the verified source wording"))
    if ctx.mode == Mode.strict and change != ChangeType.unchanged:
        flags.append(Flag(code="strict", message="strict mode: every rewrite needs my approval"))
    if any(f.id in ctx.new_fact_ids for f in facts):
        flags.append(Flag(code="interview", message="built from a fact added in this run's interview"))
    # Unchanged, already-verified wording needs no approval. An unapproved draft
    # bullet from the master file is not "verified", so its flags still count.
    trusted_unchanged = change == ChangeType.unchanged and kind != "draft_bullet"
    pending = bool(flags) and not trusted_unchanged
    # In strict mode the verified wording stays on the resume until I approve the rewrite.
    final = original if (ctx.mode == Mode.strict and pending) else text
    return TailoredBullet(
        rewritten_text=text, fact_ids=[f.id for f in facts], source_bullet_id=source.id if source else None,
        change_type=change, rationale=rationale, original_text=original, original_kind=kind,
        flags=flags if pending else [], approval=Approval.pending if pending else Approval.auto,
        final_text=final, history=list(notes or []),
    )


def _fallback(work: _Work, used_texts: list[str], ctx: ValidationContext) -> TailoredBullet | None:
    """Revert to an approved bullet, else the plain fact text, else drop (None)."""
    candidates: list[tuple[Bullet | None, Fact | None]] = []
    if work.source is not None and work.source.approved:
        candidates.append((work.source, None))
    cited = {f.id for f in work.facts}
    for bullet in work.entry.bullets:
        if bullet.approved and cited & set(bullet.fact_ids) and bullet is not work.source:
            candidates.append((bullet, None))
    candidates.extend((None, fact) for fact in work.facts)
    for bullet, fact in candidates:
        text = bullet.text if bullet is not None else fact.text  # type: ignore[union-attr]
        if not any(textutil.same_text(text, used) for used in used_texts):
            kind = "an approved bullet" if bullet is not None else "the plain fact text"
            return _verified_bullet(bullet, fact, f"reverted to {kind} after a failed validation retry")
    return None


def _select_entries(drafts: list[EntryDraft], kind: str, ctx: ValidationContext, violations: list[Violation],
                    taken: set[str]) -> list[tuple[Entry, EntryDraft | None, str | None]]:
    """Resolve the AI's selections; replace invalid ones with verified content."""
    resolved: list[tuple[Entry | None, EntryDraft | None, str | None]] = []
    for draft in drafts:
        entry = ctx.master.entry(draft.source_id)
        if entry is None or entry.kind != kind:
            violations.append(Violation(
                code="A4", severity=Severity.hard, entry_id=draft.source_id,
                message=f'selected {kind} "{draft.source_id}" ("{draft.org}"), which does not exist in the master resume',
                token=draft.source_id, resolution="removed; replaced with verified content"))
            resolved.append((None, None, f'replaced the invalid selection "{draft.source_id}"'))
        elif entry.id in taken:
            violations.append(Violation(
                code="A4", severity=Severity.hard, entry_id=entry.id,
                message=f"selected {entry.id} more than once", token=entry.id, resolution="duplicate removed"))
        else:
            mismatch = header_mismatch(draft, entry)
            if mismatch:
                violations.append(Violation(
                    code="A5", severity=Severity.hard, entry_id=entry.id,
                    message=f"altered the {mismatch}", token=mismatch.split(":")[0],
                    resolution="selection removed; replaced with verified content"))
                resolved.append((None, None, f"replaced an altered selection of {entry.id}"))
            else:
                taken.add(entry.id)
                resolved.append((entry, draft, None))
    pool = list(ctx.master.experience if kind == "experience" else ctx.master.projects)
    out: list[tuple[Entry, EntryDraft | None, str | None]] = []
    for entry, draft, note in resolved:
        if entry is not None:
            out.append((entry, draft, None))
            continue
        candidates = ctx.relevance.rank_entries([e for e in pool if e.id not in taken and (e.facts or e.bullets)])
        if candidates:
            taken.add(candidates[0].id)
            out.append((candidates[0], None, note))
    return out


def enforce(draft: TailoredResumeDraft, ctx: ValidationContext, retry_fn: RetryFn | None = None) -> ValidationOutcome:
    """Validate a draft and return a resume in which every bullet is backed by evidence."""
    violations: list[Violation] = []
    taken: set[str] = set()
    selections = {
        "experience": _select_entries(draft.experiences, "experience", ctx, violations, taken),
        "project": _select_entries(draft.projects, "project", ctx, violations, taken),
    }

    # Pass 1: resolve ids and run the evidence checks.
    works: dict[str, list[_Work]] = {}
    for kind in ("experience", "project"):
        for entry, entry_draft, _note in selections[kind]:
            if entry_draft is None:
                continue
            for index, bullet in enumerate(entry_draft.bullets):
                text = textutil.plain(bullet.rewritten_text)
                if not text:
                    continue
                facts, source, structural = _resolve_ids(entry, bullet.fact_ids, bullet.source_bullet_id, index, ctx)
                work = _Work(entry, index, text, bullet.change_type, bullet.rationale, facts, source, structural)
                if not facts:
                    work.evidence = [Violation(code="A1", severity=Severity.hard, entry_id=entry.id, bullet_index=index,
                                               message="cites no valid fact, so nothing supports it")]
                else:
                    work.evidence = check_bullet(text, gather_evidence(entry, facts, source), ctx, entry.id, index)
                works.setdefault(entry.id, []).append(work)

    # Pass 2: one retry for bullets with hard violations.
    failing = [w for group in works.values() for w in group if w.hard()]
    retried = 0
    if failing and retry_fn is not None:
        requests = [BulletFixRequest(key=w.key, entry_id=w.entry.id, text=w.text,
                                     fact_ids=[f.id for f in w.facts], violations=[v.message for v in w.hard()])
                    for w in failing]
        fixes = {fix.key: fix for fix in retry_fn(requests)}
        retried = len(requests)
        for work in failing:
            fix = fixes.get(work.key)
            first_try = work.hard()
            for v in first_try:
                v.resolution = "sent back to Claude once"
            violations.extend(work.structural + first_try)
            work.structural = []
            if fix is None or not textutil.plain(fix.rewritten_text):
                continue                                    # still failing -> fallback below
            facts, _source, structural = _resolve_ids(work.entry, fix.fact_ids, None, work.index, ctx)
            if not facts:
                violations.extend(structural)
                continue
            new_text = textutil.plain(fix.rewritten_text)
            evidence = check_bullet(new_text, gather_evidence(work.entry, facts, work.source), ctx,
                                    work.entry.id, work.index)
            work.structural = structural
            if not any(v.severity == Severity.hard for v in evidence):
                work.text, work.facts, work.evidence = new_text, facts, evidence
                work.notes.append("rewritten once after a validation failure")
                for v in first_try:
                    v.resolution = "fixed by the one allowed retry"
            else:
                work.evidence = evidence
                for v in first_try:
                    v.resolution = "sent back to Claude once; the retry still failed"

    # Pass 3: assemble entries; fall back or drop what still fails.
    resume = TailoredResume(mode=ctx.mode)
    for kind in ("experience", "project"):
        target = resume.experiences if kind == "experience" else resume.projects
        for entry, entry_draft, note in selections[kind]:
            if entry_draft is None:
                target.append(verified_entry(entry, ctx, note or "verified content"))
                continue
            built: list[TailoredBullet] = []
            for work in works.get(entry.id, []):
                violations.extend(work.structural)
                if work.hard():
                    for v in work.hard():
                        v.resolution = v.resolution or "no retry available"
                    replacement = _fallback(work, [b.text for b in built], ctx)
                    outcome = "reverted to verified wording" if replacement else "bullet dropped"
                    for v in work.hard():
                        if v not in violations:
                            violations.append(v)
                        v.resolution = f"{v.resolution}; {outcome}" if v.resolution else outcome
                    if replacement is not None:
                        built.append(replacement)
                    continue
                if any(textutil.same_text(work.text, b.text) for b in built):
                    continue                                # exact duplicate within the entry
                soft = [v for v in work.evidence if v.severity == Severity.soft]
                for v in soft:
                    v.resolution = "sent to approval"
                bullet = build_bullet(work.text, work.facts, work.source, work.ai_change, work.rationale,
                                      soft, ctx, work.notes)
                if bullet.approval == Approval.pending:
                    violations.extend(soft)
                built.append(bullet)
            if not built:
                # Every bullet failed: keep the (real) experience, with verified wording.
                filled = verified_entry(entry, ctx, "all AI bullets failed validation; verified wording used", limit=1)
                built = filled.bullets
            target.append(TailoredEntry(source_id=entry.id, kind=kind, bullets=built))

    resume.skills = _check_skills(draft.skills, ctx, violations)
    resume.honors = _check_listed(draft.honors, [h for edu in ctx.master.education for h in edu.honors], "honor", violations)
    resume.coursework = _check_listed(draft.coursework, [c for edu in ctx.master.education for c in edu.coursework],
                                      "coursework item", violations)
    resume.award_ids = []
    for award_id in dict.fromkeys(draft.award_ids):
        if ctx.master.award(award_id) is None:
            violations.append(Violation(code="A6", severity=Severity.hard, message=f'award "{award_id}" does not exist',
                                        token=award_id, resolution="removed"))
        else:
            resume.award_ids.append(award_id)
    resume.section_order = _section_order(draft.section_order, resume, ctx)
    return ValidationOutcome(resume=resume, violations=violations, retried=retried)


def _check_skills(skills: list[str], ctx: ValidationContext, violations: list[Violation]) -> list[str]:
    """B5: the skills section may only list master-list skills that are verified."""
    kept: dict[str, str] = {}
    for skill in skills:
        display = ctx.master.skills.display_name(skill)
        if display is None:
            violations.append(Violation(code="B5", severity=Severity.hard,
                                        message=f'skill "{skill}" is not in the master skills list',
                                        token=skill, resolution="removed from the skills section"))
        elif not ctx.status.is_verified(skill):
            violations.append(Violation(code="B5", severity=Severity.hard,
                                        message=f'skill "{skill}" is not verified in skills_status.yaml',
                                        token=skill, resolution="removed from the skills section"))
        else:
            kept.setdefault(textutil.canon(display), display)
    return list(kept.values())


def _check_listed(chosen: list[str], allowed: list[str], label: str, violations: list[Violation]) -> list[str]:
    kept: list[str] = []
    for item in chosen:
        match = next((a for a in allowed if textutil.same_text(a, item)), None)
        if match is None:
            violations.append(Violation(code="A6", severity=Severity.hard,
                                        message=f'{label} "{item}" is not in the master resume',
                                        token=item, resolution="removed"))
        elif match not in kept:
            kept.append(match)
    return kept


def _section_order(requested: list[SectionName], resume: TailoredResume, ctx: ValidationContext) -> list[SectionName]:
    """Keep the AI's order, but make sure every section with content is present exactly once."""
    has_content = {
        SectionName.education: bool(ctx.master.education),
        SectionName.experience: any(_section_of(e, ctx) == "experience" for e in resume.experiences),
        SectionName.leadership: any(_section_of(e, ctx) == "leadership" for e in resume.experiences),
        SectionName.projects: bool(resume.projects),
        SectionName.skills: bool(resume.skills),
        SectionName.awards: bool(resume.award_ids),
        SectionName.certifications: bool(ctx.master.certifications),
    }
    order = [s for s in dict.fromkeys(requested) if has_content.get(s)]
    for name in config.DEFAULT_SECTION_ORDER:
        section = SectionName(name)
        if has_content.get(section) and section not in order:
            order.append(section)
    return order


def _section_of(entry: TailoredEntry, ctx: ValidationContext) -> str:
    source = ctx.master.entry(entry.source_id)
    return getattr(source, "section_name", "experience")


# ---------------------------------------------------------------------------
# Re-checking a single edited bullet (approval edits, page-fit tightening)
# ---------------------------------------------------------------------------


def recheck(text: str, entry_id: str, fact_ids: list[str], source_bullet_id: str | None,
            ctx: ValidationContext) -> list[Violation]:
    """Validate one stand-alone bullet text against the facts it cites."""
    entry = ctx.master.entry(entry_id)
    if entry is None:
        return [Violation(code="A4", severity=Severity.hard, entry_id=entry_id, message="unknown entry")]
    facts, source, structural = _resolve_ids(entry, fact_ids, source_bullet_id, 0, ctx)
    if not facts:
        return structural + [Violation(code="A1", severity=Severity.hard, entry_id=entry_id,
                                       message="cites no valid fact, so nothing supports it")]
    return structural + check_bullet(textutil.plain(text), gather_evidence(entry, facts, source), ctx, entry_id, None)
