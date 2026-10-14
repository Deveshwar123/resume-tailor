"""All pydantic schemas.

Two families of models live here:

* **Wire models** (`*Extract`, `*Draft`, `Judgment`, ...) describe exactly what
  Claude is asked to return through structured outputs. Every field is required
  (nullable where it can be absent) and there are no numeric constraints, so the
  schemas stay inside what structured outputs support.
* **Domain models** are what the rest of the program uses. They add fields that
  only code may set (flags, approval state, verified header data, scores).

Nothing here calls the network.
"""

import re
from enum import StrEnum
from typing import ClassVar, Iterator, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

import config
from src import textutil

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class Ownership(StrEnum):
    member = "member"
    contributor = "contributor"
    lead = "lead"
    founder = "founder"
    owner = "owner"


class FactKind(StrEnum):
    responsibility = "responsibility"
    achievement = "achievement"
    metric = "metric"
    tool_use = "tool_use"
    scope = "scope"


class ExperienceType(StrEnum):
    job = "job"
    internship = "internship"
    research = "research"
    program = "program"        # selective programs, competitions, workshops
    leadership = "leadership"
    club = "club"
    volunteer = "volunteer"


class SkillStatus(StrEnum):
    verified = "verified"
    learning = "learning"
    missing = "missing"


class JobType(StrEnum):
    software = "software"
    data = "data"
    finance = "finance"
    research = "research"
    other = "other"


class SkillKind(StrEnum):
    language = "language"      # programming / query languages
    tool = "tool"              # named products, platforms, libraries
    technical = "technical"    # technical concepts (machine learning, ETL, ...)
    domain = "domain"          # domain knowledge (accounting, equity research, ...)
    soft = "soft"              # communication, teamwork, ...


class ChangeType(StrEnum):
    unchanged = "unchanged"
    minor_wording = "minor_wording"
    reworded = "reworded"
    combined = "combined"


class FixType(StrEnum):
    REWRITE = "REWRITE"
    ASK_USER = "ASK_USER"
    REAL_GAP = "REAL_GAP"


class Tier(StrEnum):
    official = "official"
    posting = "posting"
    third_party = "third_party"
    unverified = "unverified"


class Severity(StrEnum):
    hard = "hard"
    soft = "soft"


class EligStatus(StrEnum):
    meets = "Meets"
    partial = "Partial"
    fails = "Doesn't meet"
    unknown = "Unknown"
    not_stated = "Not stated"


class Approval(StrEnum):
    auto = "auto"              # minor change, auto-approved
    pending = "pending"        # waiting for the human checkpoint
    approved = "approved"
    edited = "edited"
    reverted = "reverted"
    override = "override"      # user kept an edit that failed validation


class SectionName(StrEnum):
    education = "education"
    experience = "experience"
    projects = "projects"
    leadership = "leadership"
    skills = "skills"
    awards = "awards"
    certifications = "certifications"


class ClassStanding(StrEnum):
    freshman = "freshman"
    sophomore = "sophomore"
    junior = "junior"
    senior = "senior"
    graduate = "graduate"


class WorkAuthKind(StrEnum):
    not_stated = "not_stated"
    citizen_required = "citizen_required"            # U.S. citizenship required
    us_person_required = "us_person_required"        # citizen or permanent resident
    no_sponsorship = "no_sponsorship"                # must work without sponsorship
    sponsorship_available = "sponsorship_available"
    other = "other"


class PriorExperienceKind(StrEnum):
    internship = "internship"
    research = "research"
    work = "work"
    any = "any"


class StopReason(StrEnum):
    target_reached = "target_reached"
    no_rewrite_fixes = "no_rewrite_fixes"
    max_rounds = "max_rounds"
    plateau = "plateau"


class Mode(StrEnum):
    normal = "normal"
    strict = "strict"


# ---------------------------------------------------------------------------
# Master resume (source of truth)
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^(\d{4}(-(0[1-9]|1[0-2]))?|present|)$")


class Contact(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""


class DateRange(BaseModel):
    """Dates are "YYYY", "YYYY-MM", "present", or "" (unknown)."""

    start: str = ""
    end: str = ""

    @field_validator("start", "end", mode="before")
    @classmethod
    def _normalize(cls, v: object) -> str:
        text = "" if v is None else str(v).strip()
        if text.lower() == "present":
            return "present"
        if not _DATE_RE.match(text):
            raise ValueError(f'date "{text}" must be "YYYY", "YYYY-MM", or "present"')
        return text


class Fact(BaseModel):
    """One atomic, verified statement about the candidate."""

    id: str
    kind: FactKind = FactKind.responsibility
    text: str
    ownership: Ownership | None = None      # overrides the role-level ownership
    metrics: list[str] = Field(default_factory=list)   # exact strings allowed in output
    skills: list[str] = Field(default_factory=list)    # skills this fact evidences
    source: str | None = None               # "interview" for facts added by interview.py
    added: str | None = None                # ISO date the fact was added


class Bullet(BaseModel):
    """An optional phrasing built from facts. `approved` means the user signed it off."""

    id: str
    text: str
    fact_ids: list[str] = Field(default_factory=list)
    approved: bool = False


class Education(BaseModel):
    id: str
    school: str
    degree: str
    major: str = ""
    location: str = ""
    grad: str = ""                 # display text, e.g. "May 2030"
    class_year: int | None = None
    gpa: float | None = None
    honors: list[str] = Field(default_factory=list)
    coursework: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)   # extra display lines, copied verbatim


class Experience(BaseModel):
    """A job, internship, research role, program, club, or leadership role."""

    id: str
    title: str = ""
    org: str
    location: str = ""
    dates: DateRange = Field(default_factory=DateRange)
    type: ExperienceType = ExperienceType.job
    section: Literal["experience", "leadership"] | None = None   # override where it is listed
    ownership: Ownership = Ownership.contributor   # highest TRUE level for this role
    tags: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    bullets: list[Bullet] = Field(default_factory=list)

    @property
    def kind(self) -> str:
        return "experience"

    @property
    def section_name(self) -> str:
        """The resume section this role is listed under ("experience" or "leadership")."""
        return self.section or config.SECTION_FOR_TYPE.get(self.type.value, "experience")

    @property
    def display_org(self) -> str:
        return self.org

    @property
    def display_title(self) -> str:
        return self.title


class Project(BaseModel):
    id: str
    name: str
    role: str = ""
    link: str = ""
    dates: DateRange = Field(default_factory=DateRange)
    ownership: Ownership = Ownership.owner
    tags: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    bullets: list[Bullet] = Field(default_factory=list)

    @property
    def kind(self) -> str:
        return "project"

    @property
    def display_org(self) -> str:
        return self.name

    @property
    def display_title(self) -> str:
        return self.role

    @property
    def location(self) -> str:
        return ""


Entry = Experience | Project


class Award(BaseModel):
    id: str
    text: str


class SkillsBank(BaseModel):
    """The ONLY skills allowed in the output skills section."""

    languages: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)

    def all(self) -> list[str]:
        return [*self.languages, *self.tools, *self.concepts]

    def category_of(self, skill: str) -> str | None:
        target = textutil.canon(skill)
        for category in ("languages", "tools", "concepts"):
            if any(textutil.canon(s) == target for s in getattr(self, category)):
                return category
        return None

    def display_name(self, skill: str) -> str | None:
        """The spelling used in the master list for a (possibly synonymous) skill."""
        target = textutil.canon(skill)
        for item in self.all():
            if textutil.canon(item) == target:
                return item
        return None


class MasterResume(BaseModel):
    contact: Contact = Field(default_factory=Contact)
    education: list[Education] = Field(default_factory=list)
    work_authorization: str = ""            # used only by eligibility.py
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: SkillsBank = Field(default_factory=SkillsBank)
    certifications: list[str] = Field(default_factory=list)
    awards: list[Award] = Field(default_factory=list)

    # -- lookups -----------------------------------------------------------
    def entries(self) -> Iterator[Entry]:
        yield from self.experience
        yield from self.projects

    def entry(self, entry_id: str) -> Entry | None:
        for entry in self.entries():
            if entry.id == entry_id:
                return entry
        return None

    def fact_index(self) -> dict[str, tuple[Entry, Fact]]:
        return {fact.id: (entry, fact) for entry in self.entries() for fact in entry.facts}

    def bullet_index(self) -> dict[str, tuple[Entry, Bullet]]:
        return {b.id: (entry, b) for entry in self.entries() for b in entry.bullets}

    def award(self, award_id: str) -> Award | None:
        return next((a for a in self.awards if a.id == award_id), None)

    @staticmethod
    def effective_ownership(entry: Entry, fact: Fact) -> Ownership:
        """A fact's own ownership wins; otherwise the role-level ownership applies."""
        return fact.ownership or entry.ownership

    # -- integrity ---------------------------------------------------------
    @model_validator(mode="after")
    def _check_integrity(self) -> "MasterResume":
        problems: list[str] = []
        seen: dict[str, str] = {}

        def claim(item_id: str, where: str) -> None:
            if not item_id or not item_id.strip():
                problems.append(f"{where}: empty id")
            elif item_id in seen:
                problems.append(f'duplicate id "{item_id}" ({seen[item_id]} and {where})')
            else:
                seen[item_id] = where

        for edu in self.education:
            claim(edu.id, f"education {edu.school!r}")
            if not edu.school.strip():
                problems.append(f"education {edu.id}: school is required")
            if not edu.degree.strip():
                problems.append(f"education {edu.id}: degree is required")
        for award in self.awards:
            claim(award.id, "award")
            if not award.text.strip():
                problems.append(f"award {award.id}: text is required")
        for entry in self.entries():
            label = f"{entry.kind} {entry.display_org!r}"
            claim(entry.id, label)
            if not entry.display_org.strip():
                problems.append(f"{entry.kind} {entry.id}: {'org' if entry.kind == 'experience' else 'name'} is required")
            fact_ids = set()
            for fact in entry.facts:
                claim(fact.id, f"fact in {label}")
                fact_ids.add(fact.id)
                if not fact.text.strip():
                    problems.append(f"fact {fact.id}: text is required")
                for metric in fact.metrics:
                    if textutil.tight(metric) not in textutil.tight(fact.text):
                        problems.append(
                            f'fact {fact.id}: metric "{metric}" does not appear in the fact text'
                        )
            for bullet in entry.bullets:
                claim(bullet.id, f"bullet in {label}")
                if not bullet.text.strip():
                    problems.append(f"bullet {bullet.id}: text is required")
                if not bullet.fact_ids:
                    problems.append(f"bullet {bullet.id}: must cite at least one fact_id")
                for fid in bullet.fact_ids:
                    if fid not in fact_ids:
                        problems.append(
                            f'bullet {bullet.id}: cites "{fid}", which is not a fact of {entry.id}'
                        )
        if problems:
            raise ValueError("; ".join(problems))
        return self


class SkillsStatus(BaseModel):
    """My status per skill. Only `verified` skills may appear on the resume."""

    skills: dict[str, SkillStatus] = Field(default_factory=dict)

    def _canon_map(self) -> dict[str, SkillStatus]:
        return {textutil.canon(name): status for name, status in self.skills.items()}

    def status_of(self, skill: str) -> SkillStatus | None:
        """Status for a skill (synonym-aware); None if the skill is not listed."""
        return self._canon_map().get(textutil.canon(skill))

    def is_verified(self, skill: str) -> bool:
        return self.status_of(skill) == SkillStatus.verified

    def names_with(self, status: SkillStatus) -> list[str]:
        return [name for name, value in self.skills.items() if value == status]


# ---------------------------------------------------------------------------
# Job analysis
# ---------------------------------------------------------------------------


class WeightedSkill(BaseModel):
    name: str
    weight: int          # 1 (nice) .. 3 (central to the role)
    kind: SkillKind


class GradYearReq(BaseModel):
    """Graduation window, each bound "YYYY" or "YYYY-MM"; null when unstated."""

    earliest: str | None
    latest: str | None
    source: str | None


class ClassStandingReq(BaseModel):
    """Standing during the school year in which the candidate applies."""

    allowed: list[ClassStanding]
    source: str | None


class DegreeFieldsReq(BaseModel):
    fields: list[str]
    related_fields_ok: bool      # posting says "or related field"
    source: str | None


class MinGpaReq(BaseModel):
    value: float | None
    source: str | None


class WorkAuthReq(BaseModel):
    kind: WorkAuthKind
    source: str | None


class PriorExperienceReq(BaseModel):
    required: bool | None
    kind: PriorExperienceKind | None
    source: str | None


class OtherReq(BaseModel):
    text: str
    source: str | None


class Requirements(BaseModel):
    grad_year_range: GradYearReq
    class_standing: ClassStandingReq
    degree_fields: DegreeFieldsReq
    min_gpa: MinGpaReq
    work_authorization: WorkAuthReq
    prior_experience_required: PriorExperienceReq
    other: list[OtherReq]

    @classmethod
    def empty(cls) -> "Requirements":
        return cls(
            grad_year_range=GradYearReq(earliest=None, latest=None, source=None),
            class_standing=ClassStandingReq(allowed=[], source=None),
            degree_fields=DegreeFieldsReq(fields=[], related_fields_ok=False, source=None),
            min_gpa=MinGpaReq(value=None, source=None),
            work_authorization=WorkAuthReq(kind=WorkAuthKind.not_stated, source=None),
            prior_experience_required=PriorExperienceReq(required=None, kind=None, source=None),
            other=[],
        )


class JobAnalysisExtract(BaseModel):
    """Wire model: what analyze.py asks Claude to extract from a posting."""

    company: str
    role: str
    location: str
    job_type: JobType
    required_skills: list[WeightedSkill]
    preferred_skills: list[WeightedSkill]
    tools: list[str]
    key_phrases: list[str]
    responsibilities: list[str]
    keywords: list[str]
    requirements: Requirements


class JobAnalysis(JobAnalysisExtract):
    """Domain model: the extraction plus bookkeeping set by code."""

    id: str = ""                 # file stem under data/postings/
    url: str = ""
    analyzed_on: str = ""        # ISO date
    posting_words: int = 0
    unverified_quotes: list[str] = Field(default_factory=list)   # quotes not found in the posting

    def all_skills(self) -> list[WeightedSkill]:
        return [*self.required_skills, *self.preferred_skills]


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


class EligibilityRow(BaseModel):
    requirement: str
    status: EligStatus
    evidence: str = ""           # exact quote from the posting, when there is one
    detail: str = ""             # what was compared, missing skills, ...
    interpreted: bool = False    # true when Claude interpreted vague wording
    quote_verified: bool | None = None   # quote found verbatim in the posting text


class EligibilityReport(BaseModel):
    rows: list[EligibilityRow] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    learning_required_skills: list[str] = Field(default_factory=list)

    def flags(self) -> list[str]:
        """Human-readable flags for anything that is not a clean Meets / Not stated."""
        out = []
        for row in self.rows:
            if row.status in (EligStatus.fails, EligStatus.partial, EligStatus.unknown):
                note = f"{row.requirement}: {row.status.value}"
                if row.detail:
                    note += f" ({row.detail})"
                out.append(note)
        return out

    def has_blocker(self) -> bool:
        return any(row.status == EligStatus.fails for row in self.rows)


class InterpretedRequirements(BaseModel):
    """Wire model: Claude's reading of vaguely worded requirements."""

    grad_earliest: str | None
    grad_latest: str | None
    class_standing_allowed: list[ClassStanding]
    work_authorization: WorkAuthKind | None
    notes: str


# ---------------------------------------------------------------------------
# Company research
# ---------------------------------------------------------------------------


class BriefItemDraft(BaseModel):
    text: str
    source_url: str | None       # a URL from the search results, "posting", or null


class CompanyBriefDraft(BaseModel):
    """Wire model: Claude structures its research notes; code assigns the tiers."""

    summary: list[BriefItemDraft]
    tech_stack: list[BriefItemDraft]
    values_language: list[BriefItemDraft]
    intern_work: list[BriefItemDraft]
    recent_news: list[BriefItemDraft]


class BriefItem(BaseModel):
    text: str
    source_url: str | None = None
    tier: Tier = Tier.unverified


class BriefSource(BaseModel):
    url: str
    title: str = ""
    tier: Tier = Tier.unverified


class CompanyBrief(BaseModel):
    company: str = ""
    researched_on: str = ""
    summary: list[BriefItem] = Field(default_factory=list)
    tech_stack: list[BriefItem] = Field(default_factory=list)
    values_language: list[BriefItem] = Field(default_factory=list)
    intern_work: list[BriefItem] = Field(default_factory=list)
    recent_news: list[BriefItem] = Field(default_factory=list)
    sources: list[BriefSource] = Field(default_factory=list)

    FIELDS: ClassVar[tuple[str, ...]] = (
        "summary", "tech_stack", "values_language", "intern_work", "recent_news",
    )

    def items(self, tiers: tuple[str, ...] | None = None) -> dict[str, list[BriefItem]]:
        """Brief items by field, optionally restricted to some tiers."""
        out: dict[str, list[BriefItem]] = {}
        for name in self.FIELDS:
            kept = [i for i in getattr(self, name) if tiers is None or i.tier.value in tiers]
            if kept:
                out[name] = kept
        return out

    def is_empty(self) -> bool:
        return not any(getattr(self, name) for name in self.FIELDS)


# ---------------------------------------------------------------------------
# Tailored resume
# ---------------------------------------------------------------------------


class BulletDraft(BaseModel):
    """Wire model for one bullet proposed by Claude."""

    rewritten_text: str
    fact_ids: list[str]
    source_bullet_id: str | None
    change_type: ChangeType
    rationale: str


class EntryDraft(BaseModel):
    """Wire model for one selected experience or project.

    `org`, `title`, `start` and `end` are an echo of the master resume. They are
    never used for output (code copies the real values); they exist so
    validate.py can detect an AI that altered an employer, title, or date.
    """

    source_id: str
    org: str
    title: str
    start: str
    end: str
    bullets: list[BulletDraft]


class TailoredResumeDraft(BaseModel):
    """Wire model: the tailor's / reviser's full proposal."""

    section_order: list[SectionName]
    experiences: list[EntryDraft]
    projects: list[EntryDraft]
    skills: list[str]
    honors: list[str]
    coursework: list[str]
    award_ids: list[str]


class Flag(BaseModel):
    """A soft flag that sends a bullet to the approval checkpoint."""

    code: str
    message: str


class TailoredBullet(BaseModel):
    # -- from the AI ---------------------------------------------------------
    rewritten_text: str
    fact_ids: list[str] = Field(default_factory=list)
    source_bullet_id: str | None = None
    change_type: ChangeType = ChangeType.unchanged     # recomputed by validate.py
    rationale: str = ""
    # -- set by code only ----------------------------------------------------
    original_text: str = ""          # verified source wording (approved bullet or fact text)
    original_kind: str = ""          # "approved_bullet" | "fact" | "facts"
    flags: list[Flag] = Field(default_factory=list)
    approval: Approval = Approval.auto
    final_text: str = ""             # the text that is rendered
    history: list[str] = Field(default_factory=list)   # validation / page-fit notes

    @property
    def text(self) -> str:
        return self.final_text or self.rewritten_text


class TailoredEntry(BaseModel):
    source_id: str
    kind: str = "experience"         # "experience" | "project"
    bullets: list[TailoredBullet] = Field(default_factory=list)
    replaced: str | None = None      # note when validate.py swapped this entry in


class TailoredResume(BaseModel):
    mode: Mode = Mode.normal
    section_order: list[SectionName] = Field(default_factory=list)
    experiences: list[TailoredEntry] = Field(default_factory=list)
    projects: list[TailoredEntry] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    honors: list[str] = Field(default_factory=list)
    coursework: list[str] = Field(default_factory=list)
    award_ids: list[str] = Field(default_factory=list)

    def entries(self) -> Iterator[TailoredEntry]:
        yield from self.experiences
        yield from self.projects

    def bullets(self) -> Iterator[tuple[TailoredEntry, int, TailoredBullet]]:
        for entry in self.entries():
            for index, bullet in enumerate(entry.bullets):
                yield entry, index, bullet

    def pending(self) -> list[tuple[TailoredEntry, int, TailoredBullet]]:
        return [t for t in self.bullets() if t[2].approval == Approval.pending]

    def to_draft(self, master: MasterResume) -> TailoredResumeDraft:
        """Re-serialize as a wire draft (used to show Claude the current state)."""

        def entry_draft(entry: TailoredEntry) -> EntryDraft:
            source = master.entry(entry.source_id)
            return EntryDraft(
                source_id=entry.source_id,
                org=source.display_org if source else "",
                title=source.display_title if source else "",
                start=source.dates.start if source else "",
                end=source.dates.end if source else "",
                bullets=[
                    BulletDraft(
                        # A proposal still waiting for approval is shown as proposed, so it
                        # survives a revision round (in strict mode `text` is the original).
                        rewritten_text=b.rewritten_text if b.approval == Approval.pending else b.text,
                        fact_ids=list(b.fact_ids),
                        source_bullet_id=b.source_bullet_id,
                        change_type=b.change_type,
                        rationale=b.rationale,
                    )
                    for b in entry.bullets
                ],
            )

        return TailoredResumeDraft(
            section_order=list(self.section_order),
            experiences=[entry_draft(e) for e in self.experiences],
            projects=[entry_draft(e) for e in self.projects],
            skills=list(self.skills),
            honors=list(self.honors),
            coursework=list(self.coursework),
            award_ids=list(self.award_ids),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class Violation(BaseModel):
    code: str                        # A1..A5, B1..B5 (see validate.py)
    severity: Severity
    entry_id: str = ""
    bullet_index: int | None = None
    message: str
    token: str = ""                  # the offending word / number / id
    resolution: str = ""             # what the validator did about it


class BulletFixRequest(BaseModel):
    """Sent to Claude for the single retry after a hard violation."""

    key: str                         # "<entry_id>#<bullet_index>"
    entry_id: str
    text: str
    fact_ids: list[str]
    violations: list[str]


class BulletFix(BaseModel):
    key: str
    rewritten_text: str
    fact_ids: list[str]


class BulletFixBatch(BaseModel):
    """Wire model: Claude's corrected bullets."""

    fixes: list[BulletFix]


# ---------------------------------------------------------------------------
# Judging and scoring
# ---------------------------------------------------------------------------


class CategoryScore(BaseModel):
    score: int                       # 1-10 (clamped in code)
    justification: str


class CategoryScores(BaseModel):
    impact: CategoryScore
    relevance: CategoryScore
    clarity: CategoryScore
    language: CategoryScore


class Fix(BaseModel):
    type: FixType
    target: str                      # which bullet / section the fix is about
    detail: str                      # what to change, or the question for the user
    expected_gain: float             # judge's estimate, in quality points


class Judgment(BaseModel):
    """Wire model: the judge returns category scores and fixes. Never a total."""

    category_scores: CategoryScores
    fixes: list[Fix]

    def fixes_of(self, fix_type: FixType) -> list[Fix]:
        return [f for f in self.fixes if f.type == fix_type]


class QualityScore(BaseModel):
    total: float                               # 0-10, computed in Python
    categories: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    ats_final: bool = False                    # True once ATS was scored on the real PDF


class JobMatch(BaseModel):
    score: float                               # 0-100
    components: dict[str, float] = Field(default_factory=dict)   # each 0-1
    covered: list[str] = Field(default_factory=list)
    partial: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class KeywordCoverage(BaseModel):
    score: float                               # 0-100, weighted
    matched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class AtsCheck(BaseModel):
    name: str
    passed: bool
    score: float = 1.0                         # 0-1 partial credit
    detail: str = ""


class AtsReport(BaseModel):
    checks: list[AtsCheck] = Field(default_factory=list)
    score: float = 0.0                         # 0-10
    final: bool = False                        # False for the pre-render text-only estimate

    def failures(self) -> list[AtsCheck]:
        return [c for c in self.checks if not c.passed]


class RoundRecord(BaseModel):
    round: int
    quality: QualityScore
    judgment: Judgment
    note: str = ""


class LoopResult(BaseModel):
    best: TailoredResume
    best_round: int = 0
    history: list[RoundRecord] = Field(default_factory=list)
    stop_reason: StopReason = StopReason.target_reached
    violations: list[Violation] = Field(default_factory=list)

    @property
    def best_quality(self) -> QualityScore:
        return self.history[self.best_round].quality

    @property
    def last_judgment(self) -> Judgment:
        return self.history[-1].judgment

    @property
    def best_judgment(self) -> Judgment:
        return self.history[self.best_round].judgment


# ---------------------------------------------------------------------------
# Interview
# ---------------------------------------------------------------------------


class ProposedFact(BaseModel):
    """Wire model: a user's interview answer structured as a fact."""

    parent_id: str | None            # existing experience / project it belongs to, or null
    kind: FactKind
    text: str
    ownership: Ownership
    metrics: list[str]
    skills: list[str]


# ---------------------------------------------------------------------------
# Usage / cost
# ---------------------------------------------------------------------------


class CallUsage(BaseModel):
    purpose: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0
    cost_usd: float = 0.0
