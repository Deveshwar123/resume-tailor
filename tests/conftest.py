"""Shared fixtures: a small fictional master resume and a scripted Claude.

Tests never touch the network and never read the real master_resume.yaml
(except the one test that checks the shipped file still validates).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from src import ui
from src.llm import SearchResult, UsageTracker
from src.models import MasterResume, SkillsStatus

SAMPLE_MASTER: dict[str, Any] = {
    "contact": {
        "name": "Test Person", "email": "test.person@example.com", "phone": "(555) 010-0100",
        "location": "Austin, TX", "linkedin": "linkedin.com/in/testperson", "github": "github.com/testperson",
    },
    "education": [{
        "id": "edu_state", "school": "State University", "degree": "B.S. Computer Science",
        "major": "Computer Science", "location": "Austin, TX", "grad": "May 2028", "class_year": 2028,
        "gpa": 3.6, "honors": ["Dean's List"], "coursework": ["Data Structures", "Databases"],
    }],
    "work_authorization": "US citizen",
    "experience": [
        {
            "id": "exp_acme", "title": "Software Engineering Intern", "org": "Acme Corp",
            "location": "Austin, TX", "dates": {"start": "2025-06", "end": "2025-08"},
            "type": "internship", "ownership": "contributor", "tags": ["software", "data"],
            "skills": ["Python", "SQL"],
            "facts": [
                {"id": "exp_acme_f01", "kind": "responsibility",
                 "text": "Assisted a 4-person team in developing a Python application",
                 "ownership": "contributor", "metrics": ["4-person"], "skills": ["Python"]},
                {"id": "exp_acme_f02", "kind": "achievement",
                 "text": "Wrote SQL queries that reduced report generation time by 30%",
                 "metrics": ["30%"], "skills": ["SQL"]},
                {"id": "exp_acme_f03", "kind": "tool_use",
                 "text": "Tested REST API endpoints with pytest", "skills": ["pytest", "REST"]},
            ],
            "bullets": [
                {"id": "exp_acme_b01", "text": "Assisted a 4-person team in developing a Python application.",
                 "fact_ids": ["exp_acme_f01"], "approved": True},
                {"id": "exp_acme_b02", "text": "Wrote SQL queries that reduced report generation time by 30%.",
                 "fact_ids": ["exp_acme_f02"], "approved": True},
            ],
        },
        {
            "id": "exp_club", "title": "Vice President", "org": "Data Club", "location": "",
            "dates": {"start": "2024-09", "end": "present"}, "type": "club", "ownership": "lead",
            "tags": ["leadership", "data"], "skills": ["data analysis"],
            "facts": [
                {"id": "exp_club_f01", "kind": "responsibility",
                 "text": "Led weekly workshops for 25 members on data analysis",
                 "ownership": "lead", "metrics": ["25"], "skills": ["data analysis"]},
                {"id": "exp_club_f02", "kind": "achievement",
                 "text": "Organized a campus hackathon with 60+ participants", "metrics": ["60+"]},
            ],
            "bullets": [],
        },
        {
            "id": "exp_lab", "title": "Research Assistant", "org": "Vision Lab", "location": "Austin, TX",
            "dates": {"start": "2024-01", "end": "2024-05"}, "type": "research", "ownership": "member",
            "tags": ["research"], "skills": [],
            "facts": [
                {"id": "exp_lab_f01", "kind": "responsibility",
                 "text": "Participated in weekly lab meetings and annotated 500 images",
                 "ownership": "member", "metrics": ["500"]},
            ],
            "bullets": [],
        },
    ],
    "projects": [
        {
            "id": "proj_tracker", "name": "Budget Tracker", "role": "", "link": "https://github.com/testperson/budget",
            "dates": {"start": "2025-01", "end": "2025-03"}, "ownership": "owner", "tags": ["software", "web"],
            "skills": ["JavaScript", "React"],
            "facts": [
                {"id": "proj_tracker_f01", "kind": "tool_use",
                 "text": "Built a budget tracking web app with JavaScript and React",
                 "ownership": "owner", "skills": ["JavaScript", "React"]},
                {"id": "proj_tracker_f02", "kind": "metric",
                 "text": "Deployed the app for 40 student users", "metrics": ["40"]},
            ],
            "bullets": [
                {"id": "proj_tracker_b01", "text": "Built a budget tracking web app with JavaScript and React.",
                 "fact_ids": ["proj_tracker_f01"], "approved": True},
            ],
        },
    ],
    "skills": {
        "languages": ["Python", "SQL", "JavaScript"],
        "tools": ["React", "pytest", "Git"],
        "concepts": ["data analysis", "REST"],
    },
    "certifications": [],
    "awards": [{"id": "award_hack", "text": "Campus Hackathon winner, 2025."}],
}

SAMPLE_STATUS: dict[str, str] = {
    "Python": "verified", "SQL": "verified", "JavaScript": "verified", "React": "verified",
    "pytest": "verified", "Git": "verified", "data analysis": "verified", "REST": "verified",
    "PyTorch": "learning", "Docker": "learning", "AWS": "missing", "Tableau": "missing",
}


@pytest.fixture
def master_data() -> dict[str, Any]:
    return copy.deepcopy(SAMPLE_MASTER)


@pytest.fixture
def master() -> MasterResume:
    return MasterResume.model_validate(copy.deepcopy(SAMPLE_MASTER))


@pytest.fixture
def status() -> SkillsStatus:
    return SkillsStatus.model_validate({"skills": dict(SAMPLE_STATUS)})


class ScriptedLLM:
    """Stands in for `src.llm.LLM`: returns queued objects instead of calling Claude.

    `responses` maps a purpose ("tailor", "judge", ...) to a list of objects
    returned in order. A callable entry is invoked with the call's kwargs.
    """

    def __init__(self, responses: dict[str, list[Any]] | None = None) -> None:
        self.responses = {k: list(v) for k, v in (responses or {}).items()}
        self.calls: list[dict[str, Any]] = []
        self.tracker = UsageTracker()

    def queue(self, purpose: str, *items: Any) -> None:
        self.responses.setdefault(purpose, []).extend(items)

    def _next(self, purpose: str, kwargs: dict[str, Any]) -> Any:
        self.calls.append({"purpose": purpose, **kwargs})
        queue = self.responses.get(purpose) or []
        if not queue:
            raise AssertionError(f"ScriptedLLM: no response queued for purpose {purpose!r}")
        item = queue.pop(0) if len(queue) > 1 or not getattr(self, "repeat_last", False) else queue[0]
        return item(kwargs) if callable(item) else item

    def parse(self, *, purpose: str, model: str, system: str, user: str, schema: type,
              cache_system: bool = True, max_tokens: int | None = None) -> Any:
        result = self._next(purpose, {"model": model, "system": system, "user": user, "schema": schema})
        if isinstance(result, dict):
            result = schema.model_validate(result)
        assert isinstance(result, schema), f"{purpose}: expected {schema.__name__}, got {type(result).__name__}"
        return result

    def search(self, *, purpose: str, model: str, system: str, user: str,
               max_searches: int | None = None) -> SearchResult:
        return self._next(purpose, {"model": model, "system": system, "user": user})

    def purposes(self) -> list[str]:
        return [c["purpose"] for c in self.calls]


@pytest.fixture
def scripted() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture(autouse=True)
def _non_interactive():
    """Prompts never block in tests unless a test opts in with ui.set_input."""
    ui.set_input(None, interactive=False)
    yield
    ui.set_input(None, interactive=None)


def answers(*lines: str):
    """Feed scripted keyboard answers to ui prompts."""
    queue = list(lines)

    def _fake_input(_prompt: str = "") -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    ui.set_input(_fake_input, interactive=True)


# ---------------------------------------------------------------------------
# A sample posting and its (scripted) analysis
# ---------------------------------------------------------------------------

SAMPLE_POSTING = """Acme Corp - Data Engineering Intern (Summer 2027)
Location: Austin, TX

About the role
You will build data pipelines in Python and SQL, clean and analyze datasets, and help the
analytics team automate reporting. Interns ship production code with a mentor.

Minimum qualifications
- Currently pursuing a B.S. in Computer Science, Data Science, or a related field.
- Expected graduation between December 2027 and June 2029.
- Minimum GPA of 3.5.
- Must be authorized to work in the United States without sponsorship.
- Experience with Python and SQL.

Preferred qualifications
- Familiarity with AWS and Docker.
- Prior internship experience is a plus.
- Exposure to machine learning.
"""


def make_extract(**overrides: Any):
    """A JobAnalysisExtract for the sample posting; override any top-level field."""
    from src.models import JobAnalysisExtract

    data: dict[str, Any] = {
        "company": "Acme Corp",
        "role": "Data Engineering Intern",
        "location": "Austin, TX",
        "job_type": "data",
        "required_skills": [
            {"name": "Python", "weight": 3, "kind": "language"},
            {"name": "SQL", "weight": 3, "kind": "language"},
        ],
        "preferred_skills": [
            {"name": "AWS", "weight": 2, "kind": "tool"},
            {"name": "Docker", "weight": 1, "kind": "tool"},
            {"name": "machine learning", "weight": 1, "kind": "technical"},
        ],
        "tools": ["Python", "SQL", "AWS", "Docker"],
        "key_phrases": ["build data pipelines", "automate reporting"],
        "responsibilities": ["Build data pipelines", "Clean and analyze datasets"],
        "keywords": ["Python", "SQL", "data pipelines", "AWS", "Docker", "machine learning", "datasets"],
        "requirements": {
            "grad_year_range": {"earliest": "2027-12", "latest": "2029-06",
                                "source": "Expected graduation between December 2027 and June 2029."},
            "class_standing": {"allowed": [], "source": None},
            "degree_fields": {"fields": ["Computer Science", "Data Science"], "related_fields_ok": True,
                              "source": "Currently pursuing a B.S. in Computer Science, Data Science, or a related field."},
            "min_gpa": {"value": 3.5, "source": "Minimum GPA of 3.5."},
            "work_authorization": {"kind": "no_sponsorship",
                                   "source": "Must be authorized to work in the United States without sponsorship."},
            "prior_experience_required": {"required": False, "kind": None,
                                          "source": "Prior internship experience is a plus."},
            "other": [],
        },
    }
    for key, value in overrides.items():
        if key == "requirements":
            data["requirements"].update(value)
        else:
            data[key] = value
    return JobAnalysisExtract.model_validate(data)


def make_analysis(**overrides: Any):
    from datetime import date

    from src.analyze import finalize
    from src.fetch import JobText

    return finalize(make_extract(**overrides), JobText(SAMPLE_POSTING, "https://careers.acme.com/jobs/123"),
                    date(2026, 10, 13))


@pytest.fixture
def analysis():
    return make_analysis()


# ---------------------------------------------------------------------------
# Draft builders (what a scripted Claude "returns" from tailor / revise)
# ---------------------------------------------------------------------------


def bd(text: str, fact_ids: list[str], source: str | None = None, change: str = "reworded", why: str = ""):
    from src.models import BulletDraft

    return BulletDraft(rewritten_text=text, fact_ids=fact_ids, source_bullet_id=source, change_type=change, rationale=why)


def ed(source_id: str, bullets: list, **echo: str):
    """An EntryDraft whose header echo matches the sample master unless overridden."""
    from src.models import EntryDraft

    known = {e["id"]: e for e in SAMPLE_MASTER["experience"] + SAMPLE_MASTER["projects"]}
    source = known.get(source_id, {})
    header = {
        "org": source.get("org", source.get("name", "")),
        "title": source.get("title", source.get("role", "")),
        "start": source.get("dates", {}).get("start", ""),
        "end": source.get("dates", {}).get("end", ""),
    }
    header.update(echo)
    return EntryDraft(source_id=source_id, bullets=bullets, **header)


def make_draft(experiences: list | None = None, projects: list | None = None, skills: list[str] | None = None,
               order: list[str] | None = None, honors: list[str] | None = None,
               coursework: list[str] | None = None, award_ids: list[str] | None = None):
    from src.models import TailoredResumeDraft

    return TailoredResumeDraft(
        section_order=order or ["education", "experience", "projects", "skills"],
        experiences=experiences or [], projects=projects or [], skills=skills or [],
        honors=honors or [], coursework=coursework or [], award_ids=award_ids or [],
    )


@pytest.fixture
def ctx(master, status, analysis):
    from src.validate import ValidationContext

    return ValidationContext(master=master, status=status, analysis=analysis)
