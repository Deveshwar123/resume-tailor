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
