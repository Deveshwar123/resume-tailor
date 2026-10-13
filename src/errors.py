"""Expected-failure exceptions.

Anything raised as a `TailorError` is an *expected* failure (missing key, bad
model ID, login wall, schema problem in the master resume, ...). The CLI
prints its message cleanly and exits non-zero without a stack trace.
"""

from __future__ import annotations


class TailorError(Exception):
    """Base class for expected, user-facing failures."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message if not self.hint else f"{self.message}\n{self.hint}"


class ConfigError(TailorError):
    """Missing or invalid configuration (.env, API key, model IDs, files)."""


class MasterResumeError(TailorError):
    """master_resume.yaml or skills_status.yaml failed validation."""


class FetchError(TailorError):
    """The job posting could not be fetched or looked like a login/captcha wall."""


class LLMError(TailorError):
    """A Claude call failed in an expected way (auth, rate limit, bad output)."""


class LLMRefusal(LLMError):
    """Claude declined the request (stop_reason == "refusal")."""


class RenderError(TailorError):
    """DOCX/PDF rendering failed (no converter, template problem, ...)."""


class UserAbort(TailorError):
    """The user chose to stop (e.g. answered "n" at the eligibility prompt)."""
