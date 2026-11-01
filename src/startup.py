"""Startup checks: environment, API key, model verification, and data files.

`run_startup()` is the first thing every command does. It fails fast with a
clear message (never a stack trace) when something is missing, and returns a
`Session` holding everything the rest of the run needs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import dotenv_values

import config
from src import store, ui
from src.errors import ConfigError
from src.llm import LLM
from src.models import MasterResume, SkillsStatus


@dataclass
class Session:
    master: MasterResume
    skills_status: SkillsStatus
    llm: LLM | None = None
    warnings: list[str] = field(default_factory=list)


def load_env() -> dict[str, str]:
    """Read `.env` (and only `.env`) and apply model overrides to config."""
    values = {k: v for k, v in dotenv_values(config.ENV_PATH).items() if v is not None}
    for name in ("FAST_MODEL", "WRITE_MODEL", "LLM_BACKEND", "CLAUDE_CODE_BIN",
                 "OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_FAST_MODEL"):
        # A variable set in the shell beats .env, so `LLM_BACKEND=ollama python tailor.py ...` works for one run.
        override = (os.environ.get(name) or values.get(name) or "").strip()
        if override:
            setattr(config, name, override)
    # Other fact banks and data folders (for example a test run on the fictional example data).
    paths = {name: (os.environ.get(name) or values.get(name) or "").strip()
             for name in ("TAILOR_MASTER_RESUME", "TAILOR_SKILLS_STATUS", "TAILOR_DATA_DIR")}
    config.use_paths(master=paths["TAILOR_MASTER_RESUME"] or None, skills=paths["TAILOR_SKILLS_STATUS"] or None,
                     data_dir=paths["TAILOR_DATA_DIR"] or None)
    return values


def read_api_key(env: dict[str, str]) -> str:
    """The API key comes from `.env` only - never from code or the command line."""
    if not config.ENV_PATH.exists():
        raise ConfigError(
            f"No .env file found at {config.ENV_PATH}.",
            "Copy .env.example to .env and set ANTHROPIC_API_KEY=sk-ant-...",
        )
    key = (env.get("ANTHROPIC_API_KEY") or "").strip().strip('"').strip("'")
    if not key or key.startswith("sk-ant-your") or key.endswith("..."):
        raise ConfigError(
            "ANTHROPIC_API_KEY is missing from .env.",
            "Add a line like ANTHROPIC_API_KEY=sk-ant-... (create a key at https://platform.claude.com/settings/keys).",
        )
    return key


def _model_check_path():
    return config.CACHE_DIR / "model_check.json"


def models_recently_verified(models: list[str]) -> bool:
    """True when these models passed the check on this backend within MODEL_CHECK_TTL_HOURS.

    With Claude Code the check is two real `claude -p` calls (about 4 s); a
    model that has since become unavailable still fails clearly on its first
    real call.
    """
    try:
        saved = json.loads(_model_check_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (saved.get("backend") == config.LLM_BACKEND and saved.get("models") == sorted(set(models))
            and time.time() - float(saved.get("at", 0)) < config.MODEL_CHECK_TTL_HOURS * 3600)


def remember_model_check(models: list[str]) -> None:
    path = _model_check_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"backend": config.LLM_BACKEND, "models": sorted(set(models)), "at": time.time()}),
                        encoding="utf-8")
    except OSError:
        pass                                  # only an optimization


def backend_needs_test_call() -> bool:
    """Only the API backend needs an extra request: the others already made one while verifying models."""
    return (config.LLM_BACKEND or "anthropic").strip().lower() == "anthropic"


def make_llm(env: dict[str, str]):
    """Build the client for the configured backend (LLM_BACKEND in .env)."""
    backend = (config.LLM_BACKEND or "anthropic").strip().lower()
    if backend == "anthropic":
        return LLM.from_key(read_api_key(env))
    if backend == "claude_code":
        from src.backends import ClaudeCodeLLM

        return ClaudeCodeLLM()
    if backend == "ollama":
        from src.backends import OllamaLLM

        return OllamaLLM()
    raise ConfigError(f'Unknown LLM_BACKEND "{config.LLM_BACKEND}".', "Use anthropic, claude_code, or ollama.")


def run_startup(*, need_api: bool = True, verify_models: bool = True, quiet: bool = False,
                test_call: bool = False, paths: dict[str, str | os.PathLike | None] | None = None) -> Session:
    """Run every startup check in order and return the session.

    `need_api=False` is used by commands that never call Claude
    (`--insights`, `--find`), so they work without a key. `paths` (keys
    master, skills, data_dir) point the run at other data; they win over the
    TAILOR_* environment variables.
    """
    env = load_env()
    if paths:
        config.use_paths(master=paths.get("master"), skills=paths.get("skills"), data_dir=paths.get("data_dir"))

    master = store.load_master_resume()
    skills_status = store.load_skills_status()
    warnings = store.data_warnings(master, skills_status)

    llm = None
    if need_api:
        llm = make_llm(env)
        models = [config.FAST_MODEL, config.WRITE_MODEL]
        if verify_models and (test_call or not models_recently_verified(models)):
            llm.verify_models(models)
            remember_model_check(models)
        if test_call and backend_needs_test_call():
            llm.verify_access(config.FAST_MODEL)

    if not quiet:
        facts = sum(len(e.facts) for e in master.entries())
        ui.success(
            f"master resume: {len(master.experience)} experiences, {len(master.projects)} projects, "
            f"{facts} facts; {len(skills_status.skills)} skills tracked"
        )
        if llm is not None and verify_models:
            if config.LLM_BACKEND == "ollama":
                local = config.OLLAMA_FAST_MODEL or config.OLLAMA_MODEL
                ui.success(f"local models ready via Ollama: {config.OLLAMA_MODEL} (write), {local} (fast)")
            else:
                via = " via Claude Code (your Claude subscription)" if config.LLM_BACKEND == "claude_code" else ""
                ui.success(f"models verified{via}: {config.FAST_MODEL} (fast), {config.WRITE_MODEL} (write)")
        if llm is not None and test_call and config.LLM_BACKEND == "anthropic":
            ui.success("test request succeeded (the key can make calls and the account has credits)")
        for warning in warnings:
            ui.warn(warning)
    return Session(master=master, skills_status=skills_status, llm=llm, warnings=warnings)
