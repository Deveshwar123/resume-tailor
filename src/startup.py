"""Startup checks: environment, API key, model verification, and data files.

`run_startup()` is the first thing every command does. It fails fast with a
clear message (never a stack trace) when something is missing, and returns a
`Session` holding everything the rest of the run needs.
"""

from __future__ import annotations

import os
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
    for name in ("FAST_MODEL", "WRITE_MODEL"):
        override = (values.get(name) or os.environ.get(name) or "").strip()
        if override:
            setattr(config, name, override)
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


def run_startup(*, need_api: bool = True, verify_models: bool = True, quiet: bool = False,
                test_call: bool = False) -> Session:
    """Run every startup check in order and return the session.

    `need_api=False` is used by commands that never call Claude
    (`--insights`, `--find`), so they work without a key.
    """
    env = load_env()

    master = store.load_master_resume()
    skills_status = store.load_skills_status()
    warnings = store.data_warnings(master, skills_status)

    llm: LLM | None = None
    if need_api:
        llm = LLM.from_key(read_api_key(env))
        if verify_models:
            llm.verify_models([config.FAST_MODEL, config.WRITE_MODEL])
        if test_call:
            llm.verify_access(config.FAST_MODEL)

    if not quiet:
        facts = sum(len(e.facts) for e in master.entries())
        ui.success(
            f"master resume: {len(master.experience)} experiences, {len(master.projects)} projects, "
            f"{facts} facts; {len(skills_status.skills)} skills tracked"
        )
        if llm is not None and verify_models:
            ui.success(f"models verified: {config.FAST_MODEL} (fast), {config.WRITE_MODEL} (write)")
        if llm is not None and test_call:
            ui.success("test request succeeded (the key can make calls and the account has credits)")
        for warning in warnings:
            ui.warn(warning)
    return Session(master=master, skills_status=skills_status, llm=llm, warnings=warnings)
