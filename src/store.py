"""Loading and saving the two YAML files I maintain by hand.

* `master_resume.yaml` - the verified fact bank
* `skills_status.yaml` - my status per skill

Loading validates against the pydantic schemas and turns any problem into a
readable `MasterResumeError`. Saving always makes a timestamped backup first
and keeps the leading comment block of the file intact.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

import config
from src import textutil
from src.errors import MasterResumeError
from src.models import MasterResume, SkillsStatus, SkillStatus


def _read_yaml(path: Path, label: str) -> Any:
    if not path.exists():
        raise MasterResumeError(f"{label} not found at {path}", f"Create {path.name} (see README.md).")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        raise MasterResumeError(f"{path.name} is not valid YAML{where}: {getattr(exc, 'problem', exc)}") from exc


def _format_validation(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "(file)"
        message = err["msg"].removeprefix("Value error, ")
        # The integrity validator joins several problems with "; ".
        for part in message.split("; "):
            lines.append(f"  - {location}: {part}" if location != "(file)" else f"  - {part}")
    return "\n".join(lines)


def load_master_resume(path: Path | None = None) -> MasterResume:
    path = path or config.MASTER_RESUME_PATH
    data = _read_yaml(path, "Master resume")
    if not isinstance(data, dict):
        raise MasterResumeError(f"{path.name} must be a YAML mapping at the top level.")
    try:
        return MasterResume.model_validate(data)
    except ValidationError as exc:
        raise MasterResumeError(
            f"{path.name} failed validation:\n{_format_validation(exc)}",
            "Fix the entries above; every id must be unique and every bullet must cite facts of its own entry.",
        ) from exc


def load_skills_status(path: Path | None = None) -> SkillsStatus:
    """Accepts either grouped lists (verified: [...]) or a flat `skills:` mapping."""
    path = path or config.SKILLS_STATUS_PATH
    data = _read_yaml(path, "Skills status file")
    if not isinstance(data, dict):
        raise MasterResumeError(f"{path.name} must be a YAML mapping at the top level.")
    flat: dict[str, str] = {}
    conflicts: list[str] = []

    def put(name: Any, status: Any) -> None:
        name, status = str(name).strip(), str(status).strip().lower()
        key = textutil.canon(name)
        existing = next((n for n in flat if textutil.canon(n) == key), None)
        if existing and flat[existing] != status:
            conflicts.append(f'"{name}" is listed as both {flat[existing]} and {status}')
        elif not existing:
            flat[name] = status

    for status in SkillStatus:
        for name in data.get(status.value) or []:
            put(name, status.value)
    for name, status in (data.get("skills") or {}).items():
        put(name, status)
    if conflicts:
        raise MasterResumeError(f"{path.name} has conflicting entries: " + "; ".join(conflicts))
    try:
        return SkillsStatus(skills=flat)  # type: ignore[arg-type]
    except ValidationError as exc:
        raise MasterResumeError(
            f"{path.name} failed validation:\n{_format_validation(exc)}",
            "Each skill must be verified, learning, or missing.",
        ) from exc


def data_warnings(master: MasterResume, status: SkillsStatus) -> list[str]:
    """Non-fatal inconsistencies worth showing at startup."""
    warnings: list[str] = []
    if not master.contact.name or not master.contact.email or not master.contact.phone:
        warnings.append("contact is missing a name, email, or phone (ATS checks will fail).")
    if not master.education:
        warnings.append("no education entries.")
    for skill in master.skills.all():
        if not status.is_verified(skill):
            warnings.append(
                f'skill "{skill}" is in the master skills list but is not "verified" in '
                f"{config.SKILLS_STATUS_PATH.name}; it will be kept off the resume."
            )
    unverified: dict[str, list[str]] = {}
    for entry in master.entries():
        for fact in entry.facts:
            for skill in fact.skills:
                if not status.is_verified(skill):
                    unverified.setdefault(skill, []).append(fact.id)
    for skill, fact_ids in unverified.items():
        warnings.append(
            f'facts {", ".join(fact_ids[:3])}{"..." if len(fact_ids) > 3 else ""} cite skill "{skill}", '
            f'which is not "verified"; bullets naming it will be rejected.'
        )
    return warnings


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def backup_file(path: Path, backup_dir: Path | None = None) -> Path:
    """Copy `path` to data/backups/<name>.<timestamp>.yaml and return the copy."""
    backup_dir = backup_dir or config.BACKUP_DIR
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"{path.stem}.{stamp}{path.suffix}"
    counter = 1
    while target.exists():
        target = backup_dir / f"{path.stem}.{stamp}-{counter}{path.suffix}"
        counter += 1
    shutil.copy2(path, target)
    return target


def _leading_comments(path: Path) -> str:
    """The comment / blank lines at the top of a file, kept across rewrites."""
    if not path.exists():
        return ""
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            kept.append(line)
        else:
            break
    return ("\n".join(kept).rstrip() + "\n\n") if any(k.strip() for k in kept) else ""


def _prune(value: Any) -> Any:
    """Drop None values so optional fact fields do not clutter the YAML."""
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def save_master_resume(master: MasterResume, path: Path | None = None, backup: bool = True) -> Path | None:
    """Write the master resume. Returns the backup path (if one was made)."""
    path = path or config.MASTER_RESUME_PATH
    backup_path = backup_file(path) if backup and path.exists() else None
    header = _leading_comments(path)
    body = yaml.safe_dump(
        _prune(master.model_dump(mode="json")),
        sort_keys=False, allow_unicode=True, width=100, default_flow_style=False,
    )
    path.write_text(header + body, encoding="utf-8")
    return backup_path


def save_skills_status(status: SkillsStatus, path: Path | None = None, backup: bool = True) -> Path | None:
    path = path or config.SKILLS_STATUS_PATH
    backup_path = backup_file(path) if backup and path.exists() else None
    header = _leading_comments(path)
    grouped = {s.value: sorted(status.names_with(s), key=str.lower) for s in SkillStatus}
    body = yaml.safe_dump(grouped, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(header + body, encoding="utf-8")
    return backup_path
