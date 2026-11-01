"""--auto / tailor_for_job(): never prompts, writes report.json, restores global state."""

import json
from pathlib import Path

import pytest

import config
import tailor
from src import auto, loop, startup, ui
from src.errors import TailorError


@pytest.fixture
def fake_run(monkeypatch):
    seen = {}

    def run_startup(**kwargs):
        seen["paths"] = kwargs.get("paths")
        return "session"

    def run_job(session, *, url=None, posting_text=None, options=None, out_dir=None, **_):
        seen.update(url=url, posting_text=posting_text, options=options, out_dir=out_dir)
        # Inside the run nothing can wait for a person: prompts return their default at once.
        seen["interactive"] = ui.is_interactive()
        seen["answer"] = ui.ask("Continue?", "default")
        (out_dir / "report.json").write_text(json.dumps({"ok": True, "out_dir": str(out_dir),
                                                         "files": {"resume_pdf": str(out_dir / "resume.pdf")}}))
        return None

    monkeypatch.setattr(startup, "run_startup", run_startup)
    monkeypatch.setattr(loop, "run_job", run_job)
    return seen


def test_link_runs_with_auto_options_and_returns_the_report(fake_run, tmp_path):
    before = config.use_paths()
    result = auto.tailor_for_job("https://job-boards.greenhouse.io/acme/jobs/1", tmp_path / "out",
                                 master_path=tmp_path / "m.yaml")
    assert result["ok"] and result["files"]["resume_pdf"].endswith("resume.pdf")
    assert fake_run["url"] == "https://job-boards.greenhouse.io/acme/jobs/1" and fake_run["posting_text"] is None
    opts = fake_run["options"]
    assert (opts.auto, opts.interview, opts.cover_letter) == (True, False, True)
    assert fake_run["paths"]["master"] == tmp_path / "m.yaml"
    assert (fake_run["interactive"], fake_run["answer"]) == (False, "default")
    assert config.use_paths() == before                        # paths restored afterwards


def test_posting_text_is_passed_as_text(fake_run, tmp_path):
    auto.tailor_for_job("Software Engineer Intern at Acme. Python required.", tmp_path / "out")
    assert fake_run["url"] is None and fake_run["posting_text"].startswith("Software Engineer Intern")


def test_expected_failure_returns_ok_false_and_writes_the_report(monkeypatch, tmp_path):
    def run_startup(**kwargs):
        raise TailorError("Your Claude subscription's usage limit has been reached.", "Wait for it to reset.")
    monkeypatch.setattr(startup, "run_startup", run_startup)
    result = auto.tailor_for_job("https://x.example/job", tmp_path / "out")
    assert result["ok"] is False and "usage limit" in result["error"]
    assert json.loads(Path(result["files"]["report_json"]).read_text())["ok"] is False


def test_cli_auto(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(auto, "tailor_for_job", lambda source, out: calls.append((source, out)) or {"ok": True, "out_dir": str(out)})
    assert tailor.main(["https://job-boards.greenhouse.io/acme/jobs/1", "--auto", "--out", str(tmp_path)]) == 0
    assert calls == [("https://job-boards.greenhouse.io/acme/jobs/1", tmp_path)]
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert tailor.main(["--auto"]) == 1                        # needs a link or --text


def test_reexported_for_apps():
    from tailor import tailor_for_job
    assert tailor_for_job is auto.tailor_for_job
