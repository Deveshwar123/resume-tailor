"""Running without API credits: the Claude Code (subscription) and Ollama (local) backends."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests

import config
from src import backends, startup
from src.backends import ClaudeCodeLLM, OllamaLLM
from src.errors import ConfigError, LLMError
from src.models import Judgment
from tests.conftest import make_judgment


class Fake:
    """Records subprocess.run calls and returns canned stdout."""

    def __init__(self, *stdouts: str, returncode: int = 0):
        self.stdouts = list(stdouts)
        self.calls: list[dict] = []
        self.returncode = returncode

    def __call__(self, args, **kwargs):
        self.calls.append({"args": args, **kwargs})
        return SimpleNamespace(stdout=self.stdouts.pop(0), stderr="", returncode=self.returncode)


@pytest.fixture
def claude_cli(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda name: f"/usr/local/bin/{name}")

    def install(*stdouts: str, returncode: int = 0) -> Fake:
        fake = Fake(*stdouts, returncode=returncode)
        monkeypatch.setattr(backends.subprocess, "run", fake)
        return fake

    return install


def result(structured=None, *, is_error=False, text="", usage=None) -> str:
    return json.dumps({"type": "result", "is_error": is_error, "result": text, "structured_output": structured,
                       "usage": usage or {"input_tokens": 1200, "output_tokens": 300, "cache_read_input_tokens": 0,
                                          "cache_creation_input_tokens": 0}})


# ---------------------------------------------------------------------------
# Claude Code (`claude -p`)
# ---------------------------------------------------------------------------


def test_claude_code_parse_sends_a_slim_structured_request(claude_cli):
    judgment = make_judgment(8, 7, 9, 6)
    fake = claude_cli(result(judgment.model_dump(mode="json")))
    llm = ClaudeCodeLLM()
    out = llm.parse(purpose="judge", model="claude-haiku-5-5", system="You judge resumes.", user="<resume>...</resume>",
                    schema=Judgment)
    assert out == judgment
    call = fake.calls[0]
    args = call["args"]
    assert args[:2] == ["claude", "-p"] and args[args.index("--model") + 1] == "claude-haiku-5-5"
    assert args[args.index("--system-prompt") + 1] == "You judge resumes."
    assert args[args.index("--tools") + 1] == ""                       # no tools: just answer
    assert args[args.index("--setting-sources") + 1] == "" and "--strict-mcp-config" in args
    assert "--no-session-persistence" in args and args[args.index("--effort") + 1] == config.EFFORT["judge"]
    schema = json.loads(args[args.index("--json-schema") + 1])
    assert schema["additionalProperties"] is False and "category_scores" in schema["required"]
    assert call["input"] == "<resume>...</resume>"
    assert llm.tracker.calls[0].input_tokens == 1200 and llm.billing == "subscription"


def test_claude_code_falls_back_to_the_result_text(claude_cli):
    claude_cli(result(None, text=json.dumps(make_judgment().model_dump(mode="json"))))
    assert ClaudeCodeLLM().parse(purpose="judge", model="m", system="s", user="u", schema=Judgment).category_scores


@pytest.mark.parametrize("message,expected", [
    ("Claude AI usage limit reached|1760000000", "usage limit has been reached"),
    ("Invalid API key · Please run /login", "not logged in"),
    ("something odd", "reported an error"),
])
def test_claude_code_errors_are_plain(claude_cli, message, expected):
    claude_cli(result(None, is_error=True, text=message))
    with pytest.raises(LLMError, match=expected):
        ClaudeCodeLLM().parse(purpose="judge", model="m", system="s", user="u", schema=Judgment)


def test_claude_code_missing_or_garbled(claude_cli, monkeypatch):
    claude_cli("not json at all")
    with pytest.raises(LLMError, match="something other than JSON"):
        ClaudeCodeLLM().parse(purpose="judge", model="m", system="s", user="u", schema=Judgment)
    monkeypatch.setattr(backends.shutil, "which", lambda name: None)
    with pytest.raises(ConfigError, match="was not found"):
        ClaudeCodeLLM().parse(purpose="judge", model="m", system="s", user="u", schema=Judgment)


def test_claude_code_research_collects_the_urls_its_searches_returned(claude_cli):
    stream = "\n".join(json.dumps(event) for event in [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebSearch", "input": {"query": "Acme"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": [
            {"type": "text", "text": 'Links: [{"title":"Careers","url":"https://acme.com/careers"},'
                                     '{"title":"News","url":"https://www.reuters.com/acme"}]'}]}]}},
        {"type": "result", "is_error": False, "result": "Acme builds rockets (https://acme.com/careers).",
         "usage": {"input_tokens": 5000, "output_tokens": 400}},
    ])
    fake = claude_cli(stream)
    found = ClaudeCodeLLM().search(purpose="research", model="claude-sonnet-5-5", system="s", user="Acme")
    args = fake.calls[0]["args"]
    assert args[args.index("--tools") + 1:args.index("--tools") + 3] == ["WebSearch", "WebFetch"]
    assert "stream-json" in args
    assert found.text.startswith("Acme builds rockets") and found.searches == 1
    assert [url for url, _ in found.sources] == ["https://acme.com/careers", "https://www.reuters.com/acme"]


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class FakeHTTP:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.posts: list[dict] = []

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(status_code=200, json=lambda: reply, text="")


def chat(content: str, prompt=900, generated=250) -> dict:
    return {"message": {"role": "assistant", "content": content}, "prompt_eval_count": prompt, "eval_count": generated}


def test_ollama_parse_uses_the_schema_and_retries_once(monkeypatch):
    good = json.dumps(make_judgment(7, 7, 7, 7).model_dump(mode="json"))
    http = FakeHTTP(chat('{"category_scores": "oops"}'), chat(good))
    monkeypatch.setattr(backends.requests, "post", http.post)
    monkeypatch.setattr(config, "OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setattr(config, "OLLAMA_FAST_MODEL", "qwen3:4b")
    llm = OllamaLLM()
    out = llm.parse(purpose="judge", model=config.FAST_MODEL, system="sys", user="usr", schema=Judgment)
    assert out.category_scores.impact.score == 7
    first = http.posts[0]["json"]
    assert http.posts[0]["url"].endswith("/api/chat") and first["model"] == "qwen3:4b"       # the fast part
    assert first["format"]["additionalProperties"] is False and first["think"] is False
    assert first["options"]["num_ctx"] == config.OLLAMA_NUM_CTX and first["stream"] is False
    retry = http.posts[1]["json"]["messages"]
    assert retry[-1]["role"] == "user" and "not valid" in retry[-1]["content"]                # told what went wrong
    assert llm.tracker.total_cost() == 0.0 and llm.billing == "local" and not llm.can_search
    assert OllamaLLM.local_model(config.WRITE_MODEL) == "qwen3:8b"


def test_ollama_gives_up_after_one_retry_and_reports_a_stopped_server(monkeypatch):
    monkeypatch.setattr(backends.requests, "post", FakeHTTP(chat("nope"), chat("still nope")).post)
    with pytest.raises(LLMError, match="did not match"):
        OllamaLLM().parse(purpose="judge", model="m", system="s", user="u", schema=Judgment)
    monkeypatch.setattr(backends.requests, "post", FakeHTTP(requests.ConnectionError("refused")).post)
    with pytest.raises(LLMError, match="Ollama is not running"):
        OllamaLLM().parse(purpose="judge", model="m", system="s", user="u", schema=Judgment)
    with pytest.raises(LLMError, match="cannot search"):
        OllamaLLM().search(purpose="research", model="m", system="s", user="u")


def test_ollama_verify_models_names_the_pull_command(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setattr(config, "OLLAMA_FAST_MODEL", "")
    monkeypatch.setattr(backends.requests, "get",
                        lambda url, timeout=None: SimpleNamespace(json=lambda: {"models": [{"name": "lfm2:latest"}]}))
    with pytest.raises(ConfigError) as info:
        OllamaLLM().verify_models([config.FAST_MODEL, config.WRITE_MODEL])
    assert "ollama pull qwen3:8b" in info.value.hint and "lfm2:latest" in info.value.hint
    monkeypatch.setattr(backends.requests, "get",
                        lambda url, timeout=None: SimpleNamespace(json=lambda: {"models": [{"name": "qwen3:8b"}]}))
    OllamaLLM().verify_models([config.FAST_MODEL, config.WRITE_MODEL])


# ---------------------------------------------------------------------------
# Choosing a backend
# ---------------------------------------------------------------------------


def test_backend_is_chosen_from_settings_and_the_shell_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LLM_BACKEND=claude_code\nOLLAMA_MODEL=qwen3:8b\n")
    monkeypatch.setattr(config, "ENV_PATH", env)
    for name in ("LLM_BACKEND", "OLLAMA_MODEL"):
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert isinstance(startup.make_llm(startup.load_env()), ClaudeCodeLLM)         # no API key needed
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    assert isinstance(startup.make_llm(startup.load_env()), OllamaLLM)             # one-off override from the shell
    monkeypatch.setenv("LLM_BACKEND", "carrier-pigeon")
    with pytest.raises(ConfigError, match="Unknown LLM_BACKEND"):
        startup.make_llm(startup.load_env())
