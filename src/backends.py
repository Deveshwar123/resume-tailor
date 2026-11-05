"""Alternative ways to run the Claude calls, with no API credits.

Choose one with `LLM_BACKEND=` in `.env`:

    anthropic     the Anthropic API with an API key (the spec's default; billed per token)
    claude_code   the Claude Code command line (`claude -p`), using the Claude subscription
                  you are logged into: the same Claude models, no API credits, counted
                  against your plan's usage limits
    ollama        a model running locally through Ollama: free and offline, weaker writing,
                  and no web research

All three expose the same methods as `src.llm.LLM` (`parse`, `search`,
`verify_models`, `verify_access`, `tracker`), so nothing else in the program
knows which one is in use. Whatever backend wrote a draft, `validate.py` still
checks every bullet against the fact bank in code.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
from typing import Any, TypeVar

import requests
from anthropic import transform_schema
from pydantic import BaseModel, TypeAdapter, ValidationError

import config
from src.errors import ConfigError, LLMError
from src.llm import SearchResult, UsageTracker

T = TypeVar("T", bound=BaseModel)
_URL_RE = re.compile(r"https?://[^\s\"'<>\])}]+")


def strict_schema(model: type[BaseModel]) -> dict:
    """The same closed JSON schema the Anthropic SDK would send for `model`."""
    return transform_schema(TypeAdapter(model).json_schema())


def _validate(schema: type[T], data: Any, purpose: str) -> T:
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise LLMError(f"The model's {purpose} output did not match the expected structure: {exc.errors()[:2]}") from exc


# ---------------------------------------------------------------------------
# Claude Code (`claude -p`) - the Claude subscription you are logged into
# ---------------------------------------------------------------------------


class ClaudeCodeLLM:
    """Claude through the Claude Code CLI in print mode.

    Each call runs `claude -p` with its own system prompt, no tools (except web
    search for research), no settings, memory, or MCP servers, and
    `--json-schema` for structured output. Nothing is saved as a session.
    """

    billing = "subscription"
    can_search = True

    def __init__(self, executable: str | None = None, tracker: UsageTracker | None = None) -> None:
        self.executable = executable or config.CLAUDE_CODE_BIN
        self.tracker = tracker or UsageTracker()

    def _base(self, model: str, system: str, purpose: str) -> list[str]:
        args = [self.executable, "-p", "--model", model, "--no-session-persistence",
                "--system-prompt", system, "--setting-sources", "", "--strict-mcp-config"]
        effort = config.EFFORT.get(purpose)
        if effort:
            args += ["--effort", effort]
        return args

    def _run(self, args: list[str], user: str, purpose: str) -> str:
        if shutil.which(self.executable) is None:
            raise ConfigError(f'The Claude Code command "{self.executable}" was not found.',
                              "Install Claude Code, or set LLM_BACKEND=ollama in .env.")
        try:
            # A neutral working folder, so no project instructions are picked up.
            result = subprocess.run(args, input=user, capture_output=True, text=True,
                                    timeout=config.CLAUDE_CODE_TIMEOUT_SECONDS, cwd=tempfile.gettempdir())
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"Claude Code took longer than {config.CLAUDE_CODE_TIMEOUT_SECONDS}s on the {purpose} step.") from exc
        if result.returncode != 0 and not result.stdout.strip():
            raise LLMError(f"Claude Code failed on the {purpose} step: {(result.stderr or 'no output').strip()[:300]}")
        return result.stdout

    @staticmethod
    def _check(data: dict, purpose: str) -> None:
        if not data.get("is_error"):
            return
        message = str(data.get("result") or data.get("subtype") or "unknown error")
        lowered = message.lower()
        if "usage limit" in lowered or "limit reached" in lowered or "rate limit" in lowered:
            raise LLMError("Your Claude subscription's usage limit has been reached.",
                           "Wait for it to reset, or set LLM_BACKEND=ollama in .env to use the local model.")
        if "login" in lowered or "authenticat" in lowered or "api key" in lowered:
            raise LLMError("Claude Code is not logged in.", "Run `claude` once and sign in, then try again.")
        raise LLMError(f"Claude Code reported an error on the {purpose} step: {message[:300]}")

    def _record(self, purpose: str, model: str, data: dict, seconds: float = 0.0) -> None:
        usage = data.get("usage") or {}
        self.tracker.record(purpose, model, SimpleNamespace(
            input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            server_tool_use=SimpleNamespace(web_search_requests=usage.get("server_tool_use", {}).get("web_search_requests", 0)
                                            if isinstance(usage.get("server_tool_use"), dict) else 0)), seconds)

    def parse(self, *, purpose: str, model: str, system: str, user: str, schema: type[T],
              cache_system: bool = True, max_tokens: int | None = None) -> T:
        args = self._base(model, system, purpose) + [
            "--tools", "", "--output-format", "json", "--json-schema", json.dumps(strict_schema(schema))]
        started = time.monotonic()
        raw = self._run(args, user, purpose)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Claude Code returned something other than JSON on the {purpose} step: {raw[:200]}") from exc
        self._check(data, purpose)
        self._record(purpose, model, data, time.monotonic() - started)
        structured = data.get("structured_output")
        if structured is None:
            try:
                structured = json.loads(data.get("result") or "")
            except json.JSONDecodeError as exc:
                raise LLMError(f"Claude Code returned no structured output for the {purpose} step.") from exc
        return _validate(schema, structured, purpose)

    def search(self, *, purpose: str, model: str, system: str, user: str, max_searches: int | None = None) -> SearchResult:
        """Research with Claude Code's own web search; sources come from the search results it saw."""
        args = self._base(model, system, purpose) + [
            "--tools", "WebSearch", "WebFetch", "--allowedTools", "WebSearch", "WebFetch",
            "--output-format", "stream-json", "--verbose"]
        started = time.monotonic()
        raw = self._run(args, user, purpose)
        text, final, sources, searches = "", None, [], 0
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                final = event
            message = event.get("message") or {}
            for block in message.get("content") or [] if isinstance(message.get("content"), list) else []:
                if block.get("type") == "tool_use" and block.get("name") == "WebSearch":
                    searches += 1
                if block.get("type") == "tool_result":
                    content = block.get("content")
                    blob = json.dumps(content) if not isinstance(content, str) else content
                    for url in _URL_RE.findall(blob):
                        url = url.rstrip(".,;:\\")
                        if url not in [u for u, _t in sources]:
                            sources.append((url, ""))
        if final is None:
            raise LLMError("Claude Code returned no result for the research step.")
        self._check(final, purpose)
        self._record(purpose, model, final, time.monotonic() - started)
        text = str(final.get("result") or "")
        return SearchResult(text=text.strip(), sources=sources, searches=searches)

    def verify_models(self, model_ids: list[str]) -> None:
        for model_id in dict.fromkeys(model_ids):
            self.verify_access(model_id)

    def verify_access(self, model: str) -> None:
        probe = type("Probe", (BaseModel,), {"__annotations__": {"ok": bool}})
        try:
            self.parse(purpose="check", model=model, system="Answer with ok = true.", user="Are you there?", schema=probe)
        except LLMError as exc:
            raise ConfigError(f"Claude Code could not run {model}: {exc.message}", exc.hint) from exc


# ---------------------------------------------------------------------------
# Ollama - a local model
# ---------------------------------------------------------------------------


class OllamaLLM:
    """A model served by Ollama on this machine (or an Ollama cloud model).

    Structured output uses Ollama's `format` (a JSON schema). Web research is
    not available, so research is skipped with this backend.
    """

    billing = "local"
    can_search = False

    def __init__(self, host: str | None = None, tracker: UsageTracker | None = None) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.tracker = tracker or UsageTracker()

    @staticmethod
    def local_model(model: str) -> str:
        """Map the configured Claude model for a step to the local model that plays its part."""
        if model == config.FAST_MODEL and config.OLLAMA_FAST_MODEL:
            return config.OLLAMA_FAST_MODEL
        return config.OLLAMA_MODEL

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        try:
            response = requests.post(f"{self.host}{path}", json=payload, timeout=timeout)
        except requests.ConnectionError as exc:
            raise LLMError("Ollama is not running.", "Open the Ollama app (or run `ollama serve`), then try again.") from exc
        except requests.Timeout as exc:
            raise LLMError(f"The local model took longer than {timeout:.0f}s.", "Try a smaller model.") from exc
        if response.status_code == 404:
            raise LLMError(f'Ollama does not have the model "{payload.get("model")}".',
                           f"Run: ollama pull {payload.get('model')}")
        if response.status_code >= 400:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
        return response.json()

    def parse(self, *, purpose: str, model: str, system: str, user: str, schema: type[T],
              cache_system: bool = True, max_tokens: int | None = None) -> T:
        local = self.local_model(model)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = ""
        for attempt in range(2):          # one retry with the parse error, small models slip sometimes
            started = time.monotonic()
            data = self._post("/api/chat", {
                "model": local, "messages": messages, "stream": False, "think": False,
                "format": strict_schema(schema),
                "options": {"num_ctx": config.OLLAMA_NUM_CTX, "temperature": 0.2},
            }, timeout=config.OLLAMA_TIMEOUT_SECONDS)
            self.tracker.record(purpose, local, SimpleNamespace(
                input_tokens=data.get("prompt_eval_count", 0), output_tokens=data.get("eval_count", 0),
                cache_read_input_tokens=0, cache_creation_input_tokens=0, server_tool_use=None),
                time.monotonic() - started)
            content = (data.get("message") or {}).get("content") or ""
            try:
                return schema.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)[:300]
                messages += [{"role": "assistant", "content": content},
                             {"role": "user", "content": f"That was not valid for the required JSON schema ({last_error}). "
                                                         "Return the complete JSON again, exactly matching the schema."}]
        raise LLMError(f"The local model's {purpose} output did not match the expected structure: {last_error}")

    def search(self, **_kwargs) -> SearchResult:
        raise LLMError("The local model cannot search the web, so company research is skipped.")

    def verify_models(self, model_ids: list[str]) -> None:
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=10).json()
        except requests.RequestException as exc:
            raise ConfigError("Ollama is not running.", "Open the Ollama app (or run `ollama serve`), then try again.") from exc
        have = {m.get("name", "") for m in tags.get("models", [])}
        wanted = {self.local_model(m) for m in model_ids}
        missing = [m for m in wanted if m not in have and f"{m}:latest" not in have]
        if missing:
            raise ConfigError("Ollama does not have: " + ", ".join(missing),
                              "Download it with: " + " && ".join(f"ollama pull {m}" for m in missing)
                              + "\nInstalled: " + (", ".join(sorted(have)) or "none"))

    def verify_access(self, model: str) -> None:
        probe = type("Probe", (BaseModel,), {"__annotations__": {"ok": bool}})
        try:
            self.parse(purpose="check", model=model, system="Answer with ok = true.", user="Are you there?", schema=probe)
        except LLMError as exc:
            raise ConfigError(f"The local model could not answer: {exc.message}", exc.hint) from exc


class HttpJsonLLM:
    """OpenAI, DeepSeek or Google Gemini over HTTPS, with the key in TAILOR_API_KEY (Job Agent passes it from
    the macOS Keychain; it is never written to a file or a log). Replies are JSON checked against the
    schema, with one retry like the local model. No web research: research is skipped."""

    billing = "api"
    can_search = False
    URLS = {"openai": "https://api.openai.com/v1/chat/completions",
            "deepseek": "https://api.deepseek.com/chat/completions",      # OpenAI-compatible
            "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"}

    def __init__(self, provider: str, tracker: UsageTracker | None = None) -> None:
        self.provider = provider
        self.key = os.environ.get("TAILOR_API_KEY", "")
        if not self.key:
            raise LLMError(f"No API key for {provider}.", "Job Agent passes it as TAILOR_API_KEY.")
        self.tracker = tracker or UsageTracker()

    def model_for(self, model: str) -> str:
        fast = model == config.FAST_MODEL
        if self.provider == "openai":
            return os.environ.get("OPENAI_FAST_MODEL" if fast else "OPENAI_MODEL") or ("gpt-4.1-mini" if fast else "gpt-4.1")
        if self.provider == "deepseek":
            return os.environ.get("DEEPSEEK_FAST_MODEL" if fast else "DEEPSEEK_MODEL") or "deepseek-chat"
        return os.environ.get("GEMINI_FAST_MODEL" if fast else "GEMINI_MODEL") or ("gemini-flash-latest" if fast else "gemini-pro-latest")

    def _ask(self, model: str, system: str, messages: list[dict], schema: dict, max_tokens: int | None) -> tuple[str, int, int]:
        if self.provider in ("openai", "deepseek"):
            body = {"model": model, "messages": [{"role": "system", "content": system}, *messages],
                    "response_format": {"type": "json_object"}, "temperature": 0.2}
            if max_tokens and self.provider == "openai":
                body["max_completion_tokens"] = max_tokens
            elif self.provider == "deepseek":
                body["max_tokens"] = min(max_tokens or 8192, 8192)          # deepseek-chat's output limit
            try:   # a site that can't be reached fails in 15 s, so the next backend starts quickly
                r = requests.post(self.URLS[self.provider], json=body, timeout=(15, 300),
                                  headers={"Authorization": f"Bearer {self.key}"})
            except requests.RequestException as exc:
                raise LLMError(f"{self.provider} couldn't be reached ({type(exc).__name__}).") from None
        else:
            return self._gemini(model, system, messages, max_tokens)
        if r.status_code != 200:
            raise LLMError(f"{self.provider} returned HTTP {r.status_code}: {r.text[:200]}")
        d = r.json()
        u = d.get("usage") or {}
        return d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    _working: dict[str, str] = {}      # wanted model -> the Gemini model that answered (busy/quota ones skipped)

    def _gemini(self, model: str, system: str, messages: list[dict], max_tokens: int | None) -> tuple[str, int, int]:
        chain = [model, "gemini-flash-latest", "gemini-3.5-flash", "gemini-flash-lite-latest"]
        chain = list(dict.fromkeys([self._working.get(model, model), *chain]))
        last = ""
        for name in chain:
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                                 for m in messages],
                    "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2,
                                         **({"maxOutputTokens": max_tokens} if max_tokens else {})}}
            r = requests.post(self.URLS["gemini"].format(model=name), json=body, timeout=300,
                              headers={"x-goog-api-key": self.key})
            if r.status_code in (429, 503):               # busy or rate-limited: often gone in seconds, once more
                time.sleep(10)
                r = requests.post(self.URLS["gemini"].format(model=name), json=body, timeout=300,
                                  headers={"x-goog-api-key": self.key})
            if r.status_code in (404, 429, 500, 503):     # not for this key, out of quota, or busy: next model
                last = f"{name}: HTTP {r.status_code}"
                continue
            if r.status_code != 200:
                raise LLMError(f"gemini returned HTTP {r.status_code}: {r.text[:200]}")
            self._working[model] = name
            d = r.json()
            u = d.get("usageMetadata") or {}
            parts = ((d.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts), u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
        raise LLMError(f"No Gemini model answered ({last}).")

    def parse(self, *, purpose: str, model: str, system: str, user: str, schema: type[T],
              cache_system: bool = True, max_tokens: int | None = None) -> T:
        name = self.model_for(model)
        shape = json.dumps(strict_schema(schema))
        system = f"{system}\n\nReply with JSON only, matching this JSON schema exactly:\n{shape}"
        messages = [{"role": "user", "content": user}]
        last_error = ""
        for _attempt in range(2):
            started = time.monotonic()
            content, tin, tout = self._ask(name, system, messages, strict_schema(schema), max_tokens)
            self.tracker.record(purpose, name, SimpleNamespace(
                input_tokens=tin, output_tokens=tout, cache_read_input_tokens=0, cache_creation_input_tokens=0,
                server_tool_use=None), time.monotonic() - started)
            try:
                return schema.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)[:300]
                messages += [{"role": "assistant", "content": content},
                             {"role": "user", "content": f"That was not valid for the required JSON schema ({last_error}). "
                                                         "Return the complete JSON again, exactly matching the schema."}]
        raise LLMError(f"{self.provider}'s {purpose} output did not match the expected structure: {last_error}")

    def search(self, **_kwargs) -> SearchResult:
        raise LLMError(f"{self.provider} research is off here, so company research is skipped.")

    def verify_models(self, model_ids: list[str]) -> None:
        return None
