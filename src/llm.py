"""Thin wrapper around the Anthropic SDK.

Every Claude call in the project goes through `LLM`:

* `LLM.parse()`  - one structured-output call, validated into a pydantic model
* `LLM.search()` - one research call using the server-side web search tool
* `LLM.verify_models()` - startup check that the configured model IDs exist

The wrapper owns the cross-cutting concerns the spec asks for: retries with
exponential backoff (delegated to the SDK), refusal handling, token usage
logging, run-cost accounting, and friendly errors instead of stack traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

import config
from src.errors import ConfigError, LLMError, LLMRefusal
from src.models import CallUsage

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Usage and cost
# ---------------------------------------------------------------------------


def _rates(model: str, prompt_tokens: int) -> tuple[float, float]:
    """(input, output) USD per million tokens for `model`."""
    best: tuple[float, float] | None = None
    best_len = -1
    for prefix, rates in config.PRICING.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best, best_len = rates, len(prefix)
    if prompt_tokens > config.LONG_PROMPT_THRESHOLD:
        for prefix, rates in config.LONG_PROMPT_PRICING.items():
            if model.startswith(prefix):
                best = rates
    return best or (0.0, 0.0)


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  cache_read: int = 0, cache_write: int = 0, web_searches: int = 0) -> float:
    in_rate, out_rate = _rates(model, input_tokens + cache_read + cache_write)
    tokens_cost = (
        input_tokens * in_rate
        + cache_read * in_rate * config.CACHE_READ_MULTIPLIER
        + cache_write * in_rate * config.CACHE_WRITE_MULTIPLIER
        + output_tokens * out_rate
    ) / 1_000_000
    return tokens_cost + web_searches * config.WEB_SEARCH_COST_PER_SEARCH


@dataclass
class UsageTracker:
    """Collects token usage for every call so the run can report its cost."""

    calls: list[CallUsage] = field(default_factory=list)

    def record(self, purpose: str, model: str, usage: Any) -> CallUsage:
        server_tools = getattr(usage, "server_tool_use", None)
        call = CallUsage(
            purpose=purpose,
            model=model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            web_searches=int(getattr(server_tools, "web_search_requests", 0) or 0),
        )
        call.cost_usd = estimate_cost(
            model, call.input_tokens, call.output_tokens,
            call.cache_read_tokens, call.cache_write_tokens, call.web_searches,
        )
        self.calls.append(call)
        return call

    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(c.input_tokens + c.cache_read_tokens + c.cache_write_tokens for c in self.calls),
            sum(c.output_tokens for c in self.calls),
        )

    def by_purpose(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for call in self.calls:
            row = out.setdefault(call.purpose, {"calls": 0, "input": 0, "output": 0, "cost": 0.0})
            row["calls"] += 1
            row["input"] += call.input_tokens + call.cache_read_tokens + call.cache_write_tokens
            row["output"] += call.output_tokens
            row["cost"] += call.cost_usd
        return out


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    text: str                                   # Claude's research notes
    sources: list[tuple[str, str]]              # (url, title) actually returned by the search tool
    searches: int = 0
    errors: list[str] = field(default_factory=list)


def _walk(node: Any, found: list[tuple[str, str]], errors: list[str]) -> None:
    """Collect every web search result / citation URL nested anywhere in a response."""
    if isinstance(node, dict):
        kind = node.get("type")
        if kind in ("web_search_result", "web_search_result_location") and node.get("url"):
            found.append((str(node["url"]), str(node.get("title") or "")))
        if kind == "web_search_tool_result_error" and node.get("error_code"):
            errors.append(str(node["error_code"]))
        for value in node.values():
            _walk(value, found, errors)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _walk(value, found, errors)


def _block_dicts(content: Any) -> list[dict]:
    out = []
    for block in content or []:
        if isinstance(block, dict):
            out.append(block)
        elif hasattr(block, "model_dump"):
            out.append(block.model_dump())
    return out


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


BILLING_HINT = ("Add credits under Plans & Billing in the Claude Console (https://platform.claude.com), "
                "then run `python tailor.py --check`. API usage is billed separately from a Claude.ai subscription.")


def api_message(exc: Exception) -> str:
    """The API's own error message, without the SDK's "Error code: 400 - {...}" wrapper."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if body.get("message"):
            return str(body["message"])
    return str(getattr(exc, "message", "") or exc)


def is_out_of_credits(exc: Exception) -> bool:
    text = api_message(exc).lower()
    return "credit balance" in text or getattr(exc, "type", None) == "billing_error" or getattr(exc, "status_code", None) == 402


class LLM:
    """All Claude access for the project."""

    def __init__(self, client: Any, tracker: UsageTracker | None = None) -> None:
        self.client = client
        self.tracker = tracker or UsageTracker()
        self._fallback_ok = config.ENABLE_REFUSAL_FALLBACK

    @classmethod
    def from_key(cls, api_key: str) -> "LLM":
        client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=config.API_MAX_RETRIES,      # exponential backoff on 429 / 5xx / 529
            timeout=config.API_TIMEOUT_SECONDS,
        )
        return cls(client)

    # -- internals ---------------------------------------------------------
    def _wants_fallback(self, model: str) -> bool:
        return self._fallback_ok and model.startswith(config.FALLBACK_CAPABLE_PREFIXES)

    def _request(self, method: str, model: str, kwargs: dict[str, Any]) -> Any:
        """Call messages.<method>, mapping SDK errors to friendly `LLMError`s.

        WRITE-tier models opt into Anthropic's server-side refusal fallback. If
        the account rejects that beta, it is switched off for the session and
        the request is sent again without it.
        """
        try:
            if self._wants_fallback(model):
                try:
                    return getattr(self.client.beta.messages, method)(
                        betas=[config.FALLBACK_BETA], fallbacks="default", **kwargs
                    )
                except anthropic.BadRequestError as exc:
                    if "fallback" not in str(exc).lower():
                        raise
                    self._fallback_ok = False
            return getattr(self.client.messages, method)(**kwargs)
        except ValidationError as exc:
            raise LLMError(f"Claude returned data that does not match the expected schema: {exc.errors()[:3]}") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("The Anthropic API rejected the API key.", "Check ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(f"The API key is not allowed to use {model}: {api_message(exc)}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f'Model "{model}" was not found.', "Set FAST_MODEL / WRITE_MODEL in config.py or .env.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(
                f"Rate limited by the Anthropic API (gave up after {config.API_MAX_RETRIES} retries).",
                "Wait a minute and run the command again.",
            ) from exc
        except anthropic.BadRequestError as exc:
            if is_out_of_credits(exc):
                raise LLMError("Your Anthropic API account is out of credits.", BILLING_HINT) from exc
            raise LLMError(f"The Anthropic API rejected the request: {api_message(exc)}") from exc
        except anthropic.APIStatusError as exc:
            if is_out_of_credits(exc):
                raise LLMError("Your Anthropic API account is out of credits.", BILLING_HINT) from exc
            kind = "overloaded" if exc.status_code == 529 else "unavailable"
            raise LLMError(
                f"The Anthropic API is {kind} (HTTP {exc.status_code}; gave up after {config.API_MAX_RETRIES} retries).",
                "Try again shortly.",
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("The request to the Anthropic API timed out.", "Check your connection and retry.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API.", "Check your internet connection.") from exc

    @staticmethod
    def _system_blocks(system: str, cache: bool) -> list[dict[str, Any]]:
        block: dict[str, Any] = {"type": "text", "text": system}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}   # reused across revision rounds
        return [block]

    def _check_stop(self, response: Any, purpose: str) -> None:
        reason = getattr(response, "stop_reason", None)
        if reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise LLMRefusal(
                f"Claude declined the {purpose} request (category: {category}).",
                "Nothing was written. Try --text with a cleaned-up job description.",
            )
        if reason == "max_tokens":
            raise LLMError(
                f"Claude's {purpose} response was cut off at max_tokens.",
                "Raise MAX_TOKENS in config.py or lower the effort for this step.",
            )

    # -- public API --------------------------------------------------------
    def parse(self, *, purpose: str, model: str, system: str, user: str,
              schema: type[T], cache_system: bool = True, max_tokens: int | None = None) -> T:
        """One structured-output call. Returns a validated instance of `schema`."""
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or config.MAX_TOKENS,
            "system": self._system_blocks(system, cache_system),
            "messages": [{"role": "user", "content": user}],
            "output_format": schema,
        }
        effort = config.EFFORT.get(purpose)
        if effort:
            kwargs["output_config"] = {"effort": effort}
        response = self._request("parse", model, kwargs)
        self.tracker.record(purpose, getattr(response, "model", model) or model, getattr(response, "usage", None))
        self._check_stop(response, purpose)
        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise LLMError(f"Claude returned no structured output for the {purpose} request.")
        if not isinstance(parsed, schema):
            try:
                parsed = schema.model_validate(parsed)
            except ValidationError as exc:
                raise LLMError(f"Claude's {purpose} output failed validation: {exc.errors()[:3]}") from exc
        return parsed

    def search(self, *, purpose: str, model: str, system: str, user: str,
               max_searches: int | None = None) -> SearchResult:
        """One research call with the server-side web search tool.

        Handles `pause_turn` (the server-side tool loop hit its iteration limit)
        by re-sending the conversation so the server resumes where it stopped.
        """
        tool = {
            "type": config.WEB_SEARCH_TOOL_TYPE,
            "name": "web_search",
            "max_uses": max_searches or config.MAX_RESEARCH_SEARCHES,
        }
        first_message = {"role": "user", "content": user}
        messages: list[dict[str, Any]] = [first_message]
        texts: list[str] = []
        sources: list[tuple[str, str]] = []
        errors: list[str] = []
        searches = 0
        for _ in range(config.MAX_PAUSE_CONTINUATIONS + 1):
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": config.MAX_TOKENS,
                "system": self._system_blocks(system, cache=False),
                "messages": messages,
                "tools": [tool],
            }
            effort = config.EFFORT.get(purpose)
            if effort:
                kwargs["output_config"] = {"effort": effort}
            response = self._request("create", model, kwargs)
            call = self.tracker.record(purpose, getattr(response, "model", model) or model, getattr(response, "usage", None))
            searches += call.web_searches
            self._check_stop(response, purpose)
            blocks = _block_dicts(response.content)
            texts.extend(b.get("text", "") for b in blocks if b.get("type") == "text")
            _walk(blocks, sources, errors)
            if response.stop_reason != "pause_turn":
                break
            # Resume: same user turn plus the paused assistant turn, nothing added.
            messages = [first_message, {"role": "assistant", "content": response.content}]
        unique: dict[str, str] = {}
        for url, title in sources:
            unique.setdefault(url, title)
        return SearchResult("".join(texts).strip(), list(unique.items()), searches, sorted(set(errors)))

    def verify_access(self, model: str) -> None:
        """One tiny request, so a key that lists models but cannot make calls fails here.

        Costs a few tokens. Used by `--check` only.
        """
        try:
            self.client.messages.create(
                model=model, max_tokens=64, output_config={"effort": "low"},
                messages=[{"role": "user", "content": "Reply with the word ok."}],
            )
        except anthropic.AuthenticationError as exc:
            raise ConfigError("The Anthropic API rejected the API key.", "Check ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.APIStatusError as exc:
            if is_out_of_credits(exc):
                raise ConfigError("Your Anthropic API account is out of credits.", BILLING_HINT) from exc
            raise ConfigError(f"A test request to {model} failed (HTTP {exc.status_code}): {api_message(exc)}") from exc
        except anthropic.APIConnectionError as exc:
            raise ConfigError("Could not reach the Anthropic API.", "Check your internet connection.") from exc

    def verify_models(self, model_ids: list[str]) -> None:
        """Fail fast when a configured model ID is not available to this key."""
        missing: list[str] = []
        try:
            for model_id in dict.fromkeys(model_ids):
                try:
                    self.client.models.retrieve(model_id)
                except (anthropic.NotFoundError, anthropic.PermissionDeniedError):
                    missing.append(model_id)
            if missing:
                available = sorted(m.id for m in self.client.models.list())
                listing = "\n".join(f"  - {m}" for m in available) or "  (none returned)"
                raise ConfigError(
                    "These configured models are not available: " + ", ".join(missing),
                    "Available model IDs:\n" + listing + "\nSet FAST_MODEL / WRITE_MODEL in config.py or .env.",
                )
        except anthropic.AuthenticationError as exc:
            raise ConfigError("The Anthropic API rejected the API key.", "Check ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.APIStatusError as exc:
            raise ConfigError(f"Could not verify models (HTTP {exc.status_code}): {api_message(exc)}") from exc
        except anthropic.APIConnectionError as exc:
            raise ConfigError("Could not reach the Anthropic API to verify models.", "Check your internet connection.") from exc
