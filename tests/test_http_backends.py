"""The API-key backends (OpenAI, DeepSeek, Gemini): what they send, and failing fast when a site
can't be reached so Job Agent's next backend starts. No request leaves the machine."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests

import config
from src import backends
from src.backends import HttpJsonLLM
from src.errors import LLMError
from src.models import Judgment
from tests.conftest import make_judgment


class Post:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, json=None, timeout=None, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def ok(content: str):
    return SimpleNamespace(status_code=200, text="", json=lambda: {
        "choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def test_deepseek_sends_an_openai_style_request_to_deepseek_only(monkeypatch):
    monkeypatch.setenv("TAILOR_API_KEY", "test-key")
    post = Post(ok(json.dumps(make_judgment().model_dump())))
    monkeypatch.setattr(backends.requests, "post", post)
    llm = HttpJsonLLM("deepseek")
    out = llm.parse(purpose="judge", model=config.FAST_MODEL, system="s", user="u", schema=Judgment, max_tokens=20000)
    assert isinstance(out, Judgment)
    call = post.calls[0]
    assert call["url"] == "https://api.deepseek.com/chat/completions"
    assert call["headers"] == {"Authorization": "Bearer test-key"}
    assert call["json"]["model"] == "deepseek-chat" and call["json"]["response_format"] == {"type": "json_object"}
    assert call["json"]["max_tokens"] == 8192 and "max_completion_tokens" not in call["json"]
    assert call["timeout"][0] <= 15                       # can't connect: fail in seconds, not minutes


def test_an_unreachable_site_is_an_llm_error_without_the_key(monkeypatch):
    monkeypatch.setenv("TAILOR_API_KEY", "sk-secret-123")
    monkeypatch.setattr(backends.requests, "post", Post(requests.ConnectionError("reset by peer, key sk-secret-123")))
    with pytest.raises(LLMError) as err:
        HttpJsonLLM("deepseek").parse(purpose="judge", model=config.FAST_MODEL, system="s", user="u", schema=Judgment)
    assert "couldn't be reached" in str(err.value) and "sk-secret" not in str(err.value)


def test_deepseek_is_a_backend_you_can_pick(monkeypatch):
    from src import startup

    monkeypatch.setenv("TAILOR_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_BACKEND", "deepseek")
    assert startup.make_llm({}).provider == "deepseek"
