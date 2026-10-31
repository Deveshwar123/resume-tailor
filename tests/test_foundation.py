"""Milestone 1: schemas, text helpers, data loading, startup checks, Claude wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import BaseModel, ValidationError

import config
from src import startup, store, textutil
from src.errors import ConfigError, LLMError, LLMRefusal, MasterResumeError
from src.llm import LLM, estimate_cost
from src.models import DateRange, MasterResume, Ownership, SkillStatus

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def test_sample_master_validates(master):
    assert master.entry("exp_acme").org == "Acme Corp"
    entry, fact = master.fact_index()["exp_acme_f02"]
    assert entry.id == "exp_acme" and "30%" in fact.text


def test_fact_ownership_overrides_role(master):
    club = master.entry("exp_club")
    assert master.effective_ownership(club, club.facts[0]) == Ownership.lead
    acme = master.entry("exp_acme")
    assert master.effective_ownership(acme, acme.facts[1]) == Ownership.contributor  # falls back to role


def test_duplicate_ids_are_rejected(master_data):
    master_data["experience"][1]["facts"][0]["id"] = "exp_acme_f01"
    with pytest.raises(ValidationError, match="duplicate id"):
        MasterResume.model_validate(master_data)


def test_bullet_must_cite_facts_of_its_own_entry(master_data):
    master_data["experience"][0]["bullets"][0]["fact_ids"] = ["exp_club_f01"]
    with pytest.raises(ValidationError, match="not a fact of exp_acme"):
        MasterResume.model_validate(master_data)


def test_metric_must_appear_in_fact_text(master_data):
    master_data["experience"][0]["facts"][0]["metrics"] = ["12-person"]
    with pytest.raises(ValidationError, match="does not appear in the fact text"):
        MasterResume.model_validate(master_data)


@pytest.mark.parametrize("value", ["2024", "2024-09", "present", "Present", ""])
def test_valid_dates(value):
    assert DateRange(start=value, end=value)


@pytest.mark.parametrize("value", ["Sept 2024", "2024-13", "24-09", "09/2024"])
def test_invalid_dates(value):
    with pytest.raises(ValidationError):
        DateRange(start=value, end="present")


def test_skill_status_is_synonym_aware(status):
    assert status.is_verified("python3")
    assert status.status_of("Structured Query Language") == SkillStatus.verified
    assert status.status_of("pytorch") == SkillStatus.learning
    assert status.status_of("Kubernetes") is None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def test_terms_respect_word_boundaries():
    vocab = ["Java", "JavaScript", "C", "C++", "Git", "GitHub", "GitHub Pages", "machine learning", "R"]
    found = textutil.find_terms("Built in JavaScript and C++, hosted on GitHub Pages, with ML models", vocab)
    assert set(found) == {"javascript", "c++", "github pages", "machine learning"}


def test_short_terms_match_case_sensitively():
    assert textutil.contains_term("Modeled returns in R and Excel", "R")
    assert not textutil.contains_term("our research was thorough", "R")
    assert not textutil.contains_term("students who excel at math", "Excel")


def test_synonyms_canonicalize():
    assert textutil.canon("ML") == "machine learning"
    assert textutil.canon("Postgres") == "postgresql"
    assert textutil.canon("Full-Stack") == textutil.canon("full stack")
    assert textutil.canon("scikit learn") == "scikit-learn"
    assert textutil.contains_term("Trained ML models", "machine learning")
    assert textutil.contains_term("Stored data in PostgreSQL", "postgres")


def test_numeric_atoms_cover_numbers_words_and_units():
    atoms = {(a.value, a.unit, a.more) for a in textutil.numeric_atoms(
        "Interviewed 20+ candidates, cut cost 40% ($5k) on a five-month, 4-person 3D project")}
    assert ("20", "", True) in atoms
    assert ("40", "%", False) in atoms
    assert ("5000", "$", False) in atoms
    assert ("5", "", False) in atoms          # "five-month"
    assert ("4", "", False) in atoms          # "4-person"
    assert ("3d", "tok", False) in atoms


def test_unsupported_atoms_flags_invented_and_inflated_numbers():
    evidence = ["Interviewed 20 candidates for a 4-person team"]
    assert not textutil.unsupported_atoms("Interviewed 20 candidates", evidence)
    assert [a.raw for a in textutil.unsupported_atoms("Interviewed 25 candidates", evidence)] == ["25"]
    # "20+" and "over 20" claim more than the evidence does.
    assert textutil.unsupported_atoms("Interviewed 20+ candidates", evidence)
    assert textutil.unsupported_atoms("Interviewed over 20 candidates", evidence)
    # A duration is not a "more than" claim.
    assert not textutil.unsupported_atoms("Presented over two years", ["across a two-year program"])


def test_ownership_terms_and_exclusions():
    assert textutil.ownership_terms("Assisted a team in developing an app") == {"assist": 2, "develop": 2}
    assert textutil.ownership_terms("Led development of an app") == {"lead": 3}
    assert "lead" not in textutil.ownership_terms("Cut latency, leading to faster reports")
    assert textutil.ownership_terms("Co-founded the club and managed its budget") == {"co-found": 4, "manage": 3}
    assert "noun:founder" in textutil.ownership_terms("Founder of a tutoring startup")


def test_date_formatting_and_sort_keys():
    assert textutil.format_date_range("2024-11", "2024-12") == "Nov – Dec 2024"
    assert textutil.format_date_range("2024-10", "present") == "Oct 2024 – Present"
    assert textutil.format_date_range("2025-10", "2025-10") == "Oct 2025"
    assert textutil.parse_ym("present") > textutil.parse_ym("2026-05") > textutil.parse_ym("2024", end=True)


# ---------------------------------------------------------------------------
# Loading and saving
# ---------------------------------------------------------------------------


EXAMPLE_MASTER = config.ROOT / "master_resume.example.yaml"
EXAMPLE_STATUS = config.ROOT / "skills_status.example.yaml"


@pytest.mark.parametrize("master_path,status_path", [
    (EXAMPLE_MASTER, EXAMPLE_STATUS),                                 # shipped with the repo
    (config.MASTER_RESUME_PATH, config.SKILLS_STATUS_PATH),          # my own copies, when they exist (gitignored)
], ids=["examples", "personal"])
def test_shipped_data_files_validate(master_path, status_path):
    if not master_path.exists() or not status_path.exists():
        pytest.skip(f"{master_path.name} is not present (a fresh clone has only the example files)")
    master = store.load_master_resume(master_path)
    status = store.load_skills_status(status_path)
    assert master.contact.name and master.education
    assert status.names_with(SkillStatus.verified)
    # Every skill tagged on a fact must be verified, or bullets naming it get rejected.
    assert not [w for w in store.data_warnings(master, status) if "cite skill" in w]


def test_missing_master_file_is_a_clean_error(tmp_path):
    with pytest.raises(MasterResumeError, match="not found"):
        store.load_master_resume(tmp_path / "nope.yaml")


def test_invalid_yaml_reports_the_line(tmp_path):
    path = tmp_path / "master_resume.yaml"
    path.write_text("contact:\n  name: [unclosed\n")
    with pytest.raises(MasterResumeError, match="not valid YAML"):
        store.load_master_resume(path)


def test_schema_errors_are_listed(tmp_path):
    path = tmp_path / "master_resume.yaml"
    path.write_text("experience:\n  - id: e1\n    org: X\n    dates: {start: 'June 2024', end: present}\n")
    with pytest.raises(MasterResumeError) as info:
        store.load_master_resume(path)
    assert "experience.0.dates.start" in info.value.message


def test_skills_status_grouped_format_and_conflicts(tmp_path):
    path = tmp_path / "skills_status.yaml"
    path.write_text("verified: [Python, SQL]\nlearning: [PyTorch]\nmissing: [AWS]\n")
    loaded = store.load_skills_status(path)
    assert loaded.is_verified("python") and loaded.status_of("AWS") == SkillStatus.missing
    path.write_text("verified: [Python]\nmissing: [python3]\n")
    with pytest.raises(MasterResumeError, match="conflicting"):
        store.load_skills_status(path)


def test_save_keeps_header_and_makes_backup(tmp_path, master, monkeypatch):
    monkeypatch.setattr(config, "BACKUP_DIR", tmp_path / "backups")
    path = tmp_path / "master_resume.yaml"
    path.write_text("# my notes\n# second line\n\ncontact: {name: Old}\n")
    backup = store.save_master_resume(master, path)
    assert backup is not None and backup.exists() and "name: Old" in backup.read_text()
    written = path.read_text()
    assert written.startswith("# my notes\n# second line\n")
    assert store.load_master_resume(path).entry("exp_acme") is not None


# ---------------------------------------------------------------------------
# Claude wrapper
# ---------------------------------------------------------------------------


class Thing(BaseModel):
    name: str


def _usage(**kw):
    base = dict(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0,
                cache_creation_input_tokens=0, server_tool_use=None)
    return SimpleNamespace(**{**base, **kw})


def _response(parsed=None, stop_reason="end_turn", model="claude-haiku-5-5", **usage):
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason, stop_details=None,
                           model=model, usage=_usage(**usage), content=[])


class FakeMessages:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def _next(self, kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def parse(self, **kwargs):
        return self._next(kwargs)

    def create(self, **kwargs):
        return self._next(kwargs)


class FakeClient:
    def __init__(self, stable=(), beta=(), models=("claude-haiku-5-5", "claude-sonnet-5-5")):
        self.messages = FakeMessages(stable)
        self.beta = SimpleNamespace(messages=FakeMessages(beta))
        self._models = list(models)
        self.models = SimpleNamespace(retrieve=self._retrieve, list=lambda: [SimpleNamespace(id=m) for m in self._models])

    def _retrieve(self, model_id):
        if model_id not in self._models:
            raise _api_error(anthropic.NotFoundError, 404)
        return SimpleNamespace(id=model_id)


def _api_error(cls, status, message="boom"):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message, response=httpx2.Response(status, request=request), body=None)


def test_parse_returns_validated_model_and_tracks_cost():
    client = FakeClient(stable=[_response(Thing(name="x"))])
    llm = LLM(client)
    out = llm.parse(purpose="analyze", model="claude-haiku-5-5", system="s", user="u", schema=Thing)
    assert out == Thing(name="x")
    sent = client.messages.calls[0]
    assert sent["output_format"] is Thing and sent["output_config"] == {"effort": config.EFFORT["analyze"]}
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in sent                      # rejected by current models
    call = llm.tracker.calls[0]
    assert call.cost_usd == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1_000_000)


def test_cost_uses_cache_and_long_prompt_rates():
    assert estimate_cost("claude-sonnet-5-5", 1_000_000, 0) == pytest.approx(2.00)
    assert estimate_cost("claude-sonnet-5-5", 0, 0, cache_read=1_000_000) == pytest.approx(0.20)
    # Haiku 5.5 prompts above 100K tokens are billed on the higher rate card.
    assert estimate_cost("claude-haiku-5-5", 200_000, 0) == pytest.approx(200_000 * 0.50 / 1_000_000)
    assert estimate_cost("claude-sonnet-5-5", 0, 0, web_searches=5) == pytest.approx(0.05)


def test_write_model_opts_into_refusal_fallback_and_degrades_if_rejected():
    client = FakeClient(beta=[_response(Thing(name="a"), model="claude-sonnet-5-5")])
    llm = LLM(client)
    llm.parse(purpose="tailor", model="claude-sonnet-5-5", system="s", user="u", schema=Thing)
    sent = client.beta.messages.calls[0]
    assert sent["fallbacks"] == "default" and sent["betas"] == [config.FALLBACK_BETA]

    rejected = _api_error(anthropic.BadRequestError, 400, "fallbacks: Extra inputs are not permitted")
    client = FakeClient(beta=[rejected], stable=[_response(Thing(name="b")), _response(Thing(name="c"))])
    llm = LLM(client)
    assert llm.parse(purpose="tailor", model="claude-sonnet-5-5", system="s", user="u", schema=Thing).name == "b"
    assert llm.parse(purpose="tailor", model="claude-sonnet-5-5", system="s", user="u", schema=Thing).name == "c"
    assert len(client.beta.messages.calls) == 1            # not attempted again this session


def test_fast_model_never_sends_fallbacks():
    client = FakeClient(stable=[_response(Thing(name="x"))])
    LLM(client).parse(purpose="judge", model="claude-haiku-5-5", system="s", user="u", schema=Thing)
    assert not client.beta.messages.calls and "fallbacks" not in client.messages.calls[0]


def test_refusal_and_truncation_raise_clean_errors():
    llm = LLM(FakeClient(stable=[_response(None, stop_reason="refusal")]))
    with pytest.raises(LLMRefusal):
        llm.parse(purpose="judge", model="claude-haiku-5-5", system="s", user="u", schema=Thing)
    llm = LLM(FakeClient(stable=[_response(None, stop_reason="max_tokens")]))
    with pytest.raises(LLMError, match="cut off"):
        llm.parse(purpose="judge", model="claude-haiku-5-5", system="s", user="u", schema=Thing)


@pytest.mark.parametrize("cls,status,expected", [
    (anthropic.AuthenticationError, 401, "rejected the API key"),
    (anthropic.RateLimitError, 429, "Rate limited"),
    (anthropic.InternalServerError, 529, "overloaded"),
    (anthropic.BadRequestError, 400, "rejected the request"),
])
def test_api_errors_become_friendly_messages(cls, status, expected):
    llm = LLM(FakeClient(stable=[_api_error(cls, status)]))
    with pytest.raises(LLMError, match=expected):
        llm.parse(purpose="judge", model="claude-haiku-5-5", system="s", user="u", schema=Thing)


def test_search_collects_sources_and_resumes_paused_turns():
    paused = SimpleNamespace(
        stop_reason="pause_turn", stop_details=None, model="claude-sonnet-5-5",
        usage=_usage(server_tool_use=SimpleNamespace(web_search_requests=2)),
        content=[{"type": "web_search_tool_result", "content": [
            {"type": "web_search_result", "url": "https://acme.com/careers", "title": "Careers"}]}],
    )
    done = SimpleNamespace(
        stop_reason="end_turn", stop_details=None, model="claude-sonnet-5-5",
        usage=_usage(server_tool_use=SimpleNamespace(web_search_requests=1)),
        content=[{"type": "text", "text": "Acme builds rockets.", "citations": [
            {"type": "web_search_result_location", "url": "https://news.example/acme", "title": "News"}]}],
    )
    client = FakeClient(beta=[paused, done])
    result = LLM(client).search(purpose="research", model="claude-sonnet-5-5", system="s", user="u")
    assert result.text == "Acme builds rockets." and result.searches == 3
    assert [u for u, _ in result.sources] == ["https://acme.com/careers", "https://news.example/acme"]
    resumed = client.beta.messages.calls[1]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"]     # nothing extra appended
    assert client.beta.messages.calls[0]["tools"][0]["type"] == config.WEB_SEARCH_TOOL_TYPE


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def test_missing_env_file_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    with pytest.raises(ConfigError, match="No .env file"):
        startup.read_api_key(startup.load_env())


def test_placeholder_key_is_rejected(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-your-key-here\n")
    monkeypatch.setattr(config, "ENV_PATH", env)
    with pytest.raises(ConfigError, match="missing from .env"):
        startup.read_api_key(startup.load_env())


def test_env_can_override_models(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-real\nWRITE_MODEL=claude-opus-5-5\n")
    monkeypatch.setattr(config, "ENV_PATH", env)
    monkeypatch.setattr(config, "WRITE_MODEL", "claude-sonnet-5-5")
    assert startup.read_api_key(startup.load_env()) == "sk-ant-real"
    assert config.WRITE_MODEL == "claude-opus-5-5"


def test_unavailable_model_lists_what_is_available():
    llm = LLM(FakeClient(models=("claude-haiku-5-5", "claude-opus-5-5")))
    with pytest.raises(ConfigError) as info:
        llm.verify_models(["claude-haiku-5-5", "claude-sonnet-5-5"])
    assert "claude-sonnet-5-5" in info.value.message
    assert "claude-opus-5-5" in info.value.hint and "claude-haiku-5-5" in info.value.hint


def test_startup_without_api_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    # The shipped examples, so the test passes on a fresh clone without my personal files.
    monkeypatch.setattr(config, "MASTER_RESUME_PATH", EXAMPLE_MASTER)
    monkeypatch.setattr(config, "SKILLS_STATUS_PATH", EXAMPLE_STATUS)
    session = startup.run_startup(need_api=False, quiet=True)
    assert session.llm is None and session.master.experience


def test_tests_never_point_at_my_real_data_files():
    """Guard for issue #1: during tests no writable data path may live inside the repo."""
    for name in ("OUTPUT_DIR", "POSTINGS_DIR", "BACKUP_DIR", "APPLICATIONS_CSV", "CACHE_DIR"):
        path = getattr(config, name).resolve()
        assert config.ROOT.resolve() not in path.parents, f"{name} still points into the repo: {path}"


def test_declining_at_the_eligibility_prompt_is_not_an_error(monkeypatch, capsys):
    import tailor
    from src.errors import UserAbort

    def stop(args, parser):
        raise UserAbort("Stopped at the eligibility check - nothing was tailored.")

    monkeypatch.setattr(tailor, "dispatch", stop)
    assert tailor.main(["https://example.com/job"]) == 0
    assert "ERROR" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Issue #2: billing problems must surface in --check, in plain words
# ---------------------------------------------------------------------------

CREDIT_BODY = {"type": "error", "error": {"type": "invalid_request_error",
               "message": "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing."}}


def _credit_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError("Error code: 400 - " + str(CREDIT_BODY),
                                     response=httpx2.Response(400, request=request), body=CREDIT_BODY)


def test_out_of_credits_is_a_plain_message_with_the_next_step():
    llm = LLM(FakeClient(stable=[_credit_error()]))
    with pytest.raises(LLMError) as info:
        llm.parse(purpose="analyze", model="claude-haiku-5-5", system="s", user="u", schema=Thing)
    assert info.value.message == "Your Anthropic API account is out of credits."
    assert "Plans & Billing" in info.value.hint and "Error code" not in str(info.value)


def test_other_api_errors_show_the_apis_own_message():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: too large"}}
    llm = LLM(FakeClient(stable=[anthropic.BadRequestError("Error code: 400 - " + str(body),
                                                           response=httpx2.Response(400, request=request), body=body)]))
    with pytest.raises(LLMError) as info:
        llm.parse(purpose="judge", model="claude-haiku-5-5", system="s", user="u", schema=Thing)
    assert info.value.message == "The Anthropic API rejected the request: max_tokens: too large"


def test_check_makes_a_tiny_real_request_so_an_empty_account_fails_it():
    ok = FakeClient(stable=[_response(None)])
    LLM(ok).verify_access("claude-haiku-5-5")
    sent = ok.messages.calls[0]
    assert sent["max_tokens"] <= 64 and sent["model"] == "claude-haiku-5-5"
    with pytest.raises(ConfigError, match="out of credits"):
        LLM(FakeClient(stable=[_credit_error()])).verify_access("claude-haiku-5-5")
