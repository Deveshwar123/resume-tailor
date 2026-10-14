"""Every model sent to Claude as a structured-output schema must stay within what the API supports."""

from __future__ import annotations

import pytest
from anthropic import transform_schema
from pydantic import TypeAdapter

from src.cover_letter import CoverLetterDraft, SentenceFixBatch
from src.models import (
    BulletFixBatch, CompanyBriefDraft, InterpretedRequirements, JobAnalysisExtract, Judgment, ProposedFact,
    TailoredResumeDraft,
)

WIRE_MODELS = [JobAnalysisExtract, InterpretedRequirements, CompanyBriefDraft, TailoredResumeDraft, BulletFixBatch,
               Judgment, ProposedFact, CoverLetterDraft, SentenceFixBatch]
UNSUPPORTED = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern")
MAX_UNION_FIELDS = 16


def _walk(node, path, problems, counts):
    if isinstance(node, dict):
        if "properties" in node:
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: object is not closed (additionalProperties must be false)")
            optional = set(node["properties"]) - set(node.get("required", []))
            if optional:
                problems.append(f"{path}: optional properties {sorted(optional)} (give them no default)")
        if "anyOf" in node:
            counts["unions"] += 1
        problems.extend(f"{path}: unsupported keyword {key}" for key in UNSUPPORTED if key in node)
        for key, value in node.items():
            _walk(value, f"{path}.{key}", problems, counts)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]", problems, counts)


@pytest.mark.parametrize("model", WIRE_MODELS, ids=lambda m: m.__name__)
def test_wire_model_is_a_valid_structured_output_schema(model):
    schema = transform_schema(TypeAdapter(model).json_schema())      # what messages.parse() sends
    problems: list[str] = []
    counts = {"unions": 0}
    _walk(schema, model.__name__, problems, counts)
    assert problems == []
    assert counts["unions"] <= MAX_UNION_FIELDS
