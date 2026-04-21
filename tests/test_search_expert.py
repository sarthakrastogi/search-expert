"""
Tests for search_expert.

Run with:
    pytest tests/ -v

The model-loading tests are skipped in CI (no GPU) unless
SEARCH_EXPERT_RUN_MODEL_TESTS=1 is set in the environment.
"""

from __future__ import annotations

import json
import os
import pytest
import yaml

from search_expert.config import ModelFormat, build_inference_prompt, get_system_prompt
from search_expert.exceptions import ParseError
from search_expert.parser import parse_model_output
from search_expert.result import ParseResult
from search_expert import SearchExpert, ModelFormat


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_json_fields():
    return {
        "domain": "ecommerce",
        "product": "headphones",
        "price": "lt:200",
        "feature": "noise cancelling",
    }


@pytest.fixture
def sample_json_result(sample_json_fields):
    return ParseResult(
        query="noise cancelling headphones under $200",
        fields=sample_json_fields,
        raw_output=json.dumps(sample_json_fields),
        model_format="json",
    )


# ── Config tests ───────────────────────────────────────────────


def test_system_prompt_contains_format_name():
    assert "JSON" in get_system_prompt(ModelFormat.JSON)
    assert "YAML" in get_system_prompt(ModelFormat.YAML)


def test_inference_prompt_structure():
    prompt = build_inference_prompt("find a laptop", ModelFormat.JSON)
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt
    assert "find a laptop" in prompt


def test_model_format_from_string():
    assert ModelFormat("json") == ModelFormat.JSON
    assert ModelFormat("yaml") == ModelFormat.YAML


# ── Parser tests ───────────────────────────────────────────────


class TestJsonParser:
    def test_clean_json(self):
        raw = '{"domain": "ecommerce", "product": "laptop"}'
        result = parse_model_output(raw, ModelFormat.JSON)
        assert result == {"domain": "ecommerce", "product": "laptop"}

    def test_json_with_whitespace(self):
        raw = '  \n{"domain": "jobs", "job_title": "ML Engineer"}\n  '
        result = parse_model_output(raw, ModelFormat.JSON)
        assert result["domain"] == "jobs"

    def test_json_embedded_in_prose(self):
        raw = 'Sure! Here is the JSON: {"domain": "flights", "origin": "JFK"}'
        result = parse_model_output(raw, ModelFormat.JSON)
        assert result["origin"] == "JFK"

    def test_invalid_json_raises(self):
        with pytest.raises(ParseError) as exc_info:
            parse_model_output("this is not json at all", ModelFormat.JSON)
        assert exc_info.value.raw_output == "this is not json at all"

    def test_operator_value_preserved(self):
        raw = '{"price": "lt:2000", "rating": "gte:4.5"}'
        result = parse_model_output(raw, ModelFormat.JSON)
        assert result["price"] == "lt:2000"
        assert result["rating"] == "gte:4.5"


class TestYamlParser:
    def test_clean_yaml(self):
        raw = "domain: ecommerce\nproduct: laptop\nprice: lt:2000"
        result = parse_model_output(raw, ModelFormat.YAML)
        assert result == {
            "domain": "ecommerce",
            "product": "laptop",
            "price": "lt:2000",
        }

    def test_yaml_with_code_fence(self):
        raw = "```yaml\ndomain: hotels\ncity: Paris\n```"
        result = parse_model_output(raw, ModelFormat.YAML)
        assert result["city"] == "Paris"

    def test_invalid_yaml_raises(self):
        with pytest.raises(ParseError):
            parse_model_output(": : : broken yaml :", ModelFormat.YAML)


# ── ParseResult tests ──────────────────────────────────────────


class TestParseResult:
    def test_to_json(self, sample_json_result):
        out = sample_json_result.to_json()
        parsed = json.loads(out)
        assert parsed["domain"] == "ecommerce"

    def test_to_json_indent(self, sample_json_result):
        out = sample_json_result.to_json(indent=2)
        assert "\n" in out  # indented means newlines

    def test_to_yaml(self, sample_json_result):
        out = sample_json_result.to_yaml()
        parsed = yaml.safe_load(out)
        assert parsed["product"] == "headphones"

    def test_to_dict(self, sample_json_result):
        d = sample_json_result.to_dict()
        assert "query" in d
        assert "fields" in d
        assert "model_format" in d

    def test_getitem(self, sample_json_result):
        assert sample_json_result["domain"] == "ecommerce"

    def test_contains(self, sample_json_result):
        assert "product" in sample_json_result
        assert "nonexistent_field" not in sample_json_result

    def test_get_numeric_constraint_lt(self, sample_json_result):
        c = sample_json_result.get_numeric_constraint("price")
        assert c == {"operator": "lt", "value": 200.0, "value_hi": None}

    def test_get_numeric_constraint_between(self):
        result = ParseResult(
            query="jobs paying $80k-$120k",
            fields={"salary": "between:80000:120000"},
            raw_output="",
            model_format="json",
        )
        c = result.get_numeric_constraint("salary")
        assert c == {"operator": "between", "value": 80000.0, "value_hi": 120000.0}

    def test_get_numeric_constraint_missing(self, sample_json_result):
        assert sample_json_result.get_numeric_constraint("nonexistent") is None

    def test_get_numeric_constraint_plain_string(self, sample_json_result):
        assert sample_json_result.get_numeric_constraint("domain") is None

    def test_numeric_fields(self, sample_json_result):
        nf = sample_json_result.numeric_fields()
        assert "price" in nf
        assert "domain" not in nf
        assert "product" not in nf

    def test_repr(self, sample_json_result):
        r = repr(sample_json_result)
        assert "ParseResult" in r
        assert "ecommerce" in r


# ── Integration test (skipped in CI without GPU) ───────────────

SKIP_MODEL = not os.environ.get("SEARCH_EXPERT_RUN_MODEL_TESTS")


@pytest.mark.skipif(
    SKIP_MODEL, reason="Set SEARCH_EXPERT_RUN_MODEL_TESTS=1 to run model tests"
)
class TestSearchExpertIntegration:
    @pytest.fixture(scope="class")
    def expert_json(self):
        return SearchExpert(fmt=ModelFormat.JSON, eager=True)

    @pytest.fixture(scope="class")
    def expert_yaml(self):
        return SearchExpert(fmt=ModelFormat.YAML, eager=True)

    def test_json_parse_returns_result(self, expert_json):
        result = expert_json.parse("MacBook under $2000")
        assert isinstance(result, ParseResult)
        assert "domain" in result.fields

    def test_yaml_parse_returns_result(self, expert_yaml):
        result = expert_yaml.parse("3 star hotel in Paris under $150 per night")
        assert isinstance(result, ParseResult)

    def test_numeric_operator_in_output(self, expert_json):
        result = expert_json.parse("headphones under $200")
        price_constraint = result.get_numeric_constraint("price")
        # Model should produce an operator-prefixed price
        assert price_constraint is not None
        assert price_constraint["operator"] in {"lt", "lte", "approx", "between"}

    def test_empty_query_raises(self, expert_json):
        with pytest.raises(ValueError):
            expert_json.parse("")

    def test_batch_parse(self, expert_json):
        queries = [
            "Python course for beginners under $50",
            "Remote ML engineer job over $150k",
        ]
        results = expert_json.parse_batch(queries)
        assert len(results) == 2
        assert all(isinstance(r, ParseResult) for r in results)
