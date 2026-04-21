"""
Parser utilities: convert raw model output strings into Python dicts.

Each parser is lenient by design — it attempts a best-effort extraction
before raising ParseError, because small models occasionally produce
slightly malformed output.
"""

from __future__ import annotations

import json
import re

import yaml

from search_expert.config import ModelFormat
from search_expert.exceptions import ParseError


# ── JSON ──────────────────────────────────────────────────────────────


def _parse_json(text: str) -> dict:
    text = text.strip()

    # 1. Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Extract first {...} block (handles trailing text / markdown fences)
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ParseError(
        "Could not parse model output as JSON.",
        raw_output=text,
    )


# ── YAML ──────────────────────────────────────────────────────────────


def _parse_yaml(text: str) -> dict:
    text = text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:yaml)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)

    try:
        result = yaml.safe_load(text)
        if isinstance(result, dict):
            return result
    except yaml.YAMLError:
        pass

    raise ParseError(
        "Could not parse model output as YAML.",
        raw_output=text,
    )


# ── Dispatcher ────────────────────────────────────────────────────────

_PARSERS = {
    ModelFormat.JSON: _parse_json,
    ModelFormat.YAML: _parse_yaml,
}


def parse_model_output(raw: str, fmt: ModelFormat) -> dict:
    """
    Parse the raw string produced by the model into a Python dict.

    Parameters
    ----------
    raw : str
        The decoded model output (already stripped of the prompt).
    fmt : ModelFormat
        Which format the model was asked to produce.

    Returns
    -------
    dict
        Extracted field → value mapping.

    Raises
    ------
    ParseError
        If the output cannot be parsed after best-effort attempts.
    """
    parser = _PARSERS.get(fmt)
    if parser is None:
        raise ParseError(f"No parser registered for format {fmt!r}.")

    return parser(raw)
