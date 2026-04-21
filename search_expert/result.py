"""
ParseResult: the structured output returned by SearchExpert.parse().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ParseResult:
    """
    The result of parsing a natural language search query.

    Attributes
    ----------
    query : str
        The original user query, unchanged.
    fields : dict[str, Any]
        Extracted structured fields.  Numeric constraints are expressed
        with operator prefixes, e.g. ``"price": "lt:2000"``.
    raw_output : str
        The raw string produced by the model before parsing.
        Useful for debugging when fields look unexpected.
    model_format : str
        Which adapter was used ("json" or "yaml").

    Examples
    --------
    >>> result.fields
    {'domain': 'ecommerce', 'product': 'MacBook', 'price': 'lt:2000'}

    >>> result.to_json()
    '{"domain": "ecommerce", "product": "MacBook", "price": "lt:2000"}'

    >>> result.to_yaml()
    'domain: ecommerce\\nproduct: MacBook\\nprice: lt:2000\\n'

    Operator reference
    ------------------
    ``lt:N``            < N   (under, less than, below)
    ``lte:N``           ≤ N   (up to, at most, or less)
    ``gt:N``            > N   (over, more than, above)
    ``gte:N``           ≥ N   (at least, N+, starting from)
    ``approx:N``        ≈ N   (around, roughly, ~N)
    ``between:Lo:Hi``   Lo ≤ x ≤ Hi
    """

    query: str
    fields: dict[str, Any]
    raw_output: str
    model_format: str
    _extra: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self, indent: int | None = None) -> str:
        """Return fields as a JSON string."""
        return json.dumps(self.fields, ensure_ascii=False, indent=indent)

    def to_yaml(self) -> str:
        """Return fields as a YAML string."""
        return yaml.dump(self.fields, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation of the full result."""
        return {
            "query": self.query,
            "fields": self.fields,
            "model_format": self.model_format,
        }

    # ------------------------------------------------------------------
    # Operator helpers: let callers decode operator-prefixed numerics
    # ------------------------------------------------------------------

    def get_numeric_constraint(self, field_name: str) -> dict[str, Any] | None:
        """
        Decode an operator-prefixed numeric field into a structured dict.

        Returns None if the field is missing or not operator-prefixed.

        Example
        -------
        >>> result.fields["price"]
        'lt:2000'
        >>> result.get_numeric_constraint("price")
        {'operator': 'lt', 'value': 2000.0, 'value_hi': None}
        """
        raw = self.fields.get(field_name)
        if not isinstance(raw, str) or ":" not in raw:
            return None

        parts = raw.split(":")
        op = parts[0]

        if op not in {"lt", "lte", "gt", "gte", "approx", "between"}:
            return None

        try:
            if op == "between" and len(parts) == 3:
                return {
                    "operator": op,
                    "value": float(parts[1]),
                    "value_hi": float(parts[2]),
                }
            return {
                "operator": op,
                "value": float(parts[1]),
                "value_hi": None,
            }
        except (ValueError, IndexError):
            return None

    def numeric_fields(self) -> dict[str, dict[str, Any]]:
        """
        Return all fields that carry operator-prefixed numeric constraints.

        Example
        -------
        >>> result.numeric_fields()
        {'price': {'operator': 'lt', 'value': 2000.0, 'value_hi': None},
         'rating': {'operator': 'gte', 'value': 4.5, 'value_hi': None}}
        """
        out = {}
        for key in self.fields:
            decoded = self.get_numeric_constraint(key)
            if decoded is not None:
                out[key] = decoded
        return out

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ParseResult(query={self.query!r}, "
            f"fields={self.fields!r}, "
            f"model_format={self.model_format!r})"
        )

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access: result['domain']"""
        return self.fields[key]

    def __contains__(self, key: str) -> bool:
        return key in self.fields
