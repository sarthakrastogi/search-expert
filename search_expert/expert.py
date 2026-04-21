"""
SearchExpert: the main public interface.

Usage
-----
    from search_expert import SearchExpert, ModelFormat

    # Default: JSON model, 4-bit quantisation
    expert = SearchExpert()

    # YAML model, full precision
    expert = SearchExpert(fmt=ModelFormat.YAML, load_in_4bit=False)

    # Your own fine-tuned adapter
    expert = SearchExpert(model_id="your-org/your-adapter")

    result = expert.parse("MacBook under $2000 with 16GB RAM")
    print(result.fields)
    print(result.to_json(indent=2))
"""

from __future__ import annotations

import logging
from typing import Any

from search_expert.config import (
    ModelFormat,
    build_inference_prompt,
)
from search_expert.exceptions import ParseError
from search_expert.loader import Backend, load_model
from search_expert.parser import parse_model_output
from search_expert.result import ParseResult

logger = logging.getLogger(__name__)


class SearchExpert:
    """
    Parse natural language search queries into structured field dicts.

    The model is loaded lazily on the first call to :meth:`parse` (or
    eagerly if you pass ``eager=True`` to the constructor).

    Parameters
    ----------
    fmt : ModelFormat
        Which fine-tuned adapter to use.  ``ModelFormat.JSON`` (default)
        or ``ModelFormat.YAML``.
    model_id : str | None
        Override the default HuggingFace adapter repo ID.
    load_in_4bit : bool
        Use 4-bit quantisation when loading.  Reduces VRAM usage at the
        cost of a small quality degradation.  Default ``True``.
    backend : str | Backend
        Force a loading backend: ``"auto"`` (default), ``"unsloth"``,
        ``"peft"``, or ``"plain"``.
    generation_config : dict | None
        Override default ``model.generate()`` kwargs.
    eager : bool
        If ``True``, load the model immediately in ``__init__``.
        Otherwise the model is loaded on the first :meth:`parse` call.
    """

    _DEFAULT_GENERATION: dict[str, Any] = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "do_sample": True,
        "use_cache": True,
    }

    def __init__(
        self,
        fmt: ModelFormat | str = ModelFormat.JSON,
        model_id: str | None = None,
        load_in_4bit: bool = True,
        backend: Backend | str = Backend.AUTO,
        generation_config: dict[str, Any] | None = None,
        eager: bool = False,
    ) -> None:
        self.fmt = ModelFormat(fmt)
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.backend = Backend(backend)
        self.generation_cfg = {**self._DEFAULT_GENERATION, **(generation_config or {})}

        self._model = None
        self._tokenizer = None

        if eager:
            self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load (or reload) model + tokenizer."""
        self._model, self._tokenizer = load_model(
            fmt=self.fmt,
            model_id=self.model_id,
            load_in_4bit=self.load_in_4bit,
            backend=self.backend,
        )

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    def _generate(self, prompt: str) -> str:
        """Run inference and return the raw generated text (prompt stripped)."""
        import torch

        inputs = self.tokenizer(
            text=prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # Move to the same device as the model
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.generation_cfg,
            )

        # Strip the prompt tokens so we only decode the generated portion
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, query: str) -> ParseResult:
        """
        Parse a natural language search query into a :class:`ParseResult`.

        Parameters
        ----------
        query : str
            The user's search query, e.g.
            ``"MacBook under $2000 with 16GB RAM"``.

        Returns
        -------
        ParseResult
            Contains ``.fields`` (dict), ``.to_json()``, ``.to_yaml()``,
            and ``.get_numeric_constraint(field)`` helpers.

        Raises
        ------
        ParseError
            If the model output cannot be parsed into a structured dict.

        Examples
        --------
        >>> expert = SearchExpert()
        >>> r = expert.parse("2BR apartment in Austin under $1500/month")
        >>> r.fields
        {'domain': 'real_estate', 'property_type': 'apartment',
         'bedrooms': '2', 'city': 'Austin', 'price': 'lt:1500'}
        >>> r.get_numeric_constraint("price")
        {'operator': 'lt', 'value': 1500.0, 'value_hi': None}
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        prompt = build_inference_prompt(query.strip(), self.fmt)
        raw_output = self._generate(prompt)
        logger.debug("Raw model output: %r", raw_output)

        fields = parse_model_output(raw_output, self.fmt)

        return ParseResult(
            query=query,
            fields=fields,
            raw_output=raw_output,
            model_format=self.fmt.value,
        )

    def parse_batch(self, queries: list[str]) -> list[ParseResult]:
        """
        Parse multiple queries sequentially.

        This is a convenience wrapper — inference is still one-at-a-time.
        For true batched inference, use the model directly.

        Parameters
        ----------
        queries : list[str]

        Returns
        -------
        list[ParseResult]
            One result per query.  Failed parses raise :class:`ParseError`
            and abort the batch; catch it per-item if you need resilience.
        """
        return [self.parse(q) for q in queries]

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"SearchExpert("
            f"fmt={self.fmt.value!r}, "
            f"model_id={self.model_id or 'default'!r}, "
            f"loaded={loaded})"
        )
