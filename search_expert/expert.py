"""
SearchExpert: the main public interface.

Usage
-----
    from search_expert import SearchExpert, ModelFormat, ParseBackend

    # Default: JSON SLM adapter, 4-bit quantisation
    expert = SearchExpert()

    # spaCy backend — no GPU needed, ~1 ms/query
    expert = SearchExpert(parse_backend=ParseBackend.SPACY)

    # YAML SLM adapter, full precision
    expert = SearchExpert(fmt=ModelFormat.YAML, load_in_4bit=False)

    # Your own fine-tuned adapter
    expert = SearchExpert(model_id="your-org/your-adapter")

    result = expert.parse("MacBook under $2000 with 16GB RAM")
    print(result.fields)
    print(result.to_json(indent=2))
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from search_expert.config import (
    ModelFormat,
    ParseBackend,
    build_inference_prompt,
)
from search_expert.loader import Backend, load_model
from search_expert.parser import parse_model_output
from search_expert.result import ParseResult

logger = logging.getLogger(__name__)


class SearchExpert:
    """
    Parse natural language search queries into structured field dicts.

    Two backends are supported:

    **SLM** (default) — uses a fine-tuned LoRA adapter (Qwen3.5-0.8B).
    The model is loaded lazily on the first call to :meth:`parse` (or
    eagerly if you pass ``eager=True`` to the constructor).

    **spaCy** — uses rule-based NER + regex patterns. Much faster (~1 ms/q
    on CPU) but less able to generalise to novel phrasings or domains.
    Pass ``parse_backend=ParseBackend.SPACY`` to enable this backend.

    Parameters
    ----------
    fmt : ModelFormat
        Which fine-tuned adapter to use.  ``ModelFormat.JSON`` (default)
        or ``ModelFormat.YAML``.  Ignored when ``parse_backend=SPACY``.
    model_id : str | None
        Override the default HuggingFace adapter repo ID.
        Ignored when ``parse_backend=SPACY``.
    load_in_4bit : bool
        Use 4-bit quantisation when loading the SLM.  Reduces VRAM usage
        at the cost of a small quality degradation.  Default ``True``.
        Ignored when ``parse_backend=SPACY``.
    backend : str | Backend
        Force a model-loading backend: ``"auto"`` (default),
        ``"unsloth"``, ``"peft"``, or ``"plain"``.
        Ignored when ``parse_backend=SPACY``.
    generation_config : dict | None
        Override default ``model.generate()`` kwargs.
        Ignored when ``parse_backend=SPACY``.
    parse_backend : ParseBackend | str
        Top-level backend switch.  ``ParseBackend.SLM`` (default) uses
        the fine-tuned adapter; ``ParseBackend.SPACY`` uses the rule-based
        spaCy extractor.
    spacy_model : str
        spaCy model name to load when ``parse_backend=SPACY``.
        Defaults to ``"en_core_web_sm"``.
    eager : bool
        If ``True``, load the model / NLP pipeline immediately in
        ``__init__``.  Otherwise loading is deferred to the first
        :meth:`parse` call.
    """

    _DEFAULT_GENERATION: ClassVar[dict[str, Any]] = {
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
        parse_backend: ParseBackend | str = ParseBackend.SLM,
        spacy_model: str = "en_core_web_sm",
        eager: bool = False,
    ) -> None:
        self.fmt = ModelFormat(fmt)
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.backend = Backend(backend)
        self.generation_cfg = {**self._DEFAULT_GENERATION, **(generation_config or {})}
        self.parse_backend = ParseBackend(parse_backend)
        self.spacy_model = spacy_model

        self._model = None
        self._tokenizer = None
        self._spacy_extractor = None

        if eager:
            self._load()

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load (or reload) whichever backend is selected."""
        if self.parse_backend == ParseBackend.SPACY:
            self._load_spacy()
        else:
            self._load_slm()

    def _load_slm(self) -> None:
        self._model, self._tokenizer = load_model(
            fmt=self.fmt,
            model_id=self.model_id,
            load_in_4bit=self.load_in_4bit,
            backend=self.backend,
        )

    def _load_spacy(self) -> None:
        from search_expert.spacy_extractor import SpacyExtractor

        self._spacy_extractor = SpacyExtractor(model=self.spacy_model)
        # Trigger the lazy nlp load now so it's warm for inference.
        _ = self._spacy_extractor.nlp

    # ------------------------------------------------------------------
    # SLM lazy-load properties
    # ------------------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            self._load_slm()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_slm()
        return self._tokenizer

    # ------------------------------------------------------------------
    # spaCy lazy-load property
    # ------------------------------------------------------------------

    @property
    def spacy_extractor(self):
        if self._spacy_extractor is None:
            self._load_spacy()
        return self._spacy_extractor

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Run SLM inference and return the raw generated text (prompt stripped)."""
        import torch

        inputs = self.tokenizer(
            text=prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

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
        ValueError
            If query is empty.
        ParseError
            (SLM only) If the model output cannot be parsed into a dict.

        Examples
        --------
        >>> # SLM backend (default)
        >>> expert = SearchExpert()
        >>> r = expert.parse("2BR apartment in Austin under $1500/month")
        >>> r.fields
        {'domain': 'real_estate', 'property_type': 'apartment',
         'bedrooms': '2', 'city': 'Austin', 'price': 'lt:1500'}

        >>> # spaCy backend
        >>> expert = SearchExpert(parse_backend="spacy")
        >>> r = expert.parse("2BR apartment in Austin under $1500/month")
        >>> r.fields
        {'domain': 'real_estate', 'bedrooms': '2', 'city': 'Austin',
         'price': 'lt:1500'}
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if self.parse_backend == ParseBackend.SPACY:
            return self._parse_spacy(query)
        return self._parse_slm(query)

    def _parse_slm(self, query: str) -> ParseResult:
        prompt = build_inference_prompt(query.strip(), self.fmt)
        raw_output = self._generate(prompt)
        logger.debug("Raw SLM output: %r", raw_output)

        fields = parse_model_output(raw_output, self.fmt)

        return ParseResult(
            query=query,
            fields=fields,
            raw_output=raw_output,
            model_format=self.fmt.value,
        )

    def _parse_spacy(self, query: str) -> ParseResult:
        fields = self.spacy_extractor.extract(query.strip())
        logger.debug("spaCy extracted fields: %r", fields)

        return ParseResult(
            query=query,
            fields=fields,
            raw_output="",  # no raw LLM output for spaCy
            model_format="spacy",
        )

    def parse_batch(self, queries: list[str]) -> list[ParseResult]:
        """
        Parse multiple queries sequentially.

        This is a convenience wrapper — inference is still one-at-a-time.
        For true batched SLM inference, use the model directly.

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
        if self.parse_backend == ParseBackend.SPACY:
            loaded = self._spacy_extractor is not None
            return (
                f"SearchExpert("
                f"parse_backend='spacy', "
                f"spacy_model={self.spacy_model!r}, "
                f"loaded={loaded})"
            )
        loaded = self._model is not None
        return (
            f"SearchExpert("
            f"fmt={self.fmt.value!r}, "
            f"parse_backend='slm', "
            f"model_id={self.model_id or 'default'!r}, "
            f"loaded={loaded})"
        )
