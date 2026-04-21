"""
Model loader: downloads and caches the fine-tuned adapter from HuggingFace.

Supports three loading strategies depending on what's installed:
  1. unsloth  — fastest; used automatically when available (GPU recommended)
  2. peft     — standard HF PEFT; works on CPU and GPU without unsloth
  3. plain transformers — falls back for environments without PEFT

Users can force a strategy via the `backend` parameter of SearchExpert().
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from search_expert.config import BASE_MODEL_ID, DEFAULT_MODEL_IDS, ModelFormat
from search_expert.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class Backend(str, Enum):
    AUTO = "auto"
    UNSLOTH = "unsloth"
    PEFT = "peft"
    PLAIN = "plain"


def _try_unsloth(model_id: str, load_in_4bit: bool) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _try_peft(model_id: str, load_in_4bit: bool) -> tuple[Any, Any]:
    import torch
    from peft import AutoPeftModelForCausalLM  # type: ignore
    from transformers import AutoTokenizer

    quant_kwargs: dict[str, Any] = {}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception:
            logger.warning("bitsandbytes not available; loading in full precision.")

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        **quant_kwargs,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _try_plain(model_id: str) -> tuple[Any, Any]:
    """
    Last-resort: load base model only (no LoRA adapter).
    Results will be worse than the fine-tuned adapters.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.warning(
        "Loading base model without LoRA adapter (peft not installed). "
        "Output quality will be lower than the fine-tuned model."
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map="auto")
    model.eval()
    return model, tokenizer


def load_model(
    fmt: ModelFormat,
    model_id: str | None = None,
    load_in_4bit: bool = True,
    backend: Backend = Backend.AUTO,
) -> tuple[Any, Any]:
    """
    Load model + tokenizer for the given format.

    Parameters
    ----------
    fmt : ModelFormat
        JSON or YAML — selects which fine-tuned adapter to pull.
    model_id : str | None
        Override the default HuggingFace repo ID.
    load_in_4bit : bool
        Use 4-bit quantisation (saves ~60% VRAM, slight quality drop).
        Only applied when the unsloth or PEFT backend is used.
    backend : Backend
        Force a specific loading strategy; AUTO tries unsloth → peft → plain.

    Returns
    -------
    (model, tokenizer)

    Raises
    ------
    ModelLoadError
        If all attempted backends fail.
    """
    repo = model_id or DEFAULT_MODEL_IDS[fmt]
    logger.info(
        "Loading model from %s (backend=%s, 4bit=%s)", repo, backend, load_in_4bit
    )

    errors: list[str] = []

    if backend in (Backend.AUTO, Backend.UNSLOTH):
        try:
            model, tokenizer = _try_unsloth(repo, load_in_4bit)
            logger.info("Loaded via unsloth.")
            return model, tokenizer
        except ImportError:
            logger.debug("unsloth not installed, trying peft.")
        except Exception as exc:
            errors.append(f"unsloth: {exc}")
            logger.debug("unsloth failed: %s", exc)

    if backend in (Backend.AUTO, Backend.PEFT):
        try:
            model, tokenizer = _try_peft(repo, load_in_4bit)
            logger.info("Loaded via peft.")
            return model, tokenizer
        except ImportError:
            logger.debug("peft not installed, falling back to plain.")
        except Exception as exc:
            errors.append(f"peft: {exc}")
            logger.debug("peft failed: %s", exc)

    if backend in (Backend.AUTO, Backend.PLAIN):
        try:
            model, tokenizer = _try_plain(repo)
            logger.info("Loaded via plain transformers (no LoRA).")
            return model, tokenizer
        except Exception as exc:
            errors.append(f"plain: {exc}")

    raise ModelLoadError(f"Failed to load model '{repo}'. Errors: {'; '.join(errors)}")
