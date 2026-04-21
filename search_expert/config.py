"""
Configuration: model identifiers, supported formats, and prompt templates.
"""

from enum import Enum


class ModelFormat(str, Enum):
    """
    Which fine-tuned adapter to load, and which output format to expect.
    Both models return Python dicts; the format only affects the LLM's
    internal output language (JSON vs YAML), which is then parsed for you.
    """

    JSON = "json"
    YAML = "yaml"


# HuggingFace repo IDs for each adapter.
# Override these by passing `model_id=` to SearchExpert() if you host
# your own fine-tuned variants.
DEFAULT_MODEL_IDS: dict[ModelFormat, str] = {
    ModelFormat.JSON: "sarthakrastogi/search-expert-json-0.8b",
    ModelFormat.YAML: "sarthakrastogi/search-expert-yaml-0.8b",
}

# Base model that the LoRA adapters were trained on top of.
BASE_MODEL_ID = "unsloth/Qwen3.5-0.8B"

# System prompts must exactly match what was used during fine-tuning.
_SYSTEM_PROMPTS: dict[ModelFormat, str] = {
    ModelFormat.JSON: (
        "You are a structured search query parser. "
        "Given a natural language search query, extract ONLY the fields explicitly "
        "mentioned or directly implied by the query and return them as JSON. "
        "Do NOT add fields that are not present in the query. "
        "Do NOT hallucinate or invent values. "
        "Output ONLY the structured data, nothing else."
    ),
    ModelFormat.YAML: (
        "You are a structured search query parser. "
        "Given a natural language search query, extract ONLY the fields explicitly "
        "mentioned or directly implied by the query and return them as YAML. "
        "Do NOT add fields that are not present in the query. "
        "Do NOT hallucinate or invent values. "
        "Output ONLY the structured data, nothing else."
    ),
}


def get_system_prompt(fmt: ModelFormat) -> str:
    return _SYSTEM_PROMPTS[fmt]


def build_inference_prompt(query: str, fmt: ModelFormat) -> str:
    """Constructs the ChatML-formatted prompt used at inference time."""
    system = get_system_prompt(fmt)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
