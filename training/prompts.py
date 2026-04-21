# prompts.py — Shared prompt helpers for finetune.py and evaluate.py

FORMAT_LABELS = {
    "json_output": "JSON",
    "yaml_output": "YAML",
    "toml_output": "TOML",
    "csv_output": "CSV key=value",
    "xml_output": "XML",
}


def system_prompt(fmt: str) -> str:
    label = FORMAT_LABELS[fmt]
    return (
        "You are a structured search query parser. "
        "Given a natural language search query, extract ONLY the fields explicitly "
        "mentioned or directly implied by the query and return them as "
        f"{label}. "
        "Do NOT add fields that are not present in the query. "
        "Do NOT hallucinate or invent values. "
        "Output ONLY the structured data, nothing else."
    )


def make_training_prompt(query: str, output: str, fmt: str, eos: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt(fmt)}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>{eos}"
    )


def make_inference_prompt(query: str, fmt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt(fmt)}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
