# -*- coding: utf-8 -*-
"""
evaluate.py — Evaluate fine-tuned LoRA adapters on the held-out test split
===========================================================================
Loads each saved adapter, runs inference on EVAL_SAMPLES test rows,
and prints a leaderboard comparing formats by field-level F1, value
accuracy, parse rate, and latency.

Usage:
    python evaluate.py
"""

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

DATASET_REPO = "sarthakrastogi/search-expert"
EVAL_SAMPLES = 300  # test rows per format (more = slower but fairer)
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True

ALL_FORMATS = [
    "json_output",
    "yaml_output",
    # "toml_output",
    # "csv_output",
    # "xml_output",
]

MODEL_NAMES = {
    "json_output": "search-expert-json-0.8b",
    "yaml_output": "search-expert-yaml-0.8b",
    "toml_output": "search-expert-toml-0.8b",
    "csv_output": "search-expert-csv-0.8b",
    "xml_output": "search-expert-xml-0.8b",
}

# ──────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────

import gc
import json
import random
import time
import xml.etree.ElementTree as ET

import torch
import yaml
from datasets import load_dataset
from unsloth import FastLanguageModel

from prompts import FORMAT_LABELS, make_inference_prompt

# ──────────────────────────────────────────────────────────────
# FORMAT PARSERS
# ──────────────────────────────────────────────────────────────


def parse_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except Exception:
        import re

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


def parse_yaml(text: str) -> dict:
    try:
        result = yaml.safe_load(text.strip())
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def parse_toml(text: str) -> dict:
    import toml

    try:
        parsed = toml.loads(text.strip())
        if "query" in parsed and isinstance(parsed["query"], dict):
            return parsed["query"]
        return parsed
    except Exception:
        return {}


def parse_csv_kv(text: str) -> dict:
    result = {}
    for part in text.strip().split(";"):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            result[k.strip()] = v.strip()
    return result


def parse_xml(text: str) -> dict:
    try:
        root = ET.fromstring(text.strip())
        return {child.tag: child.text for child in root}
    except Exception:
        return {}


PARSERS = {
    "json_output": parse_json,
    "yaml_output": parse_yaml,
    "toml_output": parse_toml,
    "csv_output": parse_csv_kv,
    "xml_output": parse_xml,
}

# ──────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────


def evaluate(model, tokenizer, raw_test, fmt: str, n: int = 300) -> dict:
    """
    Returns:
        key_precision  — of predicted fields, fraction that exist in ground truth
        key_recall     — of ground truth fields, fraction that were predicted
        key_f1         — harmonic mean of precision and recall
        value_acc      — of correctly predicted fields, fraction with exact value match
        parse_rate     — fraction of outputs that parsed into a non-empty dict
        avg_latency_ms — mean generation time per sample
    """
    FastLanguageModel.for_inference(model)
    parser = PARSERS[fmt]
    indices = random.sample(range(len(raw_test)), min(n, len(raw_test)))

    key_hits = key_pred_total = key_gt_total = 0
    val_hits = val_total = 0
    parse_successes = 0
    total_ms = 0.0

    for i in indices:
        row = raw_test[i]
        query = row["query"]
        gt_raw = row[fmt]

        gt = parser(gt_raw)
        if not gt:
            continue

        prompt = make_inference_prompt(query, fmt)
        inputs = tokenizer(
            text=prompt, return_tensors="pt", add_special_tokens=False
        ).to("cuda")

        t0 = time.perf_counter()
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        total_ms += (time.perf_counter() - t0) * 1000

        generated = out_ids[0][inputs["input_ids"].shape[1] :]
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()

        pred = parser(decoded)
        if pred:
            parse_successes += 1

        gt_keys = set(gt.keys())
        pred_keys = set(pred.keys())

        key_hits += len(gt_keys & pred_keys)
        key_pred_total += len(pred_keys)
        key_gt_total += len(gt_keys)

        for k in gt_keys & pred_keys:
            val_total += 1
            if str(pred[k]).strip().lower() == str(gt[k]).strip().lower():
                val_hits += 1

    precision = key_hits / key_pred_total if key_pred_total else 0.0
    recall = key_hits / key_gt_total if key_gt_total else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    val_acc = val_hits / val_total if val_total else 0.0
    parse_rt = parse_successes / len(indices)
    avg_lat = total_ms / len(indices)

    return {
        "key_precision": precision,
        "key_recall": recall,
        "key_f1": f1,
        "value_acc": val_acc,
        "parse_rate": parse_rt,
        "avg_latency_ms": avg_lat,
    }


# ──────────────────────────────────────────────────────────────
# LEADERBOARD PRINTER
# ──────────────────────────────────────────────────────────────


def print_leaderboard(all_results: dict, hf_username: str = "") -> None:
    ranked = sorted(
        all_results.items(),
        key=lambda x: (x[1]["key_f1"], x[1]["value_acc"]),
        reverse=True,
    )
    medals = ["🥇", "🥈", "🥉", " 4.", " 5."]

    print("\n\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║        SEARCH QUERY EXTRACTION — FORMAT LEADERBOARD           ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print(
        f"║  {'Rank':<5} {'Format':<14} {'Key F1':>7} {'Val Acc':>8} {'Parse%':>7} {'ms/q':>6} {'Train':>7}  ║"
    )
    print("╠════════════════════════════════════════════════════════════════╣")

    for rank, (fmt, m) in enumerate(ranked):
        medal = medals[rank]
        label = FORMAT_LABELS[fmt]
        f1 = f"{m['key_f1']:.3f}"
        vacc = f"{m['value_acc']:.3f}"
        pr = f"{m['parse_rate']*100:.1f}%"
        lat = f"{m['avg_latency_ms']:.0f}"
        trn = f"{m.get('train_minutes', '-')}m"
        print(
            f"║  {medal:<5} {label:<14} {f1:>7} {vacc:>8} {pr:>7} {lat:>6} {trn:>7}  ║"
        )

    print("╚════════════════════════════════════════════════════════════════╝")

    best_fmt, best_m = ranked[0]
    print(f"\n🏆  WINNER: {FORMAT_LABELS[best_fmt]}")
    print(f"    Key F1      : {best_m['key_f1']:.3f}")
    print(f"    Value Acc   : {best_m['value_acc']:.3f}")
    print(f"    Parse Rate  : {best_m['parse_rate']*100:.1f}%")
    print(f"    Avg Latency : {best_m['avg_latency_ms']:.0f} ms/query")
    print(f"    Adapter     : {best_m['adapter']}")
    if hf_username:
        print(f"    HF Repo     : huggingface.co/{hf_username}/{best_m['adapter']}")

    for fmt, m in all_results.items():
        if m["parse_rate"] < 0.80:
            print(
                f"\n⚠  Low parse rate for {FORMAT_LABELS[fmt]}: "
                f"{m['parse_rate']*100:.1f}% — model often produced malformed output."
            )


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────


def main():
    print("Loading test split...")
    raw_test = load_dataset(DATASET_REPO, split="test")
    print(f"  Test: {len(raw_test):,} rows")

    all_results = {}

    for i, fmt in enumerate(ALL_FORMATS):
        label = FORMAT_LABELS[fmt]
        adapter_name = MODEL_NAMES[fmt]

        print(f"\n{'='*62}")
        print(f"  FORMAT {i+1}/{len(ALL_FORMATS)}: {label}")
        print(f"{'='*62}")

        print(f"  Loading adapter from ./{adapter_name}/...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_name,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
        )

        print(f"  Evaluating on {EVAL_SAMPLES} test samples...")
        metrics = evaluate(model, tokenizer, raw_test, fmt, n=EVAL_SAMPLES)
        metrics["adapter"] = adapter_name
        all_results[fmt] = metrics

        print(
            f"  Results → "
            f"F1: {metrics['key_f1']:.3f} | "
            f"Value Acc: {metrics['value_acc']:.3f} | "
            f"Parse Rate: {metrics['parse_rate']:.3f} | "
            f"Latency: {metrics['avg_latency_ms']:.0f} ms"
        )

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("  VRAM cleared.")

    print_leaderboard(all_results)

    results_path = "format_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Full results saved to {results_path}")


if __name__ == "__main__":
    main()
