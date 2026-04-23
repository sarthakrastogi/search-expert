# -*- coding: utf-8 -*-
"""
evaluate.py — Evaluate fine-tuned LoRA adapters + benchmark vs. pure vector search
====================================================================================
Two evaluation modes run on the same held-out test samples:

  1. STRUCTURED (search-expert adapters)
     Loads each LoRA adapter, generates structured output, parses it,
     and scores field-level precision / recall / F1 and value exact-match.

  2. VECTOR BASELINE
     Embeds every test query with sentence-transformers (all-MiniLM-L6-v2),
     retrieves the top-k most similar queries from the test corpus by cosine
     similarity, and treats their ground-truth structured fields as the
     "prediction" via majority vote. This simulates what a pure vector search
     would surface — semantically close results that ignore hard constraints.

The final leaderboard shows both side-by-side so the gap is clear.

Usage:
    python evaluate.py
"""

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

DATASET_REPO = "sarthakrastogi/search-expert"
EVAL_SAMPLES = 300  # test rows evaluated per structured adapter
VECTOR_SAMPLES = 300  # test rows evaluated for the vector baseline
VECTOR_TOP_K = 5  # k nearest neighbours to retrieve per query
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True

# Sentence-transformer model used for the vector baseline
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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
# SHARED SCORING HELPERS
# ──────────────────────────────────────────────────────────────


def score_prediction(gt: dict, pred: dict) -> dict:
    """
    Returns per-sample key hit / total counts and value hit / total counts.
    Used identically for structured and vector results.
    """
    gt_keys = set(gt.keys())
    pred_keys = set(pred.keys())
    shared = gt_keys & pred_keys

    val_hits = sum(
        1 for k in shared if str(pred[k]).strip().lower() == str(gt[k]).strip().lower()
    )
    return {
        "key_hits": len(shared),
        "key_pred_total": len(pred_keys),
        "key_gt_total": len(gt_keys),
        "val_hits": val_hits,
        "val_total": len(shared),
    }


def aggregate_scores(
    scores: list[dict],
    total_ms: float,
    n_samples: int,
    parse_successes: int,
) -> dict:
    key_hits = sum(s["key_hits"] for s in scores)
    key_pred_total = sum(s["key_pred_total"] for s in scores)
    key_gt_total = sum(s["key_gt_total"] for s in scores)
    val_hits = sum(s["val_hits"] for s in scores)
    val_total = sum(s["val_total"] for s in scores)

    precision = key_hits / key_pred_total if key_pred_total else 0.0
    recall = key_hits / key_gt_total if key_gt_total else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    val_acc = val_hits / val_total if val_total else 0.0
    parse_rt = parse_successes / n_samples if n_samples else 0.0
    avg_lat = total_ms / n_samples if n_samples else 0.0

    return {
        "key_precision": precision,
        "key_recall": recall,
        "key_f1": f1,
        "value_acc": val_acc,
        "parse_rate": parse_rt,
        "avg_latency_ms": avg_lat,
    }


# ──────────────────────────────────────────────────────────────
# STRUCTURED ADAPTER EVALUATION
# ──────────────────────────────────────────────────────────────


def evaluate_adapter(model, tokenizer, raw_test, fmt: str, n: int = 300) -> dict:
    """
    Runs inference with the loaded LoRA adapter and scores field-level
    precision / recall / F1 and value exact-match against ground truth.

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

    scores = []
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

        scores.append(score_prediction(gt, pred))

    return aggregate_scores(scores, total_ms, len(indices), parse_successes)


# ──────────────────────────────────────────────────────────────
# VECTOR SEARCH BASELINE
# ──────────────────────────────────────────────────────────────


def build_vector_index(queries: list[str], embed_model) -> torch.Tensor:
    """
    Embeds all queries and returns a (N, D) normalised float32 tensor.
    """
    print(f"  Embedding {len(queries):,} queries for vector index...")
    embeddings = embed_model.encode(
        queries,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    return embeddings.float()  # (N, D)


def evaluate_vector_baseline(raw_test, fmt: str, n: int = 300, top_k: int = 5) -> dict:
    """
    Simulates pure vector search retrieval — no hard constraint filtering.

    Strategy:
      For each sampled query, find the top_k most similar queries in the
      rest of the corpus by cosine similarity (dot product of normalised
      embeddings). Merge the structured fields from those top_k neighbours
      via majority vote per field. Score the merged prediction against the
      ground-truth structured output of the original query.

    This answers: "if a user issued this query and we returned semantically
    similar results WITHOUT applying any hard filters, how accurately would
    those results reflect the query's structured intent?"

    Returns the same metric keys as evaluate_adapter() for direct comparison,
    plus vector_retrieval_ms and embed_index_ms.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for the vector baseline.\n"
            "Install it with: pip install sentence-transformers"
        )

    parser = PARSERS[fmt]
    all_rows = list(raw_test)
    all_n = len(all_rows)
    all_queries = [r["query"] for r in all_rows]

    print(f"  Loading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)

    t_embed_start = time.perf_counter()
    all_embeddings = build_vector_index(all_queries, embed_model)
    embed_ms = (time.perf_counter() - t_embed_start) * 1000
    print(f"  ✓ Indexed {all_n:,} queries in {embed_ms/1000:.1f}s")

    indices = random.sample(range(all_n), min(n, all_n))

    scores = []
    total_retrieval_ms = 0.0
    parse_successes = 0

    for i in indices:
        row = all_rows[i]
        gt_raw = row[fmt]
        gt = parser(gt_raw)
        if not gt:
            continue

        # Cosine similarity (embeddings are normalised → dot product = cosine)
        query_vec = all_embeddings[i].unsqueeze(0)  # (1, D)
        sims = (all_embeddings @ query_vec.T).squeeze(1)  # (N,)
        sims[i] = -1.0  # mask the query itself out

        t0 = time.perf_counter()
        top_k_indices = torch.topk(sims, k=min(top_k, all_n - 1)).indices.tolist()
        total_retrieval_ms += (time.perf_counter() - t0) * 1000

        # Majority vote across neighbours' structured fields
        # This is the closest proxy to what vector search would return:
        # the most common field values among the top-k semantically similar queries.
        field_votes: dict[str, dict[str, int]] = {}
        for nb_idx in top_k_indices:
            nb_parsed = parser(all_rows[nb_idx][fmt])
            for field, val in nb_parsed.items():
                field_votes.setdefault(field, {})
                val_str = (
                    json.dumps(val, sort_keys=True)
                    if isinstance(val, list)
                    else str(val)
                )
                field_votes[field][val_str] = field_votes[field].get(val_str, 0) + 1

        pred = {}
        for field, votes in field_votes.items():
            best_val_str = max(votes, key=votes.__getitem__)
            try:
                pred[field] = json.loads(best_val_str)
            except Exception:
                pred[field] = best_val_str

        if pred:
            parse_successes += 1

        scores.append(score_prediction(gt, pred))

    metrics = aggregate_scores(
        scores, total_retrieval_ms, len(indices), parse_successes
    )
    metrics["vector_retrieval_ms"] = (
        total_retrieval_ms / len(indices) if indices else 0.0
    )
    metrics["embed_index_ms"] = embed_ms
    return metrics


# ──────────────────────────────────────────────────────────────
# LEADERBOARD PRINTER
# ──────────────────────────────────────────────────────────────


def print_leaderboard(
    adapter_results: dict,
    vector_results: dict,
    hf_username: str = "",
) -> None:
    """
    Prints two panels:
      Panel 1 — Structured adapter leaderboard (ranked by key_f1)
      Panel 2 — Head-to-head: best adapter vs. pure vector search
    """
    ranked = sorted(
        adapter_results.items(),
        key=lambda x: (x[1]["key_f1"], x[1]["value_acc"]),
        reverse=True,
    )
    medals = ["🥇", "🥈", "🥉", " 4.", " 5."]

    # ── Panel 1: Adapter leaderboard ──────────────────────────
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          SEARCH QUERY EXTRACTION — FORMAT LEADERBOARD           ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(
        f"║  {'Rank':<5} {'Format':<14} {'Key F1':>7} {'Val Acc':>8} {'Parse%':>7} {'ms/q':>7}  ║"
    )
    print("╠══════════════════════════════════════════════════════════════════╣")

    for rank, (fmt, m) in enumerate(ranked):
        medal = medals[rank]
        label = FORMAT_LABELS[fmt]
        f1 = f"{m['key_f1']:.3f}"
        vacc = f"{m['value_acc']:.3f}"
        pr = f"{m['parse_rate']*100:.1f}%"
        lat = f"{m['avg_latency_ms']:.0f}"
        print(f"║  {medal:<5} {label:<14} {f1:>7} {vacc:>8} {pr:>7} {lat:>7}  ║")

    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── Panel 2: Head-to-head vs. vector search ───────────────
    best_fmt, best_m = ranked[0]
    vm = vector_results

    def signed(a: float, b: float) -> str:
        d = a - b
        return f"+{d:.3f}" if d >= 0 else f"{d:.3f}"

    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         STRUCTURED (search-expert) vs. PURE VECTOR SEARCH       ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  {'Metric':<22} {'search-expert':>15} {'Vector Search':>15} {'Δ':>8}  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")

    comparison_rows = [
        ("Key Precision", best_m["key_precision"], vm["key_precision"]),
        ("Key Recall", best_m["key_recall"], vm["key_recall"]),
        ("Key F1", best_m["key_f1"], vm["key_f1"]),
        ("Value Acc", best_m["value_acc"], vm["value_acc"]),
        ("Parse Rate", best_m["parse_rate"], vm["parse_rate"]),
    ]
    for label, se_val, vs_val in comparison_rows:
        print(
            f"║  {label:<22} {se_val:>15.3f} {vs_val:>15.3f} {signed(se_val, vs_val):>8}  ║"
        )

    print("╠══════════════════════════════════════════════════════════════════╣")
    lat_se = f"{best_m['avg_latency_ms']:.0f} ms"
    lat_vs = f"{vm['avg_latency_ms']:.1f} ms"
    print(f"║  {'Avg latency/query':<22} {lat_se:>15} {lat_vs:>15} {'':>8}  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    print(f"\n🏆  Best adapter : {FORMAT_LABELS[best_fmt]}")
    if hf_username:
        print(f"    HF Repo      : huggingface.co/{hf_username}/{best_m['adapter']}")

    # Explain the gap
    gap = best_m["key_f1"] - vm["key_f1"]
    if gap > 0.05:
        print(
            f"\n📊  search-expert outperforms pure vector search on Key F1 by "
            f"{gap:.3f} ({gap*100:.1f} pp). "
            f"Vector search surfaces semantically similar results but ignores hard "
            f"constraints — price, exclusions, and categorical filters drift freely."
        )

    # Warn on low parse rates
    for fmt, m in adapter_results.items():
        if m["parse_rate"] < 0.80:
            print(
                f"\n⚠  Low parse rate for {FORMAT_LABELS[fmt]}: "
                f"{m['parse_rate']*100:.1f}% — model often produced malformed output."
            )


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────


def main():
    import os

    hf_username = os.environ.get("HF_USERNAME", "")

    print("Loading test split...")
    raw_test = load_dataset(DATASET_REPO, split="test")
    print(f"  Test: {len(raw_test):,} rows")

    adapter_results = {}

    # ── Structured adapter evaluation ─────────────────────────
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
        metrics = evaluate_adapter(model, tokenizer, raw_test, fmt, n=EVAL_SAMPLES)
        metrics["adapter"] = adapter_name
        adapter_results[fmt] = metrics

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

    # ── Vector search baseline ─────────────────────────────────
    # Uses the first format's ground truth as reference (JSON is default)
    baseline_fmt = ALL_FORMATS[0]

    print(f"\n{'='*62}")
    print(
        f"  VECTOR BASELINE  (sentence-transformers · top-{VECTOR_TOP_K} majority vote)"
    )
    print(f"{'='*62}")

    vector_metrics = evaluate_vector_baseline(
        raw_test,
        fmt=baseline_fmt,
        n=VECTOR_SAMPLES,
        top_k=VECTOR_TOP_K,
    )
    vector_metrics["adapter"] = f"vector-{EMBED_MODEL.split('/')[-1]}"

    print(
        f"  Results → "
        f"F1: {vector_metrics['key_f1']:.3f} | "
        f"Value Acc: {vector_metrics['value_acc']:.3f} | "
        f"Retrieval: {vector_metrics['avg_latency_ms']:.1f} ms/q"
    )

    # ── Leaderboards ───────────────────────────────────────────
    print_leaderboard(adapter_results, vector_metrics, hf_username=hf_username)

    # ── Save ───────────────────────────────────────────────────
    all_results = {
        "adapters": adapter_results,
        "vector_baseline": vector_metrics,
    }
    results_path = "format_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Full results saved to {results_path}")


if __name__ == "__main__":
    main()
