"""
benchmarks/amazon/evaluation.py

Evaluates THREE search pipelines on constraint satisfaction ONLY:

    Does the pipeline return products that actually satisfy the user's
    hard constraints (price limit, color exclusion, color requirement)?

    [A] Amazon's native ranking  — ground truth from HF dataset
    [B] search-expert hybrid     — structured filter → vector search
    [C] Pure vector search       — no filter, raw semantic similarity

Metrics (all over top-6 results):
    price_sat    — fraction of top-6 results within stated price limit
    color_excl   — fraction of top-6 results NOT in excluded colors
    color_req    — fraction of top-6 results IN required colors (if any)
    overall      — ALL applicable constraints satisfied simultaneously
    perfect      — 1.0 if ALL 6 results satisfy ALL constraints, else 0.0

Usage
─────
    python benchmarks/amazon/evaluation.py \
        --hf-dataset sarthakrastogi/amazon-search-dataset

    python benchmarks/amazon/evaluation.py \
        --dataset benchmarks/amazon/data/dataset.json \
        --debug        # prints filter details + colors/prices per query
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.utils import embedding_functions
    from search_expert import SearchExpert

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("⚠  search-expert or chromadb not installed.")


# ─────────────────────────────────────────────────────────────────────────────
# COLOR VOCABULARY
# Maps synonym/alias → canonical color stored in ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

COLOR_SYNONYMS: dict[str, str] = {
    # black
    "black": "black",
    "midnight black": "black",
    "jet black": "black",
    "matte black": "black",
    "piano black": "black",
    "carbon black": "black",
    "onyx": "black",
    "obsidian": "black",
    # white
    "white": "white",
    "snow": "white",
    "pearl": "white",
    "cream": "white",
    "ivory": "white",
    "chalk": "white",
    "frost": "white",
    # silver
    "silver": "silver",
    "platinum": "silver",
    "chrome": "silver",
    "aluminum": "silver",
    "aluminium": "silver",
    # blue
    "blue": "blue",
    "navy": "blue",
    "cobalt": "blue",
    "indigo": "blue",
    "sapphire": "blue",
    "arctic blue": "blue",
    "midnight blue": "blue",
    "steel blue": "blue",
    "royal blue": "blue",
    # red
    "red": "red",
    "crimson": "red",
    "scarlet": "red",
    "maroon": "red",
    "burgundy": "red",
    "cherry": "red",
    # green
    "green": "green",
    "olive": "green",
    "sage": "green",
    "forest": "green",
    "emerald": "green",
    "mint": "green",
    "lime": "green",
    # gold
    "gold": "gold",
    "golden": "gold",
    "champagne": "gold",
    "rose gold": "gold",
    "copper": "gold",
    # gray
    "gray": "gray",
    "grey": "gray",
    "charcoal": "gray",
    "graphite": "gray",
    "slate": "gray",
    "ash": "gray",
    "gunmetal": "gray",
    "space gray": "gray",
    "space grey": "gray",
    # pink
    "pink": "pink",
    "rose": "pink",
    "coral": "pink",
    "blush": "pink",
    "magenta": "pink",
    "fuchsia": "pink",
    # purple
    "purple": "purple",
    "violet": "purple",
    "lavender": "purple",
    "mauve": "purple",
    "lilac": "purple",
    "plum": "purple",
    # yellow/orange/brown
    "yellow": "yellow",
    "lemon": "yellow",
    "amber": "yellow",
    "orange": "orange",
    "burnt orange": "orange",
    "brown": "brown",
    "tan": "brown",
    "beige": "brown",
    "camel": "brown",
    "khaki": "brown",
    "mocha": "brown",
    # special
    "titanium": "titanium",
    "starlight": "starlight",
    "midnight": "gray",  # Apple "Midnight" = dark navy/charcoal
    "natural": "white",
}

# Sorted longest-first so multi-word phrases match before subwords
_COLOR_RE_LIST: list[tuple[re.Pattern, str]] = sorted(
    [
        (re.compile(rf"\b{re.escape(phrase)}\b", re.I), canonical)
        for phrase, canonical in COLOR_SYNONYMS.items()
    ],
    key=lambda x: -len(x[0].pattern),
)


def infer_color(text: str) -> str:
    """Return canonical color from arbitrary text, or '' if none found."""
    for pattern, canonical in _COLOR_RE_LIST:
        if pattern.search(text):
            return canonical
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRAINT PARSING
# ─────────────────────────────────────────────────────────────────────────────

_NEG_RE = re.compile(
    r"\b(?:not|no|except|but|excluding|avoid|without|other\s+than"
    r"|any\s+colou?r\s+(?:but|except)"
    r"|that\s+(?:are|is)\s+not)\b",
    re.I,
)


class QueryConstraints:
    def __init__(self, query: str):
        self.query = query
        self.price_limit: float | None = self._parse_price(query)
        self.excluded_colors: set[str] = self._parse_excluded(query)
        self.required_colors: set[str] = self._parse_required(query)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_price(query: str) -> float | None:
        for pat in [
            r"under\s+\$?([\d,]+(?:\.\d+)?)",
            r"below\s+\$?([\d,]+(?:\.\d+)?)",
            r"less\s+than\s+\$?([\d,]+(?:\.\d+)?)",
            r"<\s*\$?([\d,]+(?:\.\d+)?)",
            r"max(?:imum)?\s+\$?([\d,]+(?:\.\d+)?)",
            r"\$?([\d,]+(?:\.\d+)?)\s+(?:or\s+)?(?:less|cheaper)",
        ]:
            m = re.search(pat, query, re.I)
            if m:
                return float(m.group(1).replace(",", ""))
        return None

    @staticmethod
    def _parse_excluded(query: str) -> set[str]:
        excluded: set[str] = set()
        for pat in [
            r"(?:not|no|except|but|excluding|avoid|without)\s+([\w\s]+?)(?:\s+colou?red?|\s+colou?r)?(?=\s|,|\.|$|\band\b)",
            r"any\s+colou?r\s+(?:but|except)\s+([\w\s]+?)(?=\s|,|\.|$)",
            r"other\s+than\s+([\w\s]+?)(?:\s+colou?red?)?(?=\s|,|\.|$)",
            r"that\s+(?:are|is)\s+not\s+([\w\s]+?)(?:\s+colou?red?|\s+colou?r)?(?=\s|,|\.|$)",
        ]:
            for m in re.finditer(pat, query, re.I):
                phrase = m.group(1).strip()
                color = infer_color(phrase)
                if color:
                    excluded.add(color)
        return excluded

    @staticmethod
    def _parse_required(query: str) -> set[str]:
        excluded = QueryConstraints._parse_excluded(query)
        required: set[str] = set()
        for phrase, canonical in sorted(
            COLOR_SYNONYMS.items(), key=lambda x: -len(x[0])
        ):
            if canonical in excluded:
                continue
            pat = re.compile(rf"\b{re.escape(phrase)}\b", re.I)
            for m in pat.finditer(query):
                start = m.start()
                prefix = query[max(0, start - 50) : start]
                if not _NEG_RE.search(prefix):
                    required.add(canonical)
        return required

    # ── convenience ──────────────────────────────────────────────────────────

    def has_any(self) -> bool:
        return bool(
            self.price_limit is not None or self.excluded_colors or self.required_colors
        )

    def has_price(self) -> bool:
        return self.price_limit is not None

    def has_color_exclusion(self) -> bool:
        return bool(self.excluded_colors)

    def has_color_requirement(self) -> bool:
        return bool(self.required_colors)

    def __repr__(self) -> str:
        return (
            f"QueryConstraints(price≤{self.price_limit}, "
            f"exclude={self.excluded_colors}, require={self.required_colors})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRAINT SATISFACTION CHECK
# ─────────────────────────────────────────────────────────────────────────────


def get_product_color(product: dict) -> str:
    """Best-effort: stored color field → canonicalize → infer from title."""
    raw = (product.get("color") or "").strip()
    if raw:
        return infer_color(raw) or raw.lower()
    return infer_color(product.get("title") or "") or infer_color(
        product.get("description") or ""
    )


def check_product(product: dict, c: QueryConstraints) -> dict[str, bool | None]:
    """Per-constraint booleans. None = constraint not applicable."""
    result: dict[str, bool | None] = {
        "price": None,
        "color_excl": None,
        "color_req": None,
    }

    if c.has_price():
        price = float(product.get("price") or 0)
        result["price"] = None if price == 0 else (price <= c.price_limit)  # type: ignore

    if c.has_color_exclusion():
        color = get_product_color(product)
        result["color_excl"] = None if not color else (color not in c.excluded_colors)

    if c.has_color_requirement():
        color = get_product_color(product)
        result["color_req"] = None if not color else (color in c.required_colors)

    return result


def compute_constraint_metrics(
    products: list[dict],
    constraints: QueryConstraints,
    k: int = 6,
) -> dict[str, float] | None:
    if not constraints.has_any():
        return None

    top_k = products[:k]
    if not top_k:
        return None

    checked = [check_product(p, constraints) for p in top_k]

    def _rate(key: str) -> float | None:
        vals = [r[key] for r in checked if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None  # type: ignore

    out: dict[str, float] = {}

    if constraints.has_price():
        r = _rate("price")
        if r is not None:
            out["price_sat"] = r

    if constraints.has_color_exclusion():
        r = _rate("color_excl")
        if r is not None:
            out["color_excl"] = r

    if constraints.has_color_requirement():
        r = _rate("color_req")
        if r is not None:
            out["color_req"] = r

    # Overall: fraction of products satisfying ALL applicable constraints
    overall_vals: list[bool] = []
    for r in checked:
        applicable = {k: v for k, v in r.items() if v is not None}
        if applicable:
            overall_vals.append(all(applicable.values()))
    if overall_vals:
        out["overall"] = sum(overall_vals) / len(overall_vals)
        out["perfect"] = float(all(overall_vals))

    return out or None


# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB INDEX
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "benchmark_products"


def build_chroma_index(products: list[dict], db_path: str) -> "chromadb.Collection":
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    seen: set[str] = set()
    ids, docs, metas = [], [], []

    for p in products:
        asin = p.get("asin", "")
        if not asin or asin in seen:
            continue
        seen.add(asin)

        title = p.get("title", "")
        ids.append(asin)
        docs.append(p.get("description") or title)

        # Infer color from every available text field
        color = (
            infer_color(p.get("color") or "")
            or infer_color(title)
            or infer_color(p.get("description") or "")
            or infer_color(p.get("features_raw") or "")
        )

        metas.append(
            {
                "brand": str(p.get("brand", ""))[:500],
                "product": str(p.get("product_type", ""))[:200],
                "feature": str(p.get("features_raw", ""))[:500],
                "color": color,
                "price": float(p.get("price") or 0),
                "rating": float(p.get("rating") or 0),
                "category": str(p.get("category", ""))[:100],
                "title": title[:500],
            }
        )

    batch = 500
    for i in range(0, len(ids), batch):
        collection.add(
            ids=ids[i : i + batch],
            documents=docs[i : i + batch],
            metadatas=metas[i : i + batch],
        )

    n_color = sum(1 for m in metas if m["color"])
    n_price = sum(1 for m in metas if m["price"] > 0)
    print(f"  ✅  Indexed {len(ids)} unique products")
    print(f"       • {n_color}/{len(ids)} with color inferred")
    print(f"       • {n_price}/{len(ids)} with price > 0")
    return collection


# ─────────────────────────────────────────────────────────────────────────────
# OPERATOR PARSING & FILTER BUILDING  (mirrors search.py)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_operator(value: Any) -> dict:
    if not isinstance(value, str):
        return {"$eq": value}
    m = re.match(r"^(lt|lte|gt|gte|eq|ne|approx|between):(.+)$", value)
    if not m:
        return {"$eq": value}
    op, rest = m.group(1), m.group(2)

    def _num(s: str) -> float:
        return float(s.replace(",", ""))

    def _is_num(s: str) -> bool:
        try:
            _num(s)
            return True
        except ValueError:
            return False

    if op == "lt":
        return {"$lt": _num(rest)}
    if op == "lte":
        return {"$lte": _num(rest)}
    if op == "gt":
        return {"$gt": _num(rest)}
    if op == "gte":
        return {"$gte": _num(rest)}
    if op == "eq":
        return {"$eq": _num(rest)} if _is_num(rest) else {"$eq": rest}
    if op == "ne":
        return {"$ne": rest}
    if op == "approx":
        v = _num(rest)
        return {"$gte": v * 0.85, "$lte": v * 1.15}
    if op == "between":
        lo, hi = rest.split(":")
        return {"$gte": _num(lo), "$lte": _num(hi)}
    return {"$eq": rest}


def build_chroma_where(parsed: dict) -> dict | None:
    """
    Build a ChromaDB where-clause from search-expert parsed output.

    Only price and color are used as hard metadata filters — these are the
    only fields stored reliably in ChromaDB metadata AND where exact matching
    is semantically correct.

    Specifically excluded from filtering:
      - product / brand / feature: search-expert phrases these as natural
        language (e.g. "noise cancelling headphones") which won't match the
        short product_type strings in the index ("headphones"). Let vector
        search handle these.
      - rating / category: not stable enough to filter on.

    Color logic:
      - ne: values (exclusions) → each becomes its own $ne clause (AND logic
        is correct: exclude black AND exclude pink)
      - plain/eq values (requirements) → collected into $in so the user gets
        products matching ANY of their preferred colors (OR logic)
    """
    clauses = []

    # ── Price ──────────────────────────────────────────────────────────────
    price_val = parsed.get("price")
    if price_val is not None:
        op = _parse_operator(price_val)
        if "$gte" in op and "$lte" in op:
            clauses.append({"price": {"$gte": op["$gte"]}})
            clauses.append({"price": {"$lte": op["$lte"]}})
        else:
            clauses.append({"price": op})

    # ── Color ───────────────────────────────────────────────────────────────
    color_val = parsed.get("color")
    if color_val is not None:
        color_list = color_val if isinstance(color_val, list) else [color_val]

        exclusions = []  # ne: values  → AND'd together
        requirements = []  # plain/eq values → OR'd via $in

        for v in color_list:
            op = _parse_operator(v)
            if "$ne" in op:
                # Canonicalize the excluded color so it matches the index
                raw = op["$ne"]
                canonical = infer_color(raw) or raw.lower()
                exclusions.append(canonical)
            else:
                # Plain string or eq: → user wants this color (OR with others)
                raw = op.get("$eq", v)
                if isinstance(raw, str):
                    canonical = infer_color(raw) or raw.lower()
                    requirements.append(canonical)

        for excl in exclusions:
            clauses.append({"color": {"$ne": excl}})

        if len(requirements) == 1:
            clauses.append({"color": {"$eq": requirements[0]}})
        elif len(requirements) > 1:
            # ChromaDB supports $in for OR matching
            clauses.append({"color": {"$in": requirements}})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────


def run_hybrid(
    query: str,
    collection: "chromadb.Collection",
    expert: "SearchExpert",
    asin_to_product: dict[str, dict],
    k: int = 6,
    debug: bool = False,
) -> list[dict]:
    parsed = expert.parse(query).fields
    where = build_chroma_where(parsed)
    n = min(k, collection.count())

    if debug:
        print(f"    [hybrid] parsed = {parsed}")
        print(f"    [hybrid] where  = {where}")

    def _q(w: dict | None) -> list[str]:
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": n,
            "include": ["distances"],
        }
        if w:
            kwargs["where"] = w
        return collection.query(**kwargs)["ids"][0]

    try:
        ids = _q(where)
        if not ids and where:
            if debug:
                print("    [hybrid] filter → 0 results, falling back to no filter")
            ids = _q(None)
    except Exception as e:
        if debug:
            print(f"    [hybrid] filter error ({e}), falling back")
        ids = _q(None)

    return [asin_to_product[a] for a in ids if a in asin_to_product]


def run_vector(
    query: str,
    collection: "chromadb.Collection",
    asin_to_product: dict[str, dict],
    k: int = 6,
) -> list[dict]:
    n = min(k, collection.count())
    ids = collection.query(query_texts=[query], n_results=n, include=["distances"])[
        "ids"
    ][0]
    return [asin_to_product[a] for a in ids if a in asin_to_product]


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(path: str | None, hf_repo: str | None) -> list[dict]:
    if hf_repo:
        try:
            from datasets import load_dataset as hf_load
        except ImportError:
            print("❌  pip install datasets")
            sys.exit(1)
        return list(hf_load(hf_repo, split="train"))
    if path:
        with open(path) as f:
            return json.load(f)
    raise ValueError("Provide --dataset or --hf-dataset")


def group_by_query(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["query"]].append(r)
    for q in grouped:
        grouped[q].sort(key=lambda x: x["amazon_rank"])
    return dict(grouped)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    grouped: dict[str, list[dict]],
    collection: "chromadb.Collection",
    expert: "SearchExpert",
    asin_to_product: dict[str, dict],
    top_k: int = 6,
    debug: bool = False,
) -> dict:
    amz_list: list[dict] = []
    hyb_list: list[dict] = []
    vec_list: list[dict] = []
    per_query: list[dict] = []
    n_skipped = 0

    for query, products in grouped.items():
        constraints = QueryConstraints(query)

        if not constraints.has_any():
            n_skipped += 1
            continue

        # Amazon: already sorted by amazon_rank, treat that as their pipeline output
        amz_products = products
        hyb_products = run_hybrid(
            query, collection, expert, asin_to_product, k=top_k, debug=debug
        )
        vec_products = run_vector(query, collection, asin_to_product, k=top_k)

        amz_m = compute_constraint_metrics(amz_products, constraints, k=top_k) or {}
        hyb_m = compute_constraint_metrics(hyb_products, constraints, k=top_k) or {}
        vec_m = compute_constraint_metrics(vec_products, constraints, k=top_k) or {}

        if amz_m:
            amz_list.append(amz_m)
        if hyb_m:
            hyb_list.append(hyb_m)
        if vec_m:
            vec_list.append(vec_m)

        a_ov = amz_m.get("overall", float("nan"))
        h_ov = hyb_m.get("overall", float("nan"))
        v_ov = vec_m.get("overall", float("nan"))

        print(f"  {query}  AMZ={a_ov:.2f}  H={h_ov:.2f}  V={v_ov:.2f}")

        if debug:
            print(f"    constraints : {constraints}")
            print(
                f"    AMZ colors  : {[get_product_color(p) for p in amz_products[:top_k]]}"
            )
            print(
                f"    HYB colors  : {[get_product_color(p) for p in hyb_products[:top_k]]}"
            )
            print(
                f"    VEC colors  : {[get_product_color(p) for p in vec_products[:top_k]]}"
            )
            print(
                f"    AMZ prices  : {[float(p.get('price') or 0) for p in amz_products[:top_k]]}"
            )
            print(
                f"    HYB prices  : {[float(p.get('price') or 0) for p in hyb_products[:top_k]]}"
            )
            print(
                f"    VEC prices  : {[float(p.get('price') or 0) for p in vec_products[:top_k]]}"
            )

        per_query.append(
            {
                "query": query,
                "constraints": {
                    "price_limit": constraints.price_limit,
                    "excluded_colors": list(constraints.excluded_colors),
                    "required_colors": list(constraints.required_colors),
                },
                "amazon": {
                    "metrics": amz_m,
                    "top_titles": [
                        p.get("title", "")[:70] for p in amz_products[:top_k]
                    ],
                    "top_colors": [get_product_color(p) for p in amz_products[:top_k]],
                    "top_prices": [
                        float(p.get("price") or 0) for p in amz_products[:top_k]
                    ],
                },
                "hybrid": {
                    "metrics": hyb_m,
                    "top_titles": [
                        p.get("title", "")[:70] for p in hyb_products[:top_k]
                    ],
                    "top_colors": [get_product_color(p) for p in hyb_products[:top_k]],
                    "top_prices": [
                        float(p.get("price") or 0) for p in hyb_products[:top_k]
                    ],
                },
                "pure_vector": {
                    "metrics": vec_m,
                    "top_titles": [
                        p.get("title", "")[:70] for p in vec_products[:top_k]
                    ],
                    "top_colors": [get_product_color(p) for p in vec_products[:top_k]],
                    "top_prices": [
                        float(p.get("price") or 0) for p in vec_products[:top_k]
                    ],
                },
            }
        )

    def _avg(lst: list[dict]) -> dict[str, float]:
        if not lst:
            return {}
        keys = {k for d in lst for k in d}
        out = {}
        for k in keys:
            vals = [d[k] for d in lst if k in d]
            out[k] = sum(vals) / len(vals) if vals else 0.0
        return out

    return {
        "n_queries_total": len(grouped),
        "n_queries_with_constraints": len(per_query),
        "n_queries_skipped": n_skipped,
        "amazon": _avg(amz_list),
        "hybrid": _avg(hyb_list),
        "pure_vector": _avg(vec_list),
        "per_query": per_query,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "price_sat": "Price satisfaction",
    "color_excl": "Color exclusion sat.",
    "color_req": "Color match sat.",
    "overall": "Overall (all constraints)",
    "perfect": "Perfect@K (all K correct)",
}


def print_report(results: dict) -> None:
    n_total = results["n_queries_total"]
    n_cst = results["n_queries_with_constraints"]
    amz = results["amazon"]
    hyb = results["hybrid"]
    vec = results["pure_vector"]

    W = 82
    print(f"\n{'═' * W}")
    print(
        f"  CONSTRAINT SATISFACTION BENCHMARK  ({n_cst}/{n_total} queries had constraints)"
    )
    print(f"{'═' * W}")
    print(
        f"  {'Metric':<28}  {'Amazon':>9}  {'Hybrid (SE)':>11}  {'Pure Vector':>11}"
        f"  {'H–AMZ':>7}  {'H–Vec':>7}"
    )
    print(f"  {'─' * 28}  {'─' * 9}  {'─' * 11}  {'─' * 11}  {'─' * 7}  {'─' * 7}")

    any_row = False
    for key, label in METRIC_LABELS.items():
        av = amz.get(key)
        hv = hyb.get(key)
        vv = vec.get(key)
        if av is None and hv is None and vv is None:
            continue
        any_row = True
        av, hv, vv = av or 0.0, hv or 0.0, vv or 0.0
        h_a = hv - av
        h_v = hv - vv
        print(
            f"  {label:<28}  {av:>9.4f}  {hv:>11.4f}  {vv:>11.4f}"
            f"  {'+' if h_a >= 0 else ''}{h_a:>6.4f}  {'+' if h_v >= 0 else ''}{h_v:>6.4f}"
        )

    if not any_row:
        print("  (no parseable constraints found in these queries)")

    print(f"\n{'═' * W}")
    print("  Winners per metric (higher = better):")
    for key, label in METRIC_LABELS.items():
        av, hv, vv = amz.get(key, 0.0), hyb.get(key, 0.0), vec.get(key, 0.0)
        if av == hv == vv == 0:
            continue
        winner, score = max(
            [("Amazon", av), ("Hybrid", hv), ("PureVec", vv)],
            key=lambda x: x[1],
        )
        print(f"    {label:<28} →  {winner}  ({score:.4f})")

    print(f"{'═' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Constraint satisfaction benchmark")
    p.add_argument("--dataset", default="", help="Path to local dataset.json")
    p.add_argument(
        "--hf-dataset",
        default="sarthakrastogi/amazon-search-dataset",
        help="HuggingFace dataset repo ID",
    )
    p.add_argument("--report", default="benchmarks/amazon/results/report.json")
    p.add_argument(
        "--db-path", default="", help="ChromaDB persistent path (default: tmpdir)"
    )
    p.add_argument(
        "--top-k", type=int, default=6, help="Results per pipeline to evaluate"
    )
    p.add_argument("--max-queries", type=int, default=25, help="N queries to evaluate")
    p.add_argument(
        "--debug", action="store_true", help="Print filter + color/price details"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not PIPELINE_AVAILABLE:
        print("❌  Install search-expert and chromadb.")
        sys.exit(1)

    print("📂  Loading dataset…")
    rows = load_dataset(args.dataset or None, args.hf_dataset or None)
    grouped = group_by_query(rows)
    grouped = dict(list(grouped.items())[: args.max_queries])
    print(f"    {len(rows)} rows, {len(grouped)} queries")

    asin_to_product: dict[str, dict] = {}
    for row in rows:
        asin = row.get("asin", "")
        if asin and asin not in asin_to_product:
            asin_to_product[asin] = row

    db_path = args.db_path or tempfile.mkdtemp(prefix="benchmark_chroma_")
    print(f"\n🗄   Building ChromaDB index at {db_path}…")
    collection = build_chroma_index(rows, db_path=db_path)

    print("\n🤖  Loading search-expert model…")
    expert = SearchExpert()

    print(f"\n{'─' * 72}")
    print("  Running evaluation…")
    print(f"{'─' * 72}")
    results = evaluate(
        grouped,
        collection,
        expert,
        asin_to_product,
        top_k=args.top_k,
        debug=args.debug,
    )

    print_report(results)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📄  Full report saved → {report_path}")
