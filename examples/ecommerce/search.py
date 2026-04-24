"""
search.py  —  hybrid search pipeline:

    natural language query
        │
        ▼
    search-expert (structured JSON extraction)    ← hard filters
        │
        ▼
    ChromaDB metadata pre-filter
        │
        ▼
    vector similarity search on filtered set
        │
        ▼
    ranked results
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from search_expert import SearchExpert

COLLECTION_NAME = "ecommerce_products"
DB_PATH = "./chroma_db"


# ── result dataclass ──────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    id: str
    description: str
    metadata: dict[str, Any]
    score: float  # cosine similarity (higher = better)

    def __str__(self) -> str:
        price = f"${self.metadata.get('price', '?'):.2f}"
        rating = self.metadata.get("rating", "?")
        brand = self.metadata.get("brand", "")
        return (
            f"[{self.score:.3f}]  {brand}  —  {self.description[:90]}...\n"
            f"         price={price}  rating={rating}★  id={self.id}"
        )


# ── operator parsing ──────────────────────────────────────────────────────────


def _parse_operator(value: str | float | int) -> dict | None:
    """
    Turn a search-expert operator string into a ChromaDB where-clause fragment.
    e.g. "lt:200"  →  {"$lt": 200.0}
         "between:100:200" → {"$gte": 100.0, "$lte": 200.0}
         "ne:red"  →  {"$ne": "red"}
    """
    if not isinstance(value, str):
        return {"$eq": value}

    m = re.match(r"^(lt|lte|gt|gte|eq|ne|approx|between):(.+)$", value)
    if not m:
        return {"$eq": value}

    op, rest = m.group(1), m.group(2)

    def _num(s):
        return float(s.replace(",", ""))

    if op == "lt":
        return {"$lt": _num(rest)}
    if op == "lte":
        return {"$lte": _num(rest)}
    if op == "gt":
        return {"$gt": _num(rest)}
    if op == "gte":
        return {"$gte": _num(rest)}
    if op == "eq":
        try:
            return {"$eq": _num(rest)}
        except:
            return {"$eq": rest}
    if op == "ne":
        return {"$ne": rest}
    if op == "approx":
        v = _num(rest)
        return {"$gte": v * 0.85, "$lte": v * 1.15}
    if op == "between":
        lo, hi = rest.split(":")
        return {"$gte": _num(lo), "$lte": _num(hi)}

    return {"$eq": rest}


# ── filter builder ────────────────────────────────────────────────────────────

# Map from search-expert field names → ChromaDB metadata field names
FIELD_MAP = {
    "brand": "brand",
    "product": "product",
    "feature": "feature",
    "color": "color",
    "price": "price",
    "rating": "rating",
    "category": "category",
}

# Fields that are stored as comma-joined strings in ChromaDB
LIST_FIELDS = {"feature"}


def build_chroma_where(parsed: dict) -> dict | None:
    """
    Convert a search-expert JSON dict into a ChromaDB $and where-clause.
    Skips 'domain' and any fields not in FIELD_MAP.
    """
    clauses = []

    for se_field, db_field in FIELD_MAP.items():
        value = parsed.get(se_field)
        if value is None:
            continue

        # List of ne: constraints (e.g. color exclusions)
        if isinstance(value, list):
            for v in value:
                op = _parse_operator(v)
                clauses.append({db_field: op})
            continue

        # Feature matching: stored as CSV string → use $contains
        if se_field in LIST_FIELDS:
            clauses.append({db_field: {"$contains": str(value)}})
            continue

        op = _parse_operator(value)

        # Approx / between produce two keys — split into separate clauses
        if "$gte" in op and "$lte" in op:
            clauses.append({db_field: {"$gte": op["$gte"]}})
            clauses.append({db_field: {"$lte": op["$lte"]}})
        else:
            clauses.append({db_field: op})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ── main search function ──────────────────────────────────────────────────────


def hybrid_search(
    query: str,
    n_results: int = 5,
    db_path: str = DB_PATH,
    verbose: bool = True,
) -> list[SearchResult]:
    """
    Run the full hybrid search pipeline:
      1. Parse query with search-expert → structured JSON
      2. Build ChromaDB metadata filter from hard constraints
      3. Vector search within the filtered set
      4. Return ranked SearchResult objects
    """

    # 1. Parse
    expert = SearchExpert()
    parsed = expert.parse(query).fields

    if verbose:
        print(f"\n📝  Query   : {query}")
        print(f"🔍  Parsed  : {parsed}")

    # 2. Build filter
    where = build_chroma_where(parsed)

    if verbose:
        print(f"🔧  Filter  : {where}")

    # 3. Vector search
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)

    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": min(n_results, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    # 4. Build results
    results = []
    for i in range(len(raw["ids"][0])):
        doc_id = raw["ids"][0][i]
        doc_text = raw["documents"][0][i]
        meta = raw["metadatas"][0][i]
        distance = raw["distances"][0][i]
        score = 1.0 - distance  # cosine distance → similarity

        results.append(
            SearchResult(
                id=doc_id,
                description=doc_text,
                metadata=meta,
                score=score,
            )
        )

    return results


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "noise cancelling headphones any colour but black, under $200",
        "Sony wireless earbuds under $300 with hi-res audio",
        "budget laptop under $500",
        "Nike running shoes under $150 in blue",
    ]

    for q in queries:
        results = hybrid_search(q, n_results=3)
        print(f"\n{'─'*70}")
        print("Top results:")
        for r in results:
            print(f"  {r}")
        print()
