"""
benchmarks/amazon/data_generation.py

Searches Amazon for a set of benchmark queries using amazon-buddy (unofficial
scraper), extracts rich product metadata, and uploads the resulting dataset to
HuggingFace as sarthakrastogi/amazon-search-dataset.

Dataset schema
──────────────
Each row in the HuggingFace dataset represents ONE (query, product) pair:

    query               str   — the search string used
    amazon_rank         int   — position in Amazon's result list (1-based)
    is_amazon_top6      bool  — True if Amazon returned this in positions 1-6
    asin                str
    title               str
    description         str   — concatenated feature bullets (for vector search)
    brand               str
    product_type        str   — inferred coarse category ("headphones", etc.)
    price               float
    currency            str
    discounted          bool
    original_price      float
    savings_percent     float
    rating              float
    total_reviews       int
    amazon_choice       bool
    best_seller         bool
    amazon_prime        bool
    sponsored           bool
    color               str   — extracted from title/features heuristically
    features_raw        str   — comma-joined feature bullets
    category            str   — Amazon search category used
    url                 str
    thumbnail           str

Usage
─────
    # Install deps first:
    #   npm i -g amazon-buddy
    #   pip install datasets huggingface_hub tqdm

    python data_generation.py \
        --hf-token  hf_xxxx \
        --num-products 50 \
        --country US

    # Dry run (no upload):
    python data_generation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ── benchmark queries ─────────────────────────────────────────────────────────
# Chosen to cover a wide range of constraint types:
#   price-only, brand+feature, color+exclusion, category-only, rating-implicit
BENCHMARK_QUERIES = [
    # Headphones / earbuds
    "noise cancelling headphones under $200",
    "wireless noise cancelling headphones",
    "Sony noise cancelling headphones",
    "Bose noise cancelling headphones",
    "noise cancelling headphones white",
    "noise cancelling headphones blue",
    "budget noise cancelling headphones under $100",
    "professional noise cancelling headphones",
    "Sony wireless earbuds hi-res audio",
    "wireless earbuds under $100",
    "true wireless earbuds noise cancelling",
    "Apple AirPods Pro",
    # Laptops
    "budget laptop under $500",
    "gaming laptop under $1000",
    "lightweight laptop for students",
    "MacBook Air M3",
    "Dell XPS laptop",
    "2-in-1 laptop touchscreen",
    # Running shoes
    "Nike running shoes blue",
    "Nike running shoes under $150",
    "ASICS running shoes cushioned",
    "Hoka running shoes",
    "Brooks running shoes neutral",
    "Adidas running shoes",
    # Smartphones
    "Samsung Galaxy S25",
    "Google Pixel 9",
    "budget Android phone under $400",
    # Smartwatches
    "Garmin GPS running watch",
    "fitness smartwatch under $200",
    "Apple Watch Series 9",
    # Cameras
    "mirrorless camera for beginners",
    "Sony mirrorless camera vlogging",
    "Canon mirrorless camera 4K",
    # Monitors
    "4K monitor USB-C",
    "gaming monitor 144Hz under $300",
    "ultrawide curved monitor",
    # Mechanical keyboards
    "wireless mechanical keyboard",
    "compact 75% mechanical keyboard",
    # Tablets
    "iPad Air",
    "Android tablet with stylus",
    # Gaming
    "Nintendo Switch OLED",
    "PlayStation 5",
]

# ── product-type inference ────────────────────────────────────────────────────

PRODUCT_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bearbuds?\b|\bairpods?\b|\bbuds?\b", re.I), "earbuds"),
    (re.compile(r"\bheadphones?\b|\bover.?ear\b|\bon.?ear\b", re.I), "headphones"),
    (re.compile(r"\blaptop\b|\bnotebook\b|\bmacbook\b|\bchromebook\b", re.I), "laptop"),
    (
        re.compile(r"\brunning shoes?\b|\bsneakers?\b|\btrainers?\b", re.I),
        "running shoes",
    ),
    (
        re.compile(r"\bsmartwatch\b|\bwatch\b|\bfitband\b|\bfitbit\b|\bgarmin\b", re.I),
        "smartwatch",
    ),
    (re.compile(r"\bcamera\b|\bdslr\b|\bmirrorless\b", re.I), "camera"),
    (re.compile(r"\bmonitor\b|\bdisplay\b|\bscreen\b", re.I), "monitor"),
    (re.compile(r"\bkeyboard\b", re.I), "keyboard"),
    (re.compile(r"\btablet\b|\bipad\b", re.I), "tablet"),
    (
        re.compile(r"\bphone\b|\bsmartphone\b|\bpixel\b|\bgalaxy\b|\biphone\b", re.I),
        "smartphone",
    ),
    (
        re.compile(
            r"\bconsole\b|\bplaystation\b|\bxbox\b|\bnintendo\b|\bswitch\b", re.I
        ),
        "gaming console",
    ),
]

COLOR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bblack\b", re.I), "black"),
    (re.compile(r"\bwhite\b|\bplatinum\b|\bsilver\b|\bpearl\b", re.I), "white"),
    (re.compile(r"\bsilver\b", re.I), "silver"),
    (re.compile(r"\bblue\b|\bnavy\b|\bindigo\b|\bcobalt\b", re.I), "blue"),
    (re.compile(r"\bred\b|\bcrimson\b|\bscarlet\b", re.I), "red"),
    (re.compile(r"\bgreen\b|\bsage\b|\bolive\b|\bmidnight green\b", re.I), "green"),
    (re.compile(r"\bgold\b|\bchampagne\b|\brose gold\b", re.I), "gold"),
    (re.compile(r"\bgray\b|\bgrey\b|\bcharcoal\b|\bgraphite\b", re.I), "grey"),
    (re.compile(r"\bpink\b|\brose\b|\bcoral\b", re.I), "pink"),
    (re.compile(r"\bpurple\b|\bviolet\b|\blavender\b|\bmauve\b", re.I), "purple"),
    (re.compile(r"\byellow\b|\blime\b", re.I), "yellow"),
    (re.compile(r"\borange\b", re.I), "orange"),
    (re.compile(r"\bbrown\b|\btan\b|\bbeige\b|\bcamel\b", re.I), "brown"),
    (re.compile(r"\btitanium\b", re.I), "titanium"),
    (re.compile(r"\bspace gr[ae]y\b|\bmidnight\b", re.I), "space grey"),
    (re.compile(r"\bstarlight\b", re.I), "starlight"),
]

CATEGORY_MAP: dict[str, str] = {
    "headphones": "electronics",
    "earbuds": "electronics",
    "laptop": "computers",
    "running shoes": "footwear",
    "smartwatch": "wearables",
    "camera": "cameras",
    "monitor": "computers",
    "keyboard": "peripherals",
    "tablet": "computers",
    "smartphone": "phones",
    "gaming console": "gaming",
}


def infer_product_type(text: str) -> str:
    for pattern, ptype in PRODUCT_TYPE_PATTERNS:
        if pattern.search(text):
            return ptype
    return "other"


def infer_color(text: str) -> str:
    for pattern, color in COLOR_PATTERNS:
        if pattern.search(text):
            return color
    return ""


# ── amazon-buddy wrapper ──────────────────────────────────────────────────────


def scrape_amazon(
    keyword: str,
    number: int = 50,
    country: str = "US",
    retries: int = 3,
    delay: float = 2.0,
) -> list[dict]:
    """
    Call amazon-buddy CLI, parse JSON output.
    Returns list of raw product dicts (may be empty on failure).
    """
    cmd = [
        "amazon-buddy",
        "products",
        "-k",
        keyword,
        "-n",
        str(number),
        "--country",
        country,
        "--filetype",
        "",  # stdout only, no file saved
        "--timeout",
        "1500",
        "--random-ua",
    ]

    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            stdout = result.stdout.strip()
            if not stdout:
                raise ValueError("Empty output from amazon-buddy")

            # amazon-buddy prints raw JSON array to stdout when filetype=""
            data = json.loads(stdout)
            if isinstance(data, list):
                return data
            # Sometimes wrapped: {"result": [...]}
            if isinstance(data, dict) and "result" in data:
                return data["result"]
            return []

        except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as exc:
            print(f"  ⚠  Attempt {attempt}/{retries} failed for '{keyword}': {exc}")
            if attempt < retries:
                time.sleep(delay * attempt)

    return []


# ── field extraction ──────────────────────────────────────────────────────────


def extract_product(
    raw: dict,
    query: str,
    amazon_rank: int,
) -> dict[str, Any]:
    """
    Normalise one raw amazon-buddy product dict into the benchmark schema.
    """
    title = raw.get("title", "")
    asin = raw.get("asin", "")
    url = raw.get("url", f"https://www.amazon.com/dp/{asin}")

    # Price
    price_block = raw.get("price", {}) or {}
    price = float(price_block.get("current_price") or 0)
    currency = price_block.get("currency", "USD") or "USD"
    discounted = bool(price_block.get("discounted", False))
    original_price = float(price_block.get("before_price") or price)
    savings_percent = float(price_block.get("savings_percent") or 0)

    # Reviews
    reviews_block = raw.get("reviews", {}) or {}
    rating = float(reviews_block.get("rating") or 0)
    total_reviews = int(reviews_block.get("total_reviews") or 0)

    # Badges
    amazon_choice = bool(raw.get("amazonChoice", False))
    best_seller = bool(raw.get("bestSeller", False))
    amazon_prime = bool(raw.get("amazonPrime", False))
    sponsored = bool(raw.get("sponsored", False))

    # Feature bullets (from .asin() enrichment — may be empty in search results)
    feature_bullets: list[str] = raw.get("feature_bullets", []) or []
    features_raw = ",".join(str(b) for b in feature_bullets if b)

    # Description for vector search = title + bullets
    description = title
    if feature_bullets:
        description = title + ". " + ". ".join(str(b) for b in feature_bullets)

    # Inferred fields
    search_text = title + " " + features_raw
    product_type = infer_product_type(search_text)
    color = infer_color(search_text)
    category = CATEGORY_MAP.get(product_type, "other")

    # Brand: amazon-buddy sometimes has it nested; otherwise parse from title
    brand = ""
    prod_info = raw.get("product_information", {}) or {}
    raw_brand = prod_info.get("brand", "") or ""
    if raw_brand and not raw_brand.lower().startswith("visit"):
        brand = raw_brand
    if not brand:
        # First capitalised word of title is usually the brand
        match = re.match(r"^([A-Z][A-Za-z0-9\-&]+)", title)
        if match:
            brand = match.group(1)

    return {
        "query": query,
        "amazon_rank": amazon_rank,
        "is_amazon_top6": amazon_rank <= 6,
        "asin": asin,
        "title": title,
        "description": description,
        "brand": brand,
        "product_type": product_type,
        "price": price,
        "currency": currency,
        "discounted": discounted,
        "original_price": original_price,
        "savings_percent": savings_percent,
        "rating": rating,
        "total_reviews": total_reviews,
        "amazon_choice": amazon_choice,
        "best_seller": best_seller,
        "amazon_prime": amazon_prime,
        "sponsored": sponsored,
        "color": color,
        "features_raw": features_raw,
        "category": category,
        "url": url,
        "thumbnail": raw.get("thumbnail", ""),
    }


# ── main pipeline ─────────────────────────────────────────────────────────────


def generate_dataset(
    queries: list[str],
    num_products: int,
    country: str,
    output_path: Path,
    delay_between_queries: float = 3.0,
) -> list[dict]:
    all_rows: list[dict] = []

    for query in tqdm(queries, desc="Scraping Amazon"):
        print(f"\n🔍  Querying: '{query}'")
        raw_products = scrape_amazon(query, number=num_products, country=country)

        if not raw_products:
            print(f"  ⚠  No results returned for '{query}'")
            continue

        for rank, raw in enumerate(raw_products[:num_products], start=1):
            row = extract_product(raw, query=query, amazon_rank=rank)
            all_rows.append(row)

        print(
            f"  ✅  {len(raw_products)} products scraped (keeping top {min(len(raw_products), num_products)})"
        )
        time.sleep(delay_between_queries)

    # Save locally as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n💾  Saved {len(all_rows)} rows → {output_path}")

    return all_rows


def upload_to_huggingface(
    rows: list[dict],
    hf_token: str,
    repo_id: str = "sarthakrastogi/amazon-search-dataset",
) -> None:
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        print("❌  Missing deps: pip install datasets huggingface_hub")
        sys.exit(1)

    print(f"\n📤  Uploading {len(rows)} rows to HuggingFace → {repo_id}")

    ds = Dataset.from_list(rows)

    # Push with README card
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    ds.push_to_hub(
        repo_id,
        token=hf_token,
        commit_message=f"Add {len(rows)} rows across {len(set(r['query'] for r in rows))} queries",
    )

    # Write a dataset card
    card = f"""---
license: mit
task_categories:
  - text-retrieval
  - feature-extraction
language:
  - en
tags:
  - ecommerce
  - search
  - product-retrieval
  - hybrid-search
  - search-expert
---

# amazon-search-dataset

Real Amazon product data scraped across **{len(set(r['query'] for r in rows))} search queries**,
covering {len(rows)} (query, product) pairs.

Used to benchmark [search-expert](https://github.com/sarthakrastogi/search-expert)'s
hybrid search pipeline against Amazon's native search ranking.

## Schema

| Field | Type | Description |
|---|---|---|
| `query` | str | Original search query |
| `amazon_rank` | int | Position in Amazon result list (1-based) |
| `is_amazon_top6` | bool | True if Amazon returned this in positions 1-6 |
| `asin` | str | Amazon Standard Identification Number |
| `title` | str | Product title |
| `description` | str | Title + feature bullets (for vector search) |
| `brand` | str | Brand name |
| `product_type` | str | Inferred product category |
| `price` | float | Current price |
| `currency` | str | Currency code |
| `discounted` | bool | Whether price is discounted |
| `original_price` | float | Pre-discount price |
| `savings_percent` | float | Discount percentage |
| `rating` | float | Star rating (1-5) |
| `total_reviews` | int | Number of reviews |
| `amazon_choice` | bool | Amazon's Choice badge |
| `best_seller` | bool | Best Seller badge |
| `amazon_prime` | bool | Prime eligible |
| `sponsored` | bool | Sponsored listing |
| `color` | str | Inferred color from title/features |
| `features_raw` | str | Comma-joined feature bullets |
| `category` | str | Top-level category |
| `url` | str | Amazon product URL |
| `thumbnail` | str | Thumbnail image URL |

## Benchmark purpose

The `is_amazon_top6` field defines Amazon's "ground truth" for each query.
`evaluation.py` runs the search-expert hybrid pipeline on the same queries and
product pool, then computes Precision@K, Recall@K, NDCG, and MRR against this
ground truth.
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )

    print(f"✅  Upload complete: https://huggingface.co/datasets/{repo_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Amazon search benchmark dataset")
    p.add_argument("--hf-token", default="", help="HuggingFace write token")
    p.add_argument(
        "--num-products", type=int, default=50, help="Products per query (max 100)"
    )
    p.add_argument("--country", default="US", help="Amazon country code")
    p.add_argument("--output", default="benchmarks/amazon/data/dataset.json")
    p.add_argument("--delay", type=float, default=3.0, help="Seconds between queries")
    p.add_argument(
        "--dry-run", action="store_true", help="Skip upload, just scrape + save"
    )
    p.add_argument(
        "--queries-file",
        default="",
        help="Optional path to .txt file with one query per line (overrides defaults)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    queries = BENCHMARK_QUERIES
    if args.queries_file:
        queries = Path(args.queries_file).read_text().splitlines()
        queries = [q.strip() for q in queries if q.strip()]

    rows = generate_dataset(
        queries=queries,
        num_products=args.num_products,
        country=args.country,
        output_path=Path(args.output),
        delay_between_queries=args.delay,
    )

    if not args.dry_run:

        from google.colab import userdata

        hf_token = userdata.get("HF_TOKEN")
        upload_to_huggingface(rows, hf_token=hf_token)
    else:
        print("\n🏃  Dry run — skipping HuggingFace upload.")
        print(f"    Dataset saved to: {args.output}")
