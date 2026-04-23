"""
benchmarks/amazon/data_generation.py

Searches Amazon for a set of benchmark queries using SerpApi's Amazon Search API,
extracts rich product metadata, and uploads the resulting dataset to
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
    #   pip install requests datasets huggingface_hub tqdm

    python data_generation.py \
        --num-products 50

    # Dry run (no upload):
    python data_generation.py --dry-run

    # SerpApi key and HF token are read from env vars:
    #   SERPAPI_KEY   — your SerpApi API key
    #   HF_TOKEN      — your HuggingFace write token
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

# ── benchmark queries ─────────────────────────────────────────────────────────
# All queries are complex, multi-constraint queries to stress-test retrieval:
#   price + feature + color exclusion, brand + spec + budget, etc.
BENCHMARK_QUERIES = [
    # Headphones / earbuds — multi-constraint with price & color
    "noise cancelling headphones in silver or white, under $200",
    "wireless noise cancelling headphones with 30+ hour battery, black or grey, $100-$150",
    "Sony noise cancelling headphones, not black, under $250",
    "Bose noise cancelling headphones with mic, blue or navy, above $150",
    "noise cancelling earbuds under $100, not white or pink, transparency mode",
    "true wireless earbuds with ANC, white or titanium color, $80-$120",
    "over-ear headphones, foldable, space grey or black, under $200",
    "budget noise cancelling headphones, any color except gold, under $50",
    "Sony wireless earbuds, hi-res audio, silver finish, $150-$250",
    "Apple AirPods Pro alternative, not white, better battery, under $150",
    # Laptops — multi-constraint with price & color
    "lightweight laptop, silver or space grey, 16GB RAM, 12hr battery, under $600",
    "gaming laptop, black finish, RTX 4060, 144Hz, under $900",
    "MacBook Air M3 alternative, not white, Windows, thin design, under $700",
    "Dell XPS 13 competitor, OLED, silver color, under $1000",
    "2-in-1 touchscreen laptop, grey or black, stylus support, 8GB RAM, under $500",
    "business laptop, not gold, fingerprint reader, Thunderbolt 4, $600-$800",
    # Running shoes — multi-constraint with price & color
    "Nike running shoes in blue or grey, cushioned, under $120",
    "ASICS running shoes, overpronation, wide toe, not pink, under $150",
    "Hoka running shoes, not black, maximalist, grey or white, under $160",
    "Brooks running shoes women, neutral, not pink or rose, under $130",
    "Adidas running shoes, carbon plate, marathon, black or red, $150-$200",
    # Smartphones — multi-constraint with price & color
    "Samsung Galaxy S25 alternative, black or silver, 200MP camera, under $700",
    "Google Pixel 9 compact, 5G, not white, under $600",
    "budget Android phone, blue or black, 5000mAh, AMOLED, under $300",
    "flagship Android, space grey or titanium, 5G, above $600",
    # Smartwatches — multi-constraint with price & color
    "Garmin GPS running watch, black or grey, heart rate, sleep tracking, under $300",
    "fitness smartwatch, ECG, SpO2, not black, rose gold acceptable, $100-$150",
    "Apple Watch Series 9 alternative, space grey or silver, 2 week battery",
    # Cameras — multi-constraint with price & color
    "mirrorless camera, black body, in-body stabilization, beginners, under $700",
    "Sony mirrorless, vlogging, flip screen, not white, under $1200",
    "Canon mirrorless, 4K 60fps, dual card, black finish, $1500-$2000",
    # Monitors — multi-constraint with price & color
    "4K monitor, black bezel, USB-C power, IPS panel, under $400",
    "gaming monitor 144Hz, IPS not TN, silver or black stand, under $250",
    "ultrawide curved, black frame, HDR, USB hub, under $500",
    # Keyboards — multi-constraint with price & color
    "wireless mechanical keyboard, hot-swappable, white or grey, under $100",
    "compact 65% mechanical, RGB, black PBT keycaps, under $80",
    # Tablets — multi-constraint with price & color
    "iPad Air alternative, grey or silver, stylus, Android, under $400",
    "Android tablet, 12 inch, black finish, 8GB RAM, under $350",
    # Gaming — multi-constraint with price & color
    "Nintendo Switch OLED, white or grey bundle with game, under $400",
    "PlayStation 5 slim, black disc edition, extra controller, $500-$600",
    # Coffee — multi-constraint with price & color
    "organic whole bean coffee, medium roast, single-origin, under $25/lb",
    "Colombian espresso, low acidity, specialty grade, $15-$25/lb",
    # Paper goods / household — multi-constraint with price & color
    "recycled paper towels, white, FSC certified, bulk pack, under $40",
    "bamboo paper towels, reusable, not single-use, natural tan color, under $30",
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
    (re.compile(r"\bcoffee\b|\bespresso\b|\bcapuccino\b", re.I), "coffee"),
    (re.compile(r"\bpaper towel\b|\btissue\b", re.I), "paper goods"),
]

COLOR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bblack\b", re.I), "black"),
    (re.compile(r"\bwhite\b|\bplatinum\b|\bpearl\b", re.I), "white"),
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
    (re.compile(r"\brose gold\b", re.I), "rose gold"),
    (re.compile(r"\bnatural\b|\bbeige\b|\bcream\b", re.I), "natural"),
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
    "coffee": "grocery",
    "paper goods": "household",
}


def infer_product_type(text: str) -> str:
    for pattern, ptype in PRODUCT_TYPE_PATTERNS:
        if pattern.search(text):
            return ptype
    return "other"


def infer_color(text: str) -> str:
    """
    Infer the primary color from a product title or feature text.
    Returns the first matched color, or empty string if none found.
    """
    for pattern, color in COLOR_PATTERNS:
        if pattern.search(text):
            return color
    return ""


# ── SerpApi wrapper ───────────────────────────────────────────────────────────

SERPAPI_BASE = "https://serpapi.com/search.json"


def search_amazon(
    keyword: str,
    api_key: str,
    num_pages: int = 1,
    country: str = "us",
    retries: int = 3,
    delay: float = 2.0,
) -> list[dict]:
    """
    Call SerpApi Amazon Search API, collect organic_results across pages.
    Returns list of raw organic result dicts.
    """
    all_results: list[dict] = []

    for page in range(1, num_pages + 1):
        params = {
            "engine": "amazon",
            "k": keyword,
            "api_key": api_key,
            "amazon_domain": "amazon.com",
            "page": page,
        }

        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(SERPAPI_BASE, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                error = data.get("error")
                if error:
                    raise ValueError(f"SerpApi error: {error}")

                results = data.get("organic_results", [])
                all_results.extend(results)
                break  # success

            except (requests.RequestException, ValueError) as exc:
                print(
                    f"  ⚠  Attempt {attempt}/{retries} failed for '{keyword}' page {page}: {exc}"
                )
                if attempt < retries:
                    time.sleep(delay * attempt)
        else:
            print(f"  ✗  All retries exhausted for '{keyword}' page {page}")

        if page < num_pages:
            time.sleep(delay)

    return all_results


# ── field extraction ──────────────────────────────────────────────────────────


def extract_product(
    raw: dict,
    query: str,
    amazon_rank: int,
) -> dict[str, Any]:
    """
    Normalise one raw SerpApi Amazon organic result into the benchmark schema.

    SerpApi field mapping:
        asin                → asin
        title               → title
        link_clean          → url
        thumbnail           → thumbnail
        rating              → rating
        reviews             → total_reviews
        extracted_price     → price
        extracted_old_price → original_price  (used to detect discount)
        prime               → amazon_prime
        sponsored           → sponsored
        badges              → amazon_choice / best_seller detection
        brand               → brand
        delivery            → (ignored, not in schema)
    """
    title = raw.get("title", "")
    asin = raw.get("asin", "")
    url = raw.get("link_clean") or raw.get("link", f"https://www.amazon.com/dp/{asin}")
    thumbnail = raw.get("thumbnail", "")

    # Price fields
    price = float(raw.get("extracted_price") or 0)
    original_price_raw = raw.get("extracted_old_price")
    if original_price_raw:
        original_price = float(original_price_raw)
        discounted = True
        savings_percent = (
            round((original_price - price) / original_price * 100, 1)
            if original_price > 0
            else 0.0
        )
    else:
        original_price = price
        discounted = False
        savings_percent = 0.0

    # Ratings
    rating = float(raw.get("rating") or 0)
    total_reviews = int(raw.get("reviews") or 0)

    # Badges — SerpApi returns a list of badge strings
    badges: list[str] = raw.get("badges", []) or []
    badges_lower = [b.lower() for b in badges]
    amazon_choice = any(
        "amazon's choice" in b or "amazon choice" in b for b in badges_lower
    )
    best_seller = any("best seller" in b for b in badges_lower)

    amazon_prime = bool(raw.get("prime", False))
    sponsored = bool(raw.get("sponsored", False))

    # Brand — SerpApi may include a "brand" field directly
    brand = raw.get("brand", "")
    if not brand:
        # Fall back: first capitalised word of title is usually brand
        match = re.match(r"^([A-Z][A-Za-z0-9\-&]+)", title)
        if match:
            brand = match.group(1)

    # Feature bullets — not available in search results via SerpApi,
    # so description falls back to title only (consistent with amazon-buddy
    # search results which also rarely have bullets without a product API call)
    features_raw = ""
    description = title

    # Infer product type, color, category from title (features_raw is empty here)
    search_text = title
    product_type = infer_product_type(search_text)
    color = infer_color(search_text)
    category = CATEGORY_MAP.get(product_type, "other")

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
        "currency": "USD",  # SerpApi Amazon defaults to USD; adjust if using intl domains
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
        "thumbnail": thumbnail,
    }


# ── main pipeline ─────────────────────────────────────────────────────────────


def generate_dataset(
    queries: list[str],
    api_key: str,
    num_pages: int,
    output_path: Path,
    delay_between_queries: float = 3.0,
) -> list[dict]:
    all_rows: list[dict] = []

    for query in tqdm(queries, desc="Scraping Amazon via SerpApi"):
        print(f"\n🔍  Querying: '{query}'")
        raw_products = search_amazon(
            keyword=query,
            api_key=api_key,
            num_pages=num_pages,
            delay=delay_between_queries / 2,
        )

        if not raw_products:
            print(f"  ⚠  No results returned for '{query}'")
            continue

        for rank, raw in enumerate(raw_products, start=1):
            row = extract_product(raw, query=query, amazon_rank=rank)
            all_rows.append(row)

        print(f"  ✅  {len(raw_products)} products scraped")
        time.sleep(delay_between_queries)

    # Save locally as JSON (same format as original data_generation.py)
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

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    ds.push_to_hub(
        repo_id,
        token=hf_token,
        commit_message=f"Add {len(rows)} rows across {len(set(r['query'] for r in rows))} queries",
    )

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

Real Amazon product data scraped via SerpApi across **{len(set(r['query'] for r in rows))} search queries**,
covering {len(rows)} (query, product) pairs.

Queries are deliberately complex and multi-constraint (price + feature + color exclusion, brand + spec, etc.)
to stress-test retrieval systems.

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
    p = argparse.ArgumentParser(
        description="Generate Amazon search benchmark dataset via SerpApi"
    )
    p.add_argument(
        "--serpapi-key",
        default=os.environ.get("SERPAPI_KEY", ""),
        help="SerpApi API key (or set SERPAPI_KEY env var)",
    )
    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace write token (or set HF_TOKEN env var)",
    )
    p.add_argument(
        "--num-pages",
        type=int,
        default=1,
        help="Number of result pages to fetch per query (each page ~16 products)",
    )
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

    if not args.serpapi_key:
        print(
            "❌  No SerpApi key found. Set SERPAPI_KEY env var or pass --serpapi-key."
        )
        sys.exit(1)

    queries = BENCHMARK_QUERIES  # [:10]
    if args.queries_file:
        queries = Path(args.queries_file).read_text().splitlines()
        queries = [q.strip() for q in queries if q.strip()]

    rows = generate_dataset(
        queries=queries,
        api_key=args.serpapi_key,
        num_pages=args.num_pages,
        output_path=Path(args.output),
        delay_between_queries=args.delay,
    )

    if not args.dry_run:
        hf_token = args.hf_token
        if not hf_token:
            print("❌  No HF token found. Set HF_TOKEN env var or pass --hf-token.")
            sys.exit(1)
        upload_to_huggingface(rows, hf_token=hf_token)
    else:
        print("\n🏃  Dry run — skipping HuggingFace upload.")
        print(f"    Dataset saved to: {args.output}")
