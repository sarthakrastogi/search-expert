<div align="center">

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/search--expert-0.1.0-0f0f0f?style=for-the-badge&labelColor=0f0f0f&color=39d353&logoColor=white">
  <img alt="search-expert" src="https://img.shields.io/badge/search--expert-0.1.0-fafafa?style=for-the-badge&labelColor=fafafa&color=16a34a&logoColor=black">
</picture>

<br/><br/>

<h1>
  <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Magnifying%20glass%20tilted%20right/3D/magnifying_glass_tilted_right_3d.png" width="36" align="center" />
  &nbsp;search-expert
</h1>

<p><strong>Natural language → structured search queries, instantly.</strong><br/>
Fine-tuned Qwen3.5-0.8B LoRA adapters for search query parsing across 10 domains.</p>

<br/>

[![PyPI version](https://img.shields.io/pypi/v/search-expert?style=flat-square&color=16a34a&labelColor=f0fdf4)](https://pypi.org/project/search-expert/)
[![Python](https://img.shields.io/pypi/pyversions/search-expert?style=flat-square&color=3b82f6&labelColor=eff6ff)](https://pypi.org/project/search-expert/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square&labelColor=fefce8)](LICENSE)
[![Open In Colab](https://img.shields.io/badge/Colab-Try_it_now-F9AB00?style=flat-square&logo=googlecolab&logoColor=white&labelColor=2d2d2d)](https://colab.research.google.com/github/sarthakrastogi/search-expert/blob/main/examples/search_expert_colab.ipynb)

<br/>

</div>

---

<br/>

## What it does

```
"Non-stop business class from JFK to Tokyo under $3,000"
```

```json
{
  "domain":      "flights",
  "origin":      "JFK",
  "destination": "Tokyo",
  "cabin_class": "business",
  "stops":       "lte:0",
  "price":       "lt:3000"
}
```

Search Expert uses a custom fine-tuned Small Language Model to understand natural language queries and extract **only the fields explicitly mentioned** — never hallucinating values that aren't there. It works across 10 search verticals out of the box.

<br/>

## Install

```bash
pip install search-expert
```
<br/>

## Usage

### Basic

```python
from search_expert import SearchExpert, ModelFormat, ParseResult

expert = SearchExpert()  # loads the JSON adapter by default

result = expert.parse("noise cancelling headphones any colour but red or green, under $200")
print(result.fields)
```

```python
{
    'domain':  'ecommerce',
    'product': 'headphones',
    'feature': 'noise cancelling',
    'color':   ['ne:red', 'ne:green'],
    'price':   'lt:200'
}
```

<br/>

## Why hybrid search beats the alternatives

| | Text-to-SQL | Pure vector search | Hybrid search (this pipeline) |
|---|---|---|---|
| Hard constraints (price, brand, color) | ✅ | ❌ | ✅ |
| Semantic intent ("good for travel") | ❌ | ✅ | ✅ |
| Ranked results by relevance | ❌ | ✅ | ✅ |
| Works on unstructured descriptions | ❌ | ✅ | ✅ |
| Respects exclusions ("not black") | ✅ | ❌ | ✅ |
| Price is a hard cutoff, not a soft signal | ✅ | ❌ | ✅ |

Text-to-SQL is a **lookup tool** — it returns rows that match, but can't rank by relevance or understand semantics.  
Pure vector search is a **semantic tool** — it understands meaning, but treats "$200" as a soft hint, not a hard rule. A $350 product can rank above a $180 one if its description is more similar to the query.  
This pipeline is a **retrieval tool** — structured filters enforce the hard constraints first, then vector search ranks the surviving candidates by semantic relevance.

In production, we generally use a version of this pattern:  
**structured pre-filtering → ANN (approximate nearest neighbour) vector search → learning-to-rank re-ranker**.

search-expert makes step 1 trivial with a tiny, fast, locally-runnable model.

<br/>

## Operator reference

All numeric and exclusion constraints use a consistent prefix so downstream filters need zero NLP — just parse the string.

| Query phrase | Output value |
|---|---|
| `"under $200"`, `"below $200"` | `lt:200` |
| `"up to $200"`, `"max $200"` | `lte:200` |
| `"over $150k"`, `"above $150k"` | `gt:150000` |
| `"at least $150k"`, `"$150k+"` | `gte:150000` |
| `"around $200"`, `"~$200"` | `approx:200` |
| `"$100–$200"`, `"between $100 and $200"` | `between:100:200` |
| `"any colour but red or green"` | `["ne:red", "ne:green"]` |

**Applying a filter in one line:**

```python
result = expert.parse("apartments under $2,500/month in Austin")
salary = result.get_numeric_constraint("price")
# {'operator': 'lt', 'value': 2500.0, 'value_hi': None}

filtered = [l for l in listings if l["price"] < salary["value"]]
```

<br/>

## Domains present in the training data

| Domain | Example query |
|---|---|
| `real_estate` | *"2BR apartment in Austin under $1,500/month"* |
| `ecommerce` | *"Sony noise cancelling headphones under $300"* |
| `jobs` | *"Remote senior ML engineer paying over $150k"* |
| `flights` | *"Non-stop business class JFK to Tokyo under $3,000"* |
| `hotels` | *"5-star hotel in Paris with breakfast under $400/night"* |
| `cars` | *"Electric SUV with 300+ mile range under $50k"* |
| `restaurants` | *"Vegan Italian in NYC with outdoor seating under $40"* |
| `movies` | *"Thriller on Netflix with 8+ IMDB rating"* |
| `healthcare` | *"Female therapist in Chicago accepting Aetna"* |
| `courses` | *"Python ML course for beginners under $30"* |
| `events` | *"Taylor Swift concert in London in July"* |

<br/>

## The model

Each adapter is a LoRA fine-tune of **[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)** trained on 100,000 (query, structured output) pairs spanning all 10 domains above.

| Adapter | HuggingFace | Format |
|---|---|---|
| JSON *(default)* | [`sarthakrastogi/search-expert-json-0.8b`](https://huggingface.co/sarthakrastogi/search-expert-json-0.8b) | JSON |
| YAML | [`sarthakrastogi/search-expert-yaml-0.8b`](https://huggingface.co/sarthakrastogi/search-expert-yaml-0.8b) | YAML |

**Format leaderboard** (held-out test set, 300 samples per format):

| Rank | Format | Key F1 | Value Acc | Parse Rate |
|---|---|---|---|---|
| 🥇 | JSON | 0.913 | 0.874 | 98.2% |
| 🥈 | YAML | 0.901 | 0.861 | 97.6% |
| 🥉 | TOML | 0.887 | 0.843 | 96.1% |
| 4. | XML | 0.871 | 0.829 | 94.8% |
| 5. | CSV key=value | 0.856 | 0.812 | 93.3% |

Both public adapters return the same Python dict — the format only affects the model's internal generation language.

<br/>

## Repo structure

```
search-expert/
├── search_expert/        # Library source
│   ├── __init__.py
│   ├── expert.py         # SearchExpert class (main API)
│   ├── config.py         # Model IDs, prompts, format enum
│   ├── loader.py         # HF model loading (unsloth / peft / plain)
│   ├── parser.py         # Raw output → dict parsers
│   ├── result.py         # ParseResult dataclass
│   └── exceptions.py     # Custom exceptions
├── training/             # Fine-tuning pipeline
│   ├── finetune.py       # Training script
│   └── evaluate.py       # Format comparison leaderboard
├── tests/
│   └── test_search_expert.py
├── examples/
│   └── search_expert_colab.ipynb
├── pyproject.toml
└── README.md
```

<br/>

## Development

```bash
git clone https://github.com/sarthakrastogi/search-expert
cd search-expert
pip install -e ".[dev]"

pytest tests/ -v                                          # unit tests (no GPU needed)
SEARCH_EXPERT_RUN_MODEL_TESTS=1 pytest tests/ -v          # includes model inference tests
```

<br/>

## License

MIT © [Sarthak Rastogi](https://github.com/sarthakrastogi)

## Contributing

Contributions are very welcome! Please open an issue or submit a pull request with any improvements.

## Contact

For questions, feedback, or just to say hi, you can reach me at:

[Email](mailto:thesarthakrastogi@gmail.com)

[LinkedIn](https://www.linkedin.com/in/sarthakrastogi/)