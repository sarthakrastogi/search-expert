"""
examples/basic_usage.py
-----------------------
Demonstrates the core search-expert API without requiring a GPU.
The model-loading lines are commented out so this file runs in any
environment; uncomment them when you have a GPU available.
"""

# ── Imports ───────────────────────────────────────────────────
from search_expert import SearchExpert, ModelFormat, ParseResult

# ── 1. Basic usage (JSON model, default) ──────────────────────
# expert = SearchExpert()
# result = expert.parse("noise cancelling headphones under $200")
# print(result.fields)
# # → {'domain': 'ecommerce', 'product': 'headphones',
# #    'feature': 'noise cancelling', 'price': 'lt:200'}

# ── 2. YAML model ─────────────────────────────────────────────
# expert_yaml = SearchExpert(fmt=ModelFormat.YAML)
# result = expert_yaml.parse("3BR house in Austin under $600k with pool")
# print(result.to_yaml())

# ── 3. Numeric constraint decoding ────────────────────────────
# result = expert.parse("remote ML engineer job paying over $150k")
# salary = result.get_numeric_constraint("salary")
# print(salary)
# # → {'operator': 'gt', 'value': 150000.0, 'value_hi': None}
#
# # You can use this in a search filter directly:
# if salary and salary["operator"] == "gt":
#     filter_salary_gt = salary["value"]  # 150000.0

# ── 4. Serialise result ───────────────────────────────────────
# print(result.to_json(indent=2))
# print(result.to_yaml())
# print(result.to_dict())

# ── 5. Batch parsing ──────────────────────────────────────────
# queries = [
#     "Python ML course for beginners under $30",
#     "5-star hotel in Paris with breakfast under $400/night",
#     "Taylor Swift concert in London in July",
# ]
# results = expert.parse_batch(queries)
# for r in results:
#     print(r.query, "→", r.fields)

# ── 6. Custom adapter ─────────────────────────────────────────
# expert = SearchExpert(model_id="your-org/your-fine-tuned-adapter")
# result = expert.parse("...")

# ── 7. Custom generation config ───────────────────────────────
# expert = SearchExpert(
#     generation_config={"temperature": 0.0, "max_new_tokens": 128}
# )

print("See comments in this file for usage examples.")
print("Uncomment the lines after loading a model on a GPU instance.")
