"""
search-expert
=============
A lightweight library for parsing natural language search queries into
structured output using fine-tuned Qwen3.5-0.8B LoRA adapters.

Quick start
-----------
>>> from search_expert import SearchExpert
>>> expert = SearchExpert()                  # defaults to JSON model
>>> result = expert.parse("MacBook under $2000 with 16GB RAM")
>>> print(result.fields)
{'domain': 'ecommerce', 'product': 'MacBook', 'price': 'lt:2000', 'feature': '16GB RAM'}
"""

from search_expert.config import DEFAULT_MODEL_IDS, ModelFormat
from search_expert.exceptions import ModelLoadError, ParseError, SearchExpertError
from search_expert.expert import SearchExpert
from search_expert.result import ParseResult

__version__ = "0.1.0"
__author__ = "Sarthak Rastogi"
__all__ = [
    "DEFAULT_MODEL_IDS",
    "ModelFormat",
    "ModelLoadError",
    "ParseError",
    "ParseResult",
    "SearchExpert",
    "SearchExpertError",
]
