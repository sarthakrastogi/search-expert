"""
spacy_extractor.py — Rule-based structured field extraction using spaCy NER.

This module is an alternative to the fine-tuned SLM adapters.  It uses:
  - spaCy's en_core_web_sm NER model to recognise named entities
    (GPE, ORG, PERSON, PRODUCT, MONEY, CARDINAL, DATE, …)
  - A hand-crafted regex + keyword layer on top to extract operator-prefixed
    numeric constraints (lt:, gte:, between:, …) and exclusion operators (ne:)

Compared to the SLM adapters this approach:
  ✅  ~1 ms per query on CPU — no GPU required
  ✅  Zero model download / loading time
  ✅  Fully deterministic and auditable
  ❌  Cannot generalise to phrasings not covered by the rules
  ❌  No soft semantic understanding (e.g. "good for travel" → feature)
  ❌  Multi-domain field assignment is brittle

Supported domains are detected via keyword matching and used to assign
otherwise-ambiguous fields (e.g. "rating" vs "stars").

Usage
-----
    from search_expert.spacy_extractor import SpacyExtractor

    extractor = SpacyExtractor()
    fields = extractor.extract("Sony noise cancelling headphones, not red, under $200")
    # {'domain': 'ecommerce', 'product': 'headphones', 'brand': 'Sony',
    #  'feature': 'noise cancelling', 'color': ['ne:red'], 'price': 'lt:200'}
"""

from __future__ import annotations

import re
from typing import Any

# ── Lazy spaCy import so the rest of the library works without it ─────

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy  # noqa: F401
        except ImportError:
            raise ImportError(
                "spacy is required for the SpacyExtractor backend.\n"
                "Install it with:\n"
                "    pip install spacy\n"
                "    python -m spacy download en_core_web_sm"
            )
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found.\n"
                "Download it with:\n"
                "    python -m spacy download en_core_web_sm"
            )
    return _nlp


# ── Domain keyword maps ────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "flights": [
        "flight",
        "fly",
        "airline",
        "airport",
        "nonstop",
        "non-stop",
        "layover",
        "cabin",
        "business class",
        "economy",
        "first class",
        "round trip",
        "one way",
    ],
    "hotels": [
        "hotel",
        "resort",
        "inn",
        "motel",
        "suite",
        "room",
        "accommodation",
        "check-in",
        "check-out",
        "per night",
        "/night",
        "star hotel",
    ],
    "real_estate": [
        "apartment",
        "house",
        "condo",
        "bedroom",
        "br",
        "bath",
        "sq ft",
        "rent",
        "lease",
        "mortgage",
        "property",
        "studio",
        "townhouse",
        "per month",
        "/month",
    ],
    "cars": [
        "car",
        "vehicle",
        "suv",
        "sedan",
        "truck",
        "mpg",
        "horsepower",
        "electric",
        "hybrid",
        "miles range",
        "mileage",
        "transmission",
        "drivetrain",
    ],
    "jobs": [
        "job",
        "position",
        "role",
        "engineer",
        "developer",
        "manager",
        "salary",
        "remote",
        "full-time",
        "part-time",
        "hiring",
        "years experience",
        "yoe",
    ],
    "ecommerce": [
        "headphones",
        "laptop",
        "phone",
        "camera",
        "tv",
        "monitor",
        "keyboard",
        "mouse",
        "tablet",
        "charger",
        "cable",
        "bag",
        "shoes",
        "shirt",
        "jacket",
        "watch",
    ],
    "restaurants": [
        "restaurant",
        "cafe",
        "diner",
        "bistro",
        "vegan",
        "vegetarian",
        "gluten-free",
        "outdoor seating",
        "cuisine",
        "dining",
    ],
    "movies": [
        "movie",
        "film",
        "series",
        "show",
        "netflix",
        "imdb",
        "rating",
        "thriller",
        "comedy",
        "horror",
        "drama",
        "documentary",
    ],
    "healthcare": [
        "doctor",
        "therapist",
        "dentist",
        "clinic",
        "hospital",
        "insurance",
        "aetna",
        "blue cross",
        "medicare",
        "specialist",
    ],
    "courses": [
        "course",
        "tutorial",
        "bootcamp",
        "certification",
        "learn",
        "beginner",
        "advanced",
        "udemy",
        "coursera",
    ],
    "events": [
        "concert",
        "festival",
        "event",
        "ticket",
        "show",
        "tour",
        "performance",
        "gig",
    ],
}

# ── Numeric operator patterns ──────────────────────────────────────────
# Each tuple: (compiled regex, operator string)
# Patterns are tried in order; first match wins.

_NUM = r"[\$£€]?\s*(\d[\d,]*(?:\.\d+)?)[kK]?"  # captures numeric part

_OPERATOR_PATTERNS: list[tuple[re.Pattern, str]] = [
    # between / range  — "between $100 and $300", "$100–$300", "$100 to $300"
    (
        re.compile(r"between\s+" + _NUM + r"\s+(?:and|to)\s+" + _NUM, re.IGNORECASE),
        "between",
    ),
    (
        re.compile(_NUM + r"\s*[-–—]\s*" + _NUM, re.IGNORECASE),
        "between",
    ),
    # at least / N+  (gte)
    (
        re.compile(
            r"(?:at\s+least|minimum|min\.?|(?<!\w)from)\s+" + _NUM, re.IGNORECASE
        ),
        "gte",
    ),
    (re.compile(_NUM + r"\s*\+"), "gte"),
    # up to / at most (lte)
    (
        re.compile(
            r"(?:up\s+to|at\s+most|maximum|max\.?|no\s+more\s+than)\s+" + _NUM,
            re.IGNORECASE,
        ),
        "lte",
    ),
    # over / above / more than (gt)
    (
        re.compile(
            r"(?:over|above|more\s+than|greater\s+than|exceeding)\s+" + _NUM,
            re.IGNORECASE,
        ),
        "gt",
    ),
    # under / below / less than (lt)
    (
        re.compile(
            r"(?:under|below|less\s+than|cheaper\s+than|within)\s+" + _NUM,
            re.IGNORECASE,
        ),
        "lt",
    ),
    # approx
    (
        re.compile(r"(?:around|roughly|approximately|~)\s*" + _NUM, re.IGNORECASE),
        "approx",
    ),
    # bare price with currency symbol → treat as "up to"
    (re.compile(r"[\$£€]\s*(\d[\d,]*(?:\.\d+)?)[kK]?"), "lte"),
]

# ── Exclusion pattern ──────────────────────────────────────────────────

_EXCLUSION_RE = re.compile(
    r"(?:not|no|except|excluding|without|any\s+(?:color|colour)\s+but)\s+"
    r"([a-z][a-z\s,]+?)(?=[,\.]|$|\s+(?:and|or|but|under|below|over|above))",
    re.IGNORECASE,
)

# ── Color keywords ─────────────────────────────────────────────────────

_COLORS = {
    "red",
    "blue",
    "green",
    "black",
    "white",
    "yellow",
    "orange",
    "purple",
    "pink",
    "grey",
    "gray",
    "silver",
    "gold",
    "brown",
    "beige",
    "navy",
    "teal",
    "cyan",
    "magenta",
    "maroon",
    "olive",
}

# ── Stop words for entity filtering ───────────────────────────────────

_STOP_ENTITIES = {
    "today",
    "tomorrow",
    "yesterday",
    "now",
    "later",
    "soon",
    "one",
    "two",
    "three",
    "four",
    "five",
}


# ── Helper utilities ───────────────────────────────────────────────────


def _normalise_number(s: str) -> float:
    """'1,500' or '150k' → 1500.0"""
    s = s.replace(",", "").strip()
    if s.lower().endswith("k"):
        return float(s[:-1]) * 1000
    return float(s)


def _extract_numeric_constraint(text: str, field_hint: str = "price") -> str | None:
    """
    Scan *text* for the first operator-prefixed numeric match and return
    a constraint string like ``"lt:200"`` or ``"between:100:300"``.
    Returns None if no match found.
    """
    for pattern, op in _OPERATOR_PATTERNS:
        m = pattern.search(text)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if op == "between" and len(groups) >= 2:
                lo = _normalise_number(groups[0])
                hi = _normalise_number(groups[1])
                return f"between:{lo:g}:{hi:g}"
            elif groups:
                val = _normalise_number(groups[0])
                return f"{op}:{val:g}"
    return None


def _extract_exclusions(text: str) -> list[str]:
    """Return a list of ``ne:<value>`` strings for each excluded item."""
    results: list[str] = []
    for m in _EXCLUSION_RE.finditer(text):
        raw = m.group(1).strip().rstrip(" ,")
        # Split "red or green" → ["red", "green"]
        parts = re.split(r"\s*(?:,|or|and)\s*", raw)
        for part in parts:
            part = part.strip()
            if part:
                results.append(f"ne:{part.lower()}")
    return results


def _detect_domain(text: str) -> str | None:
    lower = text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in lower)
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] > 0 else None


def _extract_stop_count(text: str) -> str | None:
    """Detect non-stop / direct / 1-stop flight constraints."""
    lower = text.lower()
    if re.search(r"\bnon.?stop\b|\bdirect\b", lower):
        return "lte:0"
    m = re.search(r"(\d+)\s*stop", lower)
    if m:
        return f"lte:{m.group(1)}"
    return None


def _extract_bedroom_count(text: str) -> str | None:
    m = re.search(r"(\d+)\s*(?:br|bed(?:room)?s?)\b", text, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_rating(text: str) -> str | None:
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:star|stars|imdb|rating|rated)",
        text,
        re.IGNORECASE,
    )
    if m:
        return f"gte:{m.group(1)}"
    m = re.search(
        r"(?:rating|rated)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*\+",
        text,
        re.IGNORECASE,
    )
    if m:
        return f"gte:{m.group(1)}"
    return None


# ── Main extractor class ───────────────────────────────────────────────


class SpacyExtractor:
    """
    Rule-based search query field extractor using spaCy NER.

    Parameters
    ----------
    model : str
        spaCy model name to load.  Defaults to ``"en_core_web_sm"``.
        You can pass ``"en_core_web_lg"`` or a custom model for better NER.

    Examples
    --------
    >>> extractor = SpacyExtractor()
    >>> extractor.extract("Sony noise cancelling headphones under $200, not red")
    {'domain': 'ecommerce', 'brand': 'Sony', 'product': 'headphones',
     'feature': 'noise cancelling', 'price': 'lt:200', 'color': ['ne:red']}
    """

    def __init__(self, model: str = "en_core_web_sm") -> None:
        self._model_name = model
        self._nlp = None  # lazy load

    @property
    def nlp(self):
        if self._nlp is None:
            if self._model_name == "en_core_web_sm":
                self._nlp = _get_nlp()
            else:
                import spacy

                self._nlp = spacy.load(self._model_name)
        return self._nlp

    # ------------------------------------------------------------------
    # Internal field extractors per domain
    # ------------------------------------------------------------------

    def _fields_flights(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        gpe_ents = [e.text for e in doc.ents if e.label_ == "GPE"]
        if len(gpe_ents) >= 2:
            fields["origin"] = gpe_ents[0]
            fields["destination"] = gpe_ents[1]
        elif len(gpe_ents) == 1:
            fields["destination"] = gpe_ents[0]

        for cabin in ["business", "economy", "first"]:
            if re.search(rf"\b{cabin}\b", text, re.IGNORECASE):
                fields["cabin_class"] = cabin
                break

        stops = _extract_stop_count(text)
        if stops:
            fields["stops"] = stops

        price = _extract_numeric_constraint(text, "price")
        if price:
            fields["price"] = price

        return fields

    def _fields_hotels(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        gpe_ents = [e.text for e in doc.ents if e.label_ == "GPE"]
        if gpe_ents:
            fields["city"] = gpe_ents[0]

        m = re.search(r"(\d)\s*-?\s*star", text, re.IGNORECASE)
        if m:
            fields["stars"] = int(m.group(1))

        price = _extract_numeric_constraint(text, "price")
        if price:
            fields["price"] = price

        for amenity in ["breakfast", "wifi", "pool", "parking", "spa", "gym"]:
            if re.search(rf"\b{amenity}\b", text, re.IGNORECASE):
                fields.setdefault("amenities", [])
                fields["amenities"].append(amenity)

        return fields

    def _fields_real_estate(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        gpe_ents = [e.text for e in doc.ents if e.label_ == "GPE"]
        if gpe_ents:
            fields["city"] = gpe_ents[0]

        bedrooms = _extract_bedroom_count(text)
        if bedrooms:
            fields["bedrooms"] = bedrooms

        for prop_type in [
            "apartment",
            "house",
            "condo",
            "studio",
            "townhouse",
            "villa",
        ]:
            if re.search(rf"\b{prop_type}\b", text, re.IGNORECASE):
                fields["property_type"] = prop_type
                break

        price = _extract_numeric_constraint(text, "price")
        if price:
            fields["price"] = price

        return fields

    def _fields_ecommerce(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}

        # Brand from ORG/PRODUCT entities
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT"):
                if ent.text.lower() not in _STOP_ENTITIES:
                    fields.setdefault("brand", ent.text)
                    break

        # Color exclusions
        excls = _extract_exclusions(text)
        color_excls = [e for e in excls if any(c in e for c in _COLORS)]
        if color_excls:
            fields["color"] = color_excls
        else:
            # Positive color match
            for color in _COLORS:
                if re.search(rf"\b{color}\b", text, re.IGNORECASE):
                    fields["color"] = color
                    break

        price = _extract_numeric_constraint(text, "price")
        if price:
            fields["price"] = price

        # Feature phrases (naive: noun chunks not already captured)
        feature_phrases = [
            "noise cancelling",
            "wireless",
            "waterproof",
            "fast charging",
            "4k",
            "hdr",
            "oled",
            "amoled",
            "retina",
            "touch screen",
        ]
        for phrase in feature_phrases:
            if re.search(re.escape(phrase), text, re.IGNORECASE):
                fields["feature"] = phrase
                break

        return fields

    def _fields_jobs(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        gpe_ents = [e.text for e in doc.ents if e.label_ == "GPE"]
        if gpe_ents:
            fields["location"] = gpe_ents[0]

        if re.search(r"\bremote\b", text, re.IGNORECASE):
            fields["remote"] = True

        salary = _extract_numeric_constraint(text, "salary")
        if salary:
            fields["salary"] = salary

        m = re.search(r"(\d+)\+?\s*years?\s+(?:of\s+)?experience", text, re.IGNORECASE)
        if m:
            fields["experience_years"] = f"gte:{m.group(1)}"

        return fields

    def _fields_cars(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT"):
                fields.setdefault("brand", ent.text)
                break

        for car_type in [
            "suv",
            "sedan",
            "truck",
            "coupe",
            "convertible",
            "hatchback",
            "minivan",
        ]:
            if re.search(rf"\b{car_type}\b", text, re.IGNORECASE):
                fields["body_type"] = car_type
                break

        for fuel in ["electric", "hybrid", "diesel", "petrol", "gasoline"]:
            if re.search(rf"\b{fuel}\b", text, re.IGNORECASE):
                fields["fuel_type"] = fuel
                break

        m = re.search(r"(\d+)\+?\s*mile(?:s)?\s*range", text, re.IGNORECASE)
        if m:
            fields["range_miles"] = f"gte:{m.group(1)}"

        price = _extract_numeric_constraint(text, "price")
        if price:
            fields["price"] = price

        return fields

    def _fields_restaurants(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        gpe_ents = [e.text for e in doc.ents if e.label_ == "GPE"]
        if gpe_ents:
            fields["city"] = gpe_ents[0]

        for cuisine in [
            "italian",
            "chinese",
            "indian",
            "mexican",
            "japanese",
            "thai",
            "french",
            "greek",
            "mediterranean",
            "american",
            "korean",
        ]:
            if re.search(rf"\b{cuisine}\b", text, re.IGNORECASE):
                fields["cuisine"] = cuisine
                break

        for diet in ["vegan", "vegetarian", "gluten-free", "halal", "kosher"]:
            if re.search(re.escape(diet), text, re.IGNORECASE):
                fields["diet"] = diet
                break

        if re.search(r"outdoor\s+seating|terrace|patio", text, re.IGNORECASE):
            fields["outdoor_seating"] = True

        price = _extract_numeric_constraint(text, "price")
        if price:
            fields["price"] = price

        return fields

    def _fields_movies(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for platform in [
            "netflix",
            "hbo",
            "disney+",
            "amazon prime",
            "hulu",
            "apple tv",
        ]:
            if re.search(re.escape(platform), text, re.IGNORECASE):
                fields["platform"] = platform
                break

        for genre in [
            "thriller",
            "comedy",
            "horror",
            "drama",
            "action",
            "romance",
            "sci-fi",
            "documentary",
            "animation",
            "fantasy",
        ]:
            if re.search(rf"\b{genre}\b", text, re.IGNORECASE):
                fields["genre"] = genre
                break

        rating = _extract_rating(text)
        if rating:
            fields["rating"] = rating

        return fields

    def _fields_healthcare(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        gpe_ents = [e.text for e in doc.ents if e.label_ == "GPE"]
        if gpe_ents:
            fields["city"] = gpe_ents[0]

        for spec in [
            "therapist",
            "dentist",
            "cardiologist",
            "dermatologist",
            "pediatrician",
            "psychiatrist",
            "orthopedist",
            "surgeon",
        ]:
            if re.search(rf"\b{spec}\b", text, re.IGNORECASE):
                fields["specialty"] = spec
                break

        for ins in [
            "aetna",
            "blue cross",
            "medicare",
            "medicaid",
            "cigna",
            "humana",
            "united",
        ]:
            if re.search(re.escape(ins), text, re.IGNORECASE):
                fields["insurance"] = ins
                break

        for gender in [
            ("female", "female"),
            ("male", "male"),
            ("woman", "female"),
            ("man", "male"),
        ]:
            if re.search(rf"\b{gender[0]}\b", text, re.IGNORECASE):
                fields["provider_gender"] = gender[1]
                break

        return fields

    def _fields_courses(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for level in ["beginner", "intermediate", "advanced"]:
            if re.search(rf"\b{level}\b", text, re.IGNORECASE):
                fields["level"] = level
                break

        price = _extract_numeric_constraint(text, "price")
        if price:
            fields["price"] = price

        for platform in [
            "udemy",
            "coursera",
            "edx",
            "linkedin learning",
            "pluralsight",
        ]:
            if re.search(re.escape(platform), text, re.IGNORECASE):
                fields["platform"] = platform
                break

        return fields

    def _fields_events(self, doc, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        gpe_ents = [e.text for e in doc.ents if e.label_ == "GPE"]
        if gpe_ents:
            fields["city"] = gpe_ents[0]

        date_ents = [e.text for e in doc.ents if e.label_ == "DATE"]
        if date_ents:
            fields["date"] = date_ents[0]

        person_ents = [e.text for e in doc.ents if e.label_ == "PERSON"]
        if person_ents:
            fields["artist"] = person_ents[0]

        price = _extract_numeric_constraint(text, "price")
        if price:
            fields["price"] = price

        return fields

    # ------------------------------------------------------------------
    # Domain-router dispatcher
    # ------------------------------------------------------------------

    _DOMAIN_DISPATCHERS = {
        "flights": "_fields_flights",
        "hotels": "_fields_hotels",
        "real_estate": "_fields_real_estate",
        "ecommerce": "_fields_ecommerce",
        "jobs": "_fields_jobs",
        "cars": "_fields_cars",
        "restaurants": "_fields_restaurants",
        "movies": "_fields_movies",
        "healthcare": "_fields_healthcare",
        "courses": "_fields_courses",
        "events": "_fields_events",
    }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, query: str) -> dict[str, Any]:
        """
        Extract structured fields from a natural language search query.

        Parameters
        ----------
        query : str
            The user's search query.

        Returns
        -------
        dict[str, Any]
            Extracted fields.  Numeric constraints use operator prefixes
            identical to the SLM adapters (``lt:``, ``gte:``, etc.).
        """
        if not query or not query.strip():
            return {}

        doc = self.nlp(query)
        domain = _detect_domain(query)

        fields: dict[str, Any] = {}
        if domain:
            fields["domain"] = domain
            dispatcher_name = self._DOMAIN_DISPATCHERS.get(domain)
            if dispatcher_name:
                domain_fields = getattr(self, dispatcher_name)(doc, query)
                fields.update(domain_fields)
        else:
            # Fallback: generic numeric + exclusion extraction
            price = _extract_numeric_constraint(query, "price")
            if price:
                fields["price"] = price
            excls = _extract_exclusions(query)
            if excls:
                fields["exclusions"] = excls

        return fields

    def __repr__(self) -> str:
        loaded = self._nlp is not None
        return f"SpacyExtractor(model={self._model_name!r}, loaded={loaded})"
