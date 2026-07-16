"""Azerbaijani meal parser (NER): free text -> [(food, qty, unit)].

Pipeline (spec section 6, with one engineering addition, see DECISIONS.md #5):
  1. Number normalisation: AZ number words + digits -> float.
  2. Unit lexicon with suffix tolerance (dilim, ədəd, boşqab, stəkan, ...).
  3. Food matching: exact/substring synonym lexicon FIRST (MiniLM is
     English-centric and unreliable on Azerbaijani), then MiniLM cosine
     similarity fallback (threshold 0.55). Unknown foods are surfaced with
     food="unknown", never dropped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from config import CLASSES

log = logging.getLogger(__name__)

NUMBER_WORDS: dict[str, float] = {
    "bir": 1, "iki": 2, "üç": 3, "dörd": 4, "beş": 5, "altı": 6,
    "yeddi": 7, "səkkiz": 8, "doqquz": 9, "on": 10,
    "yarım": 0.5, "yarim": 0.5,
}
FEW_PHRASE = "bir neçə"  # ~3
FEW_VALUE = 3.0

UNITS = ["dilim", "ədəd", "boşqab", "stəkan", "qaşıq", "porsiya", "qram", "kq",
         "kasa", "kürəcik", "fincan"]

# AZ/EN synonyms -> class key (or a known non-class food kept for UX).
FOOD_LEXICON: dict[str, str] = {
    "pitsa": "pizza", "pizza": "pizza",
    "hamburger": "hamburger", "burger": "hamburger", "cheeseburger": "hamburger",
    "salmon": "grilled_salmon", "fries": "french_fries",
    "kartof fri": "french_fries", "fri": "french_fries", "free": "french_fries",
    "sezar": "caesar_salad", "sezar salatı": "caesar_salad",
    "suşi": "sushi", "sushi": "sushi",
    "steyk": "steak", "biftek": "steak",
    "qızardılmış düyü": "fried_rice", "düyü": "fried_rice", "duyu": "fried_rice",
    "spagetti": "spaghetti_bolognese", "bolonez": "spaghetti_bolognese",
    "makaron": "spaghetti_bolognese", "əriştə": "spaghetti_bolognese",
    "pankek": "pancakes", "blin": "pancakes",
    "omlet": "omelette", "qayğanaq": "omelette",
    "qızılbalıq": "grilled_salmon", "somon": "grilled_salmon", "losos": "grilled_salmon",
    "kari": "chicken_curry", "toyuq kari": "chicken_curry",
    "ponçik": "donuts", "donut": "donuts",
    "çizkeyk": "cheesecake", "cheesecake": "cheesecake",
    "dondurma": "ice_cream",
    "hot-doq": "hot_dog", "hotdoq": "hot_dog", "hot doq": "hot_dog", "sosiska": "hot_dog",
    "düşbərə": "dumplings", "xəngəl": "dumplings", "dumplinq": "dumplings",
    "falafel": "falafel", "fələfəl": "falafel",
    "yunan salatı": "greek_salad", "yunan": "greek_salad",
    "lazanya": "lasagna",
    "ramen": "ramen",
    "vafli": "waffles",
    "tako": "tacos", "taco": "tacos",
    "quakamole": "guacamole", "avokado sousu": "guacamole",
    "klub sendviç": "club_sandwich", "sendviç": "club_sandwich",
    # Common foods outside the 25 classes: recognised, flagged in_db=False.
    "kola": "cola", "cola": "cola", "çay": "tea", "kofe": "coffee",
    "çörək": "bread", "plov": "plov", "aş": "plov",
}
NON_DB_FOODS = {"cola", "tea", "coffee", "bread", "plov"}

# Spec says 0.55, but MiniLM produces surface-form false positives on unseen
# Azerbaijani words (measured: "qutab" -> guacamole at 0.70). 0.75 blocks those
# while keeping genuine cross-lingual matches; see DECISIONS.md and nlp_eval.md.
EMBED_THRESHOLD = 0.75
_embedder = None
_class_embeddings = None
_class_targets: list[str] = []
# Sticky flag: once the embedder proves unavailable (not installed, or no model
# cache and no network), stop retrying it on every parse.
_embedder_unavailable = False


@dataclass
class MealEntity:
    """One parsed food mention.

    Attributes:
        food: Class key, known non-class food, or "unknown".
        qty: Quantity as float (default 1.0).
        unit: Unit word or None.
        raw: The source text span.
        in_db: True when food is one of the 25 nutrition_db classes.
        matched_via: "lexicon" | "embedding" | "none".
        similarity: Cosine similarity when matched via embedding.
    """

    food: str
    qty: float
    unit: Optional[str]
    raw: str
    in_db: bool = True
    matched_via: str = "lexicon"
    similarity: Optional[float] = None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("i̇", "i"))


def _extract_qty(segment: str) -> tuple[float, str]:
    """Pull the quantity out of a segment; returns (qty, remainder)."""
    if FEW_PHRASE in segment:
        return FEW_VALUE, segment.replace(FEW_PHRASE, " ")
    m = re.search(r"(\d+(?:[.,]\d+)?)", segment)
    if m:
        qty = float(m.group(1).replace(",", "."))
        return qty, segment.replace(m.group(1), " ", 1)
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", segment):
            return value, re.sub(rf"\b{word}\b", " ", segment, count=1)
    return 1.0, segment


def _extract_unit(segment: str) -> tuple[Optional[str], str]:
    """Find a unit token (suffix-tolerant); returns (unit, remainder)."""
    for token in segment.split():
        for unit in UNITS:
            if token == unit or token.startswith(unit):
                return unit, segment.replace(token, " ", 1)
    return None, segment


def _lexicon_match(phrase: str) -> Optional[str]:
    """Exact then substring match against the synonym lexicon."""
    phrase = phrase.strip()
    if phrase in FOOD_LEXICON:
        return FOOD_LEXICON[phrase]
    for syn in sorted(FOOD_LEXICON, key=len, reverse=True):
        if re.search(rf"\b{re.escape(syn)}\b", phrase):
            return FOOD_LEXICON[syn]
    return None


def _load_embedder():
    """Construct the MiniLM embedder.

    Raises:
        Exception: If sentence-transformers is missing, or the model is neither
            cached nor downloadable (offline machine).
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _embedding_match(phrase: str) -> tuple[Optional[str], float]:
    """MiniLM cosine match against class names + az_names + synonyms.

    Degrades to (None, 0.0) when the embedder cannot be loaded at all, so an
    offline machine with no model cache falls back to "unknown" instead of
    raising. The lexicon is the first line of matching and needs no model, so
    the demo stays fully usable without a network.
    """
    global _embedder, _class_embeddings, _class_targets, _embedder_unavailable
    if _embedder_unavailable:
        return None, 0.0
    if _embedder is None:
        import json as _json

        from config import settings as _settings

        try:
            _embedder = _load_embedder()
        except Exception:  # noqa: BLE001 - missing dep or offline must not kill the demo
            log.warning(
                "MiniLM unavailable (not installed, offline, or no model cache); "
                "falling back to lexicon-only matching, unmatched foods become "
                "'unknown'.")
            _embedder_unavailable = True
            return None, 0.0
        names: list[str] = []
        targets: list[str] = []
        az_names: dict[str, str] = {}
        if _settings.nutrition_db_path.exists():
            with _settings.nutrition_db_path.open(encoding="utf-8") as f:
                az_names = {k: v["az_name"] for k, v in _json.load(f).items()}
        for cls in CLASSES:
            variants = {cls.replace("_", " ")}
            if cls in az_names:
                variants.add(az_names[cls].lower())
            variants |= {s for s, tgt in FOOD_LEXICON.items() if tgt == cls}
            for v in variants:
                names.append(v)
                targets.append(cls)
        _class_embeddings = _embedder.encode(names, normalize_embeddings=True)
        _class_targets = targets

    q = _embedder.encode([phrase], normalize_embeddings=True)
    sims = (_class_embeddings @ q.T).ravel()
    best = int(sims.argmax())
    if float(sims[best]) >= EMBED_THRESHOLD:
        return _class_targets[best], float(sims[best])
    return None, float(sims[best])


STOPWORDS = {"bugün", "dünən", "səhər", "səhərə", "nahar", "nahara", "naharda",
             "axşam", "axşama", "günorta", "yedim", "yemişəm", "içdim", "içmişəm",
             "yeyəcəm", "yedik", "var", "idi", "də", "da", "ki", "mən", "biz",
             "üçün", "sonra", "əvvəl", "yeməyinə", "yeməyində"}

NEGATION_PATTERNS = ["heç nə", "hec ne", "yeməmişəm", "yemədim", "içməmişəm"]


def parse_meal(text: str) -> list[MealEntity]:
    """Parse free Azerbaijani text into MealEntity list.

    Example:
        >>> parse_meal("2 dilim pitsa və bir stəkan kola")
        [MealEntity(food='pizza', qty=2.0, unit='dilim', ...),
         MealEntity(food='cola', qty=1.0, unit='stəkan', ...)]
    """
    norm = _normalise(text)
    if any(p in norm for p in NEGATION_PATTERNS) and _lexicon_match(norm) is None:
        return []

    segments = re.split(r"\bvə\b|,|;", norm)
    entities: list[MealEntity] = []
    for seg in segments:
        raw = seg.strip()
        if not raw:
            continue
        qty, rest = _extract_qty(raw)
        unit, rest = _extract_unit(rest)
        phrase = " ".join(t for t in rest.split() if t not in STOPWORDS).strip()
        if not phrase:
            continue

        food = _lexicon_match(phrase)
        via, sim = "lexicon", None
        if food is None:
            food, sim = _embedding_match(phrase)
            via = "embedding"
        if food is None:
            entities.append(MealEntity("unknown", qty, unit, raw, in_db=False,
                                       matched_via="none", similarity=sim))
        else:
            entities.append(MealEntity(food, qty, unit, raw,
                                       in_db=food in CLASSES,
                                       matched_via=via, similarity=sim))
    return entities


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    for s in ("bugün nahara 2 dilim pitsa və bir stəkan kola içdim",
              "yarım boşqab plov", "heç nə yeməmişəm"):
        print(s, "->", parse_meal(s))
