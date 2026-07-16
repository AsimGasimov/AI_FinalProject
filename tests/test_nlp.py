"""Meal parser tests: 16 Azerbaijani sentences including tricky cases."""

from __future__ import annotations

import pytest

from src.nlp.meal_parser import MealEntity, parse_meal


def first(text: str) -> MealEntity:
    entities = parse_meal(text)
    assert entities, f"nothing parsed from: {text}"
    return entities[0]


class TestQuantities:
    def test_digit_qty(self) -> None:
        e = first("3 dilim pitsa yedim")
        assert e.food == "pizza" and e.qty == 3.0 and e.unit == "dilim"

    def test_word_qty_bir(self) -> None:
        e = first("bir hamburger yedim")
        assert e.food == "hamburger" and e.qty == 1.0

    def test_word_qty_iki(self) -> None:
        e = first("iki ədəd ponçik")
        assert e.food == "donuts" and e.qty == 2.0 and e.unit == "ədəd"

    def test_yarim_bosqab_plov(self) -> None:
        e = first("yarım boşqab plov")
        assert e.qty == 0.5 and e.unit == "boşqab"
        assert e.food == "plov" and e.in_db is False  # surfaced, not dropped

    def test_uc_qasiq_duyu(self) -> None:
        e = first("3 qaşıq düyü yedim")
        assert e.qty == 3.0 and e.unit == "qaşıq" and e.food == "fried_rice"

    def test_bir_nece(self) -> None:
        e = first("bir neçə vafli yedim")
        assert e.food == "waffles" and e.qty == 3.0

    def test_decimal_qty(self) -> None:
        e = first("1.5 porsiya lazanya")
        assert e.food == "lasagna" and e.qty == 1.5 and e.unit == "porsiya"

    def test_default_qty_is_one(self) -> None:
        e = first("səhər omlet yedim")
        assert e.food == "omelette" and e.qty == 1.0


class TestMultiEntity:
    def test_two_foods_with_ve(self) -> None:
        ents = parse_meal("2 dilim pitsa və bir stəkan kola")
        assert len(ents) == 2
        assert ents[0].food == "pizza" and ents[0].qty == 2.0 and ents[0].unit == "dilim"
        assert ents[1].food == "cola" and ents[1].qty == 1.0 and ents[1].unit == "stəkan"
        assert ents[1].in_db is False

    def test_comma_separated(self) -> None:
        ents = parse_meal("bir omlet, 2 vafli")
        assert [e.food for e in ents] == ["omelette", "waffles"]

    def test_three_foods(self) -> None:
        ents = parse_meal("bugün 1 hamburger, 1 porsiya kartof fri və 1 stəkan kola yedim")
        assert [e.food for e in ents] == ["hamburger", "french_fries", "cola"]


class TestEdgeCases:
    def test_hec_ne(self) -> None:
        assert parse_meal("heç nə yeməmişəm") == []

    def test_unknown_food_surfaced(self) -> None:
        ents = parse_meal("2 ədəd qutab yedim")
        assert len(ents) == 1
        assert ents[0].in_db is False  # unknown or non-db, but never dropped

    def test_synonym_dusbere(self) -> None:
        e = first("bir boşqab düşbərə")
        assert e.food == "dumplings"

    def test_time_words_ignored(self) -> None:
        e = first("axşam bir dilim çizkeyk yedim")
        assert e.food == "cheesecake" and e.unit == "dilim"

    def test_gram_unit(self) -> None:
        e = first("200 qram steyk")
        assert e.food == "steak" and e.qty == 200.0 and e.unit == "qram"

    def test_unloadable_embedder_degrades_to_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An embedder that cannot load (offline, no model cache) must not raise.

        LLM_PROVIDER=template keeps the demo offline, but the parser's MiniLM
        fallback would still reach for huggingface.co on an unseen food; that
        must degrade to "unknown", never crash the text tab.
        """
        from src.nlp import meal_parser as mp

        monkeypatch.setattr(mp, "_embedder", None)
        monkeypatch.setattr(mp, "_embedder_unavailable", False)
        monkeypatch.setattr(
            mp, "_load_embedder",
            lambda: (_ for _ in ()).throw(OSError("no network, no cache")))

        ents = mp.parse_meal("2 ədəd qutab yedim")  # not in the lexicon
        assert len(ents) == 1
        assert ents[0].food == "unknown" and ents[0].in_db is False
        assert mp._embedder_unavailable is True  # sticky: no retry storm


@pytest.mark.parametrize("text,expected_food", [
    ("suşi yedim", "sushi"),
    ("bir kasa ramen", "ramen"),
    ("toyuq kari yedim", "chicken_curry"),
    ("yunan salatı yedim", "greek_salad"),
])
def test_food_matching(text: str, expected_food: str) -> None:
    assert first(text).food == expected_food
