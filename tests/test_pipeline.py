"""End-to-end pipeline test on one real Food101 image.

Skipped automatically when the dataset or the effnet checkpoint is missing
(e.g. on a fresh clone before training).
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from config import settings
from src import db
from src.schemas import UserProfile

# splits.json and the checkpoints are committed, but data/raw/ is gitignored, so
# the image files are the only reliable probe for "dataset present".
needs_data = pytest.mark.skipif(
    not (settings.data_dir / "splits.json").exists()
    or not (settings.raw_dir / "food-101" / "images").is_dir()
    or not (settings.models_dir / "effnet_best.pt").exists(),
    reason="dataset or effnet checkpoint missing")


def _one_test_image() -> np.ndarray:
    with (settings.data_dir / "splits.json").open(encoding="utf-8") as f:
        splits = json.load(f)
    rel, _ = splits["test"][0]
    img = cv2.imread(str(settings.raw_dir / "food-101" / rel))
    assert img is not None
    return img


@needs_data
def test_analyze_end_to_end() -> None:
    from src.pipeline import analyze

    meal = analyze(_one_test_image(), UserProfile(daily_kcal_target=2000))
    assert meal.ok, meal.quality_reasons_az
    assert meal.food_class
    assert meal.grams > 0
    assert meal.macros is not None and meal.macros.kcal > 0
    assert meal.advice_az
    assert "tibbi məsləhət deyil" in meal.advice_az


@needs_data
def test_quality_gate_rejects_dark() -> None:
    from src.pipeline import analyze

    dark = np.zeros((300, 300, 3), np.uint8)
    meal = analyze(dark)
    assert not meal.ok
    assert meal.quality_reasons_az


def test_daily_kcal_target_mifflin() -> None:
    # 30yo male, 175cm, 75kg, light, maintain: BMR=1698.75, x1.375 = 2336
    assert db.daily_kcal_target(30, "m", 175, 75, "light", "maintain") == 2336
    # Female variant is 166 kcal lower at BMR level.
    assert db.daily_kcal_target(30, "f", 175, 75, "light", "maintain") < 2336


def test_macros_scaling() -> None:
    from src.pipeline import macros_for

    macros, tags, az = macros_for("pizza", 200)
    assert macros.kcal == pytest.approx(532, abs=1)
    assert "high_sodium" in tags
    assert az == "Pitsa"
