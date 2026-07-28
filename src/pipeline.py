"""End-to-end pipeline: image -> CV -> CNN -> nutrition -> NLP advice.

``analyze()`` returns a fully populated MealAnalysis; if the quality gate
fails it returns early with the Azerbaijani reasons.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

import numpy as np

from config import CONFIDENCE_MIN, settings
from src.cnn.predict import get_predictor
from src.cv.portion import estimate_portion
from src.cv.quality import check_quality
from src.cv.segment import detect_plate, masked_crop, segment_food
from src.nlp.advisor import advise
from src.schemas import Macros, MealAnalysis, UserProfile

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_nutrition_db() -> dict:
    with settings.nutrition_db_path.open(encoding="utf-8") as f:
        return json.load(f)


def macros_for(food_class: str, grams: float) -> tuple[Macros, list[str], str]:
    """Scale per-100g nutrition to grams; returns (macros, tags, az_name)."""
    entry = load_nutrition_db()[food_class]
    p = entry["per_100g"]
    k = grams / 100.0
    macros = Macros(kcal=round(p["kcal"] * k, 1),
                    protein_g=round(p["protein_g"] * k, 1),
                    carb_g=round(p["carb_g"] * k, 1),
                    fat_g=round(p["fat_g"] * k, 1),
                    fiber_g=round(p["fiber_g"] * k, 1),
                    sugar_g=round(p["sugar_g"] * k, 1),
                    sodium_mg=round(p["sodium_mg"] * k, 1))
    return macros, list(entry.get("tags", [])), entry["az_name"]


def analyze(img_bgr: np.ndarray, profile: UserProfile | None = None,
            model_name: str = "effnet", with_advice: bool = True,
            grams_override: float | None = None) -> MealAnalysis:
    """Run the full photo pipeline on a BGR image.

    Args:
        img_bgr: Input image (BGR uint8).
        profile: User profile for advice; default profile when None.
        model_name: Which trained checkpoint to use.
        with_advice: Skip the NLP step when False (faster for API bulk use).
        grams_override: Manual portion override from the UI slider.

    Returns:
        MealAnalysis; ``ok=False`` with quality_reasons_az on gate failure.
    """
    profile = profile or UserProfile()

    quality = check_quality(img_bgr)
    if not quality.ok:
        return MealAnalysis(ok=False, quality_reasons_az=quality.reasons_az)

    plate = detect_plate(img_bgr)
    mask = segment_food(img_bgr, plate)

    predictor = get_predictor(model_name)
    # Classify the FULL image. The CNN was trained on whole Food101 frames, so
    # feeding a background-masked crop is a train/test mismatch that badly hurts
    # accuracy (e.g. a burger gets called sushi). Segmentation is used only for
    # portion estimation and the mask overlay, never as classifier input.
    top5 = predictor.predict(img_bgr, topk=5)
    food_class, confidence = top5[0]

    portion = estimate_portion(mask, plate, food_class)
    grams = grams_override if grams_override else portion.grams
    macros, tags, az_name = macros_for(food_class, grams)

    meal = MealAnalysis(ok=True, food_class=food_class, az_name=az_name,
                        confidence=round(confidence, 3),
                        low_confidence=confidence < CONFIDENCE_MIN,
                        top5=top5, grams=grams,
                        portion_bucket=portion.bucket,
                        portion_confidence=portion.confidence,
                        macros=macros, tags=tags, source="photo")
    if with_advice:
        meal.advice_az = advise(meal, profile)
    return meal


def analysis_extras(img_bgr: np.ndarray, model_name: str = "effnet") -> dict:
    """Visual artefacts for the UI: plate circle, mask, crop, Grad-CAM overlay."""
    import torch

    from src.cnn.gradcam import GradCAM, overlay_heatmap, target_layer

    plate = detect_plate(img_bgr)
    mask = segment_food(img_bgr, plate)
    crop = masked_crop(img_bgr, mask)
    predictor = get_predictor(model_name)
    cam = GradCAM(predictor.model, target_layer(predictor.model, model_name))
    # Grad-CAM on the full image so the heatmap matches what was classified.
    with torch.enable_grad():
        heat = cam(predictor.to_tensor(img_bgr))
    cam.close()
    overlay = overlay_heatmap(img_bgr, heat)
    return {"plate": plate, "mask": mask, "crop": crop, "gradcam": overlay}


if __name__ == "__main__":
    import sys

    import cv2

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.pipeline <image>")
    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit(f"Could not read image: {sys.argv[1]}")
    result = analyze(img)
    print(result)
