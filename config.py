"""Central configuration for FoodLens.

Everything path- or environment-dependent lives here. No module may
hardcode paths; import ``settings`` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent

# Full Food101 class set (all 101 classes, alphabetical = Food101 meta order).
# Expanded from the original 25-class subset so the model can recognise the
# whole Food101 catalogue. Retraining on GPU required after any change here.
CLASSES: list[str] = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles",
]

SEED = 42
IMG_SIZE = 224
NUM_CLASSES = len(CLASSES)

# Minimum top-1 softmax probability to treat a prediction as confident.
# Chosen empirically: on a held-out sample the EfficientNet keeps ~83% of
# genuine in-class predictions above this value, while flagging the least
# certain half of out-of-distribution foods (dishes outside the 25 classes,
# e.g. plov/kabab). A 25-class closed-set model cannot reliably recognise
# arbitrary foods, so below this threshold the UI shows the top-3 candidates
# and lets the user correct the class instead of asserting a single answer.
CONFIDENCE_MIN = 0.50


class Settings(BaseSettings):
    """Runtime settings, overridable via .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: Literal["template", "local", "anthropic", "openai"] = "template"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Optional OpenAI-compatible base URL (e.g. Groq/Gemini free tiers). Empty =
    # real OpenAI. Groq: https://api.groq.com/openai/v1
    openai_base_url: str = ""
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_workers: int = 0
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'foodlens.db'}"

    # Paths (derived, not env-driven)
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def nutrition_db_path(self) -> Path:
        return self.data_dir / "nutrition_db.json"

    @property
    def guidelines_dir(self) -> Path:
        return self.data_dir / "guidelines"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    def resolve_device(self) -> str:
        """Return the actual torch device string ('cuda' or 'cpu')."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


settings = Settings()

if __name__ == "__main__":
    print(f"root={PROJECT_ROOT}")
    print(f"classes={NUM_CLASSES}, provider={settings.llm_provider}, device={settings.resolve_device()}")
