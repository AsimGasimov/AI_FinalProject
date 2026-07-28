"""Shared dataclasses used across CV, CNN, NLP and API layers.

Kept in one module to avoid circular imports between pipeline, advisor and db.
(Addition to the spec layout; see DECISIONS.md.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DISCLAIMER_AZ = ("Qeyd: Bu, tibbi məsləhət deyil. Fərdi tövsiyələr üçün həkim "
                 "və ya diyetoloqla məsləhətləşin.")


@dataclass
class Macros:
    """Absolute macro amounts for a serving."""

    kcal: float
    protein_g: float
    carb_g: float
    fat_g: float
    fiber_g: float = 0.0
    sugar_g: float = 0.0
    sodium_mg: float = 0.0


@dataclass
class UserProfile:
    """User attributes needed for targets and advice."""

    name: str = "Qonaq"
    age: int = 30
    sex: str = "m"  # "m" | "f"
    height_cm: float = 175.0
    weight_kg: float = 75.0
    activity: str = "light"  # sedentary|light|moderate|active|very_active
    goal: str = "maintain"  # lose|maintain|gain
    daily_kcal_target: Optional[float] = None


@dataclass
class MealAnalysis:
    """Full result of analysing one meal photo (or text entry)."""

    ok: bool
    food_class: str = ""
    az_name: str = ""
    confidence: float = 0.0
    low_confidence: bool = False
    top5: list[tuple[str, float]] = field(default_factory=list)
    grams: float = 0.0
    portion_bucket: str = ""
    portion_confidence: str = ""
    macros: Optional[Macros] = None
    tags: list[str] = field(default_factory=list)
    quality_reasons_az: list[str] = field(default_factory=list)
    advice_az: str = ""
    source: str = "photo"  # photo | text


@dataclass
class DailySummary:
    """Daily log rollup + next-day plan, all user-facing text in Azerbaijani."""

    date: str
    total: Macros
    target_kcal: float
    summary_az: str
    plan_az: str
