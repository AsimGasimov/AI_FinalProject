"""Conversational nutrition chatbot (RAG-grounded, goal-aware).

Answers free-text questions in Azerbaijani, tailored to the user's goal
(``lose`` | ``maintain`` | ``gain``). Like the advisor it retrieves guideline
chunks first (real RAG, not a bare LLM call) and then either sends them to a
generative provider or, in the default ``template`` mode, builds a deterministic
grounded answer from the retrieved text. Every reply ends with the disclaimer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.nlp import llm
from src.nlp.retriever import Chunk, retrieve
from src.schemas import DISCLAIMER_AZ, UserProfile

log = logging.getLogger(__name__)

SYSTEM_AZ = (
    "Sən FoodLens qidalanma köməkçisisən. İstifadəçinin məqsədini (arıqlama, "
    "saxlama və ya kütlə artırma) və verilən konteksti nəzərə alaraq qısa "
    "(3-5 cümlə), konkret və praktiki cavab ver. Yalnız verilən konteksti "
    "istifadə et, uydurma. Tibbi diaqnoz qoyma. Cavabı Azərbaycan dilində ver."
)

GOAL_AZ = {"lose": "arıqlama", "maintain": "çəkini saxlama", "gain": "kütlə artırma"}

# (principle, how-much) phrasing per goal for the deterministic intro line.
GOAL_DIRECTION = {
    "lose": ("kalori defisiti", "gündə təxminən 300-500 kkal az qəbul edin"),
    "maintain": ("kalori tarazlığı", "hədəf kalori ətrafında qalın"),
    "gain": ("kalori profisiti", "gündə təxminən 250-500 kkal çox qəbul edin"),
}

# Query expansion so retrieval surfaces goal-relevant guideline chunks even
# when the user's question is short.
GOAL_QUERY = {
    "lose": "arıqlama kalori defisiti zülal lif porsiya nəzarəti",
    "maintain": "kalori tarazlığı balanslı qidalanma zülal",
    "gain": "kütlə artırma kalori profisiti zülal əzələ öyün",
}

# Protein target band (g per kg body weight) per goal.
PROTEIN_PER_KG = {"lose": (1.2, 1.6), "maintain": (1.2, 1.6), "gain": (1.6, 2.2)}


@dataclass
class ChatReply:
    """One chatbot turn: user-facing Azerbaijani text plus cited sources."""

    text_az: str
    sources: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    """Split a chunk into trimmed sentences, dropping markdown header lines."""
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]


def _relevant_sentences(question: str, chunk: Chunk, limit: int = 2) -> list[str]:
    """Pick up to ``limit`` sentences from a chunk, preferring ones that share
    a word stem with the question; falls back to the opening sentences."""
    q_tokens = [t[:5] for t in re.findall(r"\w+", question.lower()) if len(t) >= 4]
    sents = _sentences(chunk.text)
    if q_tokens:
        matched = [s for s in sents
                   if any(stem in s.lower() for stem in q_tokens)]
        if matched:
            return matched[:limit]
    return sents[:limit]


def _protein_hint(profile: UserProfile) -> str:
    lo, hi = PROTEIN_PER_KG.get(profile.goal, (1.2, 1.6))
    return f"{lo * profile.weight_kg:.0f}-{hi * profile.weight_kg:.0f} q"


def _template_answer(question: str, profile: UserProfile,
                     chunks: list[Chunk], remaining_kcal: float | None) -> str:
    """Deterministic, retrieval-grounded answer for the no-API default mode."""
    goal = profile.goal if profile.goal in GOAL_AZ else "maintain"
    principle, howmuch = GOAL_DIRECTION[goal]
    target = profile.daily_kcal_target or 2000.0

    lines = [
        f"Məqsədiniz **{GOAL_AZ[goal]}**. Əsas prinsip {principle}: {howmuch}. "
        f"Gündəlik hədəfiniz təxminən {target:.0f} kkal, zülal isə "
        f"{_protein_hint(profile)} olsun."
    ]
    if remaining_kcal is not None:
        if remaining_kcal > 0:
            lines.append(f"Bu gün hədəfinizə çatmaq üçün daha "
                         f"~{remaining_kcal:.0f} kkal yeriniz var.")
        else:
            lines.append(f"Bu gün hədəfi ~{-remaining_kcal:.0f} kkal keçmisiniz, "
                         f"qalan öyünləri yüngül saxlayın.")

    guideline = next((c for c in chunks if c.source.endswith(".md")), None)
    if guideline is not None:
        tips = " ".join(_relevant_sentences(question, guideline, limit=2))
        if tips:
            lines.append(f"Təlimatlardan: {tips}")
    return " ".join(lines)


def answer(question: str, profile: UserProfile,
           history: list[tuple[str, str]] | None = None,
           remaining_kcal: float | None = None) -> ChatReply:
    """Answer a nutrition question, grounded in retrieved guidelines.

    Args:
        question: The user's free-text message (Azerbaijani).
        profile: User profile; ``goal`` steers the advice direction.
        history: Prior (role, text) turns for LLM context; ignored in template
            mode. ``role`` is "user" or "assistant".
        remaining_kcal: Optional kcal left against today's target, surfaced in
            the reply when provided.

    Returns:
        ChatReply with Azerbaijani text (disclaimer appended) and cited sources.
    """
    query = f"{question} {GOAL_QUERY.get(profile.goal, '')}".strip()
    chunks = retrieve(query, k=4)
    sources = list(dict.fromkeys(c.source for c in chunks))

    if llm.active_provider() == "template":
        body = _template_answer(question, profile, chunks, remaining_kcal)
    else:
        context = "\n---\n".join(f"[{c.source}] {c.text}" for c in chunks)
        convo = ""
        for role, text in (history or [])[-4:]:
            who = "İstifadəçi" if role == "user" else "Köməkçi"
            convo += f"{who}: {text}\n"
        target = profile.daily_kcal_target or 2000.0
        user = (
            f"Kontekst (təlimatlardan):\n{context}\n\n"
            f"Profil: {profile.age} yaş, məqsəd: {GOAL_AZ.get(profile.goal, profile.goal)}, "
            f"çəki {profile.weight_kg:.0f} kq, gündəlik hədəf {target:.0f} kkal, "
            f"tövsiyə olunan zülal {_protein_hint(profile)}.\n"
            + (f"Bugünkü qalan kalori: {remaining_kcal:.0f} kkal.\n"
               if remaining_kcal is not None else "")
            + (f"\nSöhbət tarixçəsi:\n{convo}" if convo else "")
            + f"\nSual: {question}")
        body = llm.generate(SYSTEM_AZ, user)

    return ChatReply(text_az=f"{body}\n\n{DISCLAIMER_AZ}", sources=sources)


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    for g in ("lose", "gain"):
        prof = UserProfile(goal=g, weight_kg=80, daily_kcal_target=2200)
        r = answer("Necə başlamalıyam, nə yeməliyəm?", prof)
        print(f"\n=== goal={g} ===\n{r.text_az}\nMənbələr: {r.sources}")
