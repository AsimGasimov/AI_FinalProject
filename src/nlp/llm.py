"""LLM provider abstraction: anthropic | local (flan-t5-base) | template.

``generate(system, user)`` has the same signature for all providers.
``template`` is the default and the defence-day safety net: deterministic
Jinja rendering, zero network, zero API keys. Falls back template <- local
<- anthropic on any provider error so the demo can never crash here.
"""

from __future__ import annotations

import logging

from config import settings

log = logging.getLogger(__name__)

_local_pipe = None


def _generate_anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=600, system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _generate_openai(system: str, user: str) -> str:
    from openai import OpenAI

    # base_url lets the same client target OpenAI-compatible endpoints such as
    # Groq or Gemini (free tiers); empty base_url -> real OpenAI.
    client = OpenAI(api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url or None)
    resp = client.chat.completions.create(
        model=settings.openai_model, max_tokens=600,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return (resp.choices[0].message.content or "").strip()


def _generate_local(system: str, user: str) -> str:
    global _local_pipe
    if _local_pipe is None:
        from transformers import pipeline

        _local_pipe = pipeline("text2text-generation", model="google/flan-t5-base",
                               device=-1)
    out = _local_pipe(f"{system}\n\n{user}", max_new_tokens=300, do_sample=False)
    return out[0]["generated_text"].strip()


def _generate_template(system: str, user: str) -> str:
    """Deterministic fallback: the caller (advisor/summarizer) is expected to
    pass a pre-rendered template text as ``user`` when provider=template.

    For template mode the "generation" is identity: advisor and summarizer
    build the full Azerbaijani text themselves via Jinja and route it through
    here so all three providers share one code path.
    """
    return user.strip()


def generate(system: str, user: str) -> str:
    """Generate text with the active provider, falling back safely."""
    provider = active_provider()
    if provider == "anthropic":
        try:
            return _generate_anthropic(system, user)
        except Exception:  # noqa: BLE001 - any API failure must not kill the demo
            log.exception("anthropic provider failed, falling back to local")
            provider = "local"
    if provider == "openai":
        try:
            return _generate_openai(system, user)
        except Exception:  # noqa: BLE001
            log.exception("openai provider failed, falling back to local")
            provider = "local"
    if provider == "local":
        try:
            return _generate_local(system, user)
        except Exception:  # noqa: BLE001
            log.exception("local provider failed, falling back to template")
    return _generate_template(system, user)


def active_provider() -> str:
    """The provider that would actually be used right now (key-aware)."""
    p = settings.llm_provider
    if p == "anthropic" and settings.anthropic_api_key:
        return "anthropic"
    if p == "openai" and settings.openai_api_key:
        return "openai"
    if p == "local":
        return "local"
    return "template"


if __name__ == "__main__":
    print(f"provider={active_provider()}")
    print(generate("test system", "salam, bu template cavabıdır"))
