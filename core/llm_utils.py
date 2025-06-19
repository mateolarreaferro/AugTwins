"""
Centralised helper for OpenAI ChatCompletion.

Voice-synthesis lives in Agent.speak;
this module focuses solely on LLM calls.
"""
from __future__ import annotations
import os
from typing import List, Dict, Optional

import openai

# ── API key ──────────────────────────────────────────────────
try:
    from settings import OPENAI_API_KEY as _OPENAI_KEY
except (ModuleNotFoundError, ImportError):
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

openai.api_key = _OPENAI_KEY


# ------------------------------------------------------------------
def chat(
    messages: List[Dict[str, str]],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    """Basic wrapper that returns *only* the assistant reply string."""
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message["content"].strip()


# convenience alias for the code-assistant REPL --------------------
def gen_oai(history: List[Dict[str, str]], *, model: str = "gpt-4o-mini",
            temperature: float = 0.2) -> str:
    """
    Thin wrapper identical to the original script’s expectation:
    takes the running chat *history* and returns the assistant text.
    """
    return chat(history, model=model, temperature=temperature)
