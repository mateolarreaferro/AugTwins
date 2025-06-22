"""
Centralised helper for OpenAI ChatCompletion.
this module focuses solely on LLM calls.
"""
from __future__ import annotations
import os
from typing import List, Dict, Optional

try:
    from openai import OpenAI
except ModuleNotFoundError:  # allow tests without openai installed
    OpenAI = None

# API key
try:
    from settings import OPENAI_API_KEY as _OPENAI_KEY
except (ModuleNotFoundError, ImportError):
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

if OpenAI:
    client = OpenAI(api_key=_OPENAI_KEY)
else:
    client = None


def chat(
    messages: List[Dict[str, str]],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    """Basic wrapper that returns *only* the assistant reply string."""
    if not client:
        raise RuntimeError("OpenAI client unavailable")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# convenience alias for the code-assistant REPL
def gen_oai(history: List[Dict[str, str]], *, model: str = "gpt-4o-mini",
            temperature: float = 0.2) -> str:
    """
    Thin wrapper identical to the original script’s expectation:
    takes the running chat *history* and returns the assistant text.
    """
    return chat(history, model=model, temperature=temperature)