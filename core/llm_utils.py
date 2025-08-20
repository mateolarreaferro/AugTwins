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

# API key - load from centralized config
from config import OPENAI_API_KEY

if OpenAI:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None


def chat(
    messages: List[Dict[str, str]],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
) -> str:
    """Basic wrapper that returns *only* the assistant reply string."""
    if not client:
        raise RuntimeError("OpenAI client unavailable")
    
    # Use max_completion_tokens for GPT-5 models, max_tokens for others
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    if model.startswith("gpt-5"):
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        elif max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
    else:
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
    
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    return content.strip() if content else ""


# convenience alias for the code-assistant REPL
def gen_oai(history: List[Dict[str, str]], *, model: str = "gpt-4o-mini",
            temperature: float = 0.2) -> str:
    """
    Thin wrapper identical to the original script’s expectation:
    takes the running chat *history* and returns the assistant text.
    """
    return chat(history, model=model, temperature=temperature)