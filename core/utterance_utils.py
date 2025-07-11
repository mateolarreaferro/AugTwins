"""Utilities for crafting agent utterances with transcript-based style cues.

The transcript snippets describe how an agent speaks—not their favorite topics
or catchphrases. They guide delivery, phrasing, and rhythm only."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

from . import llm_utils

_TRANS_DIR = Path("transcripts")


def load_transcript(name: str) -> str:
    """Return style text from transcripts/<name>.json or <name>.txt if available."""
    json_path = _TRANS_DIR / f"{name.lower()}.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            style = data.get("style_guide", "")
            phrases = data.get("sample_phrases", [])
            phrase_block = "\n".join(f"- {p}" for p in phrases) if phrases else ""
            return f"{style}\n{phrase_block}".strip()
        except Exception:
            return json_path.read_text(encoding="utf-8").strip()

    path = _TRANS_DIR / f"{name.lower()}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def generate_utterance(
    *,
    agent_name: str,
    personality: str,
    user_msg: str,
    relevant: str,
    graph_info: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
) -> str:
    """Generate a reply in the style of *agent_name*, referencing transcripts."""
    transcript = load_transcript(agent_name)
    prompt = (
        f"You are {agent_name}. Personality: {personality}\n"
        "Speak casually, like a normal person. It's okay to use fragments or short answers. "
        "Only end with a question if it feels natural.\n"
    )
    if transcript:
        prompt += f"Example speech from transcript:\n{transcript}\n\n"
    prompt += (
        f"Relevant memories:\n{relevant}\n"
        f"Graph context: {graph_info}\n\n"
        f"User: {user_msg}\n{agent_name}:"
    )
    answer = llm_utils.chat(
        [{"role": "system", "content": prompt}],
        model=model,
        temperature=temperature,
    )
    cleaned = answer.lstrip()
    prefix = f"{agent_name}:"
    if cleaned.lower().startswith(prefix.lower()):
        cleaned = cleaned[len(prefix):].lstrip()
    return cleaned
