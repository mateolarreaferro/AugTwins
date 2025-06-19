"""Disk persistence + LLM summarisation utilities.

This version avoids circular-import problems by:
  • NOT importing Agent/Memory at module load time
  • Importing Memory lazily inside load_memories()
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, TYPE_CHECKING

import llm_utils

# ------------------------------------------------------------------
# Optional forward references for static type checkers only
if TYPE_CHECKING:          # <- evaluated by tools like mypy, ignored at runtime
    from .agent import Agent, Memory

# ------------------------------------------------------------------
# storage path
_DIR = Path("memories")
_DIR.mkdir(exist_ok=True)


def _path(name: str) -> Path:
    return _DIR / f"{name.lower()}_memories.json"


# ------------------------------------------------------------------
# save / load
def save_memories(agent: "Agent") -> None:          # quotes avoid runtime eval
    with _path(agent.name).open("w", encoding="utf-8") as f:
        json.dump([m.__dict__ for m in agent.memory], f, ensure_ascii=False, indent=2)


def load_memories(name: str) -> List["Memory"]:
    """
    Lazy-import Memory *inside* the function to avoid circular imports.
    Called only after core.agent has finished initialising.
    """
    from .agent import Memory   # deferred import – safe now
    p = _path(name)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [Memory(**d) for d in data]


# ------------------------------------------------------------------
# summarisation helpers
def llm_summarise_block(
    block: str,
    *,
    agent_name: str,
    model: str = "gpt-4o-mini",
) -> str:
    prompt = (
        f"You are helping {agent_name} condense memories.\n"
        "Rewrite the following block into 3–4 concise sentences:\n\n"
        f"{block}\n\nSummary:"
    )
    return llm_utils.chat([{"role": "system", "content": prompt}], model=model)


def summarize_recent(agent: "Agent", window: int = 20) -> str:
    if not agent.memory:
        return "No memories yet."
    return llm_summarise_block(
        "\n".join(m.text for m in agent.memory[-window:]), agent_name=agent.name
    )
