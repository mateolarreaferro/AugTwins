"""Disk persistence + LLM summarisation utilities.

This version avoids circular-import problems by:
  • NOT importing Agent/Memory at module load time
  • Importing Memory lazily inside load_memories()
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, TYPE_CHECKING

from . import mem0_backend
from . import llm_utils

# Optional forward references for static type checkers only
if TYPE_CHECKING:          # <- evaluated by tools like mypy, ignored at runtime
    from .agent import Agent, Memory

# Mem0 API key lookup
try:
    from settings import MEM0_API_KEY as _MEM0_KEY
except (ModuleNotFoundError, ImportError):
    _MEM0_KEY = os.getenv("MEM0_API_KEY", "")

# storage path
_DIR = Path("memories")
_DIR.mkdir(exist_ok=True)


def _use_remote() -> bool:
    return mem0_backend.init_client() is not None


def _path(name: str, mode: str = "combined") -> Path:
    return _DIR / f"{name.lower()}_{mode}_memories.json"


# save / load
def save_memories(agent: "Agent", mode: str = "combined") -> None:  # quotes avoid runtime eval
    data = [m.__dict__ for m in agent.memory]
    if _use_remote():
        for entry in data:
            mem0_backend.add_memory(agent.name, entry)
    else:
        with _path(agent.name, mode).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_memories(name: str, mode: str = "combined") -> List["Memory"]:
    """
    Lazy-import Memory *inside* the function to avoid circular imports.
    Called only after core.agent has finished initialising.
    """
    from .agent import Memory   # deferred import – safe now
    data = []
    if _use_remote():
        data = mem0_backend.get_all_memories(name)
    if not data:
        p = _path(name, mode)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
    return [Memory(**d) for d in data]


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