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

try:
    import requests
except ModuleNotFoundError:  # allow tests without requests installed
    requests = None

from . import llm_utils

# Optional forward references for static type checkers only
if TYPE_CHECKING:          # <- evaluated by tools like mypy, ignored at runtime
    from .agent import Agent, Memory

# Mem0 API key lookup
try:
    from settings import MEM0_API_KEY as _MEM0_KEY
except (ModuleNotFoundError, ImportError):
    _MEM0_KEY = os.getenv("MEM0_API_KEY", "")

_BASE_URL = "https://api.mem0.ai/v1"

# storage path
_DIR = Path("memories")
_DIR.mkdir(exist_ok=True)


def _use_remote() -> bool:
    return bool(_MEM0_KEY and requests)


def _path(name: str, mode: str = "combined") -> Path:
    return _DIR / f"{name.lower()}_{mode}_memories.json"


def _remote_url(name: str, mode: str) -> str:
    return f"{_BASE_URL}/agents/{name.lower()}/memories/{mode}"


# save / load
def save_memories(agent: "Agent", mode: str = "combined") -> None:  # quotes avoid runtime eval
    data = [m.__dict__ for m in agent.memory]
    if _use_remote():
        headers = {"Authorization": f"Bearer {_MEM0_KEY}", "Content-Type": "application/json"}
        try:
            r = requests.post(_remote_url(agent.name, mode), json=data, headers=headers, timeout=30)
            r.raise_for_status()
        except Exception as e:
            if getattr(e, "response", None) and getattr(e.response, "status_code", None) == 404:
                print("[Mem0 save error] remote agent or mode not found; using local file")
            else:
                print("[Mem0 save error]", e)
            with _path(agent.name, mode).open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
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
        headers = {"Authorization": f"Bearer {_MEM0_KEY}"}
        try:
            r = requests.get(_remote_url(name, mode), headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            if getattr(e, "response", None) and getattr(e.response, "status_code", None) == 404:
                print("[Mem0 load error] remote memories not found; falling back to local")
            else:
                print("[Mem0 load error]", e)
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
    if _use_remote():
        headers = {"Authorization": f"Bearer {_MEM0_KEY}", "Content-Type": "application/json"}
        payload = {"text": block, "agent": agent_name}
        try:
            r = requests.post(f"{_BASE_URL}/summarize", json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json().get("summary", "")
        except Exception as e:
            print("[Mem0 summarise error]", e)

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