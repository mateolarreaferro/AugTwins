"""Disk persistence + LLM summarisation utilities.

This version avoids circular-import problems by:
  • NOT importing Agent/Memory at module load time
  • Importing Memory lazily inside load_memories()
"""
from __future__ import annotations
import json
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent import Agent

try:
    import requests
except ModuleNotFoundError:  # allow tests without requests
    requests = None

# Mem0 API key - load from centralized config
from config import MEM0_API_KEY

# Optional forward references for static type checkers only
if TYPE_CHECKING:          # <- evaluated by tools like mypy, ignored at runtime
    from .agent import Agent, Memory

if MEM0_API_KEY and requests is None:
    print("[Mem0 disabled] install 'requests' to enable remote features")

_BASE_URL = "https://api.mem0.ai/v1"

# storage path
_DIR = Path("memories")
_DIR.mkdir(exist_ok=True)


def _use_remote() -> bool:
    return bool(MEM0_API_KEY and requests)


def _path(name: str) -> Path:
    """Return the local JSON file path for *name*'s memories."""
    return _DIR / f"{name.lower()}_memories.json"


def _remote_url(name: str) -> str:
    """Return the Mem0 API URL for *name*'s memories."""
    return f"{_BASE_URL}/agents/{name.lower()}/memories"


# save / load
def save_memories(agent: "Agent") -> None:  # quotes avoid runtime eval
    data = [m.__dict__ for m in agent.memory]
    if _use_remote():
        headers = {"Authorization": f"Bearer {MEM0_API_KEY}", "Content-Type": "application/json"}
        try:
            r = requests.post(_remote_url(agent.name), json=data, headers=headers, timeout=30)
            r.raise_for_status()
        except Exception as e:
            if getattr(e, "response", None) and getattr(e.response, "status_code", None) == 404:
                print("[Mem0 save error] remote agent not found; using local file")
            else:
                print("[Mem0 save error]", e)
            with _path(agent.name).open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with _path(agent.name).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_memories(name: str) -> List["Memory"]:
    """
    Lazy-import Memory *inside* the function to avoid circular imports.
    Called only after core.agent has finished initialising.
    """
    from .agent import Memory   # deferred import – safe now
    data = []
    if _use_remote():
        headers = {"Authorization": f"Bearer {MEM0_API_KEY}"}
        try:
            r = requests.get(_remote_url(name), headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            if getattr(e, "response", None) and getattr(e.response, "status_code", None) == 404:
                print("[Mem0 load error] remote memories not found; falling back to local")
            else:
                print("[Mem0 load error]", e)
    if not data:
        p = _path(name)
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
        headers = {"Authorization": f"Bearer {MEM0_API_KEY}", "Content-Type": "application/json"}
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