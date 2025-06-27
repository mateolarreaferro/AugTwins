from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Attempt to import the official SDK
try:
    from mem0ai import Mem0
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Mem0 = None  # type: ignore

_CLIENT: Optional[Any] = None
_MEMORY_STORE: Dict[str, List[Dict[str, Any]]] = {}
_API_KEY = os.getenv("MEM0_API_KEY", "")


def init_client() -> Optional[Any]:
    """Initialise and return the Mem0 client, if possible."""
    global _CLIENT
    if _CLIENT is None and Mem0 and _API_KEY:
        _CLIENT = Mem0(api_key=_API_KEY)
    return _CLIENT


def add_memory(agent: str, memory: Dict[str, Any]) -> None:
    """Store *memory* for *agent* via Mem0 or the local fallback."""
    client = init_client()
    if client:
        try:  # pragma: no cover - network call
            client.add_memory(agent=agent, **memory)
        except Exception:
            pass
    _MEMORY_STORE.setdefault(agent.lower(), []).append(memory)


def search_memory(agent: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return up to *k* memories for *agent* matching *query*."""
    client = init_client()
    if client:
        try:  # pragma: no cover - network call
            return client.search_memory(agent=agent, query=query, k=k)
        except Exception:
            return []

    results: List[Dict[str, Any]] = []
    for mem in _MEMORY_STORE.get(agent.lower(), []):
        if query.lower() in mem.get("text", "").lower():
            results.append(mem)
            if len(results) >= k:
                break
    return results


def get_all_memories(agent: str) -> List[Dict[str, Any]]:
    """Fetch all stored memories for *agent*."""
    client = init_client()
    if client:
        try:  # pragma: no cover - network call
            return client.get_all_memories(agent=agent)
        except Exception:
            return []
    return list(_MEMORY_STORE.get(agent.lower(), []))
