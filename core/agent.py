"""Digital-Twin agent: episodic memory, retrieval, LLM chat, optional TTS."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

try:
    import requests
except ModuleNotFoundError:  # optional for offline tests
    requests = None
try:
    from sentence_transformers import SentenceTransformer, util
except ModuleNotFoundError:  # graceful fallback if dependency missing
    import types

    class SentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def encode(self, texts):
            if isinstance(texts, list):
                return [[0.0, 0.0, 0.0] for _ in texts]
            return [0.0, 0.0, 0.0]

    util = types.SimpleNamespace(cos_sim=lambda a, b: 0.0)

from . import memory_utils as mu
from . import utterance_utils

# Mem0 key for remote memory operations
try:
    from settings import MEM0_API_KEY as _MEM0_KEY
except (ModuleNotFoundError, ImportError):
    _MEM0_KEY = os.getenv("MEM0_API_KEY", "")

if _MEM0_KEY and requests is None:
    print("[Mem0 disabled] install 'requests' to enable remote features")


# ElevenLabs key (settings.py → env fallback)
try:
    from settings import ELEVEN_API_KEY as _ELEVEN_KEY
except (ModuleNotFoundError, ImportError):
    try:
        from settings import ELEVENLABS_API_KEY as _ELEVEN_KEY
    except (ModuleNotFoundError, ImportError):
        _ELEVEN_KEY = os.getenv("ELEVEN_API_KEY", "") or os.getenv("ELEVENLABS_API_KEY", "")

# Shared embedder
_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class Memory:
    text: str
    timestamp: float
    embedding: List[float] = field(default_factory=list)
    is_summary: bool = False


@dataclass
class Agent:
    name: str
    personality: str
    tts_voice_id: str
    memory: List[Memory] = field(default_factory=list)
    graph: Dict[str, Set[str]] = field(default_factory=dict)
    _sync_every: int = 5
    _unsynced_count: int = 0

    def sync_memories(self) -> None:
        """Persist current memories to disk or Mem0 and reset counter."""
        mu.save_memories(self)
        self._unsynced_count = 0

    def _maybe_sync(self) -> None:
        self._unsynced_count += 1
        if self._unsynced_count >= self._sync_every:
            self.sync_memories()

    def trim_memory(self, limit: int) -> None:
        """Keep only the most recent *limit* memories and sync."""
        if len(self.memory) > limit:
            self.memory = self.memory[-limit:]
            self.sync_memories()

    # ── Embedding helpers
    def _ensure_embeddings(self, mems: Sequence[Memory]) -> None:
        missing = [m for m in mems if not m.embedding]
        if missing:
            vecs = _EMBEDDER.encode([m.text for m in missing])
            for m, v in zip(missing, vecs):
                if hasattr(v, "tolist"):
                    m.embedding = v.tolist()
                else:
                    m.embedding = list(v)

    # ── Graph helpers
    def _update_graph(self, text: str) -> None:
        """Parse simple 'A -> B' or 'A is B' patterns into graph edges."""
        if "->" in text:
            a, b = [p.strip().lower() for p in text.split("->", 1)]
            if a and b:
                self.graph.setdefault(a, set()).add(b)
        elif " is " in text:
            a, b = [p.strip().lower() for p in text.split(" is ", 1)]
            if a and b:
                self.graph.setdefault(a, set()).add(b)

    def rebuild_graph(self) -> None:
        """Recreate graph edges from stored memories."""
        self.graph.clear()
        for m in self.memory:
            self._update_graph(m.text)

    def graph_context(self, query: str, depth: int = 1) -> List[str]:
        """Return nodes related to tokens in *query* within *depth* hops."""
        found: Set[str] = set()
        tokens = {t.lower() for t in query.split()}
        seeds = [t for t in tokens if t in self.graph]
        for seed in seeds:
            to_visit = {seed}
            for _ in range(depth):
                next_visit: Set[str] = set()
                for node in to_visit:
                    for nb in self.graph.get(node, set()):
                        if nb not in found:
                            found.add(nb)
                            next_visit.add(nb)
                to_visit = next_visit
        return list(found)

    # Memory CRUD & roll-up
    _MAX_RAW = 200
    _CHUNK   = 50

    def add_memory(self, text: str, *, is_summary: bool = False) -> None:
        emb = _EMBEDDER.encode(text)
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        else:
            emb = list(emb)
        mem = Memory(
            text=text,
            timestamp=time.time(),
            embedding=emb,
            is_summary=is_summary,
        )
        self.memory.append(mem)
        self._update_graph(text)
        self._maybe_sync()
        # Disabled old Mem0 REST API - now using Mem0 Pro client in individual agents
        # if _MEM0_KEY and requests:
        #     headers = {"Authorization": f"Bearer {_MEM0_KEY}", "Content-Type": "application/json"}
        #     payload = mem.__dict__ | {"agent": self.name}
        #     try:
        #         requests.post(
        #             "https://api.mem0.ai/v1/memory",
        #             json=payload,
        #             headers=headers,
        #             timeout=30,
        #         )
        #     except Exception as e:
        #         print("[Mem0 save error]", e)
        self._auto_rollup()

    def _auto_rollup(self) -> None:
        raw = [m for m in self.memory if not m.is_summary]
        if len(raw) > self._MAX_RAW:
            oldest = raw[:self._CHUNK]
            summary = mu.llm_summarise_block(
                "\n".join(m.text for m in oldest), agent_name=self.name
            )
            self.memory = [m for m in self.memory if m not in oldest]
            self.add_memory(f"(summary) {summary}", is_summary=True)

    # Retrieval
    def retrieve_memories(self, query: str, top_k: int = 5) -> List[str]:
        results: List[str] = []
        # Disabled old Mem0 REST API - individual agents handle their own Mem0 integration
        # if _MEM0_KEY and requests:
        #     headers = {"Authorization": f"Bearer {_MEM0_KEY}", "Content-Type": "application/json"}
        #     payload = {"query": query, "k": top_k, "agent": self.name}
        #     try:
        #         r = requests.post(
        #             "https://api.mem0.ai/v1/search", json=payload, headers=headers, timeout=30
        #         )
        #         r.raise_for_status()
        #         results = r.json().get("results", [])
        #     except Exception as e:
        #         if getattr(e, "response", None) and getattr(e.response, "status_code", None) == 404:
        #             print("[Mem0 search error] remote search not found; using local memories")
        #         else:
        #             print("[Mem0 search error]", e)

        local_results: List[str] = []
        if self.memory:
            self._ensure_embeddings(self.memory)
            q_vec = _EMBEDDER.encode(query)
            scored = [(float(util.cos_sim(q_vec, m.embedding)), m) for m in self.memory]
            scored.sort(key=lambda t: t[0], reverse=True)
            local_results = [m.text for _, m in scored[:top_k]]

        # Merge remote and local results, prioritising remote
        merged: List[str] = []
        for txt in results + local_results:
            if txt not in merged:
                merged.append(txt)
            if len(merged) >= top_k:
                break
        return merged

    # LLM response
    def generate_response(self, user_msg: str, *, model: str = "gpt-4o-mini") -> str:
        relevant = "\n".join(self.retrieve_memories(user_msg))
        graph_info = ", ".join(self.graph_context(user_msg))
        response = utterance_utils.generate_utterance(
            agent_name=self.name,
            personality=self.personality,
            user_msg=user_msg,
            relevant=relevant,
            graph_info=graph_info,
            model=model,
            temperature=0.5,
        )
        self.add_memory(f"User: {user_msg}\n{self.name}: {response}")
        return response

       # Speech synthesis (delegates to tts_utils)
    def speak(self, text: str, playback_cmd: str = "afplay") -> None:
        """
        Convert *text* into audible speech via ElevenLabs.

        Implementation is delegated to `tts_utils.speak`, which already handles
        key lookup, error reporting, and playback.  If either the API key or
        this agent’s `tts_voice_id` is missing, the helper simply prints a
        notice and returns, so callers don’t need a try/except.
        """
        from . import tts_utils                       # local import avoids cycles
        tts_utils.speak(text, voice_id=self.tts_voice_id,
                        playback_cmd=playback_cmd)
