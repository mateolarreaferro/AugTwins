"""Digital-Twin agent: episodic memory, retrieval, LLM chat, optional TTS."""
from __future__ import annotations

import os
import time
from uuid import uuid4
from dataclasses import dataclass, field
from typing import List, Sequence

try:
    import requests
except ModuleNotFoundError:  # optional for offline tests
    requests = None
from sentence_transformers import SentenceTransformer, util

from . import llm_utils  
from . import tts_utils
from . import memory_utils as mu

# Mem0 key for remote memory operations
try:
    from settings import MEM0_API_KEY as _MEM0_KEY
except (ModuleNotFoundError, ImportError):
    _MEM0_KEY = os.getenv("MEM0_API_KEY", "")


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
        if _MEM0_KEY and requests:
            headers = {"Authorization": f"Bearer {_MEM0_KEY}", "Content-Type": "application/json"}
            payload = mem.__dict__ | {"agent": self.name}
            try:
                requests.post(
                    "https://api.mem0.ai/v1/memory",
                    json=payload,
                    headers=headers,
                    timeout=30,
                )
            except Exception as e:
                print("[Mem0 add_memory error]", e)
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
        if _MEM0_KEY and requests:
            headers = {"Authorization": f"Bearer {_MEM0_KEY}", "Content-Type": "application/json"}
            payload = {"query": query, "k": top_k, "agent": self.name}
            try:
                r = requests.post(
                    "https://api.mem0.ai/v1/search", json=payload, headers=headers, timeout=30
                )
                r.raise_for_status()
                return r.json().get("results", [])
            except Exception as e:
                print("[Mem0 search error]", e)

        if not self.memory:
            return []
        self._ensure_embeddings(self.memory)
        q_vec = _EMBEDDER.encode(query)
        scored = [(float(util.cos_sim(q_vec, m.embedding)), m) for m in self.memory]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m.text for _, m in scored[:top_k]]

    # LLM response
    def generate_response(self, user_msg: str, *, model: str = "gpt-4o-mini") -> str:
        relevant = "\n".join(self.retrieve_memories(user_msg))
        prompt = (
            f"You are {self.name}. Personality: {self.personality}\n"
            "Respond naturally and weave in any relevant memories.\n\n"
            f"Relevant memories:\n{relevant}\n\n"
            f"User: {user_msg}\n{self.name}:"
        )
        answer = llm_utils.chat([{"role": "system", "content": prompt}], model=model)
        self.add_memory(f"User: {user_msg}\n{self.name}: {answer}")
        return answer

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