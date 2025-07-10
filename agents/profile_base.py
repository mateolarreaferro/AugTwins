from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from generate_profile import (
    build_openai_client,
    extract_memories,
    DEFAULT_MODEL,
    DEFAULT_CHUNK_TOKENS,
)


class ProfileAgent(Agent):
    """Agent with convenience helpers for transcripts and persona."""

    transcript_path: Path
    persona: str

    def generate_memories(
        self,
        model: str = DEFAULT_MODEL,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    ) -> None:
        """Use OpenAI to extract memories from the transcript."""
        if not self.transcript_path.exists():
            raise FileNotFoundError(self.transcript_path)
        transcript = self.transcript_path.read_text(encoding="utf-8")
        client = build_openai_client()
        memories = extract_memories(client, model, transcript, chunk_tokens)
        for m in memories:
            text = m.get("memory") if isinstance(m, dict) else None
            if text:
                self.add_memory(text)
