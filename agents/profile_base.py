from __future__ import annotations

from pathlib import Path

from core.agent import Agent


class ProfileAgent(Agent):
    """Agent with convenience helpers for transcripts and persona."""

    transcript_path: Path
    persona: str

    # Note: Memory generation is now handled directly by Mem0
    # The generate_memories method has been removed as it's no longer needed
