import sys
import types
import importlib
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


def test_save_and_load_memories(tmp_path, monkeypatch):
    # Create dummy sentence_transformers module to avoid heavy dependencies
    dummy_mod = types.ModuleType("sentence_transformers")

    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            if isinstance(texts, list):
                return [[0.0] * 3 for _ in texts]
            return [0.0] * 3

    dummy_mod.SentenceTransformer = DummySentenceTransformer
    dummy_mod.util = types.SimpleNamespace(cos_sim=lambda a, b: 0.0)
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy_mod)

    from core import agent as agent_module
    importlib.reload(agent_module)
    from core import memory_utils as mu

    mem_dir = tmp_path / "memories"
    mem_dir.mkdir()
    monkeypatch.setattr(mu, "_DIR", mem_dir)

    agent = agent_module.Agent(name="Dummy", personality="none", tts_voice_id="0")
    agent.add_memory("hello world")

    mu.save_memories(agent)
    loaded = mu.load_memories(agent.name)

    assert loaded == agent.memory
    assert all(isinstance(m, agent_module.Memory) for m in loaded)
