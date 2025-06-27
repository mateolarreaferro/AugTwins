import importlib
import sys
import types


def test_remote_save_and_load(tmp_path, monkeypatch):
    """memory_utils should use the Mem0 client when the key is present."""
    calls = {"add": [], "list": []}

    class DummyClient:
        def add_memory(self, agent: str, **data):
            calls["add"].append((agent, data))

        def get_all_memories(self, agent: str):
            calls["list"].append(agent)
            return [{"text": "hello", "timestamp": 0.0, "embedding": [0, 0, 0], "is_summary": False}]

    from core import mem0_backend as mb
    monkeypatch.setattr(mb, "_CLIENT", DummyClient(), raising=False)
    monkeypatch.setattr(mb, "_API_KEY", "dummy", raising=False)
    monkeypatch.setattr(mb, "init_client", lambda: mb._CLIENT)

    from core import memory_utils as mu
    monkeypatch.setattr(mu, "_MEM0_KEY", "dummy", raising=False)

    dummy_st = types.ModuleType("sentence_transformers")

    class DummyST:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            if isinstance(texts, list):
                return [[0.0, 0.0, 0.0] for _ in texts]
            return [0.0, 0.0, 0.0]

    dummy_st.SentenceTransformer = DummyST
    dummy_st.util = types.SimpleNamespace(cos_sim=lambda a, b: 0.0)
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy_st)

    if "core.agent" in sys.modules:
        importlib.reload(sys.modules["core.agent"])
    from core import agent as agent_module

    mem_dir = tmp_path / "memories"
    mem_dir.mkdir()
    monkeypatch.setattr(mu, "_DIR", mem_dir)

    agent = agent_module.Agent(name="Dummy", personality="none", tts_voice_id="0")
    mem = agent_module.Memory(text="hello", timestamp=0.0, embedding=[0, 0, 0])
    agent.memory.append(mem)

    mu.save_memories(agent)
    assert calls["add"]

    loaded = mu.load_memories(agent.name)
    assert calls["list"]
    assert loaded and loaded[0].text == "hello"
