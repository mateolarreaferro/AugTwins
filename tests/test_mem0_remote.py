import importlib
import sys
import types


def test_remote_save_and_load(tmp_path, monkeypatch):
    """memory_utils should use the Mem0 API when the key and requests are present."""
    calls = {"post": [], "get": []}

    class DummyResponse:
        def __init__(self, data=None):
            self.status_code = 200
            self._data = data or []

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def post(url, json=None, headers=None, timeout=30):
        calls["post"].append({"url": url, "json": json, "headers": headers})
        return DummyResponse()

    def get(url, headers=None, timeout=30):
        calls["get"].append({"url": url, "headers": headers})
        data = [{"text": "hello", "timestamp": 0.0, "embedding": [0, 0, 0], "is_summary": False}]
        return DummyResponse(data)

    from core import memory_utils as mu
    dummy_requests = types.SimpleNamespace(post=post, get=get)
    monkeypatch.setattr(mu, "requests", dummy_requests)
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
    assert calls["post"]
    assert calls["post"][0]["url"] == mu._remote_url(agent.name, "combined")

    loaded = mu.load_memories(agent.name)
    assert calls["get"]
    assert calls["get"][0]["url"] == mu._remote_url(agent.name, "combined")
    assert loaded and loaded[0].text == "hello"
