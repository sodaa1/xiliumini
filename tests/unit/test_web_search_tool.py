import importlib
import json

import pytest


def api():
    return importlib.import_module("xiliumini.tools.web_search_tool")


def test_missing_key_does_not_contact_tavily(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    module = api()
    monkeypatch.setattr(module, "TavilyClient", lambda **kw: pytest.fail("unexpected network"))
    result = json.loads(module.WebSearchTool().invoke({"query": "question"}))
    assert result == {"ok": False, "query": "question", "error": "missing TAVILY_API_KEY"}


@pytest.mark.parametrize(
    "results, expected",
    [
        (None, []),
        ("bad", []),
        ([None, {}], []),
        (
            [{"url": "https://example.test", "score": 0.9}],
            [{"title": "", "url": "https://example.test", "content": "", "score": 0.9}],
        ),
    ],
)
def test_search_normalizes_provider_results(monkeypatch, results, expected):
    module = api()
    monkeypatch.setenv("TAVILY_API_KEY", "fake-secret")

    class Client:
        def __init__(self, api_key):
            assert api_key == "fake-secret"

        def search(self, **kwargs):
            assert kwargs["query"] == "question"
            assert kwargs["include_answer"] is True
            return {"answer": "answer", "results": results}

    monkeypatch.setattr(module, "TavilyClient", Client)
    result = json.loads(module.WebSearchTool().invoke({"query": "question"}))
    assert result == {"ok": True, "query": "question", "answer": "answer", "results": expected}


def test_search_redacts_provider_failure(monkeypatch):
    module = api()
    monkeypatch.setenv("TAVILY_API_KEY", "secret")

    def fail(**kwargs):
        raise RuntimeError("secret")

    monkeypatch.setattr(module, "TavilyClient", fail)
    result = json.loads(module.WebSearchTool().invoke({"query": "question"}))
    assert result["ok"] is False
    assert "secret" not in json.dumps(result)
