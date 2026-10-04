import importlib
import json

import pytest
from tavily.errors import InvalidAPIKeyError


def api():
    return importlib.import_module("xiliumini.tools.web_search_tool")


def test_missing_key_uses_tavily_keyless_search(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    module = api()

    class Client:
        def __init__(self, api_key):
            assert api_key == ""

        def search(self, **kwargs):
            assert kwargs["query"] == "question"
            return {
                "answer": "keyless answer",
                "results": [{"url": "https://example.test/keyless"}],
            }

    monkeypatch.setattr(module, "TavilyClient", Client)
    result = json.loads(module.WebSearchTool().invoke({"query": "question"}))
    assert result == {
        "ok": True,
        "query": "question",
        "answer": "keyless answer",
        "results": [
            {
                "title": "",
                "url": "https://example.test/keyless",
                "content": "",
                "score": 0.0,
            }
        ],
    }


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


def test_invalid_api_key_falls_back_to_keyless_search(monkeypatch):
    module = api()
    monkeypatch.setenv("TAVILY_API_KEY", "invalid-secret")
    api_keys = []

    class Client:
        def __init__(self, api_key):
            api_keys.append(api_key)
            self.api_key = api_key

        def search(self, **kwargs):
            del kwargs
            if self.api_key:
                raise InvalidAPIKeyError("private provider response")
            return {
                "answer": "fallback answer",
                "results": [{"url": "https://example.test/fallback"}],
            }

    monkeypatch.setattr(module, "TavilyClient", Client)

    result = json.loads(module.WebSearchTool().invoke({"query": "question"}))

    assert result == {
        "ok": True,
        "query": "question",
        "answer": "fallback answer",
        "results": [
            {
                "title": "",
                "url": "https://example.test/fallback",
                "content": "",
                "score": 0.0,
            }
        ],
    }
    assert api_keys == ["invalid-secret", ""]
