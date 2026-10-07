from __future__ import annotations

from xiliumini.agents.search_agent import mcp_source_urls


def test_mcp_sources_only_come_from_successful_remote_result():
    result = mcp_source_urls({"ok": True, "text": "See https://aihot.news/story/123 for details."})
    assert result == [
        "https://aihot.news/story/123"
    ]
    assert mcp_source_urls({"ok": False, "text": "https://aihot.news/story/fake"}) == []
    assert mcp_source_urls({"ok": True, "text": "No links available"}) == []
