from __future__ import annotations

from rich.text import Text


def _lines(logo: Text) -> list[str]:
    return logo.plain.splitlines()


def test_render_logo_contains_large_wordmark_and_personal_agent_tagline() -> None:
    from xiliumini.cli.tui.logo import render_logo

    logo = render_logo(width=80, color=False)
    lines = _lines(logo)

    assert len(lines) == 6
    assert any("🐾" in line for line in lines[:5])
    assert max(len(line) for line in lines[:5]) >= 50
    assert "你的专属智能 Agent" in lines[-1]
    assert "Stage 6" not in logo.plain


def test_render_logo_uses_brand_styles_in_color_mode(monkeypatch) -> None:
    from xiliumini.cli.tui.logo import render_logo

    monkeypatch.delenv("NO_COLOR", raising=False)
    logo = render_logo(color=True)
    styles = {str(span.style) for span in logo.spans}

    assert any("ff812b" in style for style in styles)
    assert any("20d6d0" in style for style in styles)
    assert any("2796ff" in style for style in styles)


def test_render_logo_color_off_has_no_rich_spans() -> None:
    from xiliumini.cli.tui.logo import render_logo

    assert render_logo(color=False).spans == []


def test_render_logo_no_color_environment_forces_static_plain_text(monkeypatch) -> None:
    from xiliumini.cli.tui.logo import LOGO_FRAME_COUNT, render_logo

    monkeypatch.setenv("NO_COLOR", "1")

    first = render_logo(frame=0, color=True)
    last = render_logo(frame=LOGO_FRAME_COUNT - 1, color=True)

    assert first == last
    assert first.spans == []


def test_render_logo_clamps_animation_frames_deterministically(monkeypatch) -> None:
    from xiliumini.cli.tui.logo import LOGO_FRAME_COUNT, render_logo

    monkeypatch.delenv("NO_COLOR", raising=False)
    assert render_logo(frame=-100) == render_logo(frame=0)
    assert render_logo(frame=10_000) == render_logo(frame=LOGO_FRAME_COUNT - 1)
    assert _lines(render_logo(frame=0))[-1].count("━") < _lines(render_logo(frame=None))[-1].count(
        "━"
    )


def test_render_logo_handles_width_smaller_than_subtitle() -> None:
    from xiliumini.cli.tui.logo import render_logo

    logo = render_logo(width=3, color=False)

    assert "x i l i u m i n i" in logo.plain
    assert "你的专属智能 Agent" in logo.plain
    assert len(_lines(logo)) == 2
