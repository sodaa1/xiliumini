from __future__ import annotations

import os

from rich.text import Text

LOGO_FRAME_COUNT = 5
_TAGLINE = "你的专属智能 Agent"
_COMPACT_WORDMARK = "x i l i u m i n i"
_LARGE_MIN_WIDTH = 48
_MAX_SCAN_WIDTH = 12
_PAW_STYLES = (
    "dim #8f461f",
    "#b95722",
    "#db6926",
    "bold #ed7529",
    "bold #ff812b",
)
_GLYPHS = {
    "x": ("█   █", " █ █ ", "  █  ", " █ █ ", "█   █"),
    "i": ("█████", "  █  ", "  █  ", "  █  ", "█████"),
    "l": ("█    ", "█    ", "█    ", "█    ", "█████"),
    "u": ("█   █", "█   █", "█   █", "█   █", "█████"),
    "m": ("█   █", "██ ██", "█ █ █", "█   █", "█   █"),
    "n": ("█   █", "██  █", "█ █ █", "█  ██", "█   █"),
}
_LEFT_WORD = "xiliu"
_RIGHT_WORD = "mini"


def _frame_index(frame: int | None) -> int:
    if frame is None:
        return LOGO_FRAME_COUNT - 1
    return max(0, min(LOGO_FRAME_COUNT - 1, frame))


def _scan_width(frame: int | None) -> int:
    if frame is None:
        return _MAX_SCAN_WIDTH
    return _MAX_SCAN_WIDTH * (_frame_index(frame) + 1) // LOGO_FRAME_COUNT


def _word_row(word: str, row: int) -> str:
    return " ".join(_GLYPHS[letter][row] for letter in word)


def _compact_logo(*, use_color: bool, frame: int | None) -> Text:
    if not use_color:
        return Text(f"🐾  {_COMPACT_WORDMARK}\n{_TAGLINE}")
    logo = Text()
    logo.append("🐾", style=_PAW_STYLES[_frame_index(frame)])
    logo.append("  ")
    logo.append("x i l i u", style="bold #20d6d0")
    logo.append(" m i n i", style="bold #2796ff")
    logo.append("\n")
    logo.append(_TAGLINE, style="#7890a8")
    return logo


def render_logo(*, width: int = 80, frame: int | None = None, color: bool = True) -> Text:
    """Build one deterministic Rich frame of the centered xiliumini welcome logo."""

    use_color = color and os.getenv("NO_COLOR") is None
    visible_frame = frame if use_color else None
    if width < _LARGE_MIN_WIDTH:
        return _compact_logo(use_color=use_color, frame=visible_frame)

    scan = "━" * _scan_width(visible_frame)
    if not use_color:
        rows = []
        for row in range(5):
            prefix = "🐾  " if row == 2 else "    "
            rows.append(f"{prefix}{_word_row(_LEFT_WORD, row)} {_word_row(_RIGHT_WORD, row)}")
        rows.append(f"{scan}  {_TAGLINE}  {scan}")
        return Text("\n".join(rows))

    logo = Text()
    for row in range(5):
        if row == 2:
            logo.append("🐾", style=_PAW_STYLES[_frame_index(frame)])
            logo.append("  ")
        else:
            logo.append("    ")
        logo.append(_word_row(_LEFT_WORD, row), style="bold #20d6d0")
        logo.append(" ")
        logo.append(_word_row(_RIGHT_WORD, row), style="bold #2796ff")
        logo.append("\n")
    logo.append(scan, style="#20d6d0")
    logo.append(f"  {_TAGLINE}  ", style="#a9bed0")
    logo.append(scan, style="#2796ff")
    return logo


__all__ = ["LOGO_FRAME_COUNT", "render_logo"]
