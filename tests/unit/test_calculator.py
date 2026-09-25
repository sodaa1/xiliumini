from __future__ import annotations

import pytest

from xiliumini.tools.calculator import calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", "7"),
        ("2 ** 8", "256"),
        ("7 / 2", "3.5"),
        ("-(4 - 1)", "-3"),
    ],
)
def test_calculator_evaluates_whitelisted_arithmetic(expression: str, expected: str) -> None:
    assert calculator.invoke({"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "open('secret')",
        "value.real",
        "unknown + 1",
        "True + 1",
        "2 ** 101",
        "1e100 * 10",
        "1 / 0",
        "1 + " + "1" * 257,
    ],
)
def test_calculator_returns_readable_errors_for_unsafe_or_unbounded_input(
    expression: str,
) -> None:
    result = calculator.invoke({"expression": expression})

    assert result.startswith("Error:")


def test_calculator_never_executes_function_calls(tmp_path) -> None:
    marker = tmp_path / "should-not-exist"
    expression = f"__import__('pathlib').Path({str(marker)!r}).touch()"

    result = calculator.invoke({"expression": expression})

    assert result.startswith("Error:")
    assert not marker.exists()
