import pytest
from verify import safe_eval


def test_basic_arithmetic():
    assert safe_eval("240 - (240 * 15 / 100) - 60") == 144
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(10 - 4) / 2") == 3
    assert safe_eval("2 ** 3") == 8
    assert safe_eval("-5 + 10") == 5


def test_formatting_tolerance():
    assert safe_eval("1,200 / 4") == 300
    assert safe_eval("$100 * 0.15") == 15
    assert safe_eval("6 × 7") == 42
    assert safe_eval("84 ÷ 2") == 42
    assert safe_eval("240 * 0.85 - 60.") == 144


def test_allowed_functions():
    assert safe_eval("round(10 / 3, 2)") == 3.33
    assert safe_eval("max(3, 7)") == 7
    assert safe_eval("abs(-4)") == 4


def test_rejects_dangerous():
    for expr in ("__import__('os').system('ls')", "open('/etc/passwd')",
                 "().__class__", "x + 1", "[1,2][0]", "'a' * 3",
                 "10 ** 100", "lambda: 1"):
        with pytest.raises(ValueError):
            safe_eval(expr)


def test_rejects_empty_and_huge():
    with pytest.raises(ValueError):
        safe_eval("")
    with pytest.raises(ValueError):
        safe_eval("1+" * 200 + "1")
