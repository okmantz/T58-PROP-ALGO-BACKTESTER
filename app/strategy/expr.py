"""Restricted expression evaluator used by manual and source adapters.

The visual builder does not execute arbitrary Python; it evaluates structured
conditions directly. This module remains available for legacy/manual text
expressions and imported Pine/MQL conditions.
"""
from __future__ import annotations

import re

import pandas as pd

from app.strategy.base import StrategyError

# Keep expressions deliberately boring: column names, numbers, operators,
# parentheses, whitespace and boolean keywords. Function calls, attributes,
# indexing and statement separators are not permitted.
ALLOWED_EXPR_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ .<>=!&|()+-*/'\"\t\r\n"
)

_FORBIDDEN_WORDS = {
    "import", "exec", "eval", "open", "compile", "globals", "locals",
    "builtins", "lambda", "class", "def", "os", "sys", "subprocess",
}


def validate_expression(expr: str, field_name: str) -> None:
    if not expr or not isinstance(expr, str) or not expr.strip():
        raise StrategyError(f"'{field_name}' must be a non-empty expression string.")

    bad_chars = set(expr) - ALLOWED_EXPR_CHARS
    if bad_chars:
        raise StrategyError(
            f"'{field_name}' contains disallowed characters {sorted(bad_chars)}. "
            "Only series names, numeric constants, comparisons, boolean operators, and parentheses are allowed."
        )

    if "." in expr:
        # Decimal numbers are allowed; attribute access is not.
        if re.search(r"[A-Za-z_]\s*\.\s*[A-Za-z_]", expr):
            raise StrategyError(f"'{field_name}' contains attribute access, which is not allowed.")

    if "__" in expr or ";" in expr or "[" in expr or "]" in expr or "," in expr:
        raise StrategyError(f"'{field_name}' contains syntax that is not allowed.")

    lowered = expr.lower()
    for word in _FORBIDDEN_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            raise StrategyError(f"'{field_name}' contains forbidden token '{word}'.")

    # Reject obvious function-call syntax such as foo(...). Parentheses are
    # still allowed for grouping.
    if re.search(r"[A-Za-z_]\w*\s*\(", expr):
        raise StrategyError(f"'{field_name}' contains a function call; use the visual indicator builder instead.")


def safe_eval_bool(frame: pd.DataFrame, expr: str, field_name: str) -> pd.Series:
    """Evaluate a restricted boolean expression against DataFrame columns."""
    if not expr:
        return pd.Series(False, index=frame.index)

    validate_expression(expr, field_name)

    normalized = re.sub(r"\band\b", "&", expr, flags=re.IGNORECASE)
    normalized = re.sub(r"\bor\b", "|", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bnot\b", "~", normalized, flags=re.IGNORECASE)

    try:
        result = frame.eval(normalized, engine="python", parser="pandas")
    except Exception as exc:  # noqa: BLE001
        raise StrategyError(
            f"Failed to evaluate expression '{expr}' ({field_name}): {exc}"
        ) from exc

    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=frame.index)
    return result.fillna(False).astype(bool)
