"""
Shared safe expression evaluation.

Boolean rule expressions (from the Manual Builder, or translated from
PineScript/MQL5 conditions) are evaluated with pandas.eval(engine="python")
against a restricted character whitelist -- no arbitrary Python execution,
no builtins, no attribute access beyond what pandas.eval itself permits.
"""
from __future__ import annotations

import pandas as pd

from app.strategy.base import StrategyError

ALLOWED_EXPR_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ .<>=!&|()+-*/'\""
)


def validate_expression(expr: str, field_name: str) -> None:
    if not expr or not isinstance(expr, str):
        raise StrategyError(f"'{field_name}' must be a non-empty expression string.")
    bad_chars = set(expr) - ALLOWED_EXPR_CHARS
    if bad_chars:
        raise StrategyError(
            f"'{field_name}' contains disallowed characters {sorted(bad_chars)}. "
            "Only column/series names, comparisons (< > <= >= == !=), and/or/not are permitted."
        )


def safe_eval_bool(frame: pd.DataFrame, expr: str, field_name: str) -> pd.Series:
    """Evaluate a whitelisted boolean expression against `frame`'s columns."""
    if not expr:
        return pd.Series(False, index=frame.index)
    validate_expression(expr, field_name)
    try:
        result = frame.eval(expr, engine="python")
    except Exception as exc:  # noqa: BLE001
        raise StrategyError(f"Failed to evaluate expression '{expr}' ({field_name}): {exc}") from exc
    return result.fillna(False).astype(bool)
