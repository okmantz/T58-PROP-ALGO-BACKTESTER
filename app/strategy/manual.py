"""
Manual Strategy Builder.

Lets a user define entry/exit rules as simple boolean expressions evaluated
against the OHLCV data plus a small library of precomputed indicators
(SMA, EMA, RSI). Expressions are evaluated with pandas.eval in a restricted
local namespace -- no arbitrary Python execution.

Example config:
    {
        "name": "SMA Cross",
        "indicators": [
            {"type": "sma", "period": 20, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 50, "column": "close", "as": "sma_slow"}
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit":  "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40
    }
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.strategy.base import Strategy, StrategyError, StrategyResult, signals_from_conditions
from app.strategy.expr import safe_eval_bool
from app.strategy.indicators import INDICATOR_FUNCS


class ManualStrategy(Strategy):
    source_type = "manual"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def _build_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        for ind in self.config.get("indicators", []):
            itype = ind.get("type")
            if itype not in INDICATOR_FUNCS:
                raise StrategyError(f"Unsupported indicator type '{itype}'. Supported: {list(INDICATOR_FUNCS)}")
            col = ind.get("column", "close")
            if col not in work.columns:
                raise StrategyError(f"Indicator references unknown column '{col}'.")
            period = int(ind.get("period", 14))
            alias = ind.get("as") or f"{itype}_{period}_{col}"
            work[alias] = INDICATOR_FUNCS[itype](work[col], period)
        return work

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        cfg = self.config
        work = self._build_indicators(df)

        long_entry = cfg.get("long_entry")
        long_exit = cfg.get("long_exit")
        short_entry = cfg.get("short_entry")
        short_exit = cfg.get("short_exit")

        if not long_entry and not short_entry:
            raise StrategyError("At least one of 'long_entry' or 'short_entry' must be defined.")

        long_entry_sig = safe_eval_bool(work, long_entry, "long_entry") if long_entry else pd.Series(False, index=work.index)
        long_exit_sig = safe_eval_bool(work, long_exit, "long_exit") if long_exit else pd.Series(False, index=work.index)
        short_entry_sig = safe_eval_bool(work, short_entry, "short_entry") if short_entry else pd.Series(False, index=work.index)
        short_exit_sig = safe_eval_bool(work, short_exit, "short_exit") if short_exit else pd.Series(False, index=work.index)

        raw_signals = signals_from_conditions(work.index, long_entry_sig, long_exit_sig, short_entry_sig, short_exit_sig)
        signals = self._validate_signals(raw_signals, df)

        return StrategyResult(
            name=cfg.get("name", "Manual Strategy"),
            source_type=self.source_type,
            signals=signals,
            stop_loss_pips=cfg.get("stop_loss_pips"),
            take_profit_pips=cfg.get("take_profit_pips"),
        )
