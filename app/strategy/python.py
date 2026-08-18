"""
Python Strategy Adapter.

Loads a user-supplied .py file which must define a top-level function:

    def generate_signals(df: pd.DataFrame) -> pd.Series:
        ...  # returns a series of -1/0/1 the same length as df

Optionally the module may also define:
    STOP_LOSS_PIPS = <float>
    TAKE_PROFIT_PIPS = <float>
    STRATEGY_NAME = "<str>"

The module is imported in isolation via importlib so a bad/malicious upload
cannot silently corrupt the running app's own modules; execution errors are
caught and surfaced as a clear StrategyError per the spec's requirement that
unsupported/invalid strategy code must fail loudly rather than produce a
silently inaccurate backtest.
"""
from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import pandas as pd

from app.strategy.base import Strategy, StrategyError, StrategyResult


class PythonStrategy(Strategy):
    source_type = "python"

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise StrategyError(f"Python strategy file not found: {self.file_path}")
        if self.file_path.suffix != ".py":
            raise StrategyError("Python strategy must be a .py file.")

    def _load_module(self):
        module_name = f"user_strategy_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, self.file_path)
        if spec is None or spec.loader is None:
            raise StrategyError(f"Could not load Python module from {self.file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            raise StrategyError(f"Error executing uploaded Python strategy: {exc}") from exc
        finally:
            sys.modules.pop(module_name, None)
        return module

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        module = self._load_module()

        if not hasattr(module, "generate_signals"):
            raise StrategyError(
                "Uploaded Python strategy must define a top-level function "
                "`generate_signals(df) -> pd.Series`."
            )

        try:
            raw_signals = module.generate_signals(df.copy())
        except Exception as exc:  # noqa: BLE001
            raise StrategyError(f"Uploaded strategy raised an exception: {exc}") from exc

        if not isinstance(raw_signals, pd.Series):
            try:
                raw_signals = pd.Series(raw_signals, index=df.index)
            except Exception as exc:  # noqa: BLE001
                raise StrategyError(
                    f"generate_signals() must return a pandas Series (or array-like) of -1/0/1: {exc}"
                ) from exc

        signals = self._validate_signals(raw_signals, df)

        return StrategyResult(
            name=getattr(module, "STRATEGY_NAME", self.file_path.stem),
            source_type=self.source_type,
            signals=signals,
            stop_loss_pips=getattr(module, "STOP_LOSS_PIPS", None),
            take_profit_pips=getattr(module, "TAKE_PROFIT_PIPS", None),
        )
