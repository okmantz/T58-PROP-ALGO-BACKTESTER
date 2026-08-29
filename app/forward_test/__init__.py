"""
Forward Test (MT5 Demo) -- deploy a Strategy Library strategy to a live
MetaTrader 5 demo account and watch it trade forward, bar by bar, with real
spread/slippage/broker fills instead of a CSV's historical costs.

Why MT5, specifically: it's the one path to real automated execution that
doesn't require a paid subscription anywhere in the chain. MetaTrader5 (the
official Python package) talks directly to a running MT5 terminal for free,
and virtually every prop firm offers MT5-based demo/eval/funded accounts.
TradingView's webhook alerts -- the other common automation path -- require
a paid plan, so it's deliberately not used here.

Design rule: this module NEVER re-implements strategy logic. Every session
calls the exact same `Strategy.generate(df)` used by the backtester
(app.strategy.base), and sizes positions with the exact same
`RiskConfig.position_size(...)` used by app.backtest.execution. Backtest,
forward test, and (eventually) live execution are three callers of one
signal engine -- not three separate implementations that can quietly drift
apart from each other.

Submodules:
    mt5_settings   -- persisted MT5 demo login/password/server (keyring-backed,
                       mirrors app.ai.ollama_settings's pattern)
    mt5_connector  -- thin wrapper around the `MetaTrader5` package (Windows +
                       a running MT5 terminal only; guarded import everywhere
                       else so the rest of the app is unaffected if it's absent)
    journal        -- local SQLite trade journal for forward-test fills
    engine         -- ForwardTestSession: the polling loop that turns bars
                       into signals into orders

This is demo-account forward testing only. There is no live-account order
path anywhere in this module -- see the Forward Test tab's own docstring
for why that's a deliberate, separate, later decision.
"""
