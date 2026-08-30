# Strategy File Format — what the backtester actually reads

This explains exactly what a `.py` / `.pine` / `.mq5` file needs to contain
for this app to load it, and how it knows when to enter a trade, exit,
place a stop-loss, or take profit. Each language has its own parser
(`app/strategy/python.py`, `app/strategy/pinescript.py`,
`app/strategy/mql5.py`) and each parser only understands a **subset** of
that language — this doc describes that subset directly, so you don't have
to read the parser source to write a working strategy by hand (the
Generate Strategies (AI) tab also builds against these exact same rules).

If you use the built-in **Manual Strategy Builder** (Step 02, "Manual"
mode) instead of uploading a file, none of this applies — the visual
builder always produces something the engine can read.

---

## Python (`.py`)

**Required:** a single top-level function —

```python
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    ...
    return signals  # -1 (short), 0 (flat), or 1 (long), one value per row of df
```

- `df` has columns `timestamp, open, high, low, close, volume` (lowercase)
  and nothing else — that's the entire market data your strategy sees.
- The returned Series must be the same length as `df`, containing only
  `-1`, `0`, or `1`. This is how the engine knows when to enter/exit: a
  change from `0`→`1` opens a long, `0`→`-1` opens a short, and a change
  back to `0` (or a flip to the opposite side) closes the open position.
- `generate_signals(df)` is called **once, statelessly**, over the whole
  dataset before any trade has opened or closed. It cannot see its own
  past trade outcomes — a "stop trading after N losses today" counter
  inside this function is silently a no-op. Use the engine's own
  `RiskConfig.daily_loss_limit_pct` (Step 04, Risk) for that instead.
- Only `pandas` and `numpy` may be imported. No file I/O, no network
  calls, no other third-party packages.

### Stop-loss / take-profit — two ways

**Fixed, whole-backtest (simplest):** define module-level constants —

```python
STOP_LOSS_PIPS = 20
TAKE_PROFIT_PIPS = 40
STRATEGY_NAME = "My Strategy"   # optional, shown in reports
```

**Per-trade / dynamic (an ATR multiple, a swing level, etc.):** attach
arrays to the returned Series' `.attrs`, one raw-price value per bar (only
the value on the entry bar itself is read):

```python
signals.attrs["stop_loss_distance"]     # |entry - stop|, e.g. 1.5 * atr
signals.attrs["take_profit_distance"]   # |entry - target|
signals.attrs["trailing_stop_distance"] # raw-price trailing distance
signals.attrs["breakeven_trigger_r"]    # scalar float, e.g. 1.0 == "+1R"
```

If you compute a stop/target inside your function and **never** attach it
to `.attrs`, the engine has no way to know about it — it will fall back to
its own generic protective stop and your intended risk management is
silently discarded. `.attrs` is the *only* path a computed stop/target
reaches execution.

### The #1 bug: lookahead in multi-timeframe filters

If your strategy resamples to a higher timeframe (e.g. a 1H bias filter
for a 15m entry) and filters it with something like
`htf[htf.index < timestamp]`, **this leaks the still-forming current HTF
bar** — a resampled bar is labeled by its start time, so that filter
includes a bar built from data later than `timestamp` that hasn't
happened yet. This exact bug has been found (and quietly manufactured the
entire apparent "edge") in real uploaded strategies more than once. Use
`app.strategy.mtf.completed_bars()` / `last_completed_bar()` instead,
which correctly require a bar to have fully closed before using it.

---

## PineScript (`.pine`, PineScript v5)

The parser understands a **restricted subset** — anything outside it fails
to load with a clear error naming the unsupported construct. You may
ONLY use:

- Price references: `open, high, low, close, hl2, hlc3, ohlc4`
- `x = input.int(20, ...)` / `input.float(1.5, ...)` — becomes a constant
  using the given default value; no other `input.*` types
- `x = ta.sma(src, len)`, `ta.ema(src, len)`, `ta.wma(src, len)`,
  `ta.rsi(src, len)` — no other `ta.*` functions
- `x = ta.crossover(a, b)`, `ta.crossunder(a, b)`
- Boolean rule variables built from comparisons/`and`/`or`/`not` over the
  above, e.g. `longCondition = ta.crossover(fast, slow) and rsiVal < 70`
- **Entries** (inline or inside an `if` block) — this is how the engine
  knows when to place a trade:
  ```
  strategy.entry("Long", strategy.long, when=longCondition)
  if longCondition
      strategy.entry("Long", strategy.long)
  ```
- **Exits**: `strategy.close("Long", when=exitLongCondition)`
- **Stop-loss / take-profit** as special directive comments (not
  `strategy.exit()` price offsets, which aren't portable across
  instruments):
  ```
  // T58_SL_PIPS=20
  // T58_TP_PIPS=40
  ```

**Not supported** (raises an error): custom functions, arrays/matrices,
`security()` / multi-timeframe requests, repainting constructs, plotting,
alerts, or any `ta.*` function not listed above.

---

## MQL5 (`.mq5`, Expert Advisor source)

Also a restricted subset. You may ONLY use:

- Direct-value indicator calls (the simplified/legacy calling style):
  ```
  double fastMA = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_SMA, PRICE_CLOSE);
  double slowMA = iMA(_Symbol, PERIOD_CURRENT, 30, 0, MODE_EMA, PRICE_CLOSE);
  double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
  ```
  (only `MODE_SMA`/`MODE_EMA`/`MODE_LWMA`; only `iMA` and `iRSI` as
  indicators — the symbol/timeframe/shift/applied-price arguments are
  accepted but not otherwise used, since the engine always runs on the
  single imported dataset bar-by-bar)
- Boolean conditions with C-style operators: `> < >= <= == != && || !`
- `if (condition) { ... }` or a single-statement
  `if (condition) statement;`
- **Entries** inside a condition's guard — how the engine knows when to
  place a trade:
  ```
  trade.Buy(...)  /  trade.Sell(...)
  OrderSend(..., ORDER_TYPE_BUY, ...)  /  OrderSend(..., ORDER_TYPE_SELL, ...)
  ```
  (legacy MQL4-style `OP_BUY` / `OP_SELL` constants are also accepted)
- **Exits** inside a condition's guard: `trade.PositionClose(...)` /
  `OrderClose(...)`
- **Stop-loss / take-profit** as special directive comments (point-based
  SL/TP in MQL5 aren't portable pip distances across instruments):
  ```
  // T58_SL_PIPS=20
  // T58_TP_PIPS=40
  ```

**Not supported** (raises an error): `CopyBuffer()`-based indicator
handles, custom indicators, arrays/structs, multi-symbol/multi-timeframe
logic, trailing stops, or any indicator beyond `iMA`/`iRSI`.

---

## Quick checklist before uploading a strategy

1. Does it define the one required entry point for its language
   (`generate_signals(df)` for Python; `strategy.entry(...)` calls for
   Pine; `trade.Buy/Sell(...)` or `OrderSend(...)` for MQL5)?
2. Does it use ONLY the indicators/functions listed above for that
   language? Anything else fails to load, on purpose, rather than
   silently mis-backtesting.
3. Is a stop-loss defined one of the supported ways (fixed
   `STOP_LOSS_PIPS` / `// T58_SL_PIPS=`, or Python's dynamic
   `.attrs["stop_loss_distance"]`)? A strategy with no stop at all still
   runs (the engine applies a 1%-of-price protective stop as a fallback,
   with a warning), but an explicit stop is almost always what you want.
4. For Python only: any higher-timeframe filter uses
   `app.strategy.mtf.completed_bars()`/`last_completed_bar()`, not a raw
   `htf.index < timestamp` comparison (see "The #1 bug" above).
5. Once it loads, run it through **05 Run & Report** first, then **15
   Full Pipeline** before trusting any of its numbers — this is true
   whether you wrote it, downloaded it, or generated it with the AI
   Assist tab.
