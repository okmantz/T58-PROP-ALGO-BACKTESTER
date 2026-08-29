# Ported Strategies — Ready to Test in T58

Seven Python strategy files, each following T58's `generate_signals(df) ->
pd.Series` interface (see `app/strategy/python.py`'s docstring). Drop any of
these into the Strategy tab's Python file picker and run them like any
other strategy.

## What's here

| File | Ported from | Core idea |
|---|---|---|
| `hedge_fund_smc_ea.py` | hedge-fund-smc-ea.mq5 | EMA50/200 trend + prior-bar breakout, ATR stop/target/trailing |
| `propfirm_reactive_bos.py` | propfirm-reactive-ea.mq5 | EMA200 trend + N-bar structure breakout, ATR stop/target |
| `propfirm_elite_ea.py` | propfirm-elite-ea.mq5 | Triple-EMA + BOS + FVG + RSI + volume scoring, breakeven + trailing |
| `judas_sweep_killzone.py` | dakar-sniper-v2.mq5 **and** prop-sniper-v1.mq5 | Asian range sweep + rejection, London killzone (these two originals were the same strategy — only one port) |
| `ict_fvg_liquidity_sweep.py` | ict-sniper.pine | Confirmed swing-pivot liquidity sweep + FVG zone re-entry, dynamic per-trade stop |
| `smc_quant_engine.py` | smc-quant-engine.pine | 4H HTF EMA bias + pivot BOS/sweep + ADX + volume scoring |
| `propfirm_score_engine.py` | ultimate-propfirm-strategy.pine (representing the elite/ultra/ultimate family) | EMA-stack + BOS + FVG + OB + volume scoring |

All 7 were verified against T58's actual engine before delivery: each
passes the built-in lookahead check with `bug_detected: False`, and each
produces real trades with zero engine warnings against the bundled sample
CSV (`data/examples/EURUSD_5M_sample.csv`).

## Why 7 files, not 16

Three pairs/trios of the original files were the same underlying strategy:
- `dakar-sniper-v2.mq5` and `prop-sniper-v1.mq5` — identical Judas Sweep
  logic, same author, cosmetic differences only. One port.
- `propfirm-elite-strategy.pine`, `propfirm-ultra-strategy.pine`, and
  `ultimate-propfirm-strategy.pine` — the same EMA-stack/BOS/FVG/OB scoring
  engine at three iteration stages. The elite/ultra versions add an
  adaptive score threshold, market-regime detection, and — most
  importantly — risk scaling triggered by consecutive losses. That last
  part depends on the account's trade-by-trade history, which T58's
  `generate_signals(df)` cannot see (it's called once, statelessly, before
  any P&L exists — see the adapter's own docstring). Porting all three
  "faithfully" would have meant shipping the same entry logic three times
  while quietly dropping the one thing that actually differs between them.
  `propfirm_score_engine.py` is the one, fully-expressible version of that
  shared engine. Say the word if you want the adaptive-threshold/regime
  layer built out explicitly on top of it — just know the recovery-mode
  risk scaling itself isn't portable to this architecture regardless.

## What every file necessarily dropped, and why

Every original used at least one of these; none of them are expressible in
a stateless `generate_signals(df)` call, so none of them are silently
faked — they're just not in the file:

- **Daily loss limits, daily profit targets, max-trades-per-day, "phase"
  auto-switching risk.** These all depend on the account's *running P&L
  today*, which the strategy function can't see. Set these as **T58 engine
  settings instead** — `RiskConfig.daily_loss_limit_pct`,
  `RiskConfig.max_trades_per_day`, and the Prop Rules tab's own daily-loss
  and drawdown fields. That's the correct place for them, not inside the
  strategy: T58 enforces them centrally so every strategy gets the same
  real protection instead of each one re-implementing it (badly).
- **Consecutive-loss-triggered risk reduction** (propfirm-elite/ultra's
  "recovery mode"). Same reasoning — needs to know about prior trades'
  outcomes.
- **Partial take-profits** (scale out 50% at TP1, rest to TP2) —
  `smc_quant_engine.py`'s original did this; the port targets TP1 only,
  since T58's signal model places one stop and one target per trade.

## Things you should sanity-check before trusting the numbers

- **Session/hour filters are UTC-hour guesses.** Several originals gated
  entries to specific broker-server hours (London/NY killzones, Asian
  session). I set reasonable UTC defaults and called this out in each
  file's docstring, but if your CSV's timestamps are in a different
  timezone, the "London killzone" won't actually be London's open.
  Check the `SESSION_START_HOUR`/`SESSION_END_HOUR`-style constants near
  the top of each file.
- **FVG/wick size thresholds are calibrated to a 5-digit EURUSD quote.**
  `ict_fvg_liquidity_sweep.py` in particular has a `MIN_FVG_PIPS_EQUIV`
  constant that needs rescaling for other instruments (gold, indices,
  crypto, JPY pairs) — what counts as "a real gap" in raw price units is
  completely different for XAUUSD than for EURUSD.
- **The sample-data numbers you'll see if you smoke-test these are not a
  verdict on the strategies.** They were tested only against T58's small
  bundled EURUSD 5-minute sample (a few thousand bars) purely to confirm
  each file runs cleanly end-to-end (imports, no lookahead, no crashes,
  produces trades). That dataset is far too short and single-instrument
  to say anything about whether any of these have a real edge — that's
  what tonight's actual run on your own data is for.

## One flag worth repeating from the earlier review

`multi-confluence-strategy.pine` (not ported) has a genuine lookahead bug
— its HTF bias function is commented "ANTI-REPAINT" but calls
`request.security(..., barmerge.lookahead_on)`, which does the opposite.
## Loading these in the app

You don't need to browse for these files each time — they live right here
in the repo's `strategies/python/` folder, which the app's Strategy
Library (`app/strategy/library.py`) reads directly when you run it from
source (`python -m app.web.server` / `python run_app.py`). In a packaged
`.exe` build, these same files are bundled in and copied into the app's
persistent library the first time it runs, so they show up in the
Strategy Library there too without any manual copying.

