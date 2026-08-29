# Screening Results — None of these are "winning" yet

Run 2026-08-29, against your own real `data/raw/XAUUSD/XAUUSD15.csv` (100,000
15-minute bars, May 2022 → present), $50,000 account, 1% risk/trade, 2% daily
loss limit, 10% max drawdown, realistic round-turn costs (3 pip spread, 1 pip
slippage, $5 commission). This is not a sales pitch for any of these files —
it's the opposite. Read this before trusting any of them.

## Bottom line

**All 7 strategies in `strategies/python/` currently show a negative edge on
real XAUUSD 15m data.** Every one grinds the test account down to
approximately zero over the ~100k-bar period. This was checked at the
trade level (not just the summary stats) to rule out a sizing bug: the
per-trade losses are small and consistent with the configured 1% risk —
there's no single catastrophic blowup trade. It's a slow bleed from a
persistently negative expectancy, not an engine error. This matches your
app's own prior full research pass, which reached the same conclusion
against a different but overlapping strategy set: *no existing strategy has
a validated edge.* This run is independent confirmation, not a new finding.

| Strategy | Verdict | Net profit | Profit factor | Risk of ruin |
|---|---|---|---|---|
| hedge_fund_smc_ea.py | NOT READY | ~-$50,000 | 0.00 | 14.2% |
| propfirm_reactive_bos.py | NOT READY | ~-$50,000 | 0.00 | 9.0% |
| propfirm_elite_ea.py | NOT READY | ~-$50,000 | 0.00 | 5.4% |
| judas_sweep_killzone.py | NOT READY | ~-$50,000 | 0.002 | 60.1% |
| ict_fvg_liquidity_sweep.py | NOT READY | ~-$50,000 | 0.00 | 15.9% |
| smc_quant_engine.py | NOT READY | ~-$50,000 | 0.00 | 17.8% |
| propfirm_score_engine.py | NOT READY | ~-$50,000 | 0.00 | 18.0% |

Every single strategy: 0% Monte Carlo evaluation-pass probability, 0% first-
payout probability. None of these should be forward-tested or deployed as-is.

## Important caveats about THIS specific screen

This was a fast screen, not a final verdict either way:

- **Reduced Full Pipeline settings** were used for speed (GA population 6,
  2 generations, 2 folds, 2,000 Monte Carlo sims — versus the app's normal
  defaults of population 12, 6 generations, 4 folds, 10,000 sims). A
  strategy that's merely mediocre rather than structurally broken might
  fare slightly better with a real run. None of these looked "merely
  mediocre," though — a strategy with a real (even weak) edge doesn't
  reliably erode ~100% of the account.
- **One instrument, one timeframe.** These are XAUUSD 15-minute results
  only. A strategy tuned toward a different pair or timeframe might behave
  differently — worth checking if any of these were originally designed
  for something else (`strategies/python/README.md` has the per-file
  session-hour and FVG-threshold calibration notes).
- **Before concluding anything is dead for good**, run it through the real
  Full Pipeline tab in the app (Step 15) with default settings on the
  instrument/timeframe it's actually meant for. This screen is a fast
  filter, not a replacement for that.

## Your attached `full_pipeline_report.html` (ict_fvg_liquidity_sweep on AAPL)

Checked separately — this run completed cleanly with no engine bugs, but
it's also NOT READY and not a winner: net profit essentially flat over two
years ($665 on $50k, profit factor 1.002), 47.5% historical max drawdown
against a 4% prop limit, and the single historical run breached the daily
loss limit on day one (0% pass in that run). In-sample was actually
net-negative (-$8,739) while the later holdout period was positive
(+$6,831) — that inversion, on only 36 holdout trades, reads as small-sample
noise rather than a real edge reasserting itself. Also worth flagging: AAPL
is a stock, and this strategy's FVG/wick-size thresholds were calibrated to
a 5-digit EURUSD quote (see the "sanity-check before trusting" section of
`strategies/python/README.md`) — that mismatch alone could explain why the
signal quality looks this unstable on AAPL specifically.

## What this means for Forward Test (the new tab)

The Forward Test tab will happily deploy any of these to a demo account —
it doesn't gatekeep on backtest quality, since forward-testing something
that looks bad in backtest is sometimes exactly the point (checking whether
live spreads/fills change the picture). But don't expect a demo run of any
of the 7 files above to suddenly turn profitable. If you want something
worth actually forward-testing, the honest next step is either: build a new
strategy from scratch and validate it through the real Full Pipeline first,
or point the Full Pipeline's walk-forward-aware GA at one of these on a
longer leash (full population/generations) to see if a genuinely different
parameter region holds up — not just re-running the same logic and hoping.
