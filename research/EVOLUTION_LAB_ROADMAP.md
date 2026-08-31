# Evolution Lab / Autonomous Research Agent — Scoping Note

Owen asked for three connected pieces on top of everything else in this
delivery:

1. An "Evolution Lab" tab: generate ~100 strategies, filter through
   robustness/OOS/Monte Carlo/prop-sim, cluster, keep the top 10, mutate,
   generate the next 100, repeat — runnable for hours unattended, scored
   by a composite PROP FITNESS function instead of raw profit.
2. A knowledge graph of "what works, where, under what conditions" that
   makes each new generation smarter about *why* a strategy looks
   promising, not just *that* it does.
3. A fully autonomous research agent that takes a one-line objective
   ("find a strategy for XAUUSD under 8% max DD") and runs the whole
   hypothesis → test → OOS → stress → reject/keep → mutate loop by
   itself, journaling numbered hypotheses like a human quant would.

Given the explicit ask to go easy on credits today, building all three
properly in this pass wasn't the right trade — each is a genuinely large
system on its own (the research agent in particular is close to a second
application), and a rushed version would either be too slow to actually
run for "hours in the background" usefully, or would look done without
the safeguards that make Full Pipeline trustworthy in the first place
(the OOS/ICIR/Bonferroni gates already in this repo exist because a GA
given free rein WILL find noise and call it a strategy).

What's actually ready to build next, in order of leverage per session:

**1. PROP FITNESS composite score (small, high leverage, do first).**
A single function:
```
fitness = (eval_pass_prob/100) * (payout_prob/100) * robustness_score * oos_consistency
           / max(max_drawdown_pct, 1)
         - penalties(too_few_trades, high_param_sensitivity, high_pbo,
                      is_oos_gap, concentration, losing_streak_extremity)
```
This slots directly into `app.optimize.walkforward_ga`'s existing
`fitness_metric` plumbing (see `FITNESS_METRICS` in
`app/optimize/refinement.py`) as a new named metric, so Quick Optimize
and Full Pipeline both benefit from it immediately with no new
infrastructure. This is a half-session task and should be next.

**2. Evolution Lab tab (medium, builds on #1 + existing batch search).**
The generate/filter/cluster/mutate/repeat loop Owen described is *almost*
entirely composed of pieces that already exist in this codebase:
`app.search.strategy_space` (candidate generation), the walk-forward GA's
mutation operators, `run_monte_carlo` + `simulate_account` (prop-sim),
and a new clustering step (correlation-based, on the trade-return series
of the surviving candidates — `scipy` or a simple correlation-threshold
dedupe is enough, no need for anything fancier). The main new work is a
background-thread runner with pause/resume/stop and a persistent log,
following the exact threading pattern already used for Batch Test /
Full Pipeline Batch in `app/ui/main_window.py`. Realistic estimate: a
full focused session.

**3. Knowledge graph + autonomous research agent (large, do last).**
These are genuinely R&D, not engineering-from-existing-parts like #1/#2.
A defensible v1 of the "knowledge graph" is much simpler than it sounds:
a table of (strategy features -> outcome) rows from every backtest ever
run (already logged in `data/run_history.json` and the Strategy
Library's `record_backtest_result`), queried by feature-similarity
(cosine similarity over a hand-picked feature vector: session filter,
volatility regime, mechanism type, indicator set) rather than an actual
graph database — that's buildable. The fully autonomous "objective in,
numbered-hypothesis journal out" agent on top of it is the ambitious
piece and should be scoped as its own multi-session project once #1/#2
are in place and there's a track record of what "keep vs. reject" looks
like in practice to encode into it.

None of this is blocking today's deliverables -- Quick Optimize (this
session's "Optimize" button) already gives a one-click way to search a
single strategy's parameter space, which is most of the day-to-day value
of the loop above without the multi-hour unattended run.
