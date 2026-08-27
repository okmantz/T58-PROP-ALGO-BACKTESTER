"""
T58 Search Lab.

Turns the app from "backtest one strategy configuration at a time" into
"generate, filter, refine, and validate thousands of strategy variations,
then hand back only the ones that survive honest scrutiny."

Modules:
    strategy_space   -- Stage 0: generates the candidate pool (either one
                         user-supplied strategy, or a combinatorial grid
                         across a named strategy family).
    batch_runner     -- Stages 1-5: cheap parallel filter -> genetic-algorithm
                         refinement of survivors -> a strict validation gate
                         (walk-forward, lookahead check, parameter-neighborhood
                         robustness, deflated Sharpe) -> leaderboard -> champion
                         promotion into the app's normal report pipeline.
    robustness       -- The statistical tools that keep "thousands of
                         backtests" from just becoming "thousands of chances
                         to find noise": deflated/probabilistic Sharpe ratio,
                         walk-forward efficiency, and parameter-neighborhood
                         stability.
    results_db       -- SQLite-backed storage for every candidate evaluated
                         in a search run, queryable as a leaderboard.
    search_report    -- Renders a search run's leaderboard as an HTML report.
"""
