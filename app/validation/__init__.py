"""
Validation Lab.

Statistical robustness/validation tools that go beyond a single
train/test split or a single in-sample backtest:

  walk_forward_opt.py -- first-class walk-forward OPTIMIZATION (not just
                          a holdout check): re-optimizes on each rolling
                          or anchored in-sample window and chains the
                          resulting out-of-sample segments into one
                          continuous equity curve.
  cpcv.py             -- Combinatorial Purged Cross-Validation (Lopez de
                          Prado) and the associated Probability of
                          Backtest Overfitting (PBO) statistic.
  sensitivity.py      -- 1D parameter sensitivity sweeps and 2D
                          sensitivity heatmaps ("does this strategy sit
                          on a plateau, or a cliff edge?").
"""
