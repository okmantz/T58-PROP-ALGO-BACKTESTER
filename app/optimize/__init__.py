"""
Iterative Refinement / genetic-algorithm-style parameter optimization.

This package is entirely additive: nothing in app.backtest, app.strategy,
app.prop, or app.monte_carlo is modified to support it. It only *drives*
those existing modules many times over with different Manual Strategy
configurations and ranks the results.
"""
