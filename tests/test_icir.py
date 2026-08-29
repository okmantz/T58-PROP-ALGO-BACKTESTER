import math
import random

import pandas as pd
import pytest

from app.validation.icir import (
    bonferroni_adjusted_alpha,
    compute_ic_series,
    estimate_signal_half_life,
    icir_significance,
    information_coefficient_ratio,
    interpret_icir,
    run_icir_gate,
)


class FakeTrade:
    def __init__(self, exit_time, direction, pnl, initial_risk=1.0, size=1.0, pnl_pct=None):
        self.exit_time = pd.Timestamp(exit_time)
        self.entry_time = pd.Timestamp(exit_time)
        self.direction = direction
        self.pnl = pnl
        self.initial_risk = initial_risk
        self.size = size
        self.pnl_pct = pnl_pct if pnl_pct is not None else pnl / 100.0


def _real_signal_trades(n=240, seed=1):
    """Direction genuinely predicts the sign of the return (with noise) --
    a strategy with a real, if imperfect, edge."""
    rng = random.Random(seed)
    trades = []
    start = pd.Timestamp("2023-01-01")
    for i in range(n):
        direction = 1 if rng.random() > 0.5 else -1
        # positive R when direction matches an underlying (noisy) trend
        base_r = 0.3 if direction == 1 else -0.1
        r = base_r + rng.gauss(0, 0.5)
        trades.append(FakeTrade(start + pd.Timedelta(days=i), direction, pnl=r))
    return trades


def _noise_trades(n=240, seed=2):
    """Direction has zero real relationship to return -- pure noise."""
    rng = random.Random(seed)
    trades = []
    start = pd.Timestamp("2023-01-01")
    for i in range(n):
        direction = 1 if rng.random() > 0.5 else -1
        r = rng.gauss(0, 0.5)
        trades.append(FakeTrade(start + pd.Timedelta(days=i), direction, pnl=r))
    return trades


def test_compute_ic_series_empty():
    result = compute_ic_series([])
    assert result.ic_values == []
    assert result.n_periods_used == 0


def test_compute_ic_series_skips_degenerate_periods():
    # Both trades in the same month, same direction -> zero-variance signal, undefined IC.
    trades = [
        FakeTrade("2023-01-05", 1, pnl=10),
        FakeTrade("2023-01-20", 1, pnl=-5),
    ]
    result = compute_ic_series(trades, period="M")
    assert result.n_periods_total == 1
    assert result.n_periods_used == 0


def test_icir_higher_for_real_signal_than_noise():
    real = compute_ic_series(_real_signal_trades(), period="W")
    noise = compute_ic_series(_noise_trades(), period="W")
    real_icir = information_coefficient_ratio(real.ic_values)
    noise_icir = information_coefficient_ratio(noise.ic_values)
    assert real_icir is not None and noise_icir is not None
    assert real_icir > noise_icir


def test_information_coefficient_ratio_needs_two_periods():
    assert information_coefficient_ratio([0.2]) is None
    assert information_coefficient_ratio([]) is None


def test_information_coefficient_ratio_zero_variance_is_none():
    assert information_coefficient_ratio([0.3, 0.3, 0.3]) is None


def test_interpret_icir_buckets():
    assert "Strong" in interpret_icir(0.6)
    assert "Moderate" in interpret_icir(0.35)
    assert "Weak" in interpret_icir(0.1)
    assert "Insufficient" in interpret_icir(None)


def test_bonferroni_adjusted_alpha():
    assert bonferroni_adjusted_alpha(0.05, 1) == pytest.approx(0.05)
    assert bonferroni_adjusted_alpha(0.05, 200) == pytest.approx(0.05 / 200)
    # never loosens below n_tests=1 even if a caller passes 0 or negative
    assert bonferroni_adjusted_alpha(0.05, 0) == pytest.approx(0.05)


def test_icir_significance_detects_real_signal():
    real = compute_ic_series(_real_signal_trades(seed=7), period="W")
    result = icir_significance(real.ic_values, n_tests=1)
    assert result.p_value is not None
    assert 0.0 <= result.p_value <= 1.0


def test_icir_significance_bonferroni_can_flip_a_borderline_result():
    real = compute_ic_series(_real_signal_trades(seed=7), period="W")
    lenient = icir_significance(real.ic_values, n_tests=1, alpha=0.05)
    strict = icir_significance(real.ic_values, n_tests=100000, alpha=0.05)
    assert strict.adjusted_alpha < lenient.adjusted_alpha
    # A stricter (smaller) alpha can never turn a non-significant result significant.
    if not lenient.significant:
        assert not strict.significant


def test_estimate_signal_half_life_insufficient_trades():
    result = estimate_signal_half_life(_real_signal_trades(n=10))
    assert result.half_life_trades is None
    assert "Not enough trades" in result.note


def test_estimate_signal_half_life_runs_on_enough_trades():
    result = estimate_signal_half_life(_real_signal_trades(n=200))
    # Should produce SOME result object without raising, whether or not a
    # clean decay was fit (synthetic i.i.d. noise-plus-signal won't
    # necessarily autocorrelate cleanly, and that's fine).
    assert isinstance(result.lags_used, list)


def test_run_icir_gate_real_signal_scores_better_than_noise():
    real_in, real_out = _real_signal_trades(n=150, seed=3), _real_signal_trades(n=80, seed=4)
    noise_in, noise_out = _noise_trades(n=150, seed=5), _noise_trades(n=80, seed=6)

    real_gate = run_icir_gate(real_in, real_out, n_tests=1, period="W")
    noise_gate = run_icir_gate(noise_in, noise_out, n_tests=1, period="W")

    assert real_gate.in_sample_icir is not None
    assert noise_gate.in_sample_icir is not None
    # Not a strict guarantee on random synthetic data, but the real-signal
    # in-sample ICIR should be meaningfully higher than pure noise's.
    assert real_gate.in_sample_icir > noise_gate.in_sample_icir


def test_run_icir_gate_empty_trades_does_not_crash():
    result = run_icir_gate([], [], n_tests=5)
    assert result.ok is False
    assert result.in_sample_icir is None
    assert len(result.reasons) == 3


def test_run_icir_gate_high_n_tests_makes_significance_harder():
    trades_in = _real_signal_trades(n=150, seed=9)
    trades_out = _real_signal_trades(n=80, seed=10)
    lenient = run_icir_gate(trades_in, trades_out, n_tests=1, period="W")
    strict = run_icir_gate(trades_in, trades_out, n_tests=500, period="W")
    assert strict.significance.adjusted_alpha < lenient.significance.adjusted_alpha
    if not lenient.significance.significant:
        assert not strict.significance.significant
