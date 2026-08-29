"""
Information Coefficient / ICIR, signal half-life, and a Bonferroni-corrected
out-of-sample significance gate -- the quant "loop framework" scoring
metrics (IC/ICIR, decay half-life, multiple-testing correction) applied to
this app's existing trade-level backtest output.

Why this exists (and how it differs from what's already in the app):
  app.search.robustness.deflated_sharpe_ratio() already corrects a SHARPE
  RATIO for having been selected as the best of N trials (Bailey/Lopez de
  Prado's Probabilistic Sharpe Ratio). This module is complementary, not a
  replacement: it scores a strategy's directional signal itself using the
  standard quant-research metric for that (Information Coefficient), checks
  whether that signal's predictive power decays fast enough to be
  untradeable, and applies the simpler, better-known Bonferroni correction
  (divide the significance threshold by the number of candidates tried) as
  a second, easy-to-explain out-of-sample gate.

Definitions, adapted to this app's discrete long/flat/short trade model
(no continuous per-bar predicted-return signal exists here, unlike a
factor-model quant book -- every trade already IS one directional bet):
  - "signal" for trade i is its direction, +1 (long) or -1 (short).
  - "return" for trade i is its realized R-multiple (pnl / (initial_risk *
    size)) when the strategy defined a stop, else pnl_pct. This keeps IC
    comparable across trades of different size/instrument, the same reason
    app.backtest.statistics already prefers R-multiples for average_r.
  - IC for one period is the Pearson correlation between the direction
    array and the return array of the trades that closed in that period.
    A period with only one trade, or where every trade shares the same
    direction (zero variance), has an undefined correlation and is
    skipped -- exactly the "IC is noisy for a single reading" problem the
    source material describes, which is why ICIR (below) exists.
  - ICIR = mean(IC) / stdev(IC) across periods. Undefined (None) with fewer
    than 2 usable periods.

None of this calls the optional local-Ollama assistant -- it is pure
arithmetic over numbers the backtest engine already produced, deliberately
kept deterministic so it costs nothing to run on every backtest and never
depends on a model being installed, reachable, or accurate at math.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

# Trade avoids a hard import cycle -- only used for type hints.
try:
    from app.backtest.execution import Trade
except Exception:  # pragma: no cover
    Trade = object  # type: ignore


ICIR_STRONG = 0.5
ICIR_MODERATE = 0.3
DEFAULT_LAGS = (1, 5, 10, 20, 50)
DEFAULT_HALF_LIFE_FLOOR_TRADES = 5.0   # a half-life shorter than this is "decayed before you can act on it"
DEFAULT_RETENTION_FLOOR_PCT = 50.0     # OOS ICIR must retain at least this % of in-sample ICIR
DEFAULT_ALPHA = 0.05


def _trade_return(t) -> float | None:
    """R-multiple when a stop was defined (comparable across instruments/
    sizes), else raw pnl_pct. Mirrors app.backtest.statistics's own
    preference for R-multiples. Returns None for a degenerate trade (no
    size, non-finite pnl) rather than poisoning the series with a NaN/inf,
    the same defensive pattern used throughout app.backtest.statistics."""
    pnl = getattr(t, "pnl", None)
    if pnl is None or not math.isfinite(pnl):
        return None
    risk = getattr(t, "initial_risk", None)
    size = getattr(t, "size", None)
    if risk and size and risk > 0 and size > 0:
        r = pnl / (risk * size)
        return r if math.isfinite(r) else None
    pct = getattr(t, "pnl_pct", None)
    if pct is not None and math.isfinite(pct):
        return pct
    return None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Plain Pearson correlation, no numpy/scipy dependency for such a
    small per-period sample. None when undefined (fewer than 2 points, or
    either series has zero variance -- e.g. every trade in the period was
    long)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else None


@dataclass
class ICSeriesResult:
    ic_values: list[float] = field(default_factory=list)
    period_labels: list[str] = field(default_factory=list)
    n_periods_total: int = 0        # including skipped (undefined) periods
    n_periods_used: int = 0


def compute_ic_series(trades: list, period: str = "M") -> ICSeriesResult:
    """Buckets trades by calendar period (default monthly, matching the
    source framework's "IC every month") using exit_time, and computes one
    IC reading per period. `period` accepts any pandas Period alias ("M"
    month, "W" week, "D" day) -- shorter periods suit strategies with too
    few trades per month to get a usable per-period sample.
    """
    if not trades:
        return ICSeriesResult()

    rows = []
    for t in trades:
        ret = _trade_return(t)
        exit_time = getattr(t, "exit_time", None)
        direction = getattr(t, "direction", None)
        if ret is None or exit_time is None or direction not in (1, -1):
            continue
        rows.append((pd.Timestamp(exit_time), float(direction), ret))
    if not rows:
        return ICSeriesResult()

    df = pd.DataFrame(rows, columns=["exit_time", "direction", "ret"])
    df["bucket"] = df["exit_time"].dt.to_period(period)

    ic_values, labels = [], []
    n_total = 0
    for bucket, group in df.groupby("bucket", sort=True):
        n_total += 1
        ic = _pearson(group["direction"].tolist(), group["ret"].tolist())
        if ic is not None:
            ic_values.append(ic)
            labels.append(str(bucket))

    return ICSeriesResult(
        ic_values=ic_values, period_labels=labels,
        n_periods_total=n_total, n_periods_used=len(ic_values),
    )


def information_coefficient_ratio(ic_values: list[float]) -> float | None:
    """ICIR = mean(IC) / stdev(IC). None with fewer than 2 usable periods
    or a degenerate (zero) standard deviation."""
    n = len(ic_values)
    if n < 2:
        return None
    mean_ic = sum(ic_values) / n
    variance = sum((v - mean_ic) ** 2 for v in ic_values) / (n - 1)
    stdev = math.sqrt(variance)
    if stdev <= 1e-12:
        return None
    return mean_ic / stdev


def interpret_icir(icir: float | None) -> str:
    if icir is None:
        return "Insufficient data (need at least 2 usable periods to compute ICIR)."
    if icir >= ICIR_STRONG:
        return f"Strong (ICIR {icir:.2f}) -- a genuinely consistent signal worth trading."
    if icir >= ICIR_MODERATE:
        return f"Moderate (ICIR {icir:.2f}) -- worth investigating further, needs more validation."
    return f"Weak (ICIR {icir:.2f}) -- probably noise."


@dataclass
class HalfLifeResult:
    half_life_trades: float | None = None      # in units of "trades", this app's closest analogue to "days"
    lags_used: list[int] = field(default_factory=list)
    autocorrelations: list[float] = field(default_factory=list)
    note: str = ""


def estimate_signal_half_life(trades: list, lags: tuple[int, ...] = DEFAULT_LAGS) -> HalfLifeResult:
    """Autocorrelation of the trade-level return series at increasing
    lags, fit to an exponential decay to estimate a half-life -- adapted
    from the source framework's per-day autocorrelation/half-life check.
    This app's trades are event-driven (one row per closed trade, not one
    row per calendar bar), so lags are in units of TRADES, not days;
    treat "half-life of N trades" as "roughly N trades' worth of signal
    before predictive power halves," not a calendar duration.
    """
    ordered = sorted(
        (t for t in trades if getattr(t, "exit_time", None) is not None),
        key=lambda t: t.exit_time,
    )
    returns = [r for r in (_trade_return(t) for t in ordered) if r is not None]
    n = len(returns)
    if n < max(lags) + 5:
        return HalfLifeResult(note=f"Not enough trades ({n}) to estimate decay at lags up to {max(lags)}.")

    mean_r = sum(returns) / n
    var_r = sum((r - mean_r) ** 2 for r in returns) / n
    if var_r <= 1e-12:
        return HalfLifeResult(note="Return series has no variance -- decay is undefined.")

    lags_used, autocorrs = [], []
    for lag in lags:
        if lag >= n:
            continue
        num = sum((returns[i] - mean_r) * (returns[i + lag] - mean_r) for i in range(n - lag))
        ac = (num / (n - lag)) / var_r
        lags_used.append(lag)
        autocorrs.append(ac)

    if len(lags_used) < 2:
        return HalfLifeResult(note="Not enough distinct lags available to fit a decay curve.")

    # Fit |autocorrelation| = exp(-rate * lag) via simple log-linear
    # regression on the lags whose autocorrelation is still positive (an
    # autocorrelation that's already gone negative/flat has no meaningful
    # exponential decay left to fit -- report "already decayed" instead of
    # forcing a nonsensical negative-rate fit through noise).
    usable = [(lag, ac) for lag, ac in zip(lags_used, autocorrs) if ac > 1e-6]
    if len(usable) < 2:
        return HalfLifeResult(
            half_life_trades=float(min(lags_used)),
            lags_used=lags_used, autocorrelations=autocorrs,
            note="Signal is already effectively decayed by the shortest lag checked -- "
                 "treat this as too short to trade.",
        )
    xs = [lag for lag, _ in usable]
    ys = [math.log(ac) for _, ac in usable]
    n2 = len(xs)
    mx, my = sum(xs) / n2, sum(ys) / n2
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return HalfLifeResult(
            lags_used=lags_used, autocorrelations=autocorrs,
            note="Could not fit a decay rate (all usable lags identical).",
        )
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx  # expected negative for genuine decay
    if slope >= -1e-9:
        return HalfLifeResult(
            lags_used=lags_used, autocorrelations=autocorrs,
            note="Autocorrelation isn't decaying (flat or increasing with lag) -- "
                 "half-life is undefined; treat the signal's persistence as unknown, not infinite.",
        )
    half_life = math.log(2) / (-slope)
    return HalfLifeResult(
        half_life_trades=half_life, lags_used=lags_used, autocorrelations=autocorrs,
        note=f"Fit from {n2} usable lag(s).",
    )


def bonferroni_adjusted_alpha(alpha: float, n_tests: int) -> float:
    """The standard multiple-testing correction: divide the significance
    threshold by how many independent candidates were tried. n_tests < 1
    is clamped to 1 (never loosen the threshold below the single-test
    case)."""
    n = max(1, int(n_tests))
    return alpha / n


@dataclass
class SignificanceResult:
    mean_ic: float | None = None
    t_stat: float | None = None
    p_value: float | None = None
    adjusted_alpha: float = DEFAULT_ALPHA
    n_tests: int = 1
    significant: bool = False
    note: str = ""


def _t_distribution_two_sided_p(t_stat: float, dof: int) -> float:
    """Two-sided p-value for a t-statistic, via a numerically stable
    incomplete-beta approximation (no scipy dependency in this repo).
    Accurate to a few parts in 1e-4 across the ranges this module sees
    (dof >= 1, |t| up to a few hundred) -- comfortably enough precision to
    compare against an alpha threshold like 0.05 or a Bonferroni-shrunk
    0.00025."""
    if dof <= 0:
        return 1.0
    x = dof / (dof + t_stat * t_stat)
    return _incomplete_beta(x, dof / 2.0, 0.5)


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via a continued
    fraction (Numerical Recipes' betacf), used only to get a two-sided
    Student's t p-value without a scipy dependency."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 3e-9) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def icir_significance(ic_values: list[float], n_tests: int = 1, alpha: float = DEFAULT_ALPHA) -> SignificanceResult:
    """One-sample t-test of whether mean(IC) is significantly different
    from zero, with the significance threshold Bonferroni-corrected for
    `n_tests` candidates tried (e.g. the total population*generations the
    GA evaluated before landing on this one) -- directly implementing the
    source framework's "adjust for multiple testing" step."""
    adjusted_alpha = bonferroni_adjusted_alpha(alpha, n_tests)
    n = len(ic_values)
    if n < 2:
        return SignificanceResult(
            adjusted_alpha=adjusted_alpha, n_tests=max(1, int(n_tests)),
            note="Fewer than 2 usable IC periods -- significance is undefined.",
        )
    mean_ic = sum(ic_values) / n
    variance = sum((v - mean_ic) ** 2 for v in ic_values) / (n - 1)
    stdev = math.sqrt(variance)
    if stdev <= 1e-12:
        return SignificanceResult(
            mean_ic=mean_ic, adjusted_alpha=adjusted_alpha, n_tests=max(1, int(n_tests)),
            note="IC series has no variance across periods -- significance is undefined.",
        )
    se = stdev / math.sqrt(n)
    t_stat = mean_ic / se
    p_value = _t_distribution_two_sided_p(t_stat, dof=n - 1)
    return SignificanceResult(
        mean_ic=mean_ic, t_stat=t_stat, p_value=p_value, adjusted_alpha=adjusted_alpha,
        n_tests=max(1, int(n_tests)), significant=p_value < adjusted_alpha,
        note=f"p={p_value:.5f} vs Bonferroni-adjusted alpha={adjusted_alpha:.5f} (n_tests={max(1, int(n_tests))}).",
    )


def run_icir_gate_from_backtest(
    df: pd.DataFrame,
    strategy,
    risk,
    n_tests: int = 1,
    holdout_frac: float = 0.2,
    alpha: float = DEFAULT_ALPHA,
    period: str = "M",
) -> ICIRGateResult:
    """Convenience wrapper: runs the SAME chronological in-sample/holdout
    split as app.backtest.engine.run_holdout_comparison (kept as a
    parallel, independent split here rather than modifying that
    function's return shape, which several report templates already
    depend on), then feeds both segments' trades into run_icir_gate().
    This is the "out-of-sample gate" step of the loop framework applied
    directly to one strategy + one dataset, with no re-tuning between
    segments -- consistent with every other holdout-style check in this
    app."""
    from app.backtest.engine import run_backtest  # local import: avoids a hard dependency for pure-math callers

    n = len(df)
    split_idx = int(n * (1 - holdout_frac))
    split_idx = max(1, min(split_idx, n - 1)) if n > 1 else n
    in_sample_df = df.iloc[:split_idx].reset_index(drop=True)
    holdout_df = df.iloc[split_idx:].reset_index(drop=True)

    in_trades = run_backtest(in_sample_df, strategy, risk).trades if len(in_sample_df) else []
    out_trades = run_backtest(holdout_df, strategy, risk).trades if len(holdout_df) else []

    return run_icir_gate(in_trades, out_trades, n_tests=n_tests, alpha=alpha, period=period)


@dataclass
class ICIRGateResult:
    in_sample_icir: float | None
    out_sample_icir: float | None
    icir_retention_pct: float | None       # out/in * 100, None if in-sample ICIR is None/zero
    in_sample_half_life: HalfLifeResult
    out_sample_half_life: HalfLifeResult
    significance: SignificanceResult
    ok: bool
    reasons: list[str] = field(default_factory=list)


def run_icir_gate(
    in_sample_trades: list,
    out_sample_trades: list,
    n_tests: int = 1,
    alpha: float = DEFAULT_ALPHA,
    period: str = "M",
    half_life_floor_trades: float = DEFAULT_HALF_LIFE_FLOOR_TRADES,
    retention_floor_pct: float = DEFAULT_RETENTION_FLOOR_PCT,
) -> ICIRGateResult:
    """The out-of-sample gate from the source framework, adapted to this
    app's existing in-sample/holdout trade split (see
    app.backtest.engine.run_holdout_comparison): the ICIR must hold
    out-of-sample, the decay half-life must hold, and the result must
    still be significant after a Bonferroni correction for how many
    candidates were tried to find it. All three must pass for `ok=True`
    -- exactly the source material's "only strategies that pass all three
    checks should be marked as viable."
    """
    reasons: list[str] = []

    in_ic = compute_ic_series(in_sample_trades, period=period)
    out_ic = compute_ic_series(out_sample_trades, period=period)
    in_icir = information_coefficient_ratio(in_ic.ic_values)
    out_icir = information_coefficient_ratio(out_ic.ic_values)

    retention_pct = None
    if in_icir is not None and abs(in_icir) > 1e-9 and out_icir is not None:
        retention_pct = (out_icir / in_icir) * 100.0

    in_hl = estimate_signal_half_life(in_sample_trades)
    out_hl = estimate_signal_half_life(out_sample_trades)

    significance = icir_significance(out_ic.ic_values, n_tests=n_tests, alpha=alpha)

    icir_holds = retention_pct is not None and retention_pct >= retention_floor_pct
    if in_icir is None or out_icir is None:
        reasons.append(
            "ICIR gate: not enough distinct time periods with trades to compute a reliable ICIR "
            "in-sample and out-of-sample -- treat this strategy's signal quality as UNPROVEN."
        )
    elif icir_holds:
        reasons.append(
            f"ICIR held out-of-sample: {out_icir:.2f} vs {in_icir:.2f} in-sample "
            f"({retention_pct:.0f}% retained, want {retention_floor_pct:.0f}%+)."
        )
    else:
        reasons.append(
            f"ICIR did NOT hold out-of-sample: {out_icir:.2f} vs {in_icir:.2f} in-sample "
            f"({retention_pct:.0f}% retained, below the {retention_floor_pct:.0f}% floor) -- "
            f"likely overfit to the in-sample period."
        )

    decay_holds = (
        out_hl.half_life_trades is not None and out_hl.half_life_trades >= half_life_floor_trades
    )
    if out_hl.half_life_trades is None:
        reasons.append(f"Signal decay (out-of-sample): {out_hl.note}")
    elif decay_holds:
        reasons.append(
            f"Signal decay holds: out-of-sample half-life ~{out_hl.half_life_trades:.1f} trades "
            f"(want {half_life_floor_trades:.0f}+)."
        )
    else:
        reasons.append(
            f"Signal decays too fast to trade: out-of-sample half-life ~{out_hl.half_life_trades:.1f} "
            f"trades (below the {half_life_floor_trades:.0f}-trade floor)."
        )

    if significance.significant:
        reasons.append(
            f"Out-of-sample IC is statistically significant after correcting for {significance.n_tests} "
            f"candidate(s) tried ({significance.note})"
        )
    else:
        reasons.append(
            f"Out-of-sample IC is NOT statistically significant once corrected for "
            f"{significance.n_tests} candidate(s) tried ({significance.note}) -- consistent with "
            f"having found this result by chance among everything else that was tried."
        )

    ok = icir_holds and decay_holds and significance.significant
    return ICIRGateResult(
        in_sample_icir=in_icir, out_sample_icir=out_icir, icir_retention_pct=retention_pct,
        in_sample_half_life=in_hl, out_sample_half_life=out_hl, significance=significance,
        ok=ok, reasons=reasons,
    )
