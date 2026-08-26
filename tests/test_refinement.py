import math

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import (
    GeneMeta,
    RefinementError,
    apply_genome,
    extract_genome,
)
from app.optimize.refinement import (
    Candidate,
    RefinementConfig,
    compute_fitness,
    run_iterative_refinement,
)
from app.prop.simulator import PropRules
from app.reports.refinement_report import build_refinement_report, generate_refinement_report
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy


def _trending_df(n=1200, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.00006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config():
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def _visual_builder_config():
    """Mirrors the dict shape app.ui.main_window._build_strategy() produces
    for a Manual Strategy built with the visual condition builder."""
    return {
        "name": "visual rsi",
        "market": {"instrument": "", "timeframe": "5m", "session_start": "08:30", "session_end": "15:00", "direction": "Both"},
        "entry_conditions": {
            "long": [{"left": {"type": "rsi", "period": 14, "field": "close"}, "operator": "<", "right": {"type": "value", "value": 30}}],
            "long_connectors": [],
            "short": [{"left": {"type": "rsi", "period": 14, "field": "close"}, "operator": ">", "right": {"type": "value", "value": 70}}],
            "short_connectors": [],
        },
        "exit_conditions": {"long": [], "long_connectors": [], "short": [], "short_connectors": []},
        "risk_management": {
            "stop_type": "fixed", "stop_value": 20, "stop_atr_period": 14,
            "target_type": "fixed", "target_value": 40, "target_atr_period": 14,
            "trailing_stop": {"enabled": False, "value": 1.5, "atr_period": 14},
            "break_even": {"enabled": False, "trigger_r": 1.0},
            "time_based_exit": {"enabled": False, "time": ""},
            "max_bars_in_trade": None,
            "opposite_signal_exit": True,
        },
    }


# ---------------------------------------------------------------------------
# parameter_space
# ---------------------------------------------------------------------------

def test_extract_genome_finds_legacy_indicator_periods_and_pip_stops():
    genes = extract_genome(_sma_config())
    labels = {g.label for g in genes}
    assert "indicators[0].period" in labels
    assert "indicators[1].period" in labels
    assert "stop_loss_pips" in labels
    assert "take_profit_pips" in labels
    assert len(genes) == 4


def test_extract_genome_finds_visual_builder_nested_condition_values():
    genes = extract_genome(_visual_builder_config())
    labels = {g.label for g in genes}
    # RSI periods and the 30/70 thresholds nested inside entry_conditions
    assert "entry_conditions.long[0].left.period" in labels
    assert "entry_conditions.long[0].right.value" in labels
    assert "entry_conditions.short[0].right.value" in labels
    # risk management fixed stop/target
    assert "risk_management.stop_value" in labels
    assert "risk_management.target_value" in labels
    # max_bars_in_trade is None, so it must NOT produce a gene
    assert not any("max_bars_in_trade" in g.label for g in genes)


def test_extract_genome_ignores_non_numeric_and_excluded_fields():
    genes = extract_genome(_visual_builder_config())
    for g in genes:
        assert isinstance(g.base_value, float)
    # 'name', 'market', condition 'type'/'field'/'direction' strings never appear as gene kinds
    assert all(g.kind not in {"name", "type", "field", "direction"} for g in genes)


def test_gene_bounds_are_sane_and_ordered():
    genes = extract_genome(_sma_config())
    for g in genes:
        assert g.lo < g.hi
        assert g.lo <= g.base_value <= g.hi or math.isclose(g.lo, g.base_value) or math.isclose(g.hi, g.base_value)


def test_apply_genome_round_trips_baseline_unchanged():
    config = _sma_config()
    genes = extract_genome(config)
    baseline_genome = [g.base_value for g in genes]
    rebuilt = apply_genome(config, genes, baseline_genome)
    assert rebuilt["indicators"][0]["period"] == 5
    assert rebuilt["indicators"][1]["period"] == 15
    assert rebuilt["stop_loss_pips"] == 20
    assert rebuilt["take_profit_pips"] == 40
    # original config must not be mutated (apply_genome deep-copies)
    assert config["indicators"][0]["period"] == 5


def test_apply_genome_writes_new_values_at_correct_paths():
    config = _visual_builder_config()
    genes = extract_genome(config)
    genome = [g.base_value * 2 for g in genes]
    rebuilt = apply_genome(config, genes, genome)
    rsi_gene = next(g for g in genes if g.label == "entry_conditions.long[0].left.period")
    idx = genes.index(rsi_gene)
    assert rebuilt["entry_conditions"]["long"][0]["left"]["period"] == int(round(rsi_gene.base_value * 2))


def test_apply_genome_rejects_mismatched_length():
    config = _sma_config()
    genes = extract_genome(config)
    with pytest.raises(RefinementError):
        apply_genome(config, genes, [1.0])


def test_extract_genome_empty_for_config_with_no_tunables():
    genes = extract_genome({"name": "no params", "long_entry": "close > open"})
    assert genes == []


# ---------------------------------------------------------------------------
# fitness scoring
# ---------------------------------------------------------------------------

class _FakeMC:
    def __init__(self, eval_pass=0.0, first_payout=0.0, ruin=0.0, expected_payout=0.0):
        self.evaluation_pass_probability = eval_pass
        self.first_payout_probability = first_payout
        self.risk_of_ruin_pct = ruin
        self.expected_payout = expected_payout


def test_compute_fitness_composite_prop_score():
    mc = _FakeMC(eval_pass=80.0, first_payout=60.0, ruin=10.0)
    score = compute_fitness({}, {}, mc, "composite_prop_score")
    assert score == pytest.approx(80.0 * 0.5 + 60.0 * 0.3 - 10.0 * 0.2)


def test_compute_fitness_profit_factor_caps_infinity():
    score = compute_fitness({"profit_factor": float("inf")}, {}, _FakeMC(), "profit_factor")
    assert score == 10.0


def test_compute_fitness_unknown_metric_raises():
    with pytest.raises(RefinementError):
        compute_fitness({}, {}, _FakeMC(), "not_a_real_metric")


# ---------------------------------------------------------------------------
# full GA loop (integration)
# ---------------------------------------------------------------------------

def test_refinement_config_clamps_invalid_settings():
    cfg = RefinementConfig(population_size=1, generations=0, elite_count=99, mutation_rate=5, random_immigrants_frac=5)
    assert cfg.population_size >= 4
    assert cfg.generations >= 1
    assert cfg.elite_count <= cfg.population_size - 1
    assert cfg.mutation_rate <= 1.0
    assert cfg.random_immigrants_frac <= 0.9


def test_refinement_raises_for_config_with_no_tunable_parameters():
    df = _trending_df(n=100)
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    refine_cfg = RefinementConfig(population_size=4, generations=1)
    with pytest.raises(RefinementError):
        run_iterative_refinement(
            df, ManualStrategy({"name": "no params", "long_entry": "close > open"}),
            risk, rules, mc_cfg, refine_cfg,
        )


def test_refinement_improves_or_matches_baseline_on_net_profit():
    """
    Elitism guarantees best-ever fitness is monotonically non-decreasing
    across generations, and the best-ever candidate must be at least as
    good as the baseline (the baseline is itself always a member of the
    initial population).
    """
    df = _trending_df(n=1200, seed=3, drift=0.00015)
    risk = RiskConfig(initial_balance=10_000, pip_size=0.0001)
    rules = PropRules(evaluation_profit_target_pct=4.0, min_trading_days=1)
    mc_cfg = MonteCarloConfig(n_simulations=300, random_seed=42)
    refine_cfg = RefinementConfig(
        population_size=8, generations=4, elite_count=2,
        search_monte_carlo_sims=100, random_seed=11, fitness_metric="net_profit",
    )

    result = run_iterative_refinement(df, ManualStrategy(_sma_config()), risk, rules, mc_cfg, refine_cfg)

    assert result.best.fitness >= result.baseline.fitness
    assert len(result.generation_history) == refine_cfg.generations + 1  # gen 0 + N bred generations

    # best-ever fitness must be non-decreasing generation over generation
    best_series = [g.best_fitness for g in result.generation_history]
    running_max = float("-inf")
    for b in best_series:
        assert b >= running_max - 1e-9 or not math.isfinite(running_max)
        running_max = max(running_max, b)

    # the winning genome must actually have been applied to the config
    assert result.best.config != _sma_config() or result.best.fitness == result.baseline.fitness


def test_refinement_result_has_full_objects_for_baseline_and_best_only():
    df = _trending_df(n=800, seed=5, drift=0.0001)
    risk = RiskConfig(initial_balance=10_000, pip_size=0.0001)
    rules = PropRules(evaluation_profit_target_pct=4.0, min_trading_days=1)
    mc_cfg = MonteCarloConfig(n_simulations=200, random_seed=1)
    refine_cfg = RefinementConfig(population_size=6, generations=2, search_monte_carlo_sims=80, random_seed=2)

    result = run_iterative_refinement(df, ManualStrategy(_sma_config()), risk, rules, mc_cfg, refine_cfg)

    assert result.baseline.bt_result is not None
    assert result.best.bt_result is not None
    # Leaderboard entries are lightweight (no full backtest objects
    # retained) EXCEPT the baseline candidate itself, which was evaluated
    # with keep_full=True up front and may still be present if it survived
    # via elitism -- every other candidate must carry no full bt_result.
    non_baseline = [c for c in result.leaderboard if c is not result.baseline]
    assert len(non_baseline) >= 1
    for c in non_baseline:
        assert c.bt_result is None


def test_refinement_report_generation_end_to_end(tmp_path):
    df = _trending_df(n=800, seed=5, drift=0.0001)
    risk = RiskConfig(initial_balance=10_000, pip_size=0.0001)
    rules = PropRules(evaluation_profit_target_pct=4.0, min_trading_days=1)
    mc_cfg = MonteCarloConfig(n_simulations=200, random_seed=1)
    refine_cfg = RefinementConfig(population_size=6, generations=2, search_monte_carlo_sims=80, random_seed=2)

    result = run_iterative_refinement(df, ManualStrategy(_sma_config()), risk, rules, mc_cfg, refine_cfg)
    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))

    report = build_refinement_report(result, "sma cross", "TEST.csv", "5m", period)
    assert report["parameter_count"] == 4
    assert "best_config" in report

    paths = generate_refinement_report(
        output_dir=tmp_path, result=result, strategy_name="sma cross",
        instrument="TEST.csv", timeframe="5m", backtest_period=period, price_df=df,
    )
    assert paths["html"].exists()
    assert paths["json"].exists()
    assert paths["best_config_json"].exists()

    html = paths["html"].read_text(encoding="utf-8")
    assert "Iterative Refinement Report" in html
    assert "Parameter Drift" in html
    assert "Out-of-Sample Holdout Check" in html


# ---------------------------------------------------------------------------
# Code-based strategies (Python / PineScript / MQL5)
# ---------------------------------------------------------------------------

_PYTHON_STRATEGY_SRC = '''STRATEGY_NAME = "Test EMA Cross"
EMA_FAST = 5
EMA_SLOW = 15
STOP_LOSS_PIPS = 20
TAKE_PROFIT_PIPS = 40

def generate_signals(df):
    fast = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    slow = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    sig = (fast > slow).astype(int) - (fast < slow).astype(int)
    return sig
'''

_PINESCRIPT_STRATEGY_SRC = '''//@version=5
strategy("Test", overlay=true)
fastLen = input.int(5, "Fast Length")
slowLen = input.int(15, "Slow Length")
fast = ta.ema(close, fastLen)
slow = ta.ema(close, slowLen)
longCondition = ta.crossover(fast, slow)
shortCondition = ta.crossunder(fast, slow)
if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition
    strategy.entry("Short", strategy.short)
// T58_SL_PIPS=20
// T58_TP_PIPS=40
'''

_MQL5_STRATEGY_SRC = '''void OnTick() {
   double fastMA = iMA(_Symbol, PERIOD_CURRENT, 5, 0, MODE_EMA, PRICE_CLOSE);
   double slowMA = iMA(_Symbol, PERIOD_CURRENT, 15, 0, MODE_EMA, PRICE_CLOSE);
   if (fastMA > slowMA) {
      trade.Buy(0.1, _Symbol);
   }
   if (fastMA < slowMA) {
      trade.Sell(0.1, _Symbol);
   }
   // T58_SL_PIPS=20
   // T58_TP_PIPS=40
}
'''


def _write_temp(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


def test_refinement_works_for_python_strategy(tmp_path):
    path = _write_temp(tmp_path, "strat.py", _PYTHON_STRATEGY_SRC)
    strategy = PythonStrategy(path)

    df = _trending_df(n=1000, seed=4, drift=0.00012)
    risk = RiskConfig(initial_balance=10_000, pip_size=0.0001)
    rules = PropRules(evaluation_profit_target_pct=4.0, min_trading_days=1)
    mc_cfg = MonteCarloConfig(n_simulations=200, random_seed=1)
    refine_cfg = RefinementConfig(population_size=6, generations=2, search_monte_carlo_sims=80,
                                   random_seed=3, fitness_metric="net_profit")

    result = run_iterative_refinement(df, strategy, risk, rules, mc_cfg, refine_cfg)

    assert result.source_type == "python"
    assert len(result.genes) == 4  # EMA_FAST, EMA_SLOW, STOP_LOSS_PIPS, TAKE_PROFIT_PIPS
    assert result.best.fitness >= result.baseline.fitness
    assert result.best.config is None
    assert result.best.code_text is not None
    assert result.best.code_extension == ".py"
    # the original file on disk must be untouched
    assert path.read_text(encoding="utf-8") == _PYTHON_STRATEGY_SRC
    # the patched code must still be valid, loadable Python that runs
    patched_path = tmp_path / "patched.py"
    patched_path.write_text(result.best.code_text, encoding="utf-8")
    reloaded = PythonStrategy(patched_path)
    reloaded.generate(df)  # must not raise


def test_refinement_works_for_pinescript_strategy(tmp_path):
    strategy = PineScriptStrategy(_PINESCRIPT_STRATEGY_SRC)

    df = _trending_df(n=1000, seed=4, drift=0.00012)
    risk = RiskConfig(initial_balance=10_000, pip_size=0.0001)
    rules = PropRules(evaluation_profit_target_pct=4.0, min_trading_days=1)
    mc_cfg = MonteCarloConfig(n_simulations=200, random_seed=1)
    refine_cfg = RefinementConfig(population_size=6, generations=2, search_monte_carlo_sims=80, random_seed=3)

    result = run_iterative_refinement(df, strategy, risk, rules, mc_cfg, refine_cfg)

    assert result.source_type == "pinescript"
    assert len(result.genes) == 4  # fastLen, slowLen, T58_SL_PIPS, T58_TP_PIPS
    assert result.best.config is None
    assert result.best.code_extension == ".pine"
    assert "input.int(" in result.best.code_text
    # patched code must still parse and run
    PineScriptStrategy(result.best.code_text).generate(df)


def test_refinement_works_for_mql5_strategy(tmp_path):
    strategy = MQL5Strategy(_MQL5_STRATEGY_SRC)

    df = _trending_df(n=1000, seed=4, drift=0.00012)
    risk = RiskConfig(initial_balance=10_000, pip_size=0.0001)
    rules = PropRules(evaluation_profit_target_pct=4.0, min_trading_days=1)
    mc_cfg = MonteCarloConfig(n_simulations=200, random_seed=1)
    refine_cfg = RefinementConfig(population_size=6, generations=2, search_monte_carlo_sims=80, random_seed=3)

    result = run_iterative_refinement(df, strategy, risk, rules, mc_cfg, refine_cfg)

    assert result.source_type == "mql5"
    assert len(result.genes) == 4  # 2 iMA periods, T58_SL_PIPS, T58_TP_PIPS
    assert result.best.config is None
    assert result.best.code_extension == ".mq5"
    assert "iMA(" in result.best.code_text
    MQL5Strategy(result.best.code_text).generate(df)


def test_refinement_raises_for_code_strategy_with_no_tunable_parameters():
    strategy = PineScriptStrategy(
        '//@version=5\nstrategy("t")\nif close > open\n    strategy.entry("Long", strategy.long)\n'
    )
    df = _trending_df(n=100)
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    refine_cfg = RefinementConfig(population_size=4, generations=1)
    with pytest.raises(RefinementError):
        run_iterative_refinement(df, strategy, risk, rules, mc_cfg, refine_cfg)


def test_refinement_report_generation_for_code_strategy(tmp_path):
    strategy = PineScriptStrategy(_PINESCRIPT_STRATEGY_SRC)
    df = _trending_df(n=1000, seed=4, drift=0.00012)
    risk = RiskConfig(initial_balance=10_000, pip_size=0.0001)
    rules = PropRules(evaluation_profit_target_pct=4.0, min_trading_days=1)
    mc_cfg = MonteCarloConfig(n_simulations=200, random_seed=1)
    refine_cfg = RefinementConfig(population_size=6, generations=2, search_monte_carlo_sims=80, random_seed=3)

    result = run_iterative_refinement(df, strategy, risk, rules, mc_cfg, refine_cfg)
    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))

    paths = generate_refinement_report(
        output_dir=tmp_path, result=result, strategy_name="Pine Test",
        instrument="TEST.csv", timeframe="5m", backtest_period=period, price_df=df,
    )
    assert paths["html"].exists()
    assert paths["json"].exists()
    assert paths["best_strategy_file"].exists()
    assert paths["best_strategy_file"].suffix == ".pine"
    assert "input.int(" in paths["best_strategy_file"].read_text(encoding="utf-8")

    html = paths["html"].read_text(encoding="utf-8")
    assert "Winning PineScript Strategy" in html
    assert "Parameter Drift" in html
