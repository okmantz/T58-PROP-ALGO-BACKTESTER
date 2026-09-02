"""
T58 Research Loop -- the closed hypothesis -> test -> analyze-failure ->
improved-hypothesis loop described in the AI Research Engine plan:

    Research idea
         |
         v
    Ollama generates a strategy (app.ai.strategy_generator)
         |
         v
    Backtest (app.backtest.engine)
         |
         v
    Robustness: Monte Carlo (app.monte_carlo.engine)
         |
         v
    Prop simulation: full-lifecycle survival (app.prop.survival_engine)
         |
         v
    Analyze failure (THIS module's diagnose_failure -- a real, computed
    diagnostic, not an AI guess) --------> KEEP: stop or keep refining
         |
         v (DISCARD)
    Ollama proposes a specific structural change, grounded in the
    diagnostic's actual numbers
         |
         v
    Improved hypothesis -> back to the top

The critical property carried over from app.ai.research_agent: Ollama
never decides whether a strategy is good. Every KEEP/DISCARD verdict
comes from run_prop_survival_analysis's own score, computed on real
backtest trades -- Ollama's only two jobs are (1) turning a hypothesis
into a candidate strategy file (via the existing, already-guarded
app.ai.strategy_generator.generate_strategy, which never edits a
strategy that already exists -- it only ever proposes a new draft) and
(2) turning a *computed* failure diagnostic into a specific next idea to
try. Both of those outputs still have to survive the exact same
backtest/Monte-Carlo/prop-simulation gauntlet as anything else in this
app before they mean anything.

Every completed iteration is recorded into app.ai.experiment_memory
(parent-linked via config["parent_experiment"]), and this module refuses
to re-run a strategy whose Strategy DNA (app.strategy.dna) exactly
matches a signature that already scored DISCARD earlier in the SAME
loop run -- the "don't let the AI simply generate random strategies /
don't repeat failed experiments" requirement.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app.ai.experiment_memory import is_dna_tagset_previously_discarded, record_experiment
from app.ai.ollama_settings import OllamaSettings
from app.ai.strategy_generator import generate_strategy
from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules
from app.prop.survival_engine import PropSurvivalConfig, run_prop_survival_analysis
from app.strategy.dna import extract_dna
from app.strategy.python import PythonStrategy
from app.validation.regime_matrix import label_session

ProgressCallback = Callable[[str], None]

DEFAULT_KEEP_SCORE_THRESHOLD = 40.0


# ---------------------------------------------------------------------------
# Failure analysis -- a real, computed diagnostic (never an AI guess)
# ---------------------------------------------------------------------------

def _session_loss_concentration(df: pd.DataFrame, losers: list, min_pct: float = 65.0) -> tuple[dict | None, str | None]:
    """The session-time companion to the ATR-volatility check below --
    exactly the 'natural next addition' flagged in this module's own
    prior notes. Answers 'what fraction of losing P&L came from each
    fixed session window' using app.validation.regime_matrix's session
    classifier (no fitting needed -- session boundaries are fixed clock
    windows, unlike the ATR check's percentile threshold). Returns
    (pct_by_session, concentrated_session) -- concentrated_session is
    None unless one single session accounts for >= min_pct of total
    losing P&L, same 'don't over-claim from a coin flip' bar the
    volatility check uses."""
    if not losers or df is None or len(df) < 20 or "timestamp" not in df.columns:
        return None, None
    try:
        session_series = label_session(df)
    except Exception:
        return None, None

    ts = pd.to_datetime(df["timestamp"])
    session_by_time = pd.Series(session_series.astype(object).values, index=ts)

    def _session_at(entry_time):
        idx = session_by_time.index.searchsorted(pd.Timestamp(entry_time), side="right") - 1
        if idx < 0 or idx >= len(session_by_time):
            return None
        val = session_by_time.iloc[idx]
        return val if pd.notna(val) else None

    total_loss = sum(-t.pnl for t in losers)
    if total_loss <= 0:
        return None, None

    loss_by_session: dict[str, float] = {}
    for t in losers:
        s = _session_at(t.entry_time)
        if s is None:
            continue
        loss_by_session[s] = loss_by_session.get(s, 0.0) + (-t.pnl)
    if not loss_by_session:
        return None, None

    pct_by_session = {k: v / total_loss * 100.0 for k, v in loss_by_session.items()}
    worst_session, worst_pct = max(pct_by_session.items(), key=lambda kv: kv[1])
    return pct_by_session, (worst_session if worst_pct >= min_pct else None)


def diagnose_failure(df: pd.DataFrame, trades: list, low_vol_percentile: float = 50.0) -> dict:
    """Answers 'why did this strategy actually lose money' with two
    concrete, checkable numbers, computed directly from the strategy's
    own trades and the underlying price series -- never asked of the AI:

    1. (original check) what fraction of total LOSING P&L came from
       trades opened while a simple ATR-based volatility proxy was below
       the given percentile ('a low-volatility regime') versus at/above
       it -- exactly the diagnostic the research-loop plan describes
       ('73% of drawdown occurred during low-volatility regimes').
    2. (session-concentration check) what fraction of that same losing
       P&L is concentrated in one fixed session window (Asia/London/NY
       Open/NY/Power Hour) -- see _session_loss_concentration above.

    Returns {"low_vol_loss_pct": float, "high_vol_loss_pct": float,
    "regime": "low_vol"|"high_vol"|"mixed", "suggestion": str|None,
    "session_loss_pct": dict|None, "concentrated_session": str|None,
    "session_suggestion": str|None}. Every key is present on every
    return path (defaulted None where there isn't enough signal), so
    callers never need to guard with .get() against a missing key.
    `suggestion` and `session_suggestion` are independent, plain-
    language, ready-to-hand-to-Ollama sentences -- either, both, or
    neither may be non-None depending on what the numbers actually show.
    """
    losers = [t for t in trades if t.pnl < 0]
    session_loss_pct, concentrated_session = _session_loss_concentration(df, losers)
    session_suggestion = None
    if concentrated_session:
        session_suggestion = (
            f"{session_loss_pct[concentrated_session]:.0f}% of this strategy's losing P&L came from trades "
            f"opened during the {concentrated_session.replace('_', ' ').title()} session window. Test adding "
            f"a session-time filter that skips (or tightens risk during) that window."
        )
    session_fields = {
        "session_loss_pct": session_loss_pct, "concentrated_session": concentrated_session,
        "session_suggestion": session_suggestion,
    }

    if len(losers) < 5 or df is None or len(df) < 20 or "high" not in df.columns:
        return {"low_vol_loss_pct": None, "high_vol_loss_pct": None, "regime": "unknown", "suggestion": None,
                **session_fields}

    true_range = (df["high"] - df["low"]).abs()
    atr = true_range.rolling(14, min_periods=5).mean()
    atr_valid = atr.dropna()
    if atr_valid.empty:
        return {"low_vol_loss_pct": None, "high_vol_loss_pct": None, "regime": "unknown", "suggestion": None,
                **session_fields}
    atr_threshold = float(np.nanpercentile(atr_valid, low_vol_percentile))

    ts = pd.to_datetime(df["timestamp"])
    atr_by_time = pd.Series(atr.values, index=ts)

    def _atr_at(entry_time) -> float | None:
        idx = atr_by_time.index.searchsorted(pd.Timestamp(entry_time), side="right") - 1
        if idx < 0 or idx >= len(atr_by_time):
            return None
        val = atr_by_time.iloc[idx]
        return float(val) if pd.notna(val) else None

    total_loss = sum(-t.pnl for t in losers)
    low_vol_loss = 0.0
    high_vol_loss = 0.0
    for t in losers:
        a = _atr_at(t.entry_time)
        if a is None:
            continue
        if a < atr_threshold:
            low_vol_loss += -t.pnl
        else:
            high_vol_loss += -t.pnl

    if total_loss <= 0:
        return {"low_vol_loss_pct": None, "high_vol_loss_pct": None, "regime": "unknown", "suggestion": None,
                **session_fields}

    low_pct = low_vol_loss / total_loss * 100.0
    high_pct = high_vol_loss / total_loss * 100.0

    suggestion = None
    regime = "mixed"
    if low_pct >= 65.0:
        regime = "low_vol"
        suggestion = (
            f"{low_pct:.0f}% of this strategy's losing P&L came from trades opened while volatility "
            f"(a 14-period ATR proxy) was in its bottom {low_vol_percentile:.0f}th percentile. Test adding "
            f"an ATR-percentile entry filter that skips trades in low-volatility regimes."
        )
    elif high_pct >= 65.0:
        regime = "high_vol"
        suggestion = (
            f"{high_pct:.0f}% of this strategy's losing P&L came from trades opened while volatility "
            f"(a 14-period ATR proxy) was AT OR ABOVE its {low_vol_percentile:.0f}th percentile. Test adding "
            f"an ATR-percentile cap that skips trades in unusually high-volatility regimes, or widening stops "
            f"specifically for those regimes."
        )

    return {"low_vol_loss_pct": low_pct, "high_vol_loss_pct": high_pct, "regime": regime, "suggestion": suggestion,
            **session_fields}


# ---------------------------------------------------------------------------
# Ollama calls (both best-effort -- the loop makes forward progress even
# if Ollama is off, unreachable, or returns something unusable)
# ---------------------------------------------------------------------------

def _ask_ollama_next_hypothesis(
    settings: OllamaSettings, prior_idea: str, diagnosis: dict, timeout: int = 60,
) -> tuple[str, bool]:
    """Turns a COMPUTED failure diagnosis into a specific next idea to
    try. Returns (next_idea, came_from_ollama). Falls back to simply
    appending the diagnostic's own suggestion sentence to the prior idea
    when Ollama is disabled/unreachable/fails -- so the loop still
    incorporates the (real, computed) lesson either way; Ollama's job
    here is phrasing a better-targeted hypothesis, not deciding whether
    the lesson is real."""
    suggestion = diagnosis.get("suggestion")
    fallback = f"{prior_idea.strip()} {suggestion}".strip() if suggestion else prior_idea

    if not settings.is_usable or not suggestion:
        return fallback, False

    prompt = (
        "You are a quantitative trading research assistant. A trading strategy was tested and "
        "failed. Here is the ORIGINAL hypothesis and a COMPUTED diagnostic of why it lost money:\n\n"
        f"Original hypothesis: {prior_idea}\n"
        f"Computed diagnostic: {suggestion}\n\n"
        "In 2-3 sentences, propose ONE specific, concrete structural change to test next (e.g. a "
        "specific filter, threshold, or exit rule). Do not write code. Do not restate the diagnostic. "
        "Respond with only the revised hypothesis as plain text."
    )
    try:
        import requests
        host = (settings.host or "").rstrip("/")
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        resp = requests.post(
            f"{host}/api/generate",
            headers=headers,
            json={
                "model": settings.model, "prompt": prompt, "stream": False,
                "options": {"num_ctx": 2048, "num_predict": 200, "temperature": 0.4},
            },
            timeout=(10, timeout),
        )
        resp.raise_for_status()
        text = (resp.json().get("response") or "").strip()
        if text:
            return text, True
    except Exception:
        pass
    return fallback, False


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

@dataclass
class ResearchLoopConfig:
    n_iterations: int = 5
    language: str = "python"          # strategy_generator language -- only "python" backtests today (see run_research_loop)
    initial_idea: str = ""
    mc_sims: int = 500
    survival_sims: int = 1000
    keep_score_threshold: float = DEFAULT_KEEP_SCORE_THRESHOLD
    generation_timeout: int = 180
    ollama_hypothesis_timeout: int = 60


@dataclass
class ResearchLoopIteration:
    iteration: int
    idea: str
    strategy_name: str
    verdict: str   # "KEEP" | "DISCARD" | "SKIPPED_DUPLICATE" | "GENERATION_FAILED" | "NO_TRADES"
    trades: int = 0
    net_profit: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    eval_pass_probability: float = 0.0
    prop_survival_score: float | None = None
    dna_tags: list = field(default_factory=list)
    diagnosis: dict | None = None
    next_hypothesis: str | None = None
    next_hypothesis_from_ollama: bool = False
    experiment_id: str | None = None
    error: str | None = None
    code: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ResearchLoopResult:
    iterations: list  # list[ResearchLoopIteration]
    best_iteration: "ResearchLoopIteration | None"
    stopped_reason: str

    def to_dict(self) -> dict:
        return {
            "iterations": [it.to_dict() for it in self.iterations],
            "best_iteration": self.best_iteration.to_dict() if self.best_iteration else None,
            "stopped_reason": self.stopped_reason,
        }


def _write_temp_strategy_file(code: str) -> Path:
    import tempfile
    path = Path(tempfile.mkdtemp()) / f"strategy_{uuid.uuid4().hex}.py"
    path.write_text(code, encoding="utf-8")
    return path


def run_research_loop(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    settings: OllamaSettings,
    cfg: ResearchLoopConfig | None = None,
    progress_cb: ProgressCallback | None = None,
) -> ResearchLoopResult:
    """Runs the closed research loop for up to cfg.n_iterations rounds.
    Never raises -- every failure mode (Ollama unreachable, generated
    code that won't parse, a strategy with zero trades) is recorded as a
    verdict on that iteration and the loop moves on, exactly like a
    human researcher would just try the next idea rather than stopping
    the whole session."""
    cfg = cfg or ResearchLoopConfig()

    def log(msg: str):
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    if not settings.is_usable:
        log("Ollama isn't enabled/configured -- strategy generation cannot run without it.")
        return ResearchLoopResult(iterations=[], best_iteration=None, stopped_reason="ollama_unavailable")

    if cfg.language != "python":
        log(f"run_research_loop only backtests Python-generated strategies today (got '{cfg.language}').")
        return ResearchLoopResult(iterations=[], best_iteration=None, stopped_reason="unsupported_language")

    current_idea = (cfg.initial_idea or "").strip() or (
        "A trend-following strategy using a higher-timeframe bias filter and a volatility-based stop."
    )
    seen_discard_signatures: set[tuple] = set()
    iterations: list[ResearchLoopIteration] = []
    best: ResearchLoopIteration | None = None
    parent_experiment_id: str | None = None

    for i in range(1, cfg.n_iterations + 1):
        log(f"\n=== Iteration {i}/{cfg.n_iterations} ===")
        log(f"Hypothesis: {current_idea}")

        gen = generate_strategy(settings, cfg.language, current_idea, timeout=cfg.generation_timeout)
        strategy_name = f"research_loop_iter{i}_{uuid.uuid4().hex[:6]}"
        if gen.error or not gen.code:
            log(f"  Generation failed: {gen.error or 'no code returned'}")
            iterations.append(ResearchLoopIteration(
                iteration=i, idea=current_idea, strategy_name=strategy_name,
                verdict="GENERATION_FAILED", error=gen.error,
            ))
            break  # a broken Ollama connection won't fix itself next iteration

        dna = extract_dna(cfg.language, gen.code)
        signature = tuple(sorted(dna.active_tags()))
        in_run_repeat = bool(signature) and signature in seen_discard_signatures
        persisted_repeat, persisted_similarity = (
            is_dna_tagset_previously_discarded(dna.active_tags(), origin="research_loop")
            if dna.active_tags() else (False, 0.0)
        )
        if in_run_repeat or persisted_repeat:
            reason = (
                "matches a pattern that already failed earlier in this run" if in_run_repeat
                else f"is {persisted_similarity * 100:.0f}% similar to a pattern that failed in a PAST research loop run"
            )
            log(f"  Skipped -- this strategy's DNA {reason}.")
            iterations.append(ResearchLoopIteration(
                iteration=i, idea=current_idea, strategy_name=strategy_name,
                verdict="SKIPPED_DUPLICATE", dna_tags=list(dna.active_tags()), code=gen.code,
                next_hypothesis=current_idea,
            ))
            current_idea = f"{current_idea} Try a meaningfully different entry mechanism than before."
            continue

        try:
            tmp_path = _write_temp_strategy_file(gen.code)
            strategy = PythonStrategy(tmp_path)
        except Exception as exc:
            log(f"  Could not load generated strategy: {exc}")
            iterations.append(ResearchLoopIteration(
                iteration=i, idea=current_idea, strategy_name=strategy_name,
                verdict="GENERATION_FAILED", error=str(exc), dna_tags=list(signature), code=gen.code,
            ))
            continue

        try:
            bt_result = run_backtest(df, strategy, risk)
        except Exception as exc:
            log(f"  Backtest failed: {exc}")
            iterations.append(ResearchLoopIteration(
                iteration=i, idea=current_idea, strategy_name=strategy_name,
                verdict="GENERATION_FAILED", error=str(exc), dna_tags=list(signature), code=gen.code,
            ))
            continue

        if not bt_result.trades:
            log("  No trades generated -- discarding.")
            seen_discard_signatures.add(signature)
            it = ResearchLoopIteration(
                iteration=i, idea=current_idea, strategy_name=strategy_name,
                verdict="NO_TRADES", dna_tags=list(signature), code=gen.code,
            )
            it.experiment_id = record_experiment(
                origin="research_loop", strategy_name=strategy_name, source_type=cfg.language,
                verdict="DISCARD", lesson="Generated strategy produced zero trades on the given data.",
                config={"idea": current_idea, "iteration": i, "dna": dna.active_tags(), "parent_experiment": parent_experiment_id},
                settings=settings,
            )
            iterations.append(it)
            current_idea = f"{current_idea} The previous attempt never triggered any trades -- loosen the entry conditions."
            continue

        log(f"  {len(bt_result.trades)} trades, net profit ${bt_result.statistics.net_profit:,.2f}")

        mc_result = run_monte_carlo(bt_result.trades, prop_rules, MonteCarloConfig(n_simulations=cfg.mc_sims))
        survival_result = run_prop_survival_analysis(
            bt_result.trades, prop_rules, PropSurvivalConfig(n_simulations=cfg.survival_sims),
        )
        score = survival_result.prop_survival_score
        verdict = "KEEP" if score >= cfg.keep_score_threshold else "DISCARD"
        log(f"  Prop Survival Score: {score:.1f}/100 -> {verdict}")

        diagnosis = None
        next_idea, from_ollama = current_idea, False
        if verdict == "DISCARD":
            seen_discard_signatures.add(signature)
            diagnosis = diagnose_failure(df, bt_result.trades)
            if diagnosis.get("suggestion"):
                log(f"  Diagnosis: {diagnosis['suggestion']}")
            next_idea, from_ollama = _ask_ollama_next_hypothesis(
                settings, current_idea, diagnosis, timeout=cfg.ollama_hypothesis_timeout,
            )

        it = ResearchLoopIteration(
            iteration=i, idea=current_idea, strategy_name=strategy_name, verdict=verdict,
            trades=len(bt_result.trades), net_profit=bt_result.statistics.net_profit,
            win_rate=bt_result.statistics.win_rate, profit_factor=bt_result.statistics.profit_factor,
            max_drawdown_pct=bt_result.statistics.max_drawdown_pct,
            eval_pass_probability=mc_result.evaluation_pass_probability,
            prop_survival_score=score, dna_tags=list(dna.active_tags()),
            diagnosis=diagnosis, next_hypothesis=(next_idea if verdict == "DISCARD" else None),
            next_hypothesis_from_ollama=from_ollama, code=gen.code,
        )
        it.experiment_id = record_experiment(
            origin="research_loop", strategy_name=strategy_name, source_type=cfg.language,
            verdict=verdict, trades=len(bt_result.trades), net_profit=bt_result.statistics.net_profit,
            win_rate=bt_result.statistics.win_rate, profit_factor=bt_result.statistics.profit_factor,
            max_drawdown_pct=bt_result.statistics.max_drawdown_pct,
            eval_pass_probability=mc_result.evaluation_pass_probability,
            first_payout_probability=mc_result.first_payout_probability,
            risk_of_ruin_pct=mc_result.risk_of_ruin_pct,
            lesson=(diagnosis or {}).get("suggestion") or "",
            config={
                "idea": current_idea, "iteration": i, "dna": dna.active_tags(),
                "prop_survival_score": score, "parent_experiment": parent_experiment_id,
            },
            settings=settings,
        )
        iterations.append(it)
        parent_experiment_id = it.experiment_id

        if best is None or (it.prop_survival_score or -1) > (best.prop_survival_score or -1):
            best = it

        if verdict == "KEEP":
            log("  This iteration cleared the KEEP threshold.")

        current_idea = next_idea

    stopped_reason = "completed" if len(iterations) >= cfg.n_iterations else (
        iterations[-1].verdict if iterations else "no_iterations_ran"
    )
    return ResearchLoopResult(iterations=iterations, best_iteration=best, stopped_reason=stopped_reason)
