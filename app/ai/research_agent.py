"""T58 AI Research Agent -- the "research analyst" upgrade to the AI
Assist feature.

Every other AI touchpoint in this app (app.ai.ollama_client,
app.ai.strategy_generator) is a single request/response call: describe a
situation, get numbers or code back once. This module is the agentic
version described in the T58 AI Research Engine plan -- Ollama is given
a fixed toolbox of READ-ONLY analysis actions that run the app's OWN,
already-validated engine (backtest, walk-forward, Monte Carlo, regime
test, sensitivity sweep, cost-ladder stress, prop-firm simulation), plus
two research-memory lookups (the paper library, the experiment history),
and reasons over the OBSERVATIONS those tools return across several
steps before giving a final answer.

The critical safety property, straight from the plan: "don't let the AI
determine whether a strategy is good purely from its own judgment -- make
the quantitative engine the authority." This module enforces that
structurally, not just by convention:
  - Every tool wraps a function this app already trusts elsewhere in the
    codebase (the same run_backtest/run_monte_carlo/etc. used by Full
    Pipeline, Search Lab, and the Validation Lab tabs) -- the agent can
    never invent a number, only ask the real engine to compute one.
  - The tool registry is READ-ONLY analysis. There is no
    "edit_strategy_code" or "apply_parameters" tool here -- exactly the
    same boundary app.ai.ollama_client documents ("propose numbers,
    never code, always re-validated"). An agent session produces a
    diagnosis and a recommendation IN TEXT; turning that recommendation
    into a new tested strategy still goes through Quick Optimize / the
    Iterative Refinement GA / Full Pipeline, same as a human-typed idea
    would.
  - Every tool call result is deterministic engine output; the model's
    only creative contribution is choosing which tool to call next and
    how to summarize the pattern of results at the end.

Protocol: a plain-text ReAct loop (Thought / Action / Action Input /
Observation, ending in Final Answer) rather than any model's native
"function calling" API -- this keeps the same host/model/timeout/parsing
architecture as app.ai.ollama_client and app.ai.strategy_generator (a
local model behind a bare /api/generate call, not every model's chat
template supports native tool-calling), and keeps the whole loop testable
by mocking one HTTP call per step, exactly like those two modules already
are.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from app.ai.ollama_settings import OllamaSettings
from app.backtest.engine import BacktestResult, run_backtest
from app.backtest.risk import RiskConfig
from app.backtest.statistics import compute_cost_ladder
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.strategy.base import Strategy

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_STEPS = 6
DEFAULT_MAX_TOTAL_SECONDS = 600

ProgressCallback = Callable[[str], None]


# ---------------------------------------------------------------------------
# Context: everything a tool call needs, bound once per agent run
# ---------------------------------------------------------------------------

@dataclass
class ResearchAgentContext:
    df: pd.DataFrame
    strategy_builder: Callable[[], Strategy]   # zero-arg, returns a FRESH Strategy instance each call
    strategy_name: str
    source_type: str
    risk: RiskConfig
    prop_rules: PropRules
    instrument: str = ""
    tmp_dir: Path | None = None

    # Simple per-run memoization so the agent can call the same tool
    # more than once (e.g. after changing nothing) without re-running an
    # expensive Monte Carlo -- keyed by (tool_name, sorted args) so a
    # DIFFERENT n_simulations/n_folds argument still triggers a fresh run.
    _cache: dict = field(default_factory=dict, repr=False)

    def cache_get(self, key: str):
        return self._cache.get(key)

    def cache_set(self, key: str, value) -> None:
        self._cache[key] = value


def _cache_key(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


# ---------------------------------------------------------------------------
# Tools -- each takes (ctx, args) -> JSON-serializable dict, never raises
# ---------------------------------------------------------------------------

def _get_baseline_backtest(ctx: ResearchAgentContext) -> BacktestResult:
    cached = ctx.cache_get("__baseline_bt__")
    if cached is not None:
        return cached
    bt = run_backtest(ctx.df, ctx.strategy_builder(), ctx.risk)
    ctx.cache_set("__baseline_bt__", bt)
    return bt


def _stats_summary(stats: dict) -> dict:
    keep = (
        "total_trades", "net_profit", "return_pct", "win_rate", "profit_factor",
        "expectancy", "average_r", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_drawdown_pct", "max_losing_streak",
    )
    return {k: round(stats[k], 4) if isinstance(stats.get(k), float) else stats.get(k) for k in keep}


def _tool_run_backtest(ctx: ResearchAgentContext, args: dict) -> dict:
    bt = _get_baseline_backtest(ctx)
    if bt.statistics.total_trades == 0:
        return {"warning": "Zero trades produced on this dataset -- every other tool will also be uninformative "
                            "until this is resolved. Check the strategy's entry logic and the loaded data range."}
    result = _stats_summary(bt.statistics.to_dict())
    if bt.warnings:
        result["engine_warnings"] = bt.warnings[:5]
    return result


def _tool_run_prop_simulation(ctx: ResearchAgentContext, args: dict) -> dict:
    bt = _get_baseline_backtest(ctx)
    if not bt.trades:
        return {"error": "No trades to simulate -- run_backtest first."}
    pnls = [t.pnl for t in bt.trades]
    dates = [t.entry_time for t in bt.trades]
    sim = simulate_account(pnls, dates, ctx.prop_rules)
    out = summarize_single_run(sim)
    out["passed_evaluation"] = sim.passed_evaluation
    out["failed"] = sim.failed
    out["failure_reason"] = sim.failure_reason
    out["max_drawdown_pct_reached"] = round(sim.max_drawdown_pct_reached, 3)
    out["final_balance"] = round(sim.final_balance, 2)
    return out


def _tool_run_monte_carlo(ctx: ResearchAgentContext, args: dict) -> dict:
    bt = _get_baseline_backtest(ctx)
    if not bt.trades:
        return {"error": "No trades to simulate -- run_backtest first."}
    n_sims = int(args.get("n_simulations", 1000))
    n_sims = max(100, min(n_sims, 20_000))
    key = _cache_key("monte_carlo", {"n": n_sims})
    cached = ctx.cache_get(key)
    if cached is not None:
        return cached
    cfg = MonteCarloConfig(n_simulations=n_sims)
    mc: MonteCarloResult = run_monte_carlo(bt.trades, ctx.prop_rules, cfg)
    out = {
        "n_simulations": mc.n_simulations,
        "evaluation_pass_probability": round(mc.evaluation_pass_probability, 2),
        "first_payout_probability": round(mc.first_payout_probability, 2),
        "failure_before_payout_probability": round(mc.failure_before_payout_probability, 2),
        "risk_of_ruin_pct": round(mc.risk_of_ruin_pct, 2),
        "median_return_pct": round(mc.median_return_pct, 2),
        "median_drawdown_pct": round(mc.median_drawdown_pct, 2),
        "p95_drawdown_pct": round(mc.p95_drawdown_pct, 2),
        "worst_drawdown_pct": round(mc.worst_drawdown_pct, 2),
    }
    ctx.cache_set(key, out)
    return out


def _tool_run_walk_forward(ctx: ResearchAgentContext, args: dict) -> dict:
    from app.search.robustness import run_walk_forward

    n_folds = int(args.get("n_folds", 4))
    metric = str(args.get("metric", "profit_factor"))
    result = run_walk_forward(ctx.df, ctx.strategy_builder, ctx.risk, n_folds=n_folds, metric=metric)
    if result is None:
        return {"warning": "Not enough data to build meaningful walk-forward folds -- unproven, not necessarily bad."}
    return {
        "metric": metric,
        "mean_train_metric": round(result.mean_train_metric, 4),
        "mean_test_metric": round(result.mean_test_metric, 4),
        "walk_forward_efficiency": round(result.walk_forward_efficiency, 4),
        "is_stable": result.is_stable,
        "n_folds_evaluated": len(result.folds),
    }


def _tool_run_regime_analysis(ctx: ResearchAgentContext, args: dict) -> dict:
    from app.validation.regime_testing import run_regime_test

    n_regimes = int(args.get("n_regimes", 3))
    result = run_regime_test(ctx.df, ctx.strategy_builder, ctx.risk, n_regimes=n_regimes)
    if result is None:
        return {"warning": "Not enough data to split into volatility regimes -- unproven, not necessarily bad."}
    return {
        "n_profitable_buckets": result.n_profitable_buckets,
        "n_buckets": result.n_buckets,
        "is_regime_stable": result.is_regime_stable,
        "buckets": [
            {
                "regime": b.label, "n_trades": b.n_trades, "net_profit": round(b.net_profit, 2),
                "profit_factor": round(b.profit_factor, 3), "is_profitable": b.is_profitable,
            }
            for b in result.buckets
        ],
    }


def _tool_run_parameter_sensitivity(ctx: ResearchAgentContext, args: dict) -> dict:
    from app.optimize.parameter_space import RefinementError
    from app.validation.sensitivity import compute_1d_sensitivity

    metric = str(args.get("metric", "profit_factor"))
    mc_cfg = MonteCarloConfig(n_simulations=200)
    try:
        results = compute_1d_sensitivity(
            ctx.df, ctx.strategy_builder(), ctx.risk, ctx.prop_rules, mc_cfg, metric=metric, tmp_dir=ctx.tmp_dir,
        )
    except RefinementError as exc:
        return {"warning": str(exc)}
    if not results:
        return {"warning": "No tunable numeric parameters found for a sensitivity sweep."}
    return {
        "metric": metric,
        "parameters": [
            {
                "parameter": r.gene_label, "base_value": r.base_value,
                "max_pct_drop_between_adjacent_steps": round(r.max_pct_drop_between_adjacent_steps, 1),
                "cliff_detected": r.cliff_detected,
            }
            for r in results
        ],
        "note": "cliff_detected=true means small nearby parameter values collapse the metric sharply -- "
                "a strategy whose edge only exists at one exact value is a strong overfitting signal.",
    }


def _tool_run_cost_stress(ctx: ResearchAgentContext, args: dict) -> dict:
    bt = _get_baseline_backtest(ctx)
    if not bt.trades:
        return {"error": "No trades to stress-test -- run_backtest first."}
    ladder = compute_cost_ladder(bt.trades)
    return {"cost_ladder": [
        {
            "extra_cost_pct_per_trade": rung.get("extra_cost_pct_per_trade"),
            "net_profit": round(rung.get("net_profit", 0.0), 2),
            "profit_factor": round(rung.get("profit_factor", 0.0), 3),
        }
        for rung in ladder
    ]}


def _tool_search_research(ctx: ResearchAgentContext, args: dict, settings: OllamaSettings) -> dict:
    from app.ai.research_library import find_relevant_excerpts

    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "search_research requires a non-empty 'query' argument."}
    excerpts = find_relevant_excerpts(query, max_excerpts=int(args.get("max_excerpts", 3)), settings=settings)
    if not excerpts:
        return {"result": "No matching excerpts found in the research/ library for this query."}
    return {"excerpts": excerpts}


def _tool_search_experiments(ctx: ResearchAgentContext, args: dict, settings: OllamaSettings) -> dict:
    from app.ai.experiment_memory import search_similar_experiments

    query = str(args.get("query", "")).strip() or ctx.strategy_name
    hits = search_similar_experiments(query, settings=settings, max_results=int(args.get("max_results", 5)))
    if not hits:
        return {"result": "No similar past experiments recorded yet."}
    return {"similar_experiments": [
        {
            "strategy_name": h.strategy_name, "verdict": h.verdict, "net_profit": round(h.net_profit, 2),
            "eval_pass_probability": round(h.eval_pass_probability, 1), "lesson": h.lesson,
            "similarity": round(h.similarity, 3) if h.similarity is not None else None,
        }
        for h in hits
    ]}


def _tool_compare_strategies(ctx: ResearchAgentContext, args: dict) -> dict:
    """Compares the bound strategy's baseline stats against up to 3
    other strategies already saved in the Strategy Library, by filename.
    Read-only: loads and backtests each named strategy fresh, never
    modifies the library."""
    from app.search.strategy_space import build_strategy_from_spec
    from app.strategy.library import load_strategy_text

    names = args.get("strategy_files", [])
    if not isinstance(names, list) or not names:
        return {"error": "compare_strategies requires a 'strategy_files' list, e.g. "
                          "[{'strategy_type': 'python', 'filename': 'ema_pullback.py'}]."}
    rows = [{
        "strategy": ctx.strategy_name,
        **_stats_summary(_get_baseline_backtest(ctx).statistics.to_dict()),
    }]
    for entry in names[:3]:
        try:
            stype = entry["strategy_type"]
            fname = entry["filename"]
            code_text = load_strategy_text(stype, fname)
            spec = {"source_type": stype, "code_text": code_text}
            candidate = build_strategy_from_spec(spec, tmp_dir=ctx.tmp_dir)
            bt = run_backtest(ctx.df, candidate, ctx.risk)
            row = {"strategy": fname, **_stats_summary(bt.statistics.to_dict())}
        except Exception as exc:
            row = {"strategy": entry, "error": str(exc)}
        rows.append(row)
    return {"comparison": rows}


@dataclass
class AgentTool:
    name: str
    description: str
    args_schema: str  # human-readable, shown to the model in the system prompt
    fn: Callable[[ResearchAgentContext, dict], dict]


def build_tool_registry(ctx: ResearchAgentContext, settings: OllamaSettings) -> dict[str, AgentTool]:
    return {
        "run_backtest": AgentTool(
            "run_backtest",
            "Runs the bound strategy through a full historical backtest and returns its core "
            "performance statistics. Always call this first.",
            "{} (no arguments)", lambda a: _tool_run_backtest(ctx, a),
        ),
        "run_prop_simulation": AgentTool(
            "run_prop_simulation",
            "Runs the ONE historical trade sequence through the configured prop-firm rules "
            "(evaluation pass/fail, drawdown, first payout).",
            "{} (no arguments)", lambda a: _tool_run_prop_simulation(ctx, a),
        ),
        "run_monte_carlo": AgentTool(
            "run_monte_carlo",
            "Resamples the historical trades many times to estimate the PROBABILITY of passing "
            "the prop evaluation and reaching payout (not just the one historical outcome).",
            '{"n_simulations": 1000}', lambda a: _tool_run_monte_carlo(ctx, a),
        ),
        "run_walk_forward": AgentTool(
            "run_walk_forward",
            "Splits the data into chronological folds with NO re-tuning to check whether the "
            "strategy's edge holds up out-of-sample across several distinct periods.",
            '{"n_folds": 4, "metric": "profit_factor"}', lambda a: _tool_run_walk_forward(ctx, a),
        ),
        "run_regime_analysis": AgentTool(
            "run_regime_analysis",
            "Buckets the data into volatility regimes (e.g. low/medium/high ATR) and reports "
            "whether the strategy is profitable in each one, or only in a specific market type.",
            '{"n_regimes": 3}', lambda a: _tool_run_regime_analysis(ctx, a),
        ),
        "run_parameter_sensitivity": AgentTool(
            "run_parameter_sensitivity",
            "Sweeps each tunable numeric parameter near its current value to detect fragile "
            "'cliffs' -- a strong sign of overfitting to one exact value.",
            '{"metric": "profit_factor"}', lambda a: _tool_run_parameter_sensitivity(ctx, a),
        ),
        "run_cost_stress": AgentTool(
            "run_cost_stress",
            "Re-costs the same historical trades at increasing commission/slippage friction "
            "levels to see how much of the edge survives realistic costs.",
            "{} (no arguments)", lambda a: _tool_run_cost_stress(ctx, a),
        ),
        "compare_strategies": AgentTool(
            "compare_strategies",
            "Backtests other strategies already saved in the Strategy Library against the same "
            "data and compares their stats to this one.",
            '{"strategy_files": [{"strategy_type": "python", "filename": "other.py"}]}',
            lambda a: _tool_compare_strategies(ctx, a),
        ),
        "search_research": AgentTool(
            "search_research",
            "Searches the local research/ paper library (RAG over your own trading/quant papers) "
            "for excerpts relevant to a topic.",
            '{"query": "volatility breakout regime filtering"}',
            lambda a: _tool_search_research(ctx, a, settings),
        ),
        "search_experiments": AgentTool(
            "search_experiments",
            "Searches T58's own memory of every past strategy test for similar prior experiments "
            "and what happened to them (verdict, lesson learned).",
            '{"query": "gold liquidity sweep reversal"}',
            lambda a: _tool_search_experiments(ctx, a, settings),
        ),
    }


def _safe_call(tool: AgentTool, args: dict) -> dict:
    try:
        result = tool.fn(args)
        if not isinstance(result, dict):
            return {"error": f"Tool '{tool.name}' returned a non-dict result internally."}
        return result
    except Exception as exc:
        return {"error": f"Tool '{tool.name}' failed: {exc}"}


# ---------------------------------------------------------------------------
# ReAct prompt + parsing (pure functions -- unit-testable without Ollama)
# ---------------------------------------------------------------------------

def build_system_prompt(
    strategy_name: str, source_type: str, instrument: str, tools: dict[str, AgentTool], question: str,
) -> str:
    tool_lines = "\n".join(
        f"- {t.name}({t.args_schema}): {t.description}" for t in tools.values()
    )
    return f"""You are the T58 Research Analyst, investigating a trading strategy called \
"{strategy_name}" ({source_type}) on {instrument or "the loaded dataset"}.

You do not run backtests yourself and you cannot write or change any code. You can only call \
the tools below, which run T58's own validated backtesting engine and return real results. \
Treat every tool's numbers as ground truth -- never invent or guess a statistic instead of \
calling the tool that would produce it, and never claim a strategy is good or bad without \
having called at least run_backtest and one validation tool (Monte Carlo, walk-forward, \
regime analysis, sensitivity, or cost stress) to support the claim.

Available tools:
{tool_lines}

Research question: {question}

Respond in EXACTLY this format, one step at a time. Do not call more than one tool per step.

Thought: <your reasoning about what to check next, in one or two sentences>
Action: <one tool name from the list above>
Action Input: <a single-line JSON object with that tool's arguments>

Once you have enough evidence to answer the research question, respond instead with:

Thought: <your reasoning>
Final Answer: <a clear, evidence-based answer, citing the specific numbers the tools returned. \
If you are recommending a next step (e.g. a parameter to test), state it as a plain-language \
recommendation for a human to try via Quick Optimize / Iterative Refinement -- you cannot apply \
it yourself.>

Do not include any text outside this format. Do not repeat previous Thought/Action/Observation \
lines. Begin now with your first Thought.
"""


_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n\s*(?:Action:|Final Answer:)|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)")
_ACTION_INPUT_RE = re.compile(
    r"Action Input:\s*(?:```(?:json)?\s*)?(\{.*?\})\s*(?:```\s*)?(?=\n(?:Thought:|Observation:|Action:)|\Z)",
    re.DOTALL,
)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


@dataclass
class ParsedStep:
    thought: str = ""
    action: str | None = None
    action_input: dict | None = None
    final_answer: str | None = None
    malformed_reason: str | None = None


def parse_step(raw_text: str) -> ParsedStep:
    """Pure function: parses one model response into a ReAct step. Very
    tolerant of a local model's formatting quirks (extra whitespace,
    markdown emphasis around the keywords, a trailing code fence around
    the JSON) but never guesses past a genuinely missing Action/Action
    Input pair or unparsable JSON -- that's reported via
    `malformed_reason` so the caller can nudge the model rather than
    silently proceeding with garbage arguments."""
    if not raw_text or not raw_text.strip():
        return ParsedStep(malformed_reason="Model returned an empty response.")

    text = raw_text.strip()
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = _FINAL_ANSWER_RE.search(text)
    if final_match:
        return ParsedStep(thought=thought, final_answer=final_match.group(1).strip())

    action_match = _ACTION_RE.search(text)
    if not action_match:
        return ParsedStep(thought=thought, malformed_reason="No 'Action:' or 'Final Answer:' found in response.")

    input_match = _ACTION_INPUT_RE.search(text)
    if not input_match:
        return ParsedStep(
            thought=thought, action=action_match.group(1),
            malformed_reason="Found 'Action:' but no valid 'Action Input:' JSON object.",
        )

    raw_json = input_match.group(1).strip()
    raw_json = re.sub(r"^```(json)?|```$", "", raw_json, flags=re.MULTILINE).strip()
    try:
        parsed_args = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        return ParsedStep(
            thought=thought, action=action_match.group(1),
            malformed_reason=f"Action Input was not valid JSON: {raw_json[:200]!r}",
        )
    if not isinstance(parsed_args, dict):
        return ParsedStep(
            thought=thought, action=action_match.group(1),
            malformed_reason="Action Input JSON must be an object (dict), not a list or scalar.",
        )
    return ParsedStep(thought=thought, action=action_match.group(1), action_input=parsed_args)


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

@dataclass
class AgentStepLog:
    step_index: int
    thought: str
    action: str | None
    action_input: dict | None
    observation: dict | None
    note: str | None = None  # set for malformed-step nudges / errors


@dataclass
class AgentRunResult:
    steps: list[AgentStepLog] = field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None
    stopped_reason: str = ""

    def transcript(self) -> str:
        lines = []
        for s in self.steps:
            lines.append(f"Step {s.step_index}")
            if s.thought:
                lines.append(f"  Thought: {s.thought}")
            if s.action:
                lines.append(f"  Action: {s.action}({json.dumps(s.action_input or {})})")
            if s.observation is not None:
                lines.append(f"  Observation: {json.dumps(s.observation)[:1500]}")
            if s.note:
                lines.append(f"  Note: {s.note}")
            lines.append("")
        if self.final_answer:
            lines.append(f"Final Answer: {self.final_answer}")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


class ResearchAgent:
    def __init__(
        self,
        settings: OllamaSettings,
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_total_seconds: int = DEFAULT_MAX_TOTAL_SECONDS,
    ):
        self.settings = settings
        self.max_steps = max_steps
        self.timeout = timeout
        self.max_total_seconds = max_total_seconds

    def _call_ollama(self, prompt: str) -> tuple[str | None, str | None]:
        import requests

        host = (self.settings.host or "").rstrip("/")
        if not host:
            return None, "No Ollama host configured."
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        try:
            resp = requests.post(
                f"{host}/api/generate",
                headers=headers,
                json={"model": self.settings.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", ""), None
        except requests.exceptions.ConnectionError:
            return None, f"Couldn't reach Ollama at {host} (is it running?)."
        except requests.exceptions.Timeout:
            return None, f"Ollama at {host} didn't respond in time."
        except Exception as exc:
            return None, f"Ollama request failed: {exc}"

    def run(
        self,
        question: str,
        ctx: ResearchAgentContext,
        progress_cb: ProgressCallback | None = None,
    ) -> AgentRunResult:
        result = AgentRunResult()
        if not self.settings.is_usable:
            result.error = "AI Assist is not enabled -- turn it on and confirm TEST CONNECTION works first."
            return result

        tools = build_tool_registry(ctx, self.settings)
        system_prompt = build_system_prompt(ctx.strategy_name, ctx.source_type, ctx.instrument, tools, question)
        transcript_so_far = ""
        start = time.monotonic()
        retried_malformed = False

        for step_index in range(1, self.max_steps + 1):
            if time.monotonic() - start > self.max_total_seconds:
                result.stopped_reason = "Reached the maximum total time budget for this session."
                break

            prompt = system_prompt + "\n" + transcript_so_far
            raw_text, err = self._call_ollama(prompt)
            if err is not None:
                result.error = err
                break

            parsed = parse_step(raw_text)

            if parsed.final_answer is not None:
                result.final_answer = parsed.final_answer
                result.steps.append(AgentStepLog(step_index, parsed.thought, None, None, None))
                result.stopped_reason = "Agent produced a final answer."
                if progress_cb:
                    progress_cb(f"Step {step_index}: Final Answer.")
                break

            if parsed.malformed_reason is not None:
                log = AgentStepLog(step_index, parsed.thought, parsed.action, parsed.action_input, None,
                                    note=f"Malformed step: {parsed.malformed_reason}")
                result.steps.append(log)
                if progress_cb:
                    progress_cb(f"Step {step_index}: malformed response ({parsed.malformed_reason})")
                if retried_malformed:
                    result.error = "Model produced two malformed responses in a row -- stopping."
                    break
                retried_malformed = True
                transcript_so_far += (
                    f"\n{raw_text}\nObservation: Your last response could not be parsed "
                    f"({parsed.malformed_reason}). Respond using EXACTLY the Thought/Action/Action "
                    f"Input format described above, with valid single-line JSON for Action Input.\n"
                )
                continue

            retried_malformed = False
            tool = tools.get(parsed.action)
            if tool is None:
                observation = {"error": f"Unknown tool '{parsed.action}'. Available tools: {', '.join(tools)}"}
            else:
                observation = _safe_call(tool, parsed.action_input or {})

            result.steps.append(AgentStepLog(step_index, parsed.thought, parsed.action, parsed.action_input, observation))
            if progress_cb:
                progress_cb(f"Step {step_index}: {parsed.action}({json.dumps(parsed.action_input or {})})")

            transcript_so_far += (
                f"\nThought: {parsed.thought}\nAction: {parsed.action}\n"
                f"Action Input: {json.dumps(parsed.action_input or {})}\n"
                f"Observation: {json.dumps(observation)}\n"
            )
        else:
            result.stopped_reason = f"Reached the maximum of {self.max_steps} steps without a final answer."

        if result.final_answer is None and not result.error and not result.stopped_reason:
            result.stopped_reason = f"Reached the maximum of {self.max_steps} steps without a final answer."
        return result
