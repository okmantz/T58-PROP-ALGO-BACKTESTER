"""Tests for app.ai.research_agent -- the ReAct prompt/parsing pure
functions, the tool registry's real integration with the actual
backtest/prop/Monte Carlo engine (no mocking -- this is the "engine is
the authority" guarantee), and the agent loop's step-by-step behavior
against a stubbed Ollama HTTP call."""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from app.ai import research_agent
from app.ai.ollama_settings import OllamaSettings
from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy


def _trending_df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        drift = 0.00003 + rng.normal(0, 0.00004)
        o = price
        c = o + drift
        h = max(o, c) + abs(rng.normal(0, 0.00002))
        l = min(o, c) - abs(rng.normal(0, 0.00002))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_strategy():
    return ManualStrategy({
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
    })


def _ctx():
    return research_agent.ResearchAgentContext(
        df=_trending_df(), strategy_builder=_sma_strategy, strategy_name="sma cross",
        source_type="manual", risk=RiskConfig(initial_balance=10_000, pip_size=0.0001),
        prop_rules=PropRules(account_size=10_000), instrument="EURUSD",
    )


# ---------------------------------------------------------------------------
# Tool registry -- real engine integration, no mocking
# ---------------------------------------------------------------------------

def test_run_backtest_tool_returns_real_stats():
    ctx = _ctx()
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    result = tools["run_backtest"].fn({})
    assert "total_trades" in result
    assert result["total_trades"] >= 1


def test_run_backtest_caches_across_calls():
    ctx = _ctx()
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    first = tools["run_backtest"].fn({})
    second = tools["run_backtest"].fn({})
    assert first == second
    assert ctx.cache_get("__baseline_bt__") is not None


def test_run_prop_simulation_tool():
    ctx = _ctx()
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    result = tools["run_prop_simulation"].fn({})
    assert "passed_evaluation" in result
    assert "final_balance" in result


def test_run_monte_carlo_tool_clamps_simulation_count():
    ctx = _ctx()
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    result = tools["run_monte_carlo"].fn({"n_simulations": 999999})
    assert result["n_simulations"] == 20_000
    assert "evaluation_pass_probability" in result


def test_run_cost_stress_tool():
    ctx = _ctx()
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    result = tools["run_cost_stress"].fn({})
    assert "cost_ladder" in result
    assert len(result["cost_ladder"]) == 4


def test_run_walk_forward_tool_handles_insufficient_data():
    ctx = research_agent.ResearchAgentContext(
        df=_trending_df(n=20), strategy_builder=_sma_strategy, strategy_name="sma cross",
        source_type="manual", risk=RiskConfig(), prop_rules=PropRules(),
    )
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    result = tools["run_walk_forward"].fn({"n_folds": 4})
    assert "warning" in result


def test_search_research_tool_requires_query():
    ctx = _ctx()
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    result = tools["search_research"].fn({})
    assert "error" in result


def test_compare_strategies_tool_requires_strategy_files():
    ctx = _ctx()
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    result = tools["compare_strategies"].fn({})
    assert "error" in result


def test_safe_call_never_raises_on_tool_exception():
    ctx = _ctx()

    def _boom(args):
        raise RuntimeError("kaboom")

    tool = research_agent.AgentTool("boom", "desc", "{}", _boom)
    result = research_agent._safe_call(tool, {})
    assert "error" in result
    assert "kaboom" in result["error"]


# ---------------------------------------------------------------------------
# ReAct step parsing -- pure functions
# ---------------------------------------------------------------------------

def test_parse_step_extracts_action_and_json_input():
    text = (
        'Thought: I should check the baseline first.\n'
        'Action: run_backtest\n'
        'Action Input: {}\n'
    )
    step = research_agent.parse_step(text)
    assert step.action == "run_backtest"
    assert step.action_input == {}
    assert "baseline" in step.thought


def test_parse_step_extracts_final_answer():
    text = "Thought: I have enough evidence.\nFinal Answer: This strategy looks weak because of X."
    step = research_agent.parse_step(text)
    assert step.final_answer == "This strategy looks weak because of X."
    assert step.action is None


def test_parse_step_handles_markdown_fenced_json():
    text = (
        "Thought: checking research.\n"
        "Action: search_research\n"
        "Action Input: ```json\n{\"query\": \"volatility\"}\n```\n"
    )
    step = research_agent.parse_step(text)
    assert step.action == "search_research"
    assert step.action_input == {"query": "volatility"}


def test_parse_step_missing_action_is_malformed():
    step = research_agent.parse_step("Thought: I'm thinking about it.")
    assert step.malformed_reason is not None


def test_parse_step_invalid_json_is_malformed():
    text = "Thought: x\nAction: run_backtest\nAction Input: {not valid json}\n"
    step = research_agent.parse_step(text)
    assert step.action == "run_backtest"
    assert step.malformed_reason is not None


def test_parse_step_empty_response_is_malformed():
    step = research_agent.parse_step("")
    assert step.malformed_reason is not None


def test_build_system_prompt_lists_every_tool():
    ctx = _ctx()
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    prompt = research_agent.build_system_prompt("sma cross", "manual", "EURUSD", tools, "Is this robust?")
    for name in tools:
        assert name in prompt
    assert "cannot write or change any code" in prompt


# ---------------------------------------------------------------------------
# Agent loop -- stubbed Ollama HTTP responses
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, text):
        self._text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self._text}


def _stub_requests(monkeypatch, responses):
    """responses: list of raw text strings returned on successive POST calls."""
    call_count = {"n": 0}

    def fake_post(*args, **kwargs):
        i = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return _FakeResponse(responses[i])

    fake_requests = types.SimpleNamespace(
        post=fake_post,
        exceptions=types.SimpleNamespace(ConnectionError=ConnectionError, Timeout=TimeoutError),
    )
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)
    return call_count


def test_agent_disabled_settings_returns_error_immediately():
    agent = research_agent.ResearchAgent(OllamaSettings(enabled=False))
    result = agent.run("Is this strategy robust?", _ctx())
    assert result.error is not None
    assert result.steps == []


def test_agent_runs_one_tool_then_final_answer(monkeypatch):
    responses = [
        'Thought: check baseline.\nAction: run_backtest\nAction Input: {}\n',
        'Thought: enough evidence.\nFinal Answer: The strategy trades and has a positive profit factor.\n',
    ]
    _stub_requests(monkeypatch, responses)
    agent = research_agent.ResearchAgent(OllamaSettings(enabled=True, host="http://localhost:11434"), max_steps=5)
    result = agent.run("Is this strategy robust?", _ctx())
    assert result.final_answer == "The strategy trades and has a positive profit factor."
    assert len(result.steps) == 2
    assert result.steps[0].action == "run_backtest"
    assert result.steps[0].observation is not None
    assert "total_trades" in result.steps[0].observation


def test_agent_unknown_tool_reports_error_observation(monkeypatch):
    responses = [
        'Thought: try something weird.\nAction: hack_the_mainframe\nAction Input: {}\n',
        'Thought: give up gracefully.\nFinal Answer: No conclusion.\n',
    ]
    _stub_requests(monkeypatch, responses)
    agent = research_agent.ResearchAgent(OllamaSettings(enabled=True, host="http://localhost:11434"), max_steps=5)
    result = agent.run("question", _ctx())
    assert "error" in result.steps[0].observation
    assert result.final_answer == "No conclusion."


def test_agent_stops_after_two_consecutive_malformed_responses(monkeypatch):
    responses = ["garbage response with no keywords", "still garbage"]
    _stub_requests(monkeypatch, responses)
    agent = research_agent.ResearchAgent(OllamaSettings(enabled=True, host="http://localhost:11434"), max_steps=5)
    result = agent.run("question", _ctx())
    assert result.error is not None
    assert "malformed" in result.error.lower()


def test_agent_respects_max_steps(monkeypatch):
    # Every response is a valid tool call, never a Final Answer -- the
    # agent must stop at max_steps rather than looping forever.
    responses = ['Thought: again.\nAction: run_backtest\nAction Input: {}\n']
    _stub_requests(monkeypatch, responses)
    agent = research_agent.ResearchAgent(OllamaSettings(enabled=True, host="http://localhost:11434"), max_steps=3)
    result = agent.run("question", _ctx())
    assert result.final_answer is None
    assert len(result.steps) == 3
    assert "3 steps" in result.stopped_reason


def test_agent_ollama_connection_error_is_reported(monkeypatch):
    def _raise(*a, **k):
        raise ConnectionError("down")

    fake_requests = types.SimpleNamespace(
        post=_raise,
        exceptions=types.SimpleNamespace(ConnectionError=ConnectionError, Timeout=TimeoutError),
    )
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)
    agent = research_agent.ResearchAgent(OllamaSettings(enabled=True, host="http://localhost:11434"))
    result = agent.run("question", _ctx())
    assert result.error is not None
    assert "reach Ollama" in result.error


def test_agent_run_result_transcript_includes_steps_and_final_answer(monkeypatch):
    responses = [
        'Thought: check.\nAction: run_backtest\nAction Input: {}\n',
        'Thought: done.\nFinal Answer: Looks fine.\n',
    ]
    _stub_requests(monkeypatch, responses)
    agent = research_agent.ResearchAgent(OllamaSettings(enabled=True, host="http://localhost:11434"))
    result = agent.run("question", _ctx())
    text = result.transcript()
    assert "run_backtest" in text
    assert "Looks fine." in text
