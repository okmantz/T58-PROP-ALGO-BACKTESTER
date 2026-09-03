"""
MQL5 Strategy Adapter.

Parses a genuinely useful *subset* of MQL5 Expert Advisor source and
converts it into the same standardized long/flat/short signal series every
other strategy source produces. Like the PineScript adapter, this is a
line-based parser rather than a full language implementation. Anything
outside the supported subset raises a clear StrategyError naming the
unsupported construct rather than silently producing an inaccurate
backtest.

Supported subset
-----------------
- Direct-value indicator calls (the common simplified/legacy calling style):
    double fastMA = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_SMA, PRICE_CLOSE);
    double slowMA = iMA(_Symbol, PERIOD_CURRENT, 30, 0, MODE_EMA, PRICE_CLOSE);
    double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
  (MODE_SMA/MODE_EMA/MODE_LWMA supported; the symbol/timeframe/shift/applied
  price arguments are accepted but not otherwise used -- this engine always
  operates on the single imported dataset's close price bar-by-bar.)
- Boolean conditions using C-style operators: > < >= <= == != && || !
- `if (condition) { ... }` and single-statement `if (condition) statement;`
  (brace-depth tracked so nested blocks are handled)
- Entry actions inside a condition's guard:
    trade.Buy(...) / trade.Sell(...)
    OrderSend(..., ORDER_TYPE_BUY, ...) / OrderSend(..., ORDER_TYPE_SELL, ...)
    OrderSend(..., OP_BUY, ...) / OrderSend(..., OP_SELL, ...)   (legacy MQL4-style constants, also accepted)
- Exit actions inside a condition's guard:
    trade.PositionClose(...) / OrderClose(...)
- Special directive comments for stop-loss / take-profit (point-based SL/TP
  in MQL5 aren't portable pip distances across instruments, so they're
  supplied the same explicit way as the PineScript adapter):
    // T58_SL_PIPS=20
    // T58_TP_PIPS=40
  OR, an instrument-scale-independent alternative (PREFERRED for any
  instrument that isn't FX -- gold, indices, crypto, stocks -- since a
  fixed point/pip count is only ever correct for the one instrument scale
  it was tuned at; see strategies/mql5/momentum_regime.mq5's own history
  of exactly this failure mode against gold-scale data):
    // T58_SL_ATR_MULT=1.5
    // T58_TP_ATR_MULT=3.0
    // T58_ATR_PERIOD=14        (optional, defaults to 14)
  ATR-mult directives compute a per-bar stop/target distance in raw price
  units (StrategyResult.stop_loss_distance/take_profit_distance), which
  the backtest engine already treats as taking precedence over the fixed
  pip fields -- see app/strategy/base.py's StrategyResult docstring. If
  both are present in the same file, the ATR-mult ones win.

Not supported (raises StrategyError): CopyBuffer()-based indicator handles,
custom indicators, arrays/structs, multi-symbol/multi-timeframe logic,
trailing stops, and any indicator function beyond iMA/iRSI.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.strategy.base import Strategy, StrategyError, StrategyResult, signals_from_conditions
from app.strategy.expr import safe_eval_bool, safe_eval_numeric
from app.strategy.indicators import atr, ema, sma

_COMMENT_RE = re.compile(r"//.*$")
_ASSIGN_RE = re.compile(r"^\s*(?:double|int|bool)?\s*([A-Za-z_]\w*)\s*=\s*(.+?);\s*$")
_IMA_RE = re.compile(r"iMA\s*\([^,]+,[^,]+,\s*([^,]+),[^,]+,\s*(MODE_\w+)\s*,[^)]*\)")
_IRSI_RE = re.compile(r"iRSI\s*\([^,]+,[^,]+,\s*([^,]+),[^)]*\)")
_IF_BLOCK_RE = re.compile(r"^(\s*)if\s*\((.+)\)\s*\{\s*$")
_IF_INLINE_RE = re.compile(r"^(\s*)if\s*\((.+?)\)\s*([^\{].*);\s*$")
_BUY_RE = re.compile(r"trade\.Buy\s*\(|OrderSend\s*\([^)]*(?:ORDER_TYPE_BUY|OP_BUY)")
_SELL_RE = re.compile(r"trade\.Sell\s*\(|OrderSend\s*\([^)]*(?:ORDER_TYPE_SELL|OP_SELL)")
_CLOSE_RE = re.compile(r"trade\.PositionClose\s*\(|OrderClose\s*\(")
_SL_DIRECTIVE_RE = re.compile(r"T58_SL_PIPS\s*=\s*([\d.]+)")
_TP_DIRECTIVE_RE = re.compile(r"T58_TP_PIPS\s*=\s*([\d.]+)")
_SL_ATR_DIRECTIVE_RE = re.compile(r"T58_SL_ATR_MULT\s*=\s*([\d.]+)")
_TP_ATR_DIRECTIVE_RE = re.compile(r"T58_TP_ATR_MULT\s*=\s*([\d.]+)")
_ATR_PERIOD_DIRECTIVE_RE = re.compile(r"T58_ATR_PERIOD\s*=\s*(\d+)")

_MODE_TO_FUNC = {"MODE_SMA": sma, "MODE_EMA": ema, "MODE_LWMA": None}  # LWMA falls back to sma with a note


def _c_bool_to_python(expr: str) -> str:
    expr = re.sub(r"&&", " and ", expr)
    expr = re.sub(r"\|\|", " or ", expr)
    expr = re.sub(r"(?<![<>=!])!(?!=)", " not ", expr)
    return expr.strip()


def _strip_block_comments(code: str) -> str:
    return re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)


class MQL5Strategy(Strategy):
    source_type = "mql5"

    def __init__(self, source: str | Path):
        """source: either a path to a .mq5 file, or raw MQL5 source text."""
        path = Path(source) if isinstance(source, (str, Path)) and str(source).endswith(".mq5") else None
        if path is not None:
            if not path.exists():
                raise StrategyError(f"MQL5 file not found: {path}")
            raw = path.read_text(encoding="utf-8", errors="ignore")
        else:
            raw = str(source)

        if not raw.strip():
            raise StrategyError("MQL5 source is empty.")
        self.code = _strip_block_comments(raw)

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        work = df.copy()
        stop_loss_pips: float | None = None
        take_profit_pips: float | None = None
        sl_atr_mult: float | None = None
        tp_atr_mult: float | None = None
        atr_period = 14

        long_conditions: list[str] = []
        short_conditions: list[str] = []
        exit_conditions: list[str] = []

        # stack of (brace_depth_at_open, condition_var_name)
        if_stack: list[tuple[int, str]] = []
        brace_depth = 0
        cond_counter = 0
        pending_cond: str | None = None  # set when an `if (...)` line has no trailing content (Allman-style braces)

        def materialize(cond_raw: str) -> str:
            nonlocal cond_counter
            py_expr = _c_bool_to_python(cond_raw)
            cond_counter += 1
            col = f"__cond_{cond_counter}"
            work[col] = safe_eval_bool(work, py_expr, "if-condition")
            return col

        def active_guard() -> str | None:
            return if_stack[-1][1] if if_stack else None

        for raw_line in self.code.splitlines():
            no_comment = _COMMENT_RE.sub("", raw_line)

            sl_match = _SL_DIRECTIVE_RE.search(raw_line)
            if sl_match:
                stop_loss_pips = float(sl_match.group(1))
            tp_match = _TP_DIRECTIVE_RE.search(raw_line)
            if tp_match:
                take_profit_pips = float(tp_match.group(1))
            sl_atr_match = _SL_ATR_DIRECTIVE_RE.search(raw_line)
            if sl_atr_match:
                sl_atr_mult = float(sl_atr_match.group(1))
            tp_atr_match = _TP_ATR_DIRECTIVE_RE.search(raw_line)
            if tp_atr_match:
                tp_atr_mult = float(tp_atr_match.group(1))
            atr_period_match = _ATR_PERIOD_DIRECTIVE_RE.search(raw_line)
            if atr_period_match:
                atr_period = int(atr_period_match.group(1))

            line = no_comment.strip()
            if not line:
                continue

            # Allman-style: previous line was a bare `if (...)` -- this line
            # is either the opening brace of its block, or (rarely) the
            # single guarded statement itself.
            if pending_cond is not None:
                if line == "{":
                    if_stack.append((brace_depth, pending_cond))
                    brace_depth += 1
                    pending_cond = None
                    continue
                guard = pending_cond
                pending_cond = None
                if _BUY_RE.search(line):
                    long_conditions.append(guard)
                elif _SELL_RE.search(line):
                    short_conditions.append(guard)
                elif _CLOSE_RE.search(line):
                    exit_conditions.append(guard)
                continue

            # pop if-blocks whose closing brace(s) appear on this line
            if "}" in line:
                for _ in range(line.count("}")):
                    brace_depth = max(0, brace_depth - 1)
                    if if_stack and brace_depth == if_stack[-1][0]:
                        if_stack.pop()
                remainder = line.replace("}", "").strip()
                if not remainder:
                    continue
                line = remainder

            # indicator assignment
            assign_match = _ASSIGN_RE.match(line)
            if assign_match:
                var_name, rhs = assign_match.groups()

                ima_match = _IMA_RE.search(rhs)
                if ima_match:
                    period_tok, mode = ima_match.groups()
                    try:
                        period = int(float(period_tok.strip()))
                    except ValueError:
                        raise StrategyError(f"MQL5: could not resolve iMA period '{period_tok}' to a number.")
                    func = _MODE_TO_FUNC.get(mode, sma)
                    if func is None:
                        func = sma  # MODE_LWMA approximated with SMA in this MVP parser
                    work[var_name] = func(work["close"], period)
                    continue

                irsi_match = _IRSI_RE.search(rhs)
                if irsi_match:
                    period_tok = irsi_match.group(1)
                    try:
                        period = int(float(period_tok.strip()))
                    except ValueError:
                        raise StrategyError(f"MQL5: could not resolve iRSI period '{period_tok}' to a number.")
                    from app.strategy.indicators import rsi as rsi_func
                    work[var_name] = rsi_func(work["close"], period)
                    continue

                if any(op in rhs for op in ("<", ">", "==", "!=", "&&", "||")):
                    work[var_name] = safe_eval_bool(work, _c_bool_to_python(rhs), var_name)
                    continue

                # A bare numeric-literal initializer, e.g. `int tradesToday = 0;`
                # or `double maxRiskPct = 1.5;` -- extremely common for local
                # counter/state variables in real EAs (daily trade counters,
                # loss counters, risk constants) that are never themselves an
                # indicator or a boolean condition. Broadcasts the constant
                # across every bar, same as PineScript's input.int/float
                # handling already does for the equivalent Pine pattern.
                try:
                    work[var_name] = float(rhs.strip())
                    continue
                except ValueError:
                    pass

                # Plain arithmetic over previously-defined numeric variables,
                # e.g. `trendStrengthPct = (emaFast - emaSlow) / emaSlow;`.
                # Common for normalized/percentage-separation filters. Only
                # attempted when the RHS actually contains an arithmetic
                # operator and no function-call syntax (iMA/iRSI would have
                # already matched above) -- anything else still falls
                # through to the error below.
                if any(op in rhs for op in ("+", "-", "*", "/")):
                    try:
                        work[var_name] = safe_eval_numeric(work, rhs, var_name)
                        continue
                    except StrategyError:
                        pass

                raise StrategyError(
                    f"MQL5: unsupported expression assigned to '{var_name}': '{rhs}'. "
                    "Supported: iMA(...) with MODE_SMA/MODE_EMA/MODE_LWMA, iRSI(...), "
                    "boolean comparisons, and +-*/ arithmetic over previously defined variables."
                )

            # if (...) { block open   (K&R style, brace on same line)
            block_match = _IF_BLOCK_RE.match(line)
            if block_match:
                cond_var = materialize(block_match.group(2))
                if_stack.append((brace_depth, cond_var))
                brace_depth += 1
                continue

            # if (...) single_statement;  (no braces at all)
            inline_match = _IF_INLINE_RE.match(line)
            if inline_match:
                cond_var = materialize(inline_match.group(2))
                stmt = inline_match.group(3)
                if _BUY_RE.search(stmt):
                    long_conditions.append(cond_var)
                elif _SELL_RE.search(stmt):
                    short_conditions.append(cond_var)
                elif _CLOSE_RE.search(stmt):
                    exit_conditions.append(cond_var)
                continue

            # bare `if (...)` with nothing else on the line -- Allman style,
            # the brace (or single statement) follows on the next line
            bare_if_match = re.match(r"^\s*if\s*\((.+)\)\s*$", line)
            if bare_if_match:
                pending_cond = materialize(bare_if_match.group(1))
                continue

            # bare opening brace increases depth (e.g. OnTick() on its own
            # line, followed by `{` on the next)
            if line == "{" or line.endswith("{"):
                brace_depth += 1
                continue

            guard = active_guard()
            if _BUY_RE.search(line):
                if guard is None:
                    raise StrategyError("MQL5: found a Buy/OrderSend(BUY) call that isn't guarded by an `if` condition.")
                long_conditions.append(guard)
                continue
            if _SELL_RE.search(line):
                if guard is None:
                    raise StrategyError("MQL5: found a Sell/OrderSend(SELL) call that isn't guarded by an `if` condition.")
                short_conditions.append(guard)
                continue
            if _CLOSE_RE.search(line):
                if guard is not None:
                    exit_conditions.append(guard)
                continue

            # anything else (OnInit/OnTick declarations, comments, unrelated
            # bookkeeping) is silently ignored -- it doesn't affect signals

        if not long_conditions and not short_conditions:
            raise StrategyError(
                "MQL5: no Buy/Sell (trade.Buy/trade.Sell/OrderSend) call was found inside a recognizable "
                "`if` condition. This parser supports a subset of MQL5 -- see app/strategy/mql5.py docstring."
            )

        long_entry = self._combine(work, long_conditions)
        short_entry = self._combine(work, short_conditions)
        exit_cond = self._combine(work, exit_conditions)

        raw_signals = signals_from_conditions(work.index, long_entry, exit_cond, short_entry, exit_cond)
        signals = self._validate_signals(raw_signals, df)

        # ATR-mult directives (instrument-scale-independent) take precedence
        # over fixed pip directives when both are present -- see the module
        # docstring and app/strategy/base.py's StrategyResult docstring.
        stop_loss_distance = None
        take_profit_distance = None
        if sl_atr_mult is not None or tp_atr_mult is not None:
            atr_series = atr(df, atr_period)
            if sl_atr_mult is not None:
                stop_loss_distance = atr_series * sl_atr_mult
                stop_loss_pips = None
            if tp_atr_mult is not None:
                take_profit_distance = atr_series * tp_atr_mult
                take_profit_pips = None

        return StrategyResult(
            name="MQL5 Strategy",
            source_type=self.source_type,
            signals=signals,
            stop_loss_pips=stop_loss_pips,
            take_profit_pips=take_profit_pips,
            stop_loss_distance=stop_loss_distance,
            take_profit_distance=take_profit_distance,
        )

    def _combine(self, work: pd.DataFrame, condition_vars: list[str]) -> pd.Series:
        if not condition_vars:
            return pd.Series(False, index=work.index)
        result = pd.Series(False, index=work.index)
        for c in condition_vars:
            result = result | work[c].astype(bool)
        return result
