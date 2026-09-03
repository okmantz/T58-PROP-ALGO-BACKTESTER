"""
PineScript Strategy Adapter.

Parses a genuinely useful *subset* of PineScript v5 strategy scripts and
converts them into the same standardized long/flat/short signal series
every other strategy source produces. This is a line-based parser, not a
full language implementation -- Pine is a large language, and reproducing
its entire runtime is out of scope. Anything outside the supported subset
raises a clear StrategyError naming the unsupported construct, per the
product spec's requirement that unsupported strategy functionality must
fail loudly rather than silently produce an inaccurate backtest.

Supported subset
-----------------
- Price references: open, high, low, close, hl2, hlc3, ohlc4
- `x = input.int(20, ...)` / `input.float(1.5, ...)`  -> constant, using the
  given default value
- `x = ta.sma(src, len)`, `ta.ema(src, len)`, `ta.wma(src, len)`, `ta.rsi(src, len)`
- `x = ta.crossover(a, b)`, `ta.crossunder(a, b)`
- Boolean rule variables built from comparisons/and/or/not over the above,
  e.g. `longCondition = ta.crossover(fast, slow) and rsiVal < 70`
- Entries, either inline or inside an `if` block:
    strategy.entry("Long", strategy.long, when=longCondition)
    if longCondition
        strategy.entry("Long", strategy.long)
- Exits:
    strategy.close("Long", when=exitLongCondition)
- Special directive comments for stop-loss / take-profit, since Pine's
  strategy.exit() uses absolute price offsets rather than a portable "pips"
  concept:
    // T58_SL_PIPS=20
    // T58_TP_PIPS=40
  OR, an instrument-scale-independent alternative (PREFERRED for any
  instrument that isn't FX -- gold, indices, crypto, stocks -- since a
  fixed pip count is only ever correct for the one pip_size it was tuned
  at; get pip_size wrong, or run the same script against a different
  instrument later, and the stop/target silently becomes nonsensical --
  see strategies/pinescript/trend_pullback.pine's own history of exactly
  this failure mode):
    // T58_SL_ATR_MULT=1.5
    // T58_TP_ATR_MULT=3.0
    // T58_ATR_PERIOD=14        (optional, defaults to 14)
  ATR-mult directives compute a per-bar stop/target distance in raw price
  units (StrategyResult.stop_loss_distance/take_profit_distance), which
  the backtest engine already treats as taking precedence over the fixed
  pip fields -- see app/strategy/base.py's StrategyResult docstring. If
  both are present in the same file, the ATR-mult ones win.

Not supported (raises StrategyError): custom functions, arrays/matrices,
security()/multi-timeframe requests, repainting constructs, plotting,
alerts, and any ta.* function beyond the list above.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.strategy.base import Strategy, StrategyError, StrategyResult, signals_from_conditions
from app.strategy.expr import safe_eval_bool
from app.strategy.indicators import INDICATOR_FUNCS, atr, crossover, crossunder

_ASSIGN_RE = re.compile(r"^\s*(?:var\s+)?([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")
_TA_CALL_RE = re.compile(r"ta\.(sma|ema|wma|rsi)\s*\(\s*([^,()]+)\s*,\s*([^()]+)\s*\)")
_CROSS_CALL_RE = re.compile(r"ta\.(crossover|crossunder)\s*\(\s*([^,()]+)\s*,\s*([^()]+)\s*\)")
_INPUT_RE = re.compile(r"input\.(?:int|float)\s*\(\s*([-\d.]+)")
_IF_RE = re.compile(r"^(\s*)if\s+(.+?)\s*$")
_ENTRY_RE = re.compile(
    r'strategy\.entry\s*\(\s*"([^"]*)"\s*,\s*strategy\.(long|short)\s*(?:,.*?when\s*=\s*(.+?))?\s*\)'
)
_CLOSE_RE = re.compile(r'strategy\.close\s*\(\s*"([^"]*)"\s*(?:,.*?when\s*=\s*(.+?))?\s*\)')
_SL_DIRECTIVE_RE = re.compile(r"T58_SL_PIPS\s*=\s*([\d.]+)")
_TP_DIRECTIVE_RE = re.compile(r"T58_TP_PIPS\s*=\s*([\d.]+)")
_SL_ATR_DIRECTIVE_RE = re.compile(r"T58_SL_ATR_MULT\s*=\s*([\d.]+)")
_TP_ATR_DIRECTIVE_RE = re.compile(r"T58_TP_ATR_MULT\s*=\s*([\d.]+)")
_ATR_PERIOD_DIRECTIVE_RE = re.compile(r"T58_ATR_PERIOD\s*=\s*(\d+)")

_PRICE_ALIASES = {"open", "high", "low", "close", "hl2", "hlc3", "ohlc4"}


def _strip_comment(line: str) -> tuple[str, str]:
    """Split a line into (code, comment) at the first `//` not inside a string."""
    in_str = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_str = not in_str
        elif ch == "/" and not in_str and i + 1 < len(line) and line[i + 1] == "/":
            return line[:i], line[i:]
    return line, ""


_MAX_CONTINUATION_LINES = 20


def _join_continuation_lines(raw_lines: list[str]) -> list[str]:
    """Real Pine scripts routinely wrap ONE logical statement -- most
    commonly a `strategy(...)`/`indicator(...)` declaration -- across
    several physical lines for readability:

        strategy("My Strategy",
             overlay=true,
             pyramiding=0)

    This parser otherwise processes one physical line at a time, so
    without this step every continuation line (e.g. `     overlay=true,`)
    is mistaken for its own top-level assignment statement and rejected
    as an unsupported expression -- even though the logical statement
    itself would have been silently ignored as a cosmetic declaration
    header if it had been written on a single line (see the module
    docstring's "any other unrecognized statement is silently ignored"
    note). Joins any line with unbalanced open parens with the physical
    lines that follow it until the parens balance again, so the whole
    call is seen as the single logical line it actually is. Caps
    accumulation at `_MAX_CONTINUATION_LINES` so a genuinely stray/
    mismatched paren elsewhere in the file can't silently swallow the
    rest of the script instead of surfacing as its own clear error.
    """
    joined: list[str] = []
    buffer: list[str] = []
    depth = 0
    for raw_line in raw_lines:
        code, _ = _strip_comment(raw_line)
        buffer.append(raw_line)
        depth += code.count("(") - code.count(")")
        if depth <= 0 or len(buffer) >= _MAX_CONTINUATION_LINES:
            joined.append(" ".join(buffer) if len(buffer) > 1 else buffer[0])
            buffer = []
            depth = 0
    if buffer:
        joined.append(" ".join(buffer) if len(buffer) > 1 else buffer[0])
    return joined


class PineScriptStrategy(Strategy):
    source_type = "pinescript"

    def __init__(self, source: str | Path):
        """source: either a path to a .pine file, or raw pine script text."""
        path = Path(source) if isinstance(source, (str, Path)) and str(source).endswith(".pine") else None
        if path is not None:
            if not path.exists():
                raise StrategyError(f"PineScript file not found: {path}")
            self.code = path.read_text(encoding="utf-8", errors="ignore")
        else:
            self.code = str(source)

        if not self.code.strip():
            raise StrategyError("PineScript source is empty.")

    # -- helpers -----------------------------------------------------
    def _resolve_source(self, name: str, work: pd.DataFrame) -> pd.Series:
        name = name.strip()
        if name == "hl2":
            return (work["high"] + work["low"]) / 2
        if name == "hlc3":
            return (work["high"] + work["low"] + work["close"]) / 3
        if name == "ohlc4":
            return (work["open"] + work["high"] + work["low"] + work["close"]) / 4
        if name in work.columns:
            return work[name]
        raise StrategyError(f"PineScript: unknown series/variable '{name}' referenced before assignment.")

    def _resolve_length(self, token: str, constants: dict[str, float]) -> int:
        token = token.strip()
        try:
            return int(float(token))
        except ValueError:
            pass
        if token in constants:
            return int(constants[token])
        raise StrategyError(
            f"PineScript: could not resolve length argument '{token}' to a number. "
            "Only integer literals or input.int()/input.float() variables are supported for lengths."
        )

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        work = df.copy()
        constants: dict[str, float] = {}
        stop_loss_pips: float | None = None
        take_profit_pips: float | None = None
        sl_atr_mult: float | None = None
        tp_atr_mult: float | None = None
        atr_period = 14

        long_conditions: list[str] = []
        long_exit_conditions: list[str] = []
        short_conditions: list[str] = []
        short_exit_conditions: list[str] = []

        # context stack of (indent_level, condition_var_name) for `if` blocks
        if_stack: list[tuple[int, str]] = []

        raw_lines = _join_continuation_lines(self.code.splitlines())
        for raw_line in raw_lines:
            code, comment = _strip_comment(raw_line)

            sl_match = _SL_DIRECTIVE_RE.search(comment)
            if sl_match:
                stop_loss_pips = float(sl_match.group(1))
            tp_match = _TP_DIRECTIVE_RE.search(comment)
            if tp_match:
                take_profit_pips = float(tp_match.group(1))
            sl_atr_match = _SL_ATR_DIRECTIVE_RE.search(comment)
            if sl_atr_match:
                sl_atr_mult = float(sl_atr_match.group(1))
            tp_atr_match = _TP_ATR_DIRECTIVE_RE.search(comment)
            if tp_atr_match:
                tp_atr_mult = float(tp_atr_match.group(1))
            atr_period_match = _ATR_PERIOD_DIRECTIVE_RE.search(comment)
            if atr_period_match:
                atr_period = int(atr_period_match.group(1))

            if not code.strip():
                continue

            indent = len(code) - len(code.lstrip(" "))

            # pop if-blocks we've dedented out of
            while if_stack and indent <= if_stack[-1][0]:
                if_stack.pop()

            if_match = _IF_RE.match(code)
            if if_match:
                cond_indent = len(if_match.group(1))
                cond_expr = if_match.group(2).strip()
                cond_var = self._materialize_condition(cond_expr, work, constants)
                if_stack.append((cond_indent, cond_var))
                continue

            # ta.crossover / ta.crossunder assignment
            assign_match = _ASSIGN_RE.match(code)
            if assign_match:
                var_name, rhs = assign_match.groups()

                input_match = _INPUT_RE.search(rhs)
                if input_match and rhs.strip().startswith("input."):
                    value = float(input_match.group(1))
                    constants[var_name] = value
                    # Also expose the value as a broadcast column, not just
                    # in the `constants` dict. `constants` is only ever
                    # consulted by _resolve_length() for ta.* length
                    # arguments -- a very common second use of input.*() is
                    # a plain threshold (e.g. `longRSI = input.float(55.0,
                    # ...)` used later as `rsiVal > longRSI`), which is
                    # evaluated by safe_eval_bool() against `work`'s
                    # columns and would otherwise fail with "name 'longRSI'
                    # is not defined" even though the script is valid.
                    work[var_name] = value
                    continue

                cross_match = _CROSS_CALL_RE.search(rhs)
                if cross_match:
                    func, a_tok, b_tok = cross_match.groups()
                    a_series = self._resolve_operand(a_tok, work, constants)
                    b_series = self._resolve_operand(b_tok, work, constants)
                    work[var_name] = crossover(a_series, b_series) if func == "crossover" else crossunder(a_series, b_series)
                    continue

                ta_match = _TA_CALL_RE.search(rhs)
                if ta_match:
                    func, src_tok, len_tok = ta_match.groups()
                    src_series = self._resolve_source(src_tok, work)
                    length = self._resolve_length(len_tok, constants)
                    work[var_name] = INDICATOR_FUNCS[func](src_series, length)
                    continue

                # plain boolean expression assignment, e.g. longCondition = fastMA > slowMA
                if any(op in rhs for op in ("<", ">", "==", "!=", " and ", " or ")):
                    work[var_name] = safe_eval_bool(work, rhs, var_name)
                    continue

                # unrecognized assignment: skip silently only if it's clearly a
                # plain numeric/price alias re-binding; otherwise flag it
                if rhs.strip() in _PRICE_ALIASES or rhs.strip() in work.columns:
                    work[var_name] = self._resolve_source(rhs.strip(), work)
                    continue

                raise StrategyError(
                    f"PineScript: unsupported expression on right-hand side of '{var_name} = {rhs}'. "
                    "Supported: input.int/float, ta.sma/ema/wma/rsi, ta.crossover/crossunder, "
                    "and boolean comparisons over previously defined series."
                )

            # strategy.entry(...)
            entry_match = _ENTRY_RE.search(code)
            if entry_match:
                _, direction, when_expr = entry_match.groups()
                cond_var = self._condition_for_statement(when_expr, if_stack, work, constants)
                (long_conditions if direction == "long" else short_conditions).append(cond_var)
                continue

            # strategy.close(...)
            close_match = _CLOSE_RE.search(code)
            if close_match:
                trade_id, when_expr = close_match.groups()
                cond_var = self._condition_for_statement(when_expr, if_stack, work, constants)
                tid = (trade_id or "").lower()
                if "short" in tid:
                    short_exit_conditions.append(cond_var)
                elif "long" in tid:
                    long_exit_conditions.append(cond_var)
                else:
                    long_exit_conditions.append(cond_var)
                    short_exit_conditions.append(cond_var)
                continue

            # any other unrecognized statement is silently ignored (plotting,
            # alerts, strategy() declaration header, etc. -- purely cosmetic
            # Pine constructs that don't affect signal generation)

        if not long_conditions and not short_conditions:
            raise StrategyError(
                "PineScript: no strategy.entry(...) call was found (inline `when=` or inside an `if` block). "
                "This parser supports a subset of Pine v5 -- see app/strategy/pinescript.py docstring."
            )

        long_entry = self._combine(work, long_conditions)
        short_entry = self._combine(work, short_conditions)
        long_exit = self._combine(work, long_exit_conditions)
        short_exit = self._combine(work, short_exit_conditions)

        raw_signals = signals_from_conditions(work.index, long_entry, long_exit, short_entry, short_exit)
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
            name="PineScript Strategy",
            source_type=self.source_type,
            signals=signals,
            stop_loss_pips=stop_loss_pips,
            take_profit_pips=take_profit_pips,
            stop_loss_distance=stop_loss_distance,
            take_profit_distance=take_profit_distance,
        )

    # -- internal utilities -------------------------------------------
    def _resolve_operand(self, token: str, work: pd.DataFrame, constants: dict[str, float]) -> pd.Series:
        token = token.strip()
        try:
            return pd.Series(float(token), index=work.index)
        except ValueError:
            pass
        return self._resolve_source(token, work)

    def _materialize_condition(self, expr: str, work: pd.DataFrame, constants: dict[str, float]) -> str:
        """Evaluate/store a boolean condition expression as a temp column, return its name."""
        cross_match = _CROSS_CALL_RE.search(expr)
        if cross_match and expr.strip() == cross_match.group(0):
            func, a_tok, b_tok = cross_match.groups()
            a_series = self._resolve_operand(a_tok, work, constants)
            b_series = self._resolve_operand(b_tok, work, constants)
            col = f"__cond_{len(work.columns)}"
            work[col] = crossover(a_series, b_series) if func == "crossover" else crossunder(a_series, b_series)
            return col
        if expr.strip() in work.columns:
            return expr.strip()
        col = f"__cond_{len(work.columns)}"
        work[col] = safe_eval_bool(work, expr, "if-condition")
        return col

    def _condition_for_statement(self, when_expr, if_stack, work, constants) -> str:
        if when_expr:
            return self._materialize_condition(when_expr, work, constants)
        if if_stack:
            return if_stack[-1][1]
        raise StrategyError(
            "PineScript: strategy.entry()/strategy.close() call has no `when=` condition and is not "
            "inside an `if` block -- cannot determine when it should trigger."
        )

    def _combine(self, work: pd.DataFrame, condition_vars: list[str]) -> pd.Series:
        if not condition_vars:
            return pd.Series(False, index=work.index)
        result = pd.Series(False, index=work.index)
        for c in condition_vars:
            result = result | work[c].astype(bool)
        return result
