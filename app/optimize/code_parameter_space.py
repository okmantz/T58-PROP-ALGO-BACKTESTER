"""
Code-strategy parameter space: makes Iterative Refinement work for Python,
PineScript, and MQL5 strategies, not just Manual Strategy Builder configs.

These three sources are opaque code rather than a structured dict, so there
is no config tree to walk the way app.optimize.parameter_space does for
Manual strategies. Instead, this module finds the numeric literals each
adapter's *own supported parser subset* already treats as a parameter, and
patches them directly in the source text:

  Python       -- any top-level (unindented) `SCREAMING_SNAKE_CASE = <number>`
                  assignment (the exact convention app/strategy/python.py's
                  own docstring already documents for STOP_LOSS_PIPS /
                  TAKE_PROFIT_PIPS / STRATEGY_NAME -- this generalizes it to
                  any constant a strategy author names that way, with zero
                  extra convention to learn).
  PineScript   -- every `x = input.int(20, ...)` / `input.float(1.5, ...)`
                  default value (Pine's own dedicated "this is a parameter"
                  mechanism), plus the `// T58_SL_PIPS=` / `// T58_TP_PIPS=`
                  directive values this app already defines.
  MQL5         -- every literal period argument inside an `iMA(...)` /
                  `iRSI(...)` call (the only form of period this adapter's
                  supported subset accepts at all -- see app/strategy/mql5.py),
                  plus the same `T58_SL_PIPS` / `T58_TP_PIPS` directives.

A "candidate" configuration is then just the original source text with those
specific numeric substrings replaced -- no re-parsing convention, no change
to any of the three existing adapters' parsing logic, and no risk to
unrelated numbers anywhere else in the file (each match is anchored to the
exact call/assignment shape the adapter itself already requires in order to
run at all).

This module never modifies app/strategy/python.py, pinescript.py, or
mql5.py. PineScript/MQL5 strategies accept raw source text directly in
their constructors, so a mutated candidate is just a new instance built
from patched text. Python strategies only accept a file path, so a mutated
candidate is written to a throwaway temp .py file (cleaned up by the
caller -- see app.optimize.refinement's use of `tmp_dir`).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.optimize.parameter_space import RefinementError
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy

SUPPORTED_CODE_SOURCE_TYPES = {"python", "pinescript", "mql5"}


@dataclass(frozen=True)
class CodeGene:
    name: str            # variable / constant name, for display
    kind: str             # "python_global" | "pine_input" | "pine_sl_pips" | "pine_tp_pips" |
                           # "mql5_ma_period" | "mql5_rsi_period" | "mql5_sl_pips" | "mql5_tp_pips"
    is_int: bool
    lo: float
    hi: float
    base_value: float
    label: str            # human-readable, unique across a file's genes
    line_index: int        # 0-based line this value lives on, for patching
    span: tuple[int, int]  # (start, end) character offsets of the numeric literal within that line


# ---------------------------------------------------------------------------
# Shared bounds heuristic (mirrors app.optimize.parameter_space's rationale:
# periods/pips search 0.3x-3x the current value; thresholds that could be
# zero or negative get a relative-or-absolute span instead)
# ---------------------------------------------------------------------------

def _multiplicative_bounds(value: float, is_int: bool) -> tuple[float, float]:
    floor = 1.0 if is_int else 0.1
    if value > 0:
        lo = max(value * 0.3, floor)
        hi = max(value * 3.0, lo + floor)
    else:
        lo = floor
        hi = max(floor * 10.0, 10.0)
    if is_int:
        lo, hi = float(max(1, round(lo))), float(max(round(lo) + 1, round(hi)))
    return lo, hi


def _relative_bounds(value: float) -> tuple[float, float]:
    span = max(abs(value) * 0.5, 1.0)
    return value - span, value + span


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_PY_CONST_RE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)\s*(?:#.*)?$")
# Names that, even if numeric, are never meaningful search parameters.
_PY_EXCLUDED_NAMES = {"STRATEGY_NAME"}  # (str-valued in practice; excluded defensively)

_PINE_INPUT_RE = re.compile(
    r"^\s*(?:var\s+)?[A-Za-z_]\w*\s*=\s*input\.(?:int|float)\s*\(\s*(-?\d+(?:\.\d+)?)"
)
_PINE_INPUT_NAME_RE = re.compile(r"^\s*(?:var\s+)?([A-Za-z_]\w*)\s*=")
_SL_DIRECTIVE_RE = re.compile(r"T58_SL_PIPS\s*=\s*(-?\d+(?:\.\d+)?)")
_TP_DIRECTIVE_RE = re.compile(r"T58_TP_PIPS\s*=\s*(-?\d+(?:\.\d+)?)")

_MQL5_IMA_RE = re.compile(
    r"iMA\s*\([^,]+,[^,]+,\s*(-?\d+(?:\.\d+)?)\s*,[^,]+,\s*MODE_\w+\s*,[^)]*\)"
)
_MQL5_IRSI_RE = re.compile(r"iRSI\s*\([^,]+,[^,]+,\s*(-?\d+(?:\.\d+)?)\s*,[^)]*\)")


def discover_python_parameters(file_path: str | Path) -> list[CodeGene]:
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    genes: list[CodeGene] = []
    for i, line in enumerate(text.splitlines()):
        if line != line.lstrip():
            continue  # only top-level (unindented) constants
        m = _PY_CONST_RE.match(line)
        if not m:
            continue
        name, literal = m.groups()
        if name in _PY_EXCLUDED_NAMES:
            continue
        value = float(literal)
        is_int = "." not in literal
        lo, hi = _multiplicative_bounds(value, is_int) if is_int else _relative_bounds(value)
        span = (m.start(2), m.end(2))
        genes.append(CodeGene(
            name=name, kind="python_global", is_int=is_int, lo=lo, hi=hi,
            base_value=value, label=f"{name} (line {i + 1})", line_index=i, span=span,
        ))
    return genes


def discover_pinescript_parameters(code: str) -> list[CodeGene]:
    genes: list[CodeGene] = []
    for i, line in enumerate(code.splitlines()):
        input_match = _PINE_INPUT_RE.match(line)
        if input_match:
            name_match = _PINE_INPUT_NAME_RE.match(line)
            name = name_match.group(1) if name_match else f"input_L{i + 1}"
            literal = input_match.group(1)
            value = float(literal)
            is_int = ".int(" in line
            lo, hi = _multiplicative_bounds(value, True) if is_int else _relative_bounds(value)
            genes.append(CodeGene(
                name=name, kind="pine_input", is_int=is_int, lo=lo, hi=hi,
                base_value=value, label=f"{name} = input.{'int' if is_int else 'float'}(...) (line {i + 1})",
                line_index=i, span=(input_match.start(1), input_match.end(1)),
            ))
            continue

        sl_match = _SL_DIRECTIVE_RE.search(line)
        if sl_match:
            value = float(sl_match.group(1))
            lo, hi = _multiplicative_bounds(value, False)
            genes.append(CodeGene(
                name="T58_SL_PIPS", kind="pine_sl_pips", is_int=False, lo=lo, hi=hi,
                base_value=value, label=f"T58_SL_PIPS directive (line {i + 1})",
                line_index=i, span=(sl_match.start(1), sl_match.end(1)),
            ))
        tp_match = _TP_DIRECTIVE_RE.search(line)
        if tp_match:
            value = float(tp_match.group(1))
            lo, hi = _multiplicative_bounds(value, False)
            genes.append(CodeGene(
                name="T58_TP_PIPS", kind="pine_tp_pips", is_int=False, lo=lo, hi=hi,
                base_value=value, label=f"T58_TP_PIPS directive (line {i + 1})",
                line_index=i, span=(tp_match.start(1), tp_match.end(1)),
            ))
    return genes


def discover_mql5_parameters(code: str) -> list[CodeGene]:
    genes: list[CodeGene] = []
    for i, line in enumerate(code.splitlines()):
        ima_match = _MQL5_IMA_RE.search(line)
        if ima_match:
            value = float(ima_match.group(1))
            lo, hi = _multiplicative_bounds(value, True)
            genes.append(CodeGene(
                name=f"iMA_period_L{i + 1}", kind="mql5_ma_period", is_int=True, lo=lo, hi=hi,
                base_value=value, label=f"iMA(...) period (line {i + 1})",
                line_index=i, span=(ima_match.start(1), ima_match.end(1)),
            ))
        irsi_match = _MQL5_IRSI_RE.search(line)
        if irsi_match:
            value = float(irsi_match.group(1))
            lo, hi = _multiplicative_bounds(value, True)
            genes.append(CodeGene(
                name=f"iRSI_period_L{i + 1}", kind="mql5_rsi_period", is_int=True, lo=lo, hi=hi,
                base_value=value, label=f"iRSI(...) period (line {i + 1})",
                line_index=i, span=(irsi_match.start(1), irsi_match.end(1)),
            ))
        sl_match = _SL_DIRECTIVE_RE.search(line)
        if sl_match:
            value = float(sl_match.group(1))
            lo, hi = _multiplicative_bounds(value, False)
            genes.append(CodeGene(
                name="T58_SL_PIPS", kind="mql5_sl_pips", is_int=False, lo=lo, hi=hi,
                base_value=value, label=f"T58_SL_PIPS directive (line {i + 1})",
                line_index=i, span=(sl_match.start(1), sl_match.end(1)),
            ))
        tp_match = _TP_DIRECTIVE_RE.search(line)
        if tp_match:
            value = float(tp_match.group(1))
            lo, hi = _multiplicative_bounds(value, False)
            genes.append(CodeGene(
                name="T58_TP_PIPS", kind="mql5_tp_pips", is_int=False, lo=lo, hi=hi,
                base_value=value, label=f"T58_TP_PIPS directive (line {i + 1})",
                line_index=i, span=(tp_match.start(1), tp_match.end(1)),
            ))
    return genes


def discover_code_genes(strategy) -> list[CodeGene]:
    if strategy.source_type == "python":
        return discover_python_parameters(strategy.file_path)
    if strategy.source_type == "pinescript":
        return discover_pinescript_parameters(strategy.code)
    if strategy.source_type == "mql5":
        return discover_mql5_parameters(strategy.code)
    raise RefinementError(f"'{strategy.source_type}' is not a supported code strategy source type.")


# ---------------------------------------------------------------------------
# Applying a genome back into source text
# ---------------------------------------------------------------------------

def _fmt_num(value: float) -> str:
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def apply_code_genome(text: str, genes: list[CodeGene], genome: list[float]) -> str:
    """Returns a new source string with every gene's line patched at its
    recorded span. Multiple genes on the same line are applied
    right-to-left so earlier spans' offsets stay valid even when the
    replacement text is a different length than the original literal."""
    if len(genome) != len(genes):
        raise RefinementError(
            f"Genome length ({len(genome)}) does not match the number of "
            f"tunable parameters ({len(genes)})."
        )
    had_trailing_newline = text.endswith("\n")
    lines = text.splitlines()

    by_line: dict[int, list[tuple[CodeGene, float]]] = {}
    for gene, value in zip(genes, genome):
        by_line.setdefault(gene.line_index, []).append((gene, value))

    for line_idx, items in by_line.items():
        line = lines[line_idx]
        for gene, value in sorted(items, key=lambda t: t[0].span[0], reverse=True):
            start, end = gene.span
            formatted = str(int(round(value))) if gene.is_int else _fmt_num(float(value))
            line = line[:start] + formatted + line[end:]
        lines[line_idx] = line

    patched = "\n".join(lines)
    if had_trailing_newline:
        patched += "\n"
    return patched


def materialize_python_strategy(
    file_path: str | Path, genes: list[CodeGene], genome: list[float], tmp_dir: Path,
) -> PythonStrategy:
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    patched = apply_code_genome(text, genes, genome)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"refine_{uuid.uuid4().hex}.py"
    tmp_path.write_text(patched, encoding="utf-8")
    return PythonStrategy(tmp_path)


def materialize_code_strategy(strategy, genes: list[CodeGene], genome: list[float], tmp_dir: Path):
    """Builds a fresh strategy instance of the same source type as
    `strategy`, with `genome` applied at the positions `genes` describes.
    `tmp_dir` is only used for Python strategies (which require a file path);
    PineScript/MQL5 build directly from patched in-memory text."""
    if strategy.source_type == "python":
        return materialize_python_strategy(strategy.file_path, genes, genome, tmp_dir)
    if strategy.source_type == "pinescript":
        return PineScriptStrategy(apply_code_genome(strategy.code, genes, genome))
    if strategy.source_type == "mql5":
        return MQL5Strategy(apply_code_genome(strategy.code, genes, genome))
    raise RefinementError(f"'{strategy.source_type}' is not a supported code strategy source type.")


def patched_source_for_strategy(strategy, genes: list[CodeGene], genome: list[float]) -> tuple[str, str]:
    """Returns (patched_source_text, suggested_file_extension) without
    constructing a Strategy or writing any file -- used by the report/UI to
    show and save the winning code without needing a live strategy object."""
    if strategy.source_type == "python":
        text = Path(strategy.file_path).read_text(encoding="utf-8", errors="ignore")
        return apply_code_genome(text, genes, genome), ".py"
    if strategy.source_type == "pinescript":
        return apply_code_genome(strategy.code, genes, genome), ".pine"
    if strategy.source_type == "mql5":
        return apply_code_genome(strategy.code, genes, genome), ".mq5"
    raise RefinementError(f"'{strategy.source_type}' is not a supported code strategy source type.")
