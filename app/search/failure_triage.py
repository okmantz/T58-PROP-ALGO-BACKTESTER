"""
Auto-triage failure reasons at scale.

When a Search Lab / Evolution Lab batch discards hundreds or thousands of
candidates, a per-candidate log line ("candidate abc123: failed") tells
Owen nothing about WHY the batch failed as a whole -- whether every single
candidate timed out on the same session filter, all failed the same
drawdown check, or produced zero trades. That distinction is exactly what
decides whether the fix is "run it longer" or "the search space itself is
wrong" (see app.strategy.family_taxonomy / app.search.strategy_space for
the family-library side of that same question).

This module aggregates one stage's worth of raw candidate result dicts
(the same shape app.search.batch_runner._stageN_task functions return)
into a small, ranked "reason -> count" summary, optionally broken down
per family, so a wall of thousands of individual failures collapses into
a handful of lines worth actually reading.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


# Canonical reason buckets, checked in this order (first match wins) so a
# candidate with multiple issues is still counted exactly once, under its
# most informative reason.
_ZERO_TRADES_MARKERS = ("no trades", "zero trade", "zero-trade")


def _classify_stage1_or_stage3_record(rec: dict, min_trades: int | None = None,
                                       min_profit_factor: float | None = None) -> str:
    """One candidate result dict (Stage 1 or the Stage 3 early-kill floor
    shares the same statistics shape) -> one short, human-readable reason
    string. Falls back to the raw error message (truncated) for anything
    that doesn't fit a known bucket, so a genuinely new failure mode is
    still visible rather than silently swallowed into "other"."""
    error = (rec.get("error") or "").lower()
    stats = rec.get("statistics") or {}

    if error and any(marker in error for marker in _ZERO_TRADES_MARKERS):
        return "zero trades generated"
    if "early-kill floor" in error:
        return "failed Stage 3 early-kill floor (before Monte Carlo)"
    if rec.get("lookahead", {}).get("bug_detected"):
        return "lookahead bug detected"
    if rec.get("walk_forward") is not None and not rec["walk_forward"].get("is_stable", True):
        return "walk-forward instability (in-sample edge didn't hold out-of-sample)"
    if rec.get("robustness") is not None and not rec["robustness"].get("is_stable", True):
        return "parameter-neighborhood instability (likely fit to noise)"

    n_trades = stats.get("total_trades")
    if isinstance(n_trades, (int, float)) and min_trades is not None and n_trades < min_trades:
        return f"too few trades ({int(n_trades)} < {min_trades} required)"

    pf = stats.get("profit_factor")
    if isinstance(pf, (int, float)) and min_profit_factor is not None and pf < min_profit_factor:
        return f"profit factor too low ({pf:.2f} < {min_profit_factor:.2f} required)"

    if isinstance(stats.get("net_profit"), (int, float)) and stats["net_profit"] <= 0:
        return "net loss on a plain backtest"

    if error:
        return f"error: {error[:80]}"

    return "failed filter thresholds (drawdown or other)"


@dataclass
class FailureTriageSummary:
    stage_label: str
    total_records: int
    total_failed: int
    reason_counts: "Counter[str]" = field(default_factory=Counter)
    reason_counts_by_family: dict[str, "Counter[str]"] = field(default_factory=dict)

    def top_reasons(self, n: int = 8) -> list[tuple[str, int]]:
        return self.reason_counts.most_common(n)

    def format_log_lines(self, n: int = 8) -> list[str]:
        if self.total_failed == 0:
            return [f"  {self.stage_label}: no failures to triage -- everything passed."]
        lines = [
            f"  {self.stage_label} failure triage: {self.total_failed}/{self.total_records} failed. "
            f"Top reasons:"
        ]
        for reason, count in self.top_reasons(n):
            pct = 100.0 * count / self.total_failed
            lines.append(f"    - {reason}: {count} ({pct:.0f}% of failures)")
        # Whole-family wipeouts are the single most actionable signal here --
        # if EVERY candidate in a family failed, more search time won't fix
        # it; the family itself needs redesigning (see app.search.strategy_space).
        wiped_families = [
            fam for fam, counts in self.reason_counts_by_family.items()
            if sum(counts.values()) > 0 and fam
        ]
        if wiped_families:
            lines.append(f"    Families represented among failures: {', '.join(sorted(wiped_families))}")
        return lines


def aggregate_failure_reasons(
    records: list[dict],
    stage_label: str,
    passed_key: str,
    min_trades: int | None = None,
    min_profit_factor: float | None = None,
) -> FailureTriageSummary:
    """records: the full list of one stage's result dicts (both passed and
    failed). passed_key: which dict key marks a pass ("passed_stage1" /
    "passed_stage2" / "passed_stage3_gate")."""
    failed = [r for r in records if not r.get(passed_key)]
    reason_counts: Counter = Counter()
    by_family: dict[str, Counter] = {}
    for rec in failed:
        reason = _classify_stage1_or_stage3_record(rec, min_trades=min_trades, min_profit_factor=min_profit_factor)
        reason_counts[reason] += 1
        fam = rec.get("family") or "unknown"
        by_family.setdefault(fam, Counter())[reason] += 1
    return FailureTriageSummary(
        stage_label=stage_label, total_records=len(records), total_failed=len(failed),
        reason_counts=reason_counts, reason_counts_by_family=by_family,
    )
