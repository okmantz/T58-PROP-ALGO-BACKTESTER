"""
Strategy Family Diversity.

Two related problems this module solves for the Search Lab funnel
(app.search.batch_runner.run_search):

  1. "Don't let your optimizer generate 10,000 variations of essentially
     the same strategy" -- enforce_family_diversity() caps how many
     candidates from the SAME classified family (app.strategy.
     family_taxonomy) are allowed to survive into the next stage,
     keeping only the top-scoring ones per family. Meant to run between
     Stage 1 and Stage 2 of run_search (after the cheap filter has
     already thinned the field, before the expensive Monte Carlo /
     walk-forward / robustness stages spend time re-proving the same
     hypothesis dozens of times over).

  2. "Best strategy family for this market/data" -- summarize_family_
     performance() groups a batch of already-scored records by family
     and ranks the families themselves, not just individual candidates,
     so a search run can answer a question no single candidate's score
     answers on its own: which HYPOTHESIS held up best on this
     instrument/timeframe, independent of which specific parameter
     combination happened to win inside it.

Both functions work on the plain dict records app.search.batch_runner's
_stage1_task/_stage2_task/_stage3_task already produce (and
app.search.results_db already persists) -- no new record shape, no
required change to the existing pipeline to start using this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.strategy.family_taxonomy import FAMILY_GROUPS, classify_record, family_label

# Tried in this order -- whichever is present and non-None on a record
# wins. composite_score (Stage 3/4) is the most complete signal (folds in
# pass probability, payout probability, risk of ruin, deflated Sharpe --
# see batch_runner.py's own composite_score formula); fitness (Stage 2) is
# the GA refinement objective; quick_score (Stage 1) is the cheapest,
# earliest signal, appropriate when nothing further along has run yet.
_SCORE_KEYS_IN_PRIORITY = ("composite_score", "fitness", "quick_score")


def _score_of(record: dict, score_key: str | None) -> float | None:
    if score_key:
        val = record.get(score_key)
        return float(val) if isinstance(val, (int, float)) else None
    for key in _SCORE_KEYS_IN_PRIORITY:
        val = record.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


# ---------------------------------------------------------------------------
# 1. Diversity enforcement
# ---------------------------------------------------------------------------

def enforce_family_diversity(
    records: list[dict],
    max_per_family: int,
    score_key: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Groups `records` by classified family and keeps only the top
    `max_per_family` per group (by `score_key`, or the best available
    score -- see _SCORE_KEYS_IN_PRIORITY -- if not given). Records with
    no usable score are treated as worst-ranked within their family
    (kept only if the family has fewer than `max_per_family` scored
    records ahead of them) rather than dropped outright, so a Stage 1-only
    batch (which has no composite_score/fitness yet) still gets a
    sensible diversity cap from quick_score alone.

    Returns (kept, dropped) in the SAME relative order records arrived in
    -- callers that need to persist "why was this dropped" can zip
    `dropped` against `enforce_family_diversity`'s own reasoning (it's
    always exactly "N candidates from its family already scored higher").
    Order across families is not otherwise changed; re-sort the `kept`
    list yourself if you need it ranked.
    """
    if max_per_family <= 0:
        raise ValueError("max_per_family must be >= 1.")

    by_family: dict[str, list[dict]] = {}
    for rec in records:
        by_family.setdefault(classify_record(rec), []).append(rec)

    keep_ids: set[int] = set()
    for group, group_records in by_family.items():
        ranked = sorted(
            group_records,
            key=lambda r: (_score_of(r, score_key) is not None, _score_of(r, score_key) or float("-inf")),
            reverse=True,
        )
        for rec in ranked[:max_per_family]:
            keep_ids.add(id(rec))

    kept = [r for r in records if id(r) in keep_ids]
    dropped = [r for r in records if id(r) not in keep_ids]
    return kept, dropped


# ---------------------------------------------------------------------------
# 2. Per-family performance ranking
# ---------------------------------------------------------------------------

@dataclass
class FamilyPerformance:
    group: str
    label: str
    n_candidates: int
    n_passed_gate: int          # count with passed_stage3_gate truthy, if that field is present at all
    best_score: float | None
    median_score: float | None
    best_candidate_id: str | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def summarize_family_performance(records: list[dict], score_key: str | None = None) -> list["FamilyPerformance"]:
    """One FamilyPerformance per family actually present in `records`,
    sorted best-first by best_score (families with no usable score at
    all sort last, not dropped -- an unscored family is 'not yet
    evidenced', not 'bad')."""
    by_family: dict[str, list[dict]] = {}
    for rec in records:
        by_family.setdefault(classify_record(rec), []).append(rec)

    summaries: list[FamilyPerformance] = []
    for group, group_records in by_family.items():
        scores = sorted((s for s in (_score_of(r, score_key) for r in group_records) if s is not None), reverse=True)
        best_score = scores[0] if scores else None
        median_score = scores[len(scores) // 2] if scores else None
        best_record = None
        if best_score is not None:
            best_record = max(
                (r for r in group_records if _score_of(r, score_key) == best_score),
                key=lambda r: r.get("candidate_id", ""),
            )
        n_passed = sum(1 for r in group_records if r.get("passed_stage3_gate"))
        summaries.append(FamilyPerformance(
            group=group, label=family_label(group), n_candidates=len(group_records),
            n_passed_gate=n_passed, best_score=best_score, median_score=median_score,
            best_candidate_id=(best_record or {}).get("candidate_id"),
        ))

    summaries.sort(key=lambda s: (s.best_score is not None, s.best_score if s.best_score is not None else float("-inf")), reverse=True)
    return summaries


def best_families(summaries: list[FamilyPerformance], top_n: int = 2) -> list[FamilyPerformance]:
    """The `top_n` families with an actual score, in rank order --
    `summarize_family_performance` already sorts best-first, this just
    trims to scored entries so a caller building the headline sentence
    never picks an unscored family by accident."""
    scored = [s for s in summaries if s.best_score is not None]
    return scored[:top_n]


def render_family_report(summaries: list[FamilyPerformance]) -> str:
    """Plain-text table + the headline sentence Owen asked for:
    'Best strategy family for this market/data: X (+ runner-up Y)'."""
    lines = ["Strategy Family Diversity", ""]
    header = f"{'Family':<24}{'Candidates':>12}{'Passed Gate':>13}{'Best Score':>12}{'Median':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        best = f"{s.best_score:.2f}" if s.best_score is not None else "--"
        median = f"{s.median_score:.2f}" if s.median_score is not None else "--"
        lines.append(f"{s.label:<24}{s.n_candidates:>12}{s.n_passed_gate:>13}{best:>12}{median:>10}")

    top = best_families(summaries, top_n=2)
    lines.append("")
    if not top:
        lines.append("Best strategy family for this market/data: not yet determined (no scored candidates).")
    elif len(top) == 1:
        lines.append(f"Best strategy family for this market/data: {top[0].label}.")
    else:
        lines.append(f"Best strategy family for this market/data: {top[0].label} + {top[1].label}.")
    return "\n".join(lines)
