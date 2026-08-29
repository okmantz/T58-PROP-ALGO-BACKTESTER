"""
Systematic "why did the losers fail" analysis for one GA generation --
Stage 4 ("analysis and feedback") of the quant loop framework, computed
with plain statistics instead of asking a model to guess at it.

The framework's loop asks, after each round: what market conditions/
assumptions/parameter choices broke the losing candidates, so that
information can be fed back into the next round as constraints. For this
app's genome-based search, the equivalent question is: which tunable
parameters (genes), and which VALUES of those genes, are actually
associated with high or low fitness in the population that's already
been evaluated?

That's a plain correlation/binning computation over numbers the GA
already produced -- no model call needed, no extra backtests needed, and
it's exact rather than a guess. The output is a short, structured summary
("gene X above ~30 correlates with worse fitness") that then gets handed
to the optional local-Ollama assistant as compact CONTEXT for its one
narrow job (proposing new numeric candidates) -- keeping the actual
analysis systematic and free, and limiting AI usage to the part that
genuinely benefits from it (turning "avoid this region, lean into that
one" into concrete new parameter values).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class GeneFinding:
    label: str
    correlation: float          # Pearson correlation of this gene's value with fitness across the population
    direction: str               # "higher values -> better fitness" | "lower values -> better fitness" | "no clear relationship"
    best_region: str             # human-readable summary of where the top performers cluster, e.g. "18-24"
    worst_region: str


@dataclass
class PopulationAnalysis:
    n_candidates: int
    n_finite: int
    findings: list[GeneFinding] = field(default_factory=list)
    note: str = ""

    def summary_lines(self, top_n: int = 3) -> list[str]:
        """Compact, ranked-by-strength text lines -- kept short deliberately
        since this gets embedded directly into the AI prompt and every
        extra line is tokens spent on every single generation's request."""
        if self.note:
            return [self.note]
        ranked = sorted(self.findings, key=lambda f: abs(f.correlation), reverse=True)
        lines = []
        for f in ranked[:top_n]:
            if f.direction == "no clear relationship":
                continue
            lines.append(
                f'  - "{f.label}": {f.direction} (top performers cluster around {f.best_region}, '
                f"weakest around {f.worst_region})."
            )
        return lines


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else None


def _region_label(values: list[float], is_int: bool) -> str:
    if not values:
        return "n/a"
    lo, hi = min(values), max(values)
    if is_int:
        return f"{round(lo)}-{round(hi)}" if hi > lo else f"{round(lo)}"
    return f"{lo:.3g}-{hi:.3g}" if hi > lo else f"{lo:.3g}"


CORRELATION_NOISE_FLOOR = 0.15  # |correlation| below this is reported as "no clear relationship"


def analyze_gene_fitness_correlation(
    genes: list,
    population: list[tuple[list[float], float]],
    top_fraction: float = 0.3,
) -> PopulationAnalysis:
    """
    genes: the strategy's discovered tunable parameters (gene.label,
        gene.is_int -- see app.optimize.parameter_space).
    population: [(genome, fitness), ...] for one already-evaluated
        generation. Non-finite fitness values (a candidate that produced
        zero trades, e.g.) are excluded from the correlation entirely so
        a pile of -inf failures doesn't swamp the signal from the
        candidates that actually traded.

    Returns a per-gene correlation between that gene's value and fitness
    across the population, plus which value region the top performers
    (top `top_fraction` by fitness) versus the bottom performers cluster
    around -- e.g. "SESSION_START_HOUR: higher values -> worse fitness
    (top performers cluster around 6-9, weakest around 14-22)", which is
    exactly the kind of failure-mode pattern the source framework's Stage
    4 describes finding by hand.
    """
    usable = [(genome, fitness) for genome, fitness in population if math.isfinite(fitness)]
    if len(usable) < 4 or not genes:
        return PopulationAnalysis(
            n_candidates=len(population), n_finite=len(usable),
            note="Not enough evaluated candidates yet this generation to detect a reliable pattern.",
        )

    fitnesses = [f for _, f in usable]
    findings: list[GeneFinding] = []
    n_top = max(1, round(len(usable) * top_fraction))
    ranked = sorted(usable, key=lambda gf: gf[1], reverse=True)
    top_group = ranked[:n_top]
    bottom_group = ranked[-n_top:]

    for gi, gene in enumerate(genes):
        gene_values = [genome[gi] for genome, _ in usable if gi < len(genome)]
        if len(gene_values) < 4:
            continue
        corr = _pearson(gene_values, fitnesses)
        if corr is None:
            continue
        top_values = [genome[gi] for genome, _ in top_group if gi < len(genome)]
        bottom_values = [genome[gi] for genome, _ in bottom_group if gi < len(genome)]
        is_int = bool(getattr(gene, "is_int", False))
        if abs(corr) < CORRELATION_NOISE_FLOOR:
            direction = "no clear relationship"
        elif corr > 0:
            direction = "higher values -> better fitness"
        else:
            direction = "lower values -> better fitness"
        findings.append(GeneFinding(
            label=getattr(gene, "label", f"gene_{gi}"),
            correlation=corr,
            direction=direction,
            best_region=_region_label(top_values, is_int),
            worst_region=_region_label(bottom_values, is_int),
        ))

    return PopulationAnalysis(n_candidates=len(population), n_finite=len(usable), findings=findings)
