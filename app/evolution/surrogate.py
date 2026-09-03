"""
Surrogate-model-guided candidate proposal for the Evolution Lab.

Replaces blind mutation/crossover for the "breed the elites into next
generation's children" step (app.evolution.engine.EvolutionRunner.
_generate_population) with a small Gaussian Process regressor fit on
every manual-config candidate this run has already fully evaluated
(genome values -> final fitness score), per family. Instead of randomly
perturbing an elite's genes and hoping, each new generation proposes a
pool of random candidate genomes and scores them by the GP's own
Upper-Confidence-Bound (mean + kappa * std) -- picking points the model
either expects to score well, or is still uncertain about, rather than
picking blind.

Deliberately pure numpy (no scipy/scikit-learn dependency -- see
pyproject.toml, which pins only pandas+numpy): a plain RBF-kernel GP
via Cholesky decomposition is a few dozen lines and entirely sufficient
for the low-dimensional (typically 2-8 gene), small-sample (dozens to a
few hundred observations per family per run) regime a single Evolution
Lab run operates in. This is NOT meant to compete with a production
Bayesian-optimization library -- it's meant to stop the GA from ever
proposing a child that's a pure coin flip when it already has enough
history to make an informed guess.

Falls back to returning None (caller falls back to plain mutation) when
there isn't yet enough observed history for a family, or if the GP fit
fails for any reason (e.g. a degenerate/singular kernel matrix) -- this
must never be able to crash or stall an Evolution Lab run; it can only
ever make candidate proposals SMARTER, never required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Observation:
    x: np.ndarray   # normalized genome, each dimension in [0, 1]
    y: float        # fitness (raw scale; normalized internally before fitting)


class GPSurrogate:
    """A minimal RBF-kernel Gaussian Process regressor: fit(X, y), then
    predict(X_star) -> (mean, std). Everything operates in normalized
    [0, 1]^d gene space so one length-scale works across every gene
    regardless of that gene's own lo/hi range."""

    def __init__(self, length_scale: float = 0.25, noise: float = 1e-3, signal_var: float = 1.0):
        self.length_scale = length_scale
        self.noise = noise
        self.signal_var = signal_var
        self._X: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Squared Euclidean distance via the (a-b)^2 = a^2 + b^2 - 2ab expansion,
        # vectorized -- avoids an O(n*m*d) Python-level loop.
        a2 = np.sum(A ** 2, axis=1).reshape(-1, 1)
        b2 = np.sum(B ** 2, axis=1).reshape(1, -1)
        sq_dist = np.maximum(a2 + b2 - 2 * A @ B.T, 0.0)
        return self.signal_var * np.exp(-0.5 * sq_dist / (self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Returns True on success, False if the fit couldn't be
        completed (e.g. singular matrix) -- callers must treat False the
        same as \"no surrogate available yet\"."""
        if len(X) < 3:
            return False
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        y_norm = (y - self._y_mean) / self._y_std
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        except np.linalg.LinAlgError:
            return False
        if not (np.all(np.isfinite(L)) and np.all(np.isfinite(alpha))):
            return False
        self._X, self._L, self._alpha = X, L, alpha
        return True

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) in the ORIGINAL fitness scale, for each row
        of X_star. Must only be called after a successful fit()."""
        k_star = self._kernel(X_star, self._X)          # (n_star, n_train)
        mean_norm = k_star @ self._alpha
        v = np.linalg.solve(self._L, k_star.T)           # (n_train, n_star)
        var_norm = self.signal_var - np.sum(v ** 2, axis=0)
        var_norm = np.maximum(var_norm, 1e-9)
        mean = mean_norm * self._y_std + self._y_mean
        std = np.sqrt(var_norm) * self._y_std
        return mean, std


@dataclass
class FamilySurrogateBank:
    """Owns one GPSurrogate + observation history per family for the
    lifetime of one Evolution Lab run. Not persisted across
    checkpoint/resume -- a soft search-quality optimization, never
    correctness-critical, so losing this history on resume just means
    the surrogate re-warms up over the next few generations rather than
    corrupting anything."""
    min_observations: int = 8
    max_observations_per_family: int = 400   # cap so a long run's fit cost doesn't grow unbounded
    kappa: float = 1.5                        # UCB exploration weight
    _history: dict[str, list[_Observation]] = field(default_factory=dict)

    def observe(self, family: str, genome_norm: np.ndarray, fitness: float) -> None:
        if not math.isfinite(fitness):
            return
        hist = self._history.setdefault(family, [])
        hist.append(_Observation(x=genome_norm, y=float(fitness)))
        if len(hist) > self.max_observations_per_family:
            # Drop the oldest half rather than trimming one at a time --
            # cheap, and keeps a genuine mix of early/recent observations
            # rather than only ever remembering the most recent handful.
            del hist[: len(hist) // 2]

    def n_observations(self, family: str) -> int:
        return len(self._history.get(family, []))

    def propose(
        self, family: str, n: int, rng: np.random.Generator, n_genes: int, pool_size: int = 300,
    ) -> np.ndarray | None:
        """Returns an (n, n_genes) array of normalized [0,1] genomes to
        try next for `family`, chosen by UCB over a random candidate
        pool -- or None if there isn't enough history yet / the fit
        failed, so the caller should fall back to plain mutation."""
        hist = self._history.get(family, [])
        if len(hist) < self.min_observations:
            return None
        X = np.array([o.x for o in hist])
        y = np.array([o.y for o in hist])
        if X.shape[1] != n_genes:
            return None  # gene count changed mid-run (shouldn't happen) -- be safe, not clever

        gp = GPSurrogate(length_scale=max(0.15, 1.0 / max(n_genes, 1) ** 0.5))
        if not gp.fit(X, y):
            return None

        pool = rng.uniform(0.0, 1.0, size=(pool_size, n_genes))
        mean, std = gp.predict(pool)
        ucb = mean + self.kappa * std
        top_idx = np.argsort(-ucb)[:n]
        return pool[top_idx]
