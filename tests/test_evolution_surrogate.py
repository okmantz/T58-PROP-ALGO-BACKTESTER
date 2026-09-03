"""Tests for app.evolution.surrogate -- the pure-numpy GP surrogate used to
guide Evolution Lab candidate proposal."""
from __future__ import annotations

import numpy as np

from app.evolution.surrogate import FamilySurrogateBank, GPSurrogate


def test_gp_fit_requires_minimum_points():
    gp = GPSurrogate()
    assert gp.fit(np.zeros((2, 3)), np.zeros(2)) is False


def test_gp_predicts_higher_mean_near_observed_high_fitness_point():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(30, 2))
    # Fitness peaks at the center (0.5, 0.5).
    y = -np.sum((X - 0.5) ** 2, axis=1)
    gp = GPSurrogate(length_scale=0.3)
    assert gp.fit(X, y) is True

    near_center = np.array([[0.5, 0.5]])
    near_corner = np.array([[0.0, 0.0]])
    mean_center, _ = gp.predict(near_center)
    mean_corner, _ = gp.predict(near_corner)
    assert mean_center[0] > mean_corner[0]


def test_gp_uncertainty_is_higher_far_from_observed_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.4, 0.6, size=(20, 2))  # observations clustered tightly
    y = rng.normal(0, 1, size=20)
    gp = GPSurrogate(length_scale=0.2)
    assert gp.fit(X, y) is True

    near_data, far_from_data = np.array([[0.5, 0.5]]), np.array([[0.99, 0.99]])
    _, std_near = gp.predict(near_data)
    _, std_far = gp.predict(far_from_data)
    assert std_far[0] > std_near[0]


def test_bank_returns_none_below_min_observations():
    bank = FamilySurrogateBank(min_observations=5)
    rng = np.random.default_rng(0)
    for _ in range(3):
        bank.observe("fam", rng.uniform(0, 1, size=3), 1.0)
    assert bank.propose("fam", 2, rng, 3) is None


def test_bank_proposes_once_enough_observations_exist():
    bank = FamilySurrogateBank(min_observations=5)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(10, 3))
    y = -np.sum((X - 0.5) ** 2, axis=1)
    for xi, yi in zip(X, y):
        bank.observe("fam", xi, yi)

    proposal = bank.propose("fam", 4, rng, 3, pool_size=100)
    assert proposal is not None
    assert proposal.shape == (4, 3)
    assert np.all((proposal >= 0.0) & (proposal <= 1.0))


def test_bank_ignores_non_finite_fitness():
    bank = FamilySurrogateBank(min_observations=1)
    bank.observe("fam", np.array([0.5, 0.5]), float("-inf"))
    assert bank.n_observations("fam") == 0


def test_bank_caps_history_per_family():
    bank = FamilySurrogateBank(min_observations=1, max_observations_per_family=10)
    rng = np.random.default_rng(0)
    for i in range(25):
        bank.observe("fam", rng.uniform(0, 1, size=2), float(i))
    assert bank.n_observations("fam") <= 10


def test_different_families_keep_independent_history():
    bank = FamilySurrogateBank(min_observations=1)
    bank.observe("fam_a", np.array([0.1, 0.1]), 1.0)
    assert bank.n_observations("fam_a") == 1
    assert bank.n_observations("fam_b") == 0
