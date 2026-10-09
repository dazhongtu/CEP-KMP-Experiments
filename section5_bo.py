"""Small deterministic Gaussian-process Bayesian optimizer (minimization)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr
from scipy.stats import qmc


@dataclass
class BOResult:
    x_best: np.ndarray
    y_best: float
    x_history: np.ndarray
    y_history: np.ndarray


def _matern52(x: np.ndarray, z: np.ndarray, length: float = 0.22) -> np.ndarray:
    delta = (x[:, None, :] - z[None, :, :]) / length
    r = np.sqrt(np.sum(delta * delta, axis=2) + 1e-30)
    root5r = np.sqrt(5.0) * r
    return (1.0 + root5r + 5.0 * r * r / 3.0) * np.exp(-root5r)


def minimize_ei(objective, bounds: np.ndarray, evaluations: int,
                seed: int = 2026, initial_points: int = 5,
                candidate_count: int = 384) -> BOResult:
    """EI search over a 2-D box; returns all evaluated points for auditing."""
    bounds = np.asarray(bounds, float)
    low, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    if np.any(span <= 0):
        raise ValueError("Each optimization bound must have positive width")
    rng = np.random.default_rng(seed)
    n0 = min(initial_points, evaluations)
    unit = qmc.LatinHypercube(d=len(bounds), seed=seed).random(n0)
    xs: list[np.ndarray] = []
    ys: list[float] = []

    def evaluate(u):
        physical = low + np.asarray(u) * span
        value = float(objective(*physical))
        if not np.isfinite(value):
            value = 1e100
        xs.append(np.asarray(u, float))
        ys.append(value)
        return value

    for row in unit:
        evaluate(row)

    while len(ys) < evaluations:
        x = np.asarray(xs)
        y = np.asarray(ys)
        y_center = float(np.mean(y))
        y_scale = float(np.std(y))
        if not np.isfinite(y_scale) or y_scale < 1e-12:
            y_scale = 1.0
        yn = (y - y_center) / y_scale
        kernel = _matern52(x, x)
        kernel.flat[::len(kernel) + 1] += 1e-8
        factor = cho_factor(kernel, lower=True, check_finite=False)
        alpha = cho_solve(factor, yn, check_finite=False)
        candidates = rng.random((candidate_count, bounds.shape[0]))
        cross = _matern52(candidates, x)
        mean = cross @ alpha
        solved = cho_solve(factor, cross.T, check_finite=False)
        variance = np.maximum(1.0 - np.sum(cross * solved.T, axis=1), 1e-14)
        std = np.sqrt(variance)
        best = float(np.min(yn))
        xi = 0.01 + 0.02 * min(1.0, len(ys) / max(evaluations, 1))
        improvement = best - mean - xi
        z = improvement / std
        ei = improvement * ndtr(z) + std * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        # Keep the optimizer away from exact repeats in the random candidate pool.
        duplicate_distance = np.min(np.linalg.norm(candidates[:, None, :] - x[None, :, :], axis=2), axis=1)
        ei[duplicate_distance < 1e-7] = -np.inf
        evaluate(candidates[int(np.argmax(ei))])

    history = low + np.asarray(xs) * span
    values = np.asarray(ys)
    best_id = int(np.argmin(values))
    return BOResult(history[best_id], float(values[best_id]), history, values)

