"""Independent NumPy/SciPy implementation of the section-5 experiment core.

This module expresses the experiment equations directly and does not invoke
MATLAB or import MATLAB source files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.linalg import solve
from scipy.special import logsumexp


REGULARIZATION = 4e-3
DEMO_DT = 0.01
MODEL_DT = 0.005
NUM_STATES = 8
NUM_BEZIER_SEGMENTS = 20
GLQ_X = np.array([-0.8611363115940526, -0.3399810435848563,
                  0.3399810435848563, 0.8611363115940526])
GLQ_W = np.array([0.3478548451374538, 0.6521451548625461,
                  0.6521451548625461, 0.3478548451374538])


@dataclass
class ExperimentConfig:
    label: str
    via_indices: np.ndarray
    via_bias: np.ndarray


@dataclass
class Cache:
    config: ExperimentConfig
    demos: list[np.ndarray]
    data_out: np.ndarray
    sigma_out: np.ndarray
    ref_times: np.ndarray
    ref_mean: np.ndarray
    ref_cov: np.ndarray
    via_points: np.ndarray
    ref_srvf: np.ndarray | None
    position_precision: np.ndarray
    octave_backend: Any | None = None


class OctaveKMP:
    """Persistent Octave process used only for repeated KMP linear solves."""

    def __init__(self, matlab_compatible_dir: str | Path):
        from oct2py import Oct2Py
        self.octave = Oct2Py()
        self.octave.addpath(str(Path(matlab_compatible_dir).resolve()))

    def set_problem(self, cache_parts):
        ref_times, ref_mean, ref_cov, query_times = cache_parts
        self.octave.kmp_setup_octave(
            np.asarray(ref_times, float).reshape(1, -1),
            np.asarray(ref_mean, float).T,
            np.asarray(ref_cov, float).transpose(1, 2, 0),
            np.asarray(query_times, float).reshape(1, -1), nout=0)

    def predict(self, kh: float, lam: float) -> np.ndarray:
        value = self.octave.kmp_predict_octave(float(kh), float(lam), nout=1)
        return np.asarray(value, float)

    def close(self):
        self.octave.exit()


def experiment_configs() -> list[ExperimentConfig]:
    """Section-5 via configurations used in the paper."""
    configs: list[ExperimentConfig] = []
    bias_sets = [
        ("S1-1", [0, -20, 0, 0], [0, 20, 0, 0]),
        ("S1-2", [0, -60, 0, 0], [0, 60, 0, 0]),
        ("S1-3", [0, -100, 0, 0], [0, 100, 0, 0]),
    ]
    for label, b2, b3 in bias_sets:
        configs.append(ExperimentConfig(
            label, np.array([1, 130, 240, 400]),
            np.array([[0, 0, 0, 0], b2, b3, [0, 0, 0, 0]], float)))
    bias_sets = [
        ("S2-1", [0, 20, 0, 0], [10, -10, 0, 0]),
        ("S2-2", [0, 60, 0, 0], [30, -30, 0, 0]),
        ("S2-3", [0, 100, 0, 0], [50, -50, 0, 0]),
    ]
    for label, b2, b3 in bias_sets:
        configs.append(ExperimentConfig(
            label, np.array([1, 120, 200, 400]),
            np.array([[0, 0, 0, 0], b2, b3, [0, 0, 0, 0]], float)))
    return configs


def load_demonstrations(mat_path: str | Path, count: int = 9) -> list[np.ndarray]:
    raw = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)["demos"]
    if len(raw) < count:
        raise ValueError(f"Expected {count} demos, found {len(raw)}")
    return [np.asarray(raw[i].pos, dtype=float)[:2, :] for i in range(count)]


def _build_training_data(demos: list[np.ndarray]) -> np.ndarray:
    n = demos[0].shape[1]
    blocks = []
    for pos in demos:
        vel = np.diff(pos, axis=1) / DEMO_DT
        vel = np.column_stack((vel, vel[:, -1]))
        time = np.arange(1, n + 1, dtype=float)[None, :] * DEMO_DT
        blocks.append(np.vstack((time, pos, vel)))
    return np.concatenate(blocks, axis=1)


def _initialize_gmm(data: np.ndarray, states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = data[0]
    edges = np.linspace(t.min(), t.max(), states + 1)
    priors = np.zeros(states)
    means = np.zeros((data.shape[0], states))
    covs = np.zeros((states, data.shape[0], data.shape[0]))
    for k in range(states):
        ids = (t >= edges[k]) & (t < edges[k + 1])
        subset = data[:, ids]
        if subset.shape[1] == 0:
            raise ValueError(f"Empty GMM initialization bin {k}")
        priors[k] = subset.shape[1]
        means[:, k] = subset.mean(axis=1)
        covs[k] = np.cov(subset, bias=False) + np.eye(data.shape[0]) * 1e-8
    priors /= priors.sum()
    return priors, means, covs


def _log_gaussian(data: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = data.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + np.eye(d) * 1e-9
        sign, logdet = np.linalg.slogdet(cov)
    delta = data - mean[:, None]
    mahal = np.sum(delta * np.linalg.solve(cov, delta), axis=0)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)


def _fit_gmm(data: np.ndarray, states: int = NUM_STATES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    priors, means, covs = _initialize_gmm(data, states)
    n = data.shape[1]
    previous_ll = -np.inf
    for iteration in range(100):
        logp = np.vstack([np.log(priors[k] + np.finfo(float).tiny)
                          + _log_gaussian(data, means[:, k], covs[k])
                          for k in range(states)])
        log_norm = logsumexp(logp, axis=0)
        gamma = np.exp(logp - log_norm)
        ll = float(np.sum(log_norm) / n)
        mass = gamma.sum(axis=1)
        priors = mass / n
        means = (gamma @ data.T / mass[:, None]).T
        new_covs = np.empty_like(covs)
        for k in range(states):
            delta = data - means[:, k, None]
            new_covs[k] = (delta * gamma[k][None, :]) @ delta.T / mass[k]
            new_covs[k] += np.eye(data.shape[0]) * REGULARIZATION
        covs = new_covs
        if iteration + 1 > 5 and ll - previous_ll < 1e-4:
            break
        previous_ll = ll
    return priors, means, covs


def _gmr(priors: np.ndarray, means: np.ndarray, covs: np.ndarray,
         query_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    components, out_dim = len(priors), means.shape[0] - 1
    mu_out = np.empty((out_dim, query_times.size))
    sigma_out = np.empty((query_times.size, out_dim, out_dim))
    for j, t in enumerate(query_times):
        log_h = np.empty(components)
        cond_mu = np.empty((components, out_dim))
        cond_cov = np.empty((components, out_dim, out_dim))
        for k in range(components):
            var_t = covs[k, 0, 0]
            delta = t - means[0, k]
            log_h[k] = (np.log(priors[k] + np.finfo(float).tiny)
                        - 0.5 * (np.log(2 * np.pi * var_t) + delta * delta / var_t))
            cross = covs[k, 1:, 0]
            cond_mu[k] = means[1:, k] + cross * (delta / var_t)
            cond_cov[k] = covs[k, 1:, 1:] - np.outer(cross, cross) / var_t
        h = np.exp(log_h - logsumexp(log_h))
        mean = h @ cond_mu
        second = np.zeros((out_dim, out_dim))
        for k in range(components):
            second += h[k] * (cond_cov[k] + np.outer(cond_mu[k], cond_mu[k]))
        mu_out[:, j] = mean
        sigma_out[j] = second - np.outer(mean, mean) + np.eye(out_dim) * 1e-8
    return mu_out, sigma_out


def _insert_via(times: np.ndarray, means: np.ndarray, covs: np.ndarray,
                time: float, value: np.ndarray, covariance: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nearest = int(np.argmin(np.abs(times - time)))
    if abs(times[nearest] - time) < 5e-4:
        means = means.copy()
        covs = covs.copy()
        means[nearest] = value
        covs[nearest] = covariance
    else:
        times = np.append(times, time)
        means = np.vstack((means, value))
        covs = np.concatenate((covs, covariance[None, :, :]), axis=0)
        order = np.argsort(times)
        times, means, covs = times[order], means[order], covs[order]
    return times, means, covs


def _extended_kernel(query: np.ndarray, reference: np.ndarray, kh: float) -> np.ndarray:
    fd = 0.001
    delta = query[:, None] - reference[None, :]
    k = np.exp(-kh * delta**2)
    k0d_tmp = np.exp(-kh * (delta - fd)**2)
    kd0_tmp = np.exp(-kh * (delta + fd)**2)
    k0d = (k0d_tmp - k) / fd
    kd0 = (kd0_tmp - k) / fd
    kdd = (2 * k - k0d_tmp - kd0_tmp) / fd**2
    nq, nr = len(query), len(reference)
    out = np.zeros((4 * nq, 4 * nr))
    for axis in range(2):
        qi, qv = np.arange(nq) * 4 + axis, np.arange(nq) * 4 + axis + 2
        ri, rv = np.arange(nr) * 4 + axis, np.arange(nr) * 4 + axis + 2
        out[np.ix_(qi, ri)] = k
        out[np.ix_(qi, rv)] = k0d
        out[np.ix_(qv, ri)] = kd0
        out[np.ix_(qv, rv)] = kdd
    return out


def _bezier_constraint_matrix(segments: int) -> np.ndarray:
    rows = []
    ncp = 3 * segments
    for k in range(segments - 1):
        # Position continuity: P2(k) = P0(k+1).
        row = np.zeros(ncp)
        row[3 * k + 2], row[3 * (k + 1)] = 1.0, -1.0
        rows.append(row)
        # First-derivative continuity under equal segment parameter spans.
        row = np.zeros(ncp)
        row[3 * k + 2], row[3 * k + 1] = 1.0, -1.0
        row[3 * (k + 1) + 1], row[3 * (k + 1)] = -1.0, 1.0
        rows.append(row)
    return np.asarray(rows)


def fit_piecewise_quadratic_bezier(curve: np.ndarray,
                                    segments: int = NUM_BEZIER_SEGMENTS) -> np.ndarray:
    """Least-squares C1 piecewise quadratic fit using normalized arc length."""
    xy = np.asarray(curve, float).T
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(ds)]
    if s[-1] <= np.finfo(float).eps:
        raise ValueError("Cannot fit a zero-length trajectory")
    s /= s[-1]
    seg = np.minimum((s * segments).astype(int), segments - 1)
    u = np.where(s >= 1, 1.0, s * segments - seg)
    b = np.column_stack(((1 - u) ** 2, 2 * u * (1 - u), u**2))
    design = np.zeros((len(s), 3 * segments))
    for i in range(len(s)):
        design[i, 3 * seg[i]:3 * seg[i] + 3] = b[i]
    constraint = _bezier_constraint_matrix(segments)
    # Solve equality-constrained least squares in the null space of C1 joins.
    _, singular, vh = np.linalg.svd(constraint, full_matrices=True)
    rank = int(np.sum(singular > singular[0] * 1e-12))
    null = vh[rank:].T
    reduced = design @ null
    coeff = np.linalg.lstsq(reduced, xy, rcond=None)[0]
    return (null @ coeff).reshape(segments, 3, 2)


def bezier_srvf(controls: np.ndarray) -> np.ndarray:
    nodes = 0.5 * (GLQ_X + 1.0)
    weights = 0.5 * GLQ_W
    rows = []
    for cp in controls:
        for t, w in zip(nodes, weights):
            derivative = 2 * (1 - t) * (cp[1] - cp[0]) + 2 * t * (cp[2] - cp[1])
            speed = np.linalg.norm(derivative)
            if speed < 1e-4:
                q = np.zeros(2)
            else:
                q = derivative / np.sqrt(np.sqrt(speed**2 + 1e-8))
            rows.append(q * np.sqrt(w))
    q = np.asarray(rows)
    # MATLAB unique(...,'rows','stable') removes duplicate q rows while retaining order.
    _, first = np.unique(q, axis=0, return_index=True)
    return q[np.sort(first)]


def srvf_glq_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"SRVF sample counts differ after duplicate removal: {a.shape} vs {b.shape}")
    return float(np.linalg.norm(a - b))


def prepare_cache(demos: list[np.ndarray], config: ExperimentConfig,
                  octave_backend: OctaveKMP | None = None) -> Cache:
    data = _build_training_data(demos)
    priors, means, covs = _fit_gmm(data)
    n_samples = round(demos[0].shape[1] * DEMO_DT / MODEL_DT)
    query_times = np.arange(1, n_samples + 1, dtype=float) * MODEL_DT
    data_out, sigma_out = _gmr(priors, means, covs, query_times)
    ref_times = query_times.copy()
    ref_mean = data_out.T.copy()
    ref_cov = sigma_out.copy()
    via_points = np.empty((config.via_indices.size, 4))
    via_cov = np.eye(4) * 1e-6
    via_cov[2:, 2:] = np.eye(2) * 1000.0
    for j, idx in enumerate(config.via_indices):
        via_points[j] = data_out[:, idx - 1] + config.via_bias[j]
        ref_times, ref_mean, ref_cov = _insert_via(
            ref_times, ref_mean, ref_cov, idx * MODEL_DT,
            via_points[j], via_cov)
    pos_cov = sigma_out[:, :2, :2]
    precision = np.linalg.inv(pos_cov)
    reference_controls = fit_piecewise_quadratic_bezier(data_out[:2, :])
    reference_srvf = bezier_srvf(reference_controls)
    cache = Cache(config, demos, data_out, sigma_out, ref_times, ref_mean,
                  ref_cov, via_points, reference_srvf, precision, octave_backend)
    if octave_backend is not None:
        octave_backend.set_problem((ref_times, ref_mean, ref_cov, query_times))
    return cache


def predict_kmp(cache: Cache, kh: float, lam: float) -> np.ndarray:
    if cache.octave_backend is not None:
        return cache.octave_backend.predict(kh, lam)
    train = _extended_kernel(cache.ref_times, cache.ref_times, kh)
    system = train.copy()
    # The reference covariance is block diagonal in the time-major [x,y,vx,vy] order.
    for i, cov in enumerate(cache.ref_cov):
        sl = slice(4 * i, 4 * i + 4)
        system[sl, sl] += lam * cov
    alpha = solve(system, cache.ref_mean.reshape(-1), assume_a="sym", check_finite=False)
    query_times = np.arange(1, cache.data_out.shape[1] + 1, dtype=float) * MODEL_DT
    if np.array_equal(query_times, cache.ref_times):
        query_kernel = train
    else:
        query_kernel = _extended_kernel(query_times, cache.ref_times, kh)
    pred = query_kernel @ alpha
    return pred.reshape(-1, 4).T


def objective(cache: Cache, kh: float, lam: float, metric: str) -> tuple[float, dict[str, Any]]:
    kmp = predict_kmp(cache, kh, lam)
    gmr = cache.data_out
    key = metric.upper()
    if key == "PROPOSED":
        generated_srvf = bezier_srvf(fit_piecewise_quadratic_bezier(kmp[:2, :]))
        loss = srvf_glq_distance(cache.ref_srvf, generated_srvf)
    elif key == "RSVF_ORIGIN":
        loss = float(np.linalg.norm(gmr[:2, :].T - kmp[:2, :].T))
    elif key == "MAP":
        delta = (gmr[:2, :] - kmp[:2, :]).T
        loss = float(np.mean(np.einsum("ni,nij,nj->n", delta,
                                       cache.position_precision, delta)))
    elif key == "DTW":
        loss = dtw_distance(gmr[:2, :].T, kmp[:2, :].T)
    elif key == "FRECHT":
        loss = discrete_frechet(gmr[:2, :].T, kmp[:2, :].T)
    elif key == "PEARSON":
        a = np.r_[gmr[0], gmr[1]]
        b = np.r_[kmp[0], kmp[1]]
        corr = np.corrcoef(a, b)[0, 1]
        loss = float(-corr) if np.isfinite(corr) else 1.0
    else:
        raise ValueError(f"Unknown objective {metric}")
    if not np.isfinite(loss):
        raise FloatingPointError(f"Non-finite {metric} objective")
    return loss, {"gmr": gmr, "kmp": kmp, "via_points": cache.via_points,
                  "kh": kh, "lambda": lam}


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Use the same DTW metric package as the MATLAB bridge did."""
    import similaritymeasures
    result = similaritymeasures.dtw(np.asarray(a, float), np.asarray(b, float))
    return float(result[0])


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Use the same discrete Frechet metric package as the MATLAB bridge."""
    import similaritymeasures
    return float(similaritymeasures.frechet_dist(np.asarray(a, float), np.asarray(b, float)))


def evaluate_candidates(data_file: str | Path, max_cases: int = 1) -> dict[str, Cache]:
    demos = load_demonstrations(data_file)
    configs = experiment_configs()[:max_cases]
    return {cfg.label: prepare_cache(demos, cfg) for cfg in configs}
