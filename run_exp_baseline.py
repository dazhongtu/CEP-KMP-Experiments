"""Pure-Python baseline experiment entry point (four losses per case)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat

from section5_bo import minimize_ei
from section5_core import (MODEL_DT, REGULARIZATION, experiment_configs,
                           OctaveKMP, load_demonstrations, objective, prepare_cache)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT / "data" / "section5_nokov_processed.mat"
MAX_EVALUATIONS = 300
SEED = 2026
NUM_CASES = 6
METHODS = ["RSVF_ORIGIN", "DTW", "FRECHT", "PEARSON"]
COLORS = ["#0878a4", "#d94c32", "#258a5b", "#8064a2"]


def run(args):
    data_file = DEFAULT_DATA_FILE
    out = (ROOT / "results").resolve()
    out.mkdir(parents=True, exist_ok=True)
    demos = load_demonstrations(data_file)
    configs = experiment_configs()[:NUM_CASES]
    records = []
    started = time.time()
    octave = (OctaveKMP(Path(__file__).resolve().parent)
              if args.kmp_backend == "octave" else None)
    bounds = np.array([[0.1, 100.0], [1e-3, 5.0]])
    for ci, config in enumerate(configs, 1):
        print(f"\n=== Baseline {config.label} ({ci}/{len(configs)}) ===", flush=True)
        cache = prepare_cache(demos, config, octave)
        for mi, method in enumerate(METHODS):
            print(f"-- {method}", flush=True)
            seed = args.seed
            bo = minimize_ei(lambda kh, lam: objective(cache, kh, lam, method)[0],
                             bounds, args.max_evals, seed)
            loss, details = objective(cache, *bo.x_best, method)
            record = {"case": config.label, "method": method,
                      "kh": bo.x_best[0], "lambda": bo.x_best[1],
                      "objective": loss, "checked_objective": bo.y_best,
                      "seed": seed, "gmr": details["gmr"], "kmp": details["kmp"],
                      "via_points": cache.via_points,
                      "x_history": bo.x_history, "y_history": bo.y_history}
            records.append(record)
            print(f"kh={record['kh']:.8g} lambda={record['lambda']:.8g} "
                  f"loss={loss:.10g} checked={bo.y_best:.10g}", flush=True)
            _save(out, records, args.max_evals, args.seed)
    if octave is not None:
        octave.close()
    print(f"\nCompleted {len(records)} baseline optimizations in {(time.time()-started)/60:.2f} min.")


def _save(out: Path, records: list[dict], budget: int, seed: int):
    pd.DataFrame(
        {"Case": r["case"], "Method": r["method"], "kh": r["kh"],
         "lambda": r["lambda"], "objective": r["objective"],
         "checked_objective": r["checked_objective"], "Seed": r["seed"]}
        for r in records
    ).to_csv(out / "ExpBaselineOptimizationSummary.csv", index=False)
    meta = {"regularization": REGULARIZATION, "model_dt": MODEL_DT,
            "max_evaluations": budget, "base_seed": seed,
            "methods": METHODS}
    arrays = {"metadata": np.array(str(meta))}
    for r in records:
        key = f"{r['case'].replace('-', '_')}_{r['method']}"
        arrays.update({f"gmr_{key}": r["gmr"], f"kmp_{key}": r["kmp"],
                       f"via_{key}": r["via_points"], f"bo_x_{key}": r["x_history"],
                       f"bo_y_{key}": r["y_history"]})
    np.savez_compressed(out / "ExpBaselinePythonResults.npz", **arrays)
    savemat(out / "ExpBaselinePythonResults.mat", arrays, do_compression=True)
    if records:
        cases = list(dict.fromkeys(r["case"] for r in records))
        ncols = min(5, len(cases))
        nrows = int(np.ceil(len(cases) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False)
        for ax, case in zip(axes.flat, cases):
            rs = [r for r in records if r["case"] == case]
            ax.plot(rs[0]["gmr"][0], rs[0]["gmr"][1], "k--", lw=1.1, label="GMR")
            for r, color in zip(rs, COLORS):
                ax.plot(r["kmp"][0], r["kmp"][1], color=color, lw=1.2, label=r["method"])
            ax.scatter(rs[0]["via_points"][:, 0], rs[0]["via_points"][:, 1], s=16,
                       c="#e7a51a", edgecolors="white", linewidths=.4, label="Via")
            ax.set_title(case); ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.2)
        for ax in axes.flat[len(cases):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "ExpBaselinePythonTrajectoryComparison.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-evals", type=int, default=MAX_EVALUATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--kmp-backend", choices=["python", "octave"], default="python")
    run(parser.parse_args())
