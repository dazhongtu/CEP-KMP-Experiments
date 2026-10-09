"""Pure-Python proposed section-5 experiment entry point."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from section5_bo import minimize_ei
from section5_core import (MODEL_DT, REGULARIZATION, experiment_configs,
                           OctaveKMP, load_demonstrations, objective, prepare_cache)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT / "data" / "section5_nokov_processed.mat"
MAX_EVALUATIONS = 300
SEED = 2026
NUM_CASES = 6


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
    for index, config in enumerate(configs, 1):
        print(f"\n=== Proposed {config.label} ({index}/{len(configs)}) ===", flush=True)
        cache = prepare_cache(demos, config, octave)
        bo = minimize_ei(lambda kh, lam: objective(cache, kh, lam, "PROPOSED")[0],
                         bounds, args.max_evals, args.seed)
        loss, details = objective(cache, *bo.x_best, "PROPOSED")
        record = {"case": config.label, "kh": bo.x_best[0], "lambda": bo.x_best[1],
                  "objective": bo.y_best, "checked_objective": loss,
                  "seed": args.seed, "gmr": details["gmr"],
                  "kmp": details["kmp"], "via_points": cache.via_points,
                  "x_history": bo.x_history, "y_history": bo.y_history}
        records.append(record)
        print(f"best kh={record['kh']:.8g} lambda={record['lambda']:.8g} "
              f"loss={loss:.10g} verified={bo.y_best:.10g}", flush=True)
        _save(out, records, "proposed", args.max_evals, args.seed)
    if octave is not None:
        octave.close()
    print(f"\nCompleted {len(records)} proposed cases in {(time.time()-started)/60:.2f} min.")


def _save(out: Path, records: list[dict], name: str, budget: int, seed: int):
    labels = [r["case"] for r in records]
    import pandas as pd
    pd.DataFrame(
        {"Case": r["case"], "kh": r["kh"], "lambda": r["lambda"],
         "objective": r["objective"], "checked_objective": r["checked_objective"],
         "Seed": r["seed"]}
        for r in records
    ).to_csv(out / "ExpProposedOptimizationSummary.csv", index=False)
    meta = {"regularization": REGULARIZATION, "model_dt": MODEL_DT,
            "max_evaluations": budget, "base_seed": seed, "cases": labels}
    (out / "ExpProposedPythonResults.json").write_text(json.dumps(meta, indent=2))
    arrays = {"metadata": np.array(json.dumps(meta))}
    for r in records:
        key = r["case"].replace("-", "_")
        arrays.update({f"gmr_{key}": r["gmr"], f"kmp_{key}": r["kmp"],
                       f"via_{key}": r["via_points"], f"bo_x_{key}": r["x_history"],
                       f"bo_y_{key}": r["y_history"]})
    np.savez_compressed(out / "ExpProposedPythonResults.npz", **arrays)
    savemat(out / "ExpProposedPythonResults.mat", arrays, do_compression=True)
    if records:
        ncols = min(5, len(records))
        nrows = int(np.ceil(len(records) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False)
        for ax, r in zip(axes.flat, records):
            ax.plot(r["gmr"][0], r["gmr"][1], "k--", lw=1.1, label="GMR")
            ax.plot(r["kmp"][0], r["kmp"][1], color="#0878a4", lw=1.4, label="KMP")
            ax.scatter(r["via_points"][:, 0], r["via_points"][:, 1], s=18,
                       c="#e45339", edgecolors="white", linewidths=.4, label="Via")
            ax.set_title(r["case"]); ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.2)
        for ax in axes.flat[len(records):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "ExpProposedPythonTrajectoryComparison.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-evals", type=int, default=MAX_EVALUATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--kmp-backend", choices=["python", "octave"], default="python")
    run(parser.parse_args())
