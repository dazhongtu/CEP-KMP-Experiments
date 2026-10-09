"""Create the small public Section-5 demonstration file.

The released MAT file contains only the nine processed demonstrations used by
the Python experiments. It intentionally stores the minimum required variable:
`demos`, a 1x9 MATLAB-style cell array with fields `pos`, `vel`, `acc`, and
`dt`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT.parent / "Simulation" / "section5" / "group1_LASA_style.mat"
OUTPUT_FILE = ROOT / "data" / "section5_nokov_processed.mat"


def main() -> None:
    raw = loadmat(SOURCE_FILE, squeeze_me=True, struct_as_record=False)["demos"]
    if len(raw) != 9:
        raise ValueError(f"Expected 9 processed demonstrations, found {len(raw)}")

    demos = np.empty((1, len(raw)), dtype=object)
    for i, demo in enumerate(raw):
        demos[0, i] = {
            "pos": np.asarray(demo.pos, dtype=float),
            "vel": np.asarray(demo.vel, dtype=float),
            "acc": np.asarray(demo.acc, dtype=float),
            "dt": float(np.asarray(demo.dt).squeeze()),
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    savemat(OUTPUT_FILE, {"demos": demos}, do_compression=True)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
