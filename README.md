# Python port of the Section-5 KMP experiments

The entry points do not require MATLAB. Python handles GMM/GMR, BO, KMP,
objectives, metrics, and result output by default. An optional persistent GNU
Octave backend can be selected with `--kmp-backend octave`. On the current
machine, repeated Python KMP predictions were faster than Oct2Py calls, so the
default stays Python. Both variants use GMM regularization `4e-3`.

The proposed objective follows the current section-5 flow: GMM/GMR reference,
four via-point insertions, KMP prediction, normalized-arc-length C1 quadratic
Bézier fit with 20 segments, four-node Gauss-Legendre SRVF samples, and the
Euclidean norm of the weighted SRVF vectors. There is no outer DTW in this
objective. Baselines optimize `RSVF_ORIGIN`, `DTW`, `FRECHT`, and `PEARSON`.

## Data

The default data file is:

```text
data/section5_nokov_processed.mat
```

It contains the nine processed demonstrations used in the Section-5
experiments. The variable name is `demos`; each entry has `pos`, `vel`, `acc`,
and `dt`, matching the LASA/PbDlib-style data layout. The file contains only
the data required to run the released experiments.

To regenerate it from the local processed source file, run:

```bash
python prepare_public_data.py
```

## Environment

```bash
cd pythonCode
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python run_exp_proposed.py --max-evals 300 --seed 2026
python run_exp_baseline.py --max-evals 300 --seed 2026
```

Append `--kmp-backend octave` to compare the optional Octave implementation.
Both scripts use the fixed Section-5 data file, 6 experiment cases, and the
`results/` output directory. The optimizer is an independent NumPy/SciPy
Gaussian-process expected-improvement implementation; its trajectory and
objective calculations are intended to match the MATLAB definitions, while its
acquisition implementation is not MATLAB `bayesopt`, so the selected
hyperparameters are not expected to be bit-for-bit identical.

Outputs are placed under `results/` by default. Partial outputs are written
after each completed case/method to support recovery from an interrupted run.
