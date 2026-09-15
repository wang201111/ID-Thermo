# ID-Thermo

Identifiability-driven correction of Pitzer mixing parameters for cross-system
solid-liquid equilibrium prediction.

Code and data accompanying the manuscript:

> *Identifiability-Constrained Correction of Pitzer Mixing Parameters Enables
> Cross-System Prediction for Quaternary Electrolyte Systems*

ID-Thermo uses ternary first-saturation data to determine a compact correction
to the Pitzer mixing layer. Four same-charge pair corrections are fitted
directly, while nine triplet corrections are generated from five shared ion
factors:

```text
delta_psi(i,j,k) = F_i * F_j * F_k
```

The fitted corrections are written back to a standard Pitzer theta/psi table.
Quaternary measurements are held out from training and are used only for the
transfer evaluation.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the package:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For the repository tests:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The thermodynamic calculations use double precision throughout.

## Quick start

The shortest check of the published model is the held-out quaternary transfer
calculation using the frozen parameter file:

```bash
python experiments/03_quaternary_transfer.py
```

Expected ID-Thermo results are:

| System | R2 | RMSE (wt%) |
| --- | ---: | ---: |
| Na-K-Cl-SO4 | 0.868 | 1.532 |
| Na-Mg-Cl-SO4 | 0.729 | 1.608 |
| K-Mg-Cl-SO4 | 0.811 | 1.374 |

## Reproducing the paper calculations

| Script | Calculation | Main output |
| --- | --- | --- |
| `experiments/01_identifiability.py` | Local theta-psi sensitivities and the global 9-parameter Jacobian | `results/identifiability_*` |
| `experiments/02_train.py` | Deterministic training on the 4211 ternary records | `results/training_reproduction/` |
| `experiments/03_quaternary_transfer.py` | Zero-mixing, THEREDA, and frozen ID-Thermo transfer | `results/quaternary_transfer.csv` |
| `experiments/04_ml_baselines.py` | Three feature sets by five regression models | `results/ml_baselines_reproduced.csv` |
| `experiments/05_component_ablation.py` | Post-freeze add-one and leave-one-out diagnostics | `results/component_ablation.csv` |

Run the scripts from the repository root. For example:

```bash
python experiments/01_identifiability.py
python experiments/03_quaternary_transfer.py
python experiments/05_component_ablation.py
```

The full training run is started with:

```bash
python experiments/02_train.py --out-dir results/training_reproduction
```

The machine-learning comparison can be run in full or one configuration at a
time:

```bash
python experiments/04_ml_baselines.py
python experiments/04_ml_baselines.py --feature-group T1 --model Ridge
```

## Repository layout

```text
ID-Thermo/
├── data/
│   ├── ternary/              4211 training records from nine subsystems
│   ├── quaternary/           three held-out quaternary evaluation sets
│   └── thermodynamic/        parameter tables and solid-phase inputs
├── src/idthermo/             Pitzer/SLE solver, model, training and analysis
├── experiments/              paper-level reproduction scripts
├── model/                    frozen ID-Thermo parameter file
├── results/                  reference numerical outputs
├── tests/                    data and headline-result checks
└── pyproject.toml
```

## Data

The ternary training set contains 4211 first-saturation records from nine
subsystems over 0-100 degC. The data were compiled from the four solubility
handbooks cited as references [28]-[31] in the manuscript. The public tables
use a normalized source registry: measurement rows contain a `source_id`, and
the corresponding bibliographic entry is stored once in
`data/ternary/source_entries.csv`.

The quaternary data are stored separately and are never read by the training
script. Their record counts are 75 for Na-K-Cl-SO4, 82 for Na-Mg-Cl-SO4, and
25 for K-Mg-Cl-SO4.

All CSV files are UTF-8 with a byte-order mark so they open correctly in Excel
on Windows. Dataset fields, source keys, units, and provenance tables are
documented in [`data/README.md`](data/README.md).

## Thermodynamic inputs

`data/thermodynamic/` contains the coefficient tables used by the calculation
code in this repository. The binary parameter table, THEREDA theta and psi
reference tables, solid-phase reactions, and temperature-dependent solid
`log K` fits are loaded directly by `ThermodynamicDatabase`.

The learned ID-Thermo correction is stored in
`model/idthermo_parameters.json`. It contains four constant delta-theta values,
five shared ion factors, and the nine delta-psi values generated from those
factors.

## Reproducibility notes

- ID-Thermo training uses only the 4211 ternary records.
- The loss is ordinary least squares with no parameter regularization.
- The shared model contains four delta-theta parameters and five ion factors.
- Shared-factor starts are constructed deterministically from the auxiliary
  independent-psi solution.
- The three quaternary systems do not participate in parameter fitting,
  checkpoint selection, or machine-learning hyperparameter selection.
- The Gaussian-process baseline uses the fixed 1200-record sampling rule
  described in the manuscript.

## Citation

Please cite the accompanying manuscript when using this repository or its
corrected Pitzer parameter table.
