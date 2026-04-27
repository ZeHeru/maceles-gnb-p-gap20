# MACELES-GNB Phosphorus Models

This private repository contains a cleaned MACE 0.3.14-derived code snapshot and four trained elemental-phosphorus models for the P-GAP-20 benchmark setting.

The repository is intentionally minimal: source code, final inference-ready model files, cleaned training/evaluation scripts, benchmark CSVs, and model cards. Scratch notebooks, one-off comparison scripts, intermediate checkpoints, logs, and local cluster output files were excluded.

## Repository layout

```text
mace/                         # Modified MACE source tree with LES and GNB support
tests/                        # Tests copied from the tracked source tree
models/                       # Four final stage-two phosphorus models
scripts/train_p_gap20/        # Clean distributed training launchers
scripts/eval/                 # Clean P-GAP-20 evaluation script
results/                      # Benchmark CSVs from the finished runs
metadata/                     # Model metadata and validation split indices
docs/model_cards/             # Per-model intended-use notes
```

## Model files

| Model ID | File | Components | Recommended use |
|---|---|---|---|
| `P_MACE_baseline` | `models/P_MACE_baseline_stagetwo.model` | Short-range MACE | Short-range baseline and ablation reference. |
| `P_MACELES` | `models/P_MACELES_stagetwo.model` | MACE + learned LES | Ablation model for learned long-range effects without GNB. |
| `P_MACE_gnb` | `models/P_MACE_gnb_stagetwo.model` | MACE + GNB | Recommended ablation when analytical dispersion is needed without LES. |
| `P_MACELES_gnb` | `models/P_MACELES_gnb_stagetwo.model` | MACELES + GNB | Recommended starting point for production phosphorus simulations in covered regimes. |

All models are for elemental phosphorus only. They are not validated for P-containing compounds, impurities, adsorbates, molecules with other elements, surfaces with foreign species, or reactive chemistry outside the P-GAP-20 configuration space.

## Intended regimes

These models are trained against the phosphorus GAP-style dataset associated with Deringer, Caro, and Csanyi's general-purpose phosphorus force-field work. The reference data target DFT+MBD-quality elemental phosphorus structures, including crystalline allotropes, layered/phosphorene-like configurations, molecular and network liquids, and random-structure-search configurations.

Recommended usage by regime:

| Regime | Recommended model | Notes |
|---|---|---|
| Bulk crystalline phosphorus | `P_MACE_gnb` or `P_MACELES_gnb` | Both GNB variants perform well; compare against the baseline for ablation. |
| 2D/phosphorene and layered structures | `P_MACELES_gnb` | Long-range physics matters for interlayer and exfoliation-like behavior. |
| Molecular liquid P4 | `P_MACE_gnb` or `P_MACELES_gnb` | GNB improves the energy RMSE in the local benchmark. |
| Network liquid phosphorus | `P_MACELES_gnb` or `P_MACELES` | Forces remain challenging; inspect stability in the target thermodynamic state. |
| High-pressure crystalline distortions | `P_MACELES_gnb` | Best local benchmark result among these four models. |
| Random high-energy RSS seeds | Use caution | Full RMSE can be dominated by rare pathological frames; apply geometry sanity checks. |

## Benchmark summary

The benchmark CSVs in `results/` report energy RMSE in meV/atom and force-component RMSE in meV/Angstrom. The table below is copied from `results/p_gap20_rmse_full.csv` for `P_test_set.xyz` only.

| Config type | MACE E/F | MACE+GNB E/F | MACELES E/F | MACELES+GNB E/F | GAP+R6 E/F |
|---|---:|---:|---:|---:|---:|
| `2D` | 10.49 / 45.45 | 4.04 / 43.99 | 9.71 / 52.54 | 4.50 / 40.89 | 13 / 129 |
| `cryst_dist` | 2.45 / 27.40 | 1.82 / 21.45 | 3.55 / 34.00 | 2.10 / 24.43 | 15 / 186 |
| `cryst_dist_hp` | 163.57 / 4786.91 | 877.54 / 2507.71 | 78.77 / 404.85 | 52.39 / 50.36 | - |
| `liq_12_03_01_liqP4` | 5.53 / 47.96 | 2.52 / 34.34 | 9.10 / 43.50 | 3.77 / 34.04 | 8 / 226 |
| `liq_12_03_02_network` | 5.38 / 243.07 | 3.47 / 246.22 | 6.56 / 233.02 | 3.60 / 236.37 | 11 / 201 |
| `liq_P4` | 8.03 / 126.05 | 2.79 / 124.22 | 5.28 / 121.01 | 4.03 / 124.71 | 8 / 226 |
| `rss_200` | 31.38 / 238.02 | 27.41 / 226.49 | 30.86 / 222.99 | 27.60 / 211.58 | 55 / 382 |
| `rss_3c` | 12.97 / 139.76 | 12.67 / 136.57 | 15.15 / 143.45 | 13.86 / 138.93 | 58 / 375 |
| `rss_rnd` | 180.11 / 869.86 | 77.62 / 566.90 | 76.51 / 570.19 | 72.47 / 598.99 | 116 / 694 |

Important benchmark caveat: `rss_005` contains very random high-energy structures. For `P_MACELES_gnb`, the full RMSE is dominated by rare outlier frames, while the top-2% trimmed result is much more representative. Keep both `results/p_gap20_rmse_full.csv` and `results/p_gap20_rmse_trimmed_top2pct.csv` when discussing robustness.

## Installation

Create or activate an environment with the usual MACE dependencies, then install this source tree in editable mode:

```bash
pip install -e .
```

The saved model objects should be loaded with this repository's code snapshot. Upstream MACE may not include the LES/GNB changes needed by these models.
Unrelated upstream foundation model binaries are intentionally not bundled, so this repository is focused on the four phosphorus models.

## Loading a model

```python
from mace.calculators import MACECalculator
from ase.io import read

atoms = read("structure.xyz")
calc = MACECalculator(
    model_paths="models/P_MACELES_gnb_stagetwo.model",
    device="cuda",
    default_dtype="float32",
)
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Training reproduction

The cleaned launchers are in `scripts/train_p_gap20/`. They keep the original hyperparameters but replace machine-specific paths with environment variables where possible.

Example:

```bash
export PROJECT_ROOT=/path/to/maceles-gnb-p-gap20
export PYTHON_BIN=/path/to/python
export P_GAP20_TRAIN_FILE=/path/to/P_GAP_20_fitting_data.xyz
export MLP_WORKER_GPU=4
export MLP_WORKER_NUM=1
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST=127.0.0.1
export MLP_WORKER_0_PORT=29500
bash scripts/train_p_gap20/run_distributed_maceles_gnb.sh
```

Shared training settings:

- `valid_fraction=0.02`, `seed=123`
- `E0s={15: -0.09753304}`
- `num_interactions=2`, `num_channels=192`, `max_L=1`, `correlation=3`
- Short-range cutoff `r_max=4.5` Angstrom
- GNB cutoff `cutoff_lr=12.0` Angstrom for the two GNB variants
- `forces_weight=1000`, `energy_weight=40`
- `max_num_epochs=500`, SWA from epoch 400

The training dataset itself is not bundled in this repository. Check the original data license and redistribution terms before adding it.

## Re-running the benchmark

```bash
python scripts/eval/evaluate_p_gap20.py \
  --models-dir models \
  --test-dir /path/to/testing-framework/tests/P/EFV_test \
  --output-dir results \
  --device cuda
```

This writes recomputed CSV files with full RMSE and top-2% energy-error-trimmed RMSE.

## Provenance

- Source snapshot copied from local repository: `/home/yuzhu/workspace/mace-0.3.14`
- Source base commit: `667eee4`
- Trained model source directory: `/home/yuzhu/workspace/les_fit/MACELES-OFF-P`
- Training date in local artifacts: 2026-04-23
- Validation indices are stored in `metadata/valid_indices/`
- Detailed model checksums and training metadata are stored in `metadata/models.json`

## Checksums

| File | SHA256 |
|---|---|
| `models/P_MACE_baseline_stagetwo.model` | `833fd83292a11182eed67b9533b4de5bac747e58afba847b8a42bb473907cb61` |
| `models/P_MACELES_stagetwo.model` | `5f121e379ac07c5f69fa2239c376a2c001f90b62242407e69ac682f6405397f6` |
| `models/P_MACE_gnb_stagetwo.model` | `8d32453d5c7b544ab04649f9f25708877411502fd3ac9d45ed6f5c752651aa2f` |
| `models/P_MACELES_gnb_stagetwo.model` | `e96ecefe346af501169d95643c47846479b0fb51c9c9678ba3a100f2895c7a98` |

## Citations

If these models are used in a paper or shared project, cite the original MACE work and the phosphorus GAP reference-data/benchmark work. The upstream MACE license is preserved in `LICENSE.md`.
