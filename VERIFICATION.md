# Verification record

This file records the checks run on this repository before public release, on 2026-11-12/13. It separates **pipeline validation** from **numerical reproduction** of the paper's results.

## Environment

- **Verification environment:** Linux x86-64, 2 CPU cores, no GPU, Python 3.13.18, with the exact versions in `requirements.txt`. A fresh virtual environment was built with `pip install -r requirements.txt` (exit code 0).
- **Original study environment** (Supplementary Section S8): Python 3.10.16, NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.6.1, scikit-survival 0.25.0, lifelines 0.30.0, PyCox 0.3.0, torchtuples 0.2.2, PyTorch 2.14.1+cu126 (CUDA runtime 12.6), Matplotlib 3.10.3 and PyYAML 6.0.2.
  - These are the main packages only; no complete lockfile is available. h5py and transitive dependencies, for example, are not listed.
  - Hardware: about 48 CPU cores and 2 × H100 GPUs (`full_study_config.yaml`, `resources`); neural models were trained on CUDA.

## 1. Documented commands on a fresh clone (clean virtual environment)

| Command | Result |
|---|---|
| `python download_data.py` | 3/3 HDF5 SHA-256 checksums match `DATA_PROVENANCE.md`; SUPPORT 8,873/6,036, METABRIC 1,904/1,103, Rotterdam–GBSG 2,232/1,267 records/events |
| `python stage0_audit.py --data-dir data --out-root outputs` | pass; one METABRIC record (duration 0, censored) flagged, the only exclusion, giving 1,903 analysed patients |
| `python validate_metrics.py` | 16/16 synthetic-data checks pass (IPCW Brier/IBS against complete-data truth, Uno's C, td-AUC, calibration slope/ICI/O-E, shared IBS equal to scikit-survival) |
| `python recover_gbsg_cohort.py` | `verified_complete: True`; partitions 1,546/968 (Rotterdam) and 686/299 (GBSG) |
| `python freeze_design.py` | all three designs admissible; ladders, horizons, t\* and τ as in the protocol |
| `python run_pilot.py --device cpu` | 45/45 tasks valid (3 cohorts × 3 budgets × 5 models) |
| `python estimate_feasibility.py`, `python make_report.py` | pass; report gates G1–G7 all PASS (including no test leakage and same-seed refit determinism, max difference ≤ 4.4 × 10⁻¹⁶) |
| `python run_full_study.py --n-reps 20 --prepare` | 60 split manifests; plan of 1,800 tasks and 12,780 fits |
| `python run_full_study.py --n-reps 20 --collect`, then `python analyze_full_study.py --n-reps 20` | with no tasks run, exits with a clear "no valid tasks" message (fixed in this release; previously a pandas `EmptyDataError`) |

In a separate working copy, an independent first run produced byte-identical data CSVs and split manifests. Its pilot metrics matched to within 9 × 10⁻¹⁶.

## 2. Bounded full-study integration test

This test used Rotterdam–GBSG only, 2 outer splits, all 6 budgets and all 5 models: 60 tasks with the unchanged tuning grids, bootstrap stability on split 0 and conditional IPCW. The test configuration differed from `full_study_config.yaml` only in its dataset list, paths and threads per worker.

| Step | Result |
|---|---|
| `--prepare` | split hashes identical to those of the full 20-split preparation |
| `--worker cpu`, killed with SIGKILL after 90 s | 2 tasks checkpointed and 1 stale lock left behind |
| `--worker cpu` (restart) | stale lock reclaimed; the remaining 34 CPU tasks completed |
| `--worker gpu --device cuda:0` with no GPU | falls back to CPU with a warning; 24 neural tasks completed |
| `--worker cpu` (again) | 0 tasks run (checkpoint resume) |
| `--collect` | 60/60 tasks `ok` |
| `analyze_full_study.py --n-reps 2` | tables T0–T7 and Figures 1–6 and S1 written |
| `--n-reps 3` (30 tasks missing) | `INCOMPLETE_RUN_DIAGNOSTIC.md` written and no findings produced; `--allow-incomplete` writes only to `partial_analysis/` |
| `generate_FigureS3_publication_ready.py` | Figure S3, audit table and methods JSON written |

## 3. Numerical reproduction against the archived study results

The authors' archived aggregate tables from the original full run (T1 performance, T2 paired comparisons, T3 stability) were used as reference. These tables are not part of this repository; their values are the ones reported in the manuscript and Supplementary Materials (for example Table 2 and Tables S8, S9 and S12). Tasks were rerun with the repository's own `run_task` in the verification environment.

| Rerun | Tasks | Agreement with archived means and SDs (all metrics) |
|---|---|---|
| Split sizes: n_train, events, fit/validation partition sizes | 60 splits, 18 cells | identical |
| Ridge Cox: all cohorts, 20 splits, all budgets, with stability and conditional IPCW | 360/360 `ok` | IPA, IBS, Uno's C, td-AUC, Brier, calibration slope, ICI, O/E and all four stability measures; max absolute difference ≤ 5 × 10⁻¹⁰ |
| RSF: Rotterdam–GBSG, 20 splits, all budgets | 120/120 `ok` | same metrics; max absolute difference ≤ 10⁻¹³ |
| Full-budget inference, RSF vs ridge Cox in Rotterdam–GBSG (IPA, Uno's C, Brier) | — | mean, corrected 95% interval and p-value identical to 6 decimals |
| DeepSurv: Rotterdam–GBSG, 20 splits, CPU instead of the original GPU | 120/120 `ok` | **not bit-identical**, as expected for CPU vs GPU training. Cell means: max absolute difference 0.004 in IPA, 0.001 in Uno's C and td-AUC, 0.035 in calibration slope, and 0.010–0.017 in the stability measures. Full-budget mean IPA 0.1258 vs archived 0.1248. |

## What remains unverified

- The full 1,800-task study was not rerun. GBSA, Logistic-Hazard, all neural models in SUPPORT and METABRIC, and RSF in SUPPORT and METABRIC were not compared with the archived results.
- Neural-network results depend on hardware and library versions (GPU vs CPU arithmetic, PyTorch version), so bitwise reproduction of the neural results is not expected outside the original environment.
- The figures in `figures/` are the authors' publication figures. They were not regenerated from a complete rerun.
