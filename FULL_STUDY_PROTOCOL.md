# Full-study protocol

*When Does Model Complexity Pay Off? An Event-Budget Study of Discrimination, Calibration, and
Stability in Clinical Survival Prediction*

Data provenance, units, endpoints, the analysis-set rule and the censoring assumptions are in
`DATA_PROVENANCE.md`. This protocol was fixed before any full-study result was seen.

> **Release notes (added for the public release; the protocol text below is unchanged).**
> - The title above is the working title at the time the protocol was frozen. The final
>   manuscript title is *When Does Model Complexity Pay Off? An Event-Budget Analysis of Accuracy,
>   Calibration, and Stability in Clinical Survival Prediction*.
> - "Overall survival" for METABRIC (section 2) should be read as **death (benchmark
>   coding)**: the processed data contain a death indicator without a verified cause-of-death label
>   (see `DATA_PROVENANCE.md`).
> - Section 10 describes the original project folder. This repository contains the code,
>   configurations and documentation; the data are rebuilt with `download_data.py`, and the frozen
>   design, split manifests and cohort membership are regenerated deterministically by the scripts
>   (README, *Reproducing the analysis*). Per-task records, predictions and the original
>   environment manifest are not included.

## 1. Questions (directional hypotheses are exploratory)

1. How do prediction error (IBS/IPA) and discrimination change with the number of training events?
2. Do flexible models outperform ridge-penalised Cox at larger event budgets?
3. Does model complexity affect calibration and individual-level prediction stability?
4. Do performance gains justify the additional computational cost?

Event budgets are realised by subsampling patients while keeping the development pool's event
fraction, so training-set size changes together with the event count. Results are reported as
effects of the training-set size at a given number of events, not of events alone.

## 2. Design

| element | specification |
|---|---|
| Datasets | SUPPORT (days, overall survival), METABRIC (months, overall survival), Rotterdam-GBSG (months, recurrence-free survival); analysed separately, never pooled |
| Outer splits | R repetitions (5 for verification, extended to 20). Each: stratified 70% development pool / 30% test (strata: event status; GBSG also verified cohort). Seed = `split_seed(dataset, r)`; split 0 equals the pilot split. Manifests are hashed and re-verified on every run. |
| Event budgets | Frozen ladder per dataset, budgets below development-pool events: SUPPORT 50-100-200-400-800-1600-full; METABRIC 50-100-200-400-full; GBSG 50-100-200-400-800-full. Subsets are nested within a repetition and share one test set, so comparisons are paired across models and budgets. |
| Validation partition | 20% of each training subset (stratified by event), shared by all five models; used for tuning and neural early stopping only |
| Horizons | Clinical, frozen before the pilot; support checked on development data only. SUPPORT 180/365/730 d (t* = 365, tau = 730); METABRIC 60/120/180 mo (t* = 120, tau = 180); GBSG 24/36/60 mo (t* = 60, tau = 60) |
| IBS window | Fixed per dataset: [grid_start, tau], 50 points. grid_start is the frozen development-only value (1% quantile of observed times in the split-0 development pool), identical for every repetition and never derived from test outcomes. Each split records its follow-up support. A split without follow-up beyond tau in its test set or pool makes its tasks invalid instead of changing the estimand. |

## 3. Models and fair tuning

Five families with the approved implementations (`models.py`): ridge Cox (sksurv), random
survival forest (sksurv, 300 trees), gradient-boosted Cox (sksurv), DeepSurv (pycox CoxPH) and
Logistic-Hazard (pycox, interpolated). Every family gets the same budget of **4 predefined
configurations** (`full_study_config.yaml`), which include the pilot configuration:

- Each configuration is fitted on the 80% fit partition and scored by IBS (marginal IPCW, censoring
  estimated on the fit partition) on the shared validation partition, over a grid up to
  min(tau, last validation time). The lowest validation IBS wins; ties go to the first.
- Cox / RSF / GBSA: the selected configuration is refitted on the whole training subset.
- Neural models: the selected configuration's early-stopped run is kept (20 warm-up epochs,
  patience 20, at least 4 gradient steps per epoch, Logistic-Hazard bias starts at the marginal hazards).
- Preprocessing (median imputation, z-scores, constant-column removal; predictors treated as
  numeric because their identities are anonymised) never uses validation or test rows:
  - during tuning it is fitted on the fit partition only
  - DeepSurv baseline hazards and Logistic-Hazard cut-count determination also use fitting-partition outcomes only, not validation outcomes
  - Cox/RSF/GBSA final refits use a new preprocessor fitted on the whole training subset
  - neural models keep the fit-partition preprocessor of their selected early-stopped run
  - every bootstrap refit fits its own preprocessor on its bootstrap development rows

Fits per task: 5 for Cox/RSF/GBSA (4 tuning + 1 refit), 4 for the neural models, plus 10
bootstrap refits in stability repetitions. With 18 dataset-budget cells (90 tasks per repetition):

| stage | tasks | model fits |
|---|---|---|
| 5 repetitions (with stability on reps 0-4) | 450 | 5 x 414 + 5 x 900 = **6,570** |
| extension to 20 repetitions (reps 5-19, no stability) | +1,350 | +15 x 414 = **+6,210** |
| total, 20 repetitions | 1,800 | **12,780** |

`run_full_study.py --prepare` writes the exact plan and a runtime projection from the measured
pilot timings (`outputs/full/plan_<R>reps.json`).

## 4. Evaluation (common held-out test patients of each repetition)

All metrics use the same test patients for every model and budget in a repetition.
Follow-up is administratively truncated just above tau, which leaves all estimands at t <= tau unchanged.

- **Primary:** IPA = 1 - IBS_model / IBS_KM over the fixed window [grid_start, tau], with the KM
  reference and the censoring distribution estimated on the development pool (marginal IPCW).
  The IBS uses the shared IPCW estimator. It is validated to equal scikit-survival's
  `integrated_brier_score` exactly, and that equality is re-checked per task wherever
  scikit-survival's range rule (window start >= smallest test time) allows. The shared estimator
  does not need that rule, so the window never depends on test outcomes.
- **Discrimination:** Uno's C truncated at tau; cumulative/dynamic AUC at t*. The common risk
  score is 1 - S(t*|x) for every model.
- **Calibration at t*** (test follow-up censored at t*, so no event information beyond the
  horizon enters the calibration models):
  - observed/expected ratio (evaluation-set Kaplan-Meier)
  - calibration slope (Cox regression on cloglog F(t*))
  - ICI, E50 and E90 from a 3-knot spline calibration model (Austin et al. 2020)
  - smoothed calibration curves at budgets 50 and full
- **Secondary:** Brier score at each horizon.
- **Task validity:** a task counts as complete (`status = ok`) only if all of the following hold:
  - predictions are finite, in [0,1], nonincreasing and correctly aligned with patients
  - IBS, IPA, Brier(t*), Uno's C, td-AUC, O/E and calibration slope are finite (plus the
    conditional-IPCW metrics where applicable)
  - the split has follow-up support beyond tau

  Otherwise the task is `invalid`, or `failed` if an exception occurred. Resume accepts only `ok`
  results computed on the current split hash, and retries the rest. ICI/E50/E90 are reported
  where the spline calibration model is estimable (counted in T0).

## 5. Prediction stability (critical)

In repetitions 0-4, the selected configuration of every task is refitted on 10 bootstrap samples
of its training subset. Neural refits bootstrap the fit partition and keep the validation partition.
Predicted F(t*) is recorded for the repetition's common test patients. Reported (Riley & Collins 2023):
- the instability index (mean |F_b - F_original|)
- the mean per-patient SD
- the 90th percentile of per-patient mean absolute deviation
- classification instability at the development pool's marginal risk at t*

Bootstrap refits use fixed hyperparameters, so this is instability conditional on the selected
configuration. Refits with non-finite predictions are excluded and counted against each task's requested
number of refits (table T3b).

## 6. Conditional-IPCW sensitivity (METABRIC, GBSG)

Stage 0 found censoring-model C of 0.621 (METABRIC) and 0.611 (GBSG). For these datasets, IBS,
IPA and Brier(t*) are re-estimated with G(t|x) from a Cox censoring model fitted on the development
pool only:
- covariates are the predictors, plus the verified cohort indicator for GBSG
- penaliser 0.01
- G is floored at 0.05; the share floored is reported

Marginal IPCW remains primary. The analysis reports whether the paired-comparison classifications
agree under both weightings. Discrimination metrics keep marginal weights; this is a limitation.

## 7. Statistical analysis

**Why not a corrected resampled t-test everywhere.** The Nadeau-Bengio correction (variance
factor 1/R + n_test/n_train) was derived for repeated random train/test splits of one dataset,
with the training set being the complement of the test set. That matches this design only at the
**full budget**, where training set = development pool (70/30 splits, so n_train = n_pool). At
subsampled budgets, training subsets of n_b patients are drawn inside each pool. Training sets then
overlap much less across repetitions than at the full budget, while test sets overlap just as much.
Neither n_test/n_pool nor n_test/n_b describes the resulting between-split correlation. The second
would treat the design as if test sets were complements of tiny training sets, implying near-total
correlation. No validated correction exists for this nested design, so none is claimed.

**Strategy.**
- **All budgets - effect estimates and cross-split variability (descriptive):** for each dataset x
  budget, the mean over repetitions of each metric, and of each paired difference versus ridge Cox
  on the same test patients. Also reported: the SD and 10th-90th percentiles across repetitions,
  and the win rate (share of repetitions in which the model beats ridge Cox). This describes how a
  comparison varies from one development sample of that size to another. It is not a confidence
  interval, and no p-values are attached.
- **Full budget - formal population-level inference:** corrected resampled-t 95% interval
  (Nadeau & Bengio 2003; Bouckaert & Frank 2004), factor 1/R + n_test/n_pool, df = R - 1; p-values
  with Holm adjustment across the four comparisons versus ridge Cox in each dataset x metric cell.
  The number of formal comparisons is reported. Results are framed as estimation, and nothing is
  called significant outside these cells.

**Practical thresholds** (fixed in the pilot configuration before any comparison): ΔIPA >= 0.01,
ΔC >= 0.01, calibration slope outside [0.9, 1.1], ΔICI >= 0.02. These are conventions (one point
of IPA or C is commonly treated as a minimal relevant change); full summaries are reported so
readers can apply other thresholds. Classification (higher is better; mirrored for lower):

| budget | category | rule |
|---|---|---|
| full (formal) | meaningful improvement | corrected-interval lower bound > 0 and mean >= threshold |
| full (formal) | no meaningful difference | corrected interval inside (-threshold, +threshold) |
| full (formal) | better/worse below threshold, meaningfully worse, inconclusive | analogous rules |
| subsampled (descriptive) | consistent improvement | mean >= threshold and 10th percentile > 0 |
| subsampled (descriptive) | no practically relevant difference | 10th and 90th percentiles inside (-threshold, +threshold) |
| subsampled (descriptive) | consistent deficit / variable across splits | mirror rule / otherwise |

Learning curves and paired-difference figures show means with 10th-90th percentile cross-split
bands. The cost-benefit panel (Fig. 6) shows the full-budget corrected intervals. The 5-repetition
stage verifies execution; findings use 20 repetitions.

**Completeness gate.** The analysis runs only if every expected task (dataset x budget x model x
repetition) is `ok` and every cell has R valid repetitions. Otherwise it writes
`INCOMPLETE_RUN_DIAGNOSTIC.md` and stops. A partial analysis can be forced with
`--allow-incomplete`; it is written to `outputs/full/partial_analysis/` and labelled as not findings.

## 8. GBSG cohort provenance

The GBSG file pools Rotterdam (1,546, follow-up truncated at 84 months) and GBSG (686, to 87.4
months) records. Cohort membership was recovered by value matching against the DeepSurv HDF5 and is
used only to stratify the outer splits and in the censoring model. Pooled test sets therefore keep
the cohort mix. Results describe this mixed population, and some censoring is cohort-driven
(administrative). If membership was not verified, splits are stratified by event only and this
must be stated.

## 9. Critical versus optional

**Critical (implemented):**
- paired outer splits with nested budgets
- equal 4-configuration tuning per model inside training data
- IPA/IBS, Uno's C, td-AUC, Brier, calibration slope / ICI / O/E
- bootstrap prediction stability (reps 0-4)
- cross-split summaries and win rates; corrected intervals with Holm adjustment at the full budget
- conditional-IPCW sensitivity for METABRIC and GBSG
- checkpointed, hashed, reproducible runs

**Optional (not implemented; add only if reviewers ask):**
- cohort-specific GBSG evaluation
- conditional-IPCW versions of C and AUC
- tuning on a second criterion (e.g., C-index)
- spline-Cox baseline
- more bootstrap repetitions

## 10. Reproducibility and reporting

- Everything needed to rerun the study is in the project folder:
  - fixed seeds and hashed split manifests
  - per-task JSON records containing every tuning configuration and score
  - saved predictions
  - environment manifest and data checksums
  - analysis outputs generated only from the task records
- Tables are written as CSV and Markdown; figures as vector PDF and 300-dpi PNG.
- The datasets are public, from the DeepSurv repository; cite Katzman et al. (2018) and the
  original studies.
