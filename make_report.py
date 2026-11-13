#!/usr/bin/env python3
"""Assemble STAGE0_PILOT_REPORT.md from the generated outputs (no new computation).

Automated gates are reported as PASS/FAIL; items that need human verification
(time units, endpoint definitions, provenance) are listed explicitly and never guessed.

    python make_report.py --config pilot_config.yaml
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from common import load_config, output_dir
from estimate_feasibility import load_pilot


def table(df, floatfmt="{:.4g}"):
    if df is None or df.empty:
        return "_none_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot_config.yaml")
    ap.add_argument("--report", default=None, help="default: <output_dir>/STAGE0_PILOT_REPORT.md")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    args.report = args.report or str(out / "STAGE0_PILOT_REPORT.md")
    env = json.loads((out / "environment_manifest.json").read_text())
    man = json.loads((out / "dataset_manifest.json").read_text())
    design = json.loads((out / "frozen_design.json").read_text())
    mv = pd.read_csv(out / "metric_validation.csv")
    pilot = load_pilot(out / "pilot")
    pdir = out / "pilot"
    det = pd.concat([pd.read_csv(f) for f in pdir.glob("determinism_*.csv") if f.stat().st_size > 1]
                    or [pd.DataFrame()], ignore_index=True)
    leak = {}
    for f in pdir.glob("leakage_checks_*.json"):
        leak.update(json.loads(f.read_text()))
    feas = pd.read_csv(out / "feasibility_summary.csv") if (out / "feasibility_summary.csv").exists() else None
    rec = yaml.safe_load((out / "reduced_full_study_config.yaml").read_text()) \
        if (out / "reduced_full_study_config.yaml").exists() else {}
    errors = []
    for f in list(pdir.glob("errors.log")) + list(out.glob("errors.log")):
        errors += [ln for ln in f.read_text().splitlines()
                   if "ConvergenceWarning" not in ln and ln.strip()]

    ds = [r for r in man["datasets"] if r.get("n")]
    all_status = pilot.groupby(["dataset", "budget", "model"])["status"].last()
    leak_ok = all(v["test_pool_overlap"] == 0 and all(
        b["train_test_overlap"] == 0 and b["validation_test_overlap"] == 0 and b["train_subset_of_pool"]
        and b["preprocessor_fit_rows"] == b["train_rows"] for b in v["budgets"].values())
        for v in leak.values()) and bool(leak)
    det_max = float(det["max_abs_diff_same_seed"].max()) if not det.empty else None
    gates = {
        "G1 datasets audited (outcome columns found, 0/1 events, checksums unchanged)":
            len(ds) == 3 and all(r.get("time_col") and r.get("event_col") for r in ds)
            and man.get("checksums_unchanged", False),
        "G2 metric validation on synthetic data": bool(mv["pass"].dropna().astype(bool).all()),
        "G3 frozen horizons admissible for every dataset":
            all(d.get("admissible") for d in design["datasets"].values()),
        "G4 all model runs completed with valid survival curves":
            len(all_status) > 0 and bool((all_status == "ok").all()),
        "G5 no test leakage (index overlaps 0; preprocessing fit on training rows only)": leak_ok,
        "G6 same-seed refits reproduce predictions (max diff < 1e-4)":
            det_max is not None and det_max < 1e-4,
        "G7 reduced full study fits the compute budget": bool(rec.get("recommended_repetitions")),
    }
    hard_fail = not all(gates[k] for k in list(gates)[:5])
    auto = ("NO-GO/STOP (essential gate failed)" if hard_fail else
            "REVISE (design must be reduced)" if not gates[list(gates)[6]] else
            "GO, conditional on human confirmation of units and endpoints (section A)")

    L = ["# Stage 0 and pilot report", "",
         "> Pilot results come from one outer split with fixed, untuned configurations. They "
         "validate the implementation and measure runtime; they are **not** evidence about "
         "which model is better.", ""]

    # A. Data verification
    L += ["## A. Data verification", "", "| dataset | file | n | events | censored | event frac. | "
          "time col | event col | time range | missing cells | duplicate rows | censoring-model C |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ds:
        cd = r.get("censoring_diagnostic", {})
        L.append(f"| {r['dataset']} | {Path(r['file']).name} | {r['n']} | {r['n_events']} | "
                 f"{r['n_censored']} | {r['event_fraction']:.3f} | {r['time_col']} | {r['event_col']} | "
                 f"{r['time_summary']['min']:.4g}-{r['time_summary']['max']:.4g} | "
                 f"{sum(r['missing_by_column'].values())} | {r['duplicates']['full_rows']} | "
                 f"{cd.get('censoring_model_C', float('nan')):.3f} |")
    L += ["", "**Verified from the files:** schema, outcome columns, 0/1 event coding, counts, "
          "missingness, duplicates, follow-up (KM / reverse KM), SHA-256 checksums (unchanged during "
          f"the audit: {man.get('checksums_unchanged')}).", "",
          "**Unresolved (must be confirmed from provenance documentation, not inferred):** "
          + "; ".join(man.get("unresolved_metadata", [])) + ". Predictors are anonymised and were "
          "treated as numeric; no clinical transformations were applied.", ""]
    for r in ds:
        if "gbsg_cohort" in r:
            L += [f"**GBSG cohorts (Stage 0, from the CSV alone):** {r['gbsg_cohort']['conclusion']}", ""]
    hz = pd.DataFrame([{"dataset": k, "horizons": v.get("horizons"), "primary t*": v.get("primary_horizon"),
                        "tau": v.get("tau"), "source": v.get("horizon_source"),
                        "units": v.get("units"), "notes": "; ".join(v.get("notes", []))}
                       for k, v in design["datasets"].items()])
    L += ["**Frozen horizons** (external clinical conventions in source-verified units; support checked "
          "on the development pool of outer split 0 only - test outcomes never used; frozen before any "
          "model was fit):", "",
          table(hz), "", "**Feasible event budgets (expected, 70% development pool):** " +
          "; ".join(f"{k}: {v.get('budget_ladder_feasible')}" for k, v in design["datasets"].items()), ""]
    high = cfg.get("ipcw", {}).get("conditional_ipcw_sensitivity", [])
    excl = {k: v.get("analysis_rule_exclusions", {}) for k, v in design["datasets"].items()}
    L += ["**Analysis-set rule** (identical for all datasets; CSV files unchanged): records with "
          "duration <= 0, a missing duration or an event code outside {0,1} are excluded. Excluded: " +
          "; ".join(f"{k}: {v.get('n_excluded', '?')} (CSV rows {v.get('excluded_csv_rows', [])})"
                    for k, v in excl.items()) + ".", ""]
    cohort_rep = out / "gbsg_cohort" / "gbsg_cohort_report.json"
    if cohort_rep.exists():
        cr = json.loads(cohort_rep.read_text())
        L += [f"**GBSG cohort recovery** (value matching against the DeepSurv HDF5, sha256 "
              f"{cr['h5_sha256'][:16]}...): verified_complete = {cr['verified_complete']}; checks "
              f"{cr['checks']}; CSV order equals train-then-test: {cr['csv_order_equals_train_then_test']}. "
              f"Outer splits stratified by cohort: {design['datasets'].get('gbsg', {}).get('cohort_stratified')}.", ""]
    else:
        L += ["**GBSG cohort recovery:** not run (original HDF5 not supplied); splits are stratified "
              "by event status only.", ""]
    L += ["**Censoring assumptions.** Marginal IPCW (Brier/IBS, Uno's C, time-dependent AUC) assumes "
          "censoring is independent of both event time and covariates, and that G(t) > 0 up to tau "
          "(checked on development data: G(tau-) >= 0.2). The censoring distribution is estimated by "
          "Kaplan-Meier on the development pool, never on test outcomes. The covariate-censoring "
          "diagnostic (table above) cannot prove independence. The pilot keeps marginal IPCW; "
          + (f"{high} are flagged (censoring-model C >= 0.60) for a prespecified conditional-IPCW "
             "sensitivity analysis in the full experiment (Cox censoring model fitted on the "
             "development pool only)." if high else "no dataset is flagged for conditional IPCW."), ""]

    # B. Implementation verification
    L += ["## B. Implementation verification", "", "**Metric validation (synthetic, known truth):**", "",
          table(mv), ""]
    show = pilot[["dataset", "budget", "model", "status", "n_train", "events_train", "fit_wall_s",
                  "nonincreasing", "in_unit_interval", "aligned", "IPA", "Uno_C_tau", "tdAUC_tstar",
                  "cal_slope", "ICI", "O_E"]].copy()
    L += ["**Pilot runs** (single split; technical validation only):", "", table(show), "",
          f"**Same-seed reproducibility:** max |difference| in predicted survival = {det_max}.", "",
          f"**Leakage checks:** {'PASS' if leak_ok else 'FAIL'} - test/pool and train/test index "
          "overlaps are zero, validation rows come from the training subset, preprocessing is fitted "
          "on exactly the training rows, logistic-hazard cut points and neural early stopping use "
          "training-derived data only, and the censoring distribution and KM reference come from the "
          "development pool (never the test outcomes).", "",
          "**Errors and warnings (excluding convergence warnings):** " +
          (f"{len(errors)} lines; first entries:\n\n```\n" + "\n".join(errors[:15]) + "\n```"
           if errors else "none"), ""]

    # C. Computational feasibility
    gpus = env.get("gpu", {}).get("devices", [])
    timing = pilot.groupby(["model", "dataset", "budget"])[["fit_wall_s", "predict_wall_s"]].mean().reset_index()
    L += ["## C. Computational feasibility", "",
          f"- CPUs: {env.get('cpu_count_os')} (affinity {env.get('cpu_affinity')}, cgroup "
          f"{env.get('cgroup_cpu_max')}); memory {env.get('memory')}",
          f"- GPUs: " + ("; ".join(f"{g['name']} {g['total_GiB']} GiB (free {g['free_GiB']})" for g in gpus)
                         if gpus else "none detected"),
          f"- Threads per process: {cfg['cpu_threads']}; RSF workers: {cfg['tree_n_jobs']}", "",
          "**Measured pilot runtimes (seconds):**", "", table(timing), "",
          "**Projected reduced full study (from measured timings):**", "", table(feas), "",
          f"Recommended repetitions: **{rec.get('recommended_repetitions')}** "
          f"({rec.get('status')}). Capacity used: {rec.get('capacity')}.", ""]

    # D. Scientific feasibility
    th = cfg["practical_thresholds"]
    L += ["## D. Scientific feasibility", "",
          "**Limitations.** Varying the event budget also varies the sample size (the event fraction is "
          "held at the pool value), so effects are attributed to the training-set size at a given event "
          "count, not to events alone. Results are limited to three low-dimensional benchmark cohorts with "
          "anonymised, pre-processed predictors; marginal IPCW assumes covariate-independent censoring; "
          "hyperparameter selection at the smallest budgets is inherently noisy.", "",
          f"**Practical-effect thresholds (fixed before the main experiment):** dIPA >= {th['delta_IPA']}, "
          f"dC >= {th['delta_C']}, calibration slope outside {th['slope_band']}, dICI >= {th['delta_ICI']}. "
          "They are conventions rather than derived quantities; results will also be shown as full "
          "interval estimates so readers can apply other thresholds.", "",
          "**Minimum defensible protocol.** Paired comparisons on shared outer splits and nested budgets; "
          "a small, identical tuning budget per model scored on a training-derived validation split; "
          "IPA (IBS) primary, Uno's C, time-dependent AUC, Brier(t*), calibration slope/ICI secondary; "
          "bootstrap prediction instability on a subset of repetitions.", "",
          "**Uncertainty.** Per dataset and budget, the mean paired difference vs ridge Cox across "
          "repetitions with Nadeau-Bengio corrected intervals (factor 1/R + n_test/n_pool), win rates, "
          "and Holm adjustment across the four comparisons; learning-curve shapes are described, not tested. "
          "Directional hypotheses are exploratory.", ""]

    # E. Decision
    L += ["## E. Automated gates and decision", "", "| gate | result |", "|---|---|"]
    L += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in gates.items()]
    L += ["", f"**Automated recommendation: {auto}.**", "",
          "The final GO / REVISE / NO-GO decision also requires confirming the endpoint definitions and "
          "time units listed in section A."]
    Path(args.report).write_text("\n".join(L))
    print(f"Report written: {args.report}\nAutomated recommendation: {auto}")


if __name__ == "__main__":
    main()
