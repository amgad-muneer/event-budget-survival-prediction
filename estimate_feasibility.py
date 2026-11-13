#!/usr/bin/env python3
"""Project the reduced full-study runtime from MEASURED pilot timings.

Per (model, dataset): fit log(seconds) = a + b*log(n_train) on the pilot budgets, then
predict per-fit time for every budget in the ladder. Jobs per (repetition, dataset,
budget, model) = configs_per_model tuning fits (on 80% of the subset) + 1 final fit,
plus bootstrap refits for the stability repetitions. CPU and GPU workloads are
projected separately and combined with the available parallel capacity.

    python estimate_feasibility.py --config pilot_config.yaml
"""
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from common import load_config, output_dir

NEURAL = ("deepsurv", "logistic_hazard")


def load_pilot(pdir):
    files = sorted(pdir.glob("pilot_metrics_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("No pilot_metrics_*.csv found: run run_pilot.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["status"].isin(["ok", "invalid_predictions"])]
    return df.drop_duplicates(["dataset", "rep", "budget", "model"], keep="last")


def capacity(env, cfg, rd):
    cpus = env.get("cpu_affinity") or env.get("cpu_count_os") or 1
    quota = env.get("cgroup_cpu_max")
    if quota and quota.split()[0] != "max":
        q, period = quota.split()
        cpus = min(cpus, max(1, int(float(q) / float(period))))
    gpus = len(env.get("gpu", {}).get("devices", [])) if env.get("gpu", {}).get("cuda_available") else 0
    threads = int(cfg["cpu_threads"])
    neural_slots = gpus * int(rd["neural_procs_per_gpu"]) if gpus else 1
    cpu_for_trees = max(threads, cpus - (neural_slots * threads if gpus else 0))
    return {"cpus": int(cpus), "gpus": gpus, "threads_per_worker": threads,
            "cpu_workers": max(1, cpu_for_trees // threads), "neural_workers": neural_slots}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot_config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    rd = cfg["reduced_design"]
    env = json.loads((out / "environment_manifest.json").read_text())
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    design = json.loads((out / "frozen_design.json").read_text())
    pilot = load_pilot(out / "pilot")
    pilot["job_s"] = pilot["fit_wall_s"] + pilot["predict_wall_s"] + pilot.get("metrics_wall_s", 0).fillna(0)
    cap = capacity(env, cfg, rd)

    per_fit = []
    for key in design["datasets"]:
        rep = next((r for r in manifest["datasets"] if r["dataset"] == key and r.get("n")), None)
        if rep is None:
            continue
        expected_n = {b["label"].replace("full pool", "full"): b["expected_n"]
                      for b in rep["budgets"]["budgets"]}
        for model, g in pilot[pilot["dataset"] == key].groupby("model"):
            x, y = np.log(g["n_train"].to_numpy(float)), np.log(g["job_s"].clip(lower=1e-3).to_numpy())
            b, a = np.polyfit(x, y, 1) if len(g) >= 2 else (1.0, y[0] - x[0])
            for label in rd["budget_ladder"]:
                label = str(label)
                if label not in expected_n:
                    continue
                n = expected_n[label]
                per_fit.append({"dataset": key, "model": model, "budget": label, "n_train": n,
                                "seconds_final_fit": float(np.exp(a + b * np.log(n))),
                                "seconds_tuning_fit": float(np.exp(a + b * np.log(0.8 * n))),
                                "log_log_slope": float(b), "pilot_points": len(g)})
    pf = pd.DataFrame(per_fit)
    pf.to_csv(out / "feasibility_per_fit.csv", index=False)

    rows = []
    C, B, S = int(rd["configs_per_model"]), int(rd["stability_bootstrap"]), int(rd["stability_repetitions"])
    for R in rd["repetition_options"]:
        job = C * pf["seconds_tuning_fit"] + pf["seconds_final_fit"]
        boot = B * pf["seconds_final_fit"]
        total = R * job + min(S, R) * boot
        neural = pf["model"].isin(NEURAL)
        cpu_s, gpu_s = total[~neural].sum(), total[neural].sum()
        wall_h = max(cpu_s / cap["cpu_workers"], gpu_s / cap["neural_workers"]) / 3600
        wall_h = wall_h * float(rd["safety_factor"])
        n_fits = int(len(pf) * (R * (C + 1)) + len(pf) * min(S, R) * B)
        rows.append({"repetitions": R, "n_fits": n_fits, "cpu_model_hours_serial": cpu_s / 3600,
                     "neural_hours_serial": gpu_s / 3600, "projected_compute_wall_h": wall_h,
                     "plus_analysis_wall_h": wall_h + float(rd["analysis_overhead_hours"]),
                     "fits_compute_budget": wall_h <= float(rd["compute_budget_hours"])})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "feasibility_summary.csv", index=False)

    ok = summary[summary["fits_compute_budget"]]
    R = int(ok["repetitions"].max()) if len(ok) else None
    rec = {"recommended_repetitions": R,
           "status": "feasible" if R else "NOT feasible as designed - drop the largest budgets "
                                          "for RSF/GBSA or reduce configs_per_model",
           "capacity": cap, "budget_ladder": rd["budget_ladder"],
           "tuning": {"configs_per_model": C, "selection": "IBS on one training-derived 80/20 split",
                      "neural_stopping": "early stopping on the same validation partition"},
           "stability": {"repetitions": min(S, R or S), "bootstrap_refits": B,
                         "hyperparameters": "fixed at the tuned values of that repetition"},
           "outer_split": {"test_fraction": cfg["test_fraction"], "paired": True, "nested_budgets": True},
           "uncertainty": "Nadeau-Bengio corrected resampled t intervals (pool-based factor) "
                          "for paired differences vs ridge Cox; win rates; Holm within cell",
           "practical_thresholds": cfg["practical_thresholds"],
           "censoring": {"primary": "marginal IPCW (KM on development pool)",
                         "sensitivity_conditional_ipcw": cfg.get("ipcw", {}).get(
                             "conditional_ipcw_sensitivity", [])},
           "projection_basis": "measured pilot runtimes, log-log interpolation in n_train",
           "projection": summary.to_dict(orient="records")}
    (out / "reduced_full_study_config.yaml").write_text(yaml.safe_dump(rec, sort_keys=False))
    print(summary.to_string(index=False))
    print("\nCapacity:", cap)
    print("Recommended repetitions:", R)


if __name__ == "__main__":
    main()
