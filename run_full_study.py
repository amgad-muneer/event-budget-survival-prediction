#!/usr/bin/env python3
"""Full event-budget study: tuning, final fits, evaluation, stability and conditional-IPCW
sensitivity, with split manifests, per-task checkpoints and safe resume.

A task = (dataset, outer repetition, event budget, model). Each finished task writes
outputs/full/tasks/<dataset>/repXX/<budget>__<model>.json (+ .npz predictions) atomically;
existing results are never recomputed, so extending from 5 to 20 repetitions only runs the
new tasks. Workers claim tasks with lock files, so several CPU and GPU workers can run at once.

    python run_full_study.py --config full_study_config.yaml --n-reps 5 --prepare
    python run_full_study.py --config full_study_config.yaml --n-reps 5 --worker cpu --worker-id cpu01
    python run_full_study.py --config full_study_config.yaml --n-reps 5 --worker gpu --device cuda:0 --worker-id gpu0
    python run_full_study.py --config full_study_config.yaml --n-reps 5 --collect
(Start several --worker processes in parallel; they share tasks through lock files.)
"""
import argparse
import os
import sys

import yaml


def _set_threads():
    """Thread limits must be set before numpy/torch are imported."""
    argv = sys.argv
    path = argv[argv.index("--config") + 1] if "--config" in argv else "full_study_config.yaml"
    with open(path) as fh:
        res = (yaml.safe_load(fh).get("full_study") or {}).get("resources", {})
    kind = argv[argv.index("--worker") + 1] if "--worker" in argv else "cpu"
    n = str(res.get("threads_per_gpu_worker" if kind == "gpu" else "threads_per_cpu_worker", 4))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = n
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return int(n)


THREADS = _set_threads()

import hashlib  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (CoxCensoringModel, Preprocessor, check_curves, cohort_membership,  # noqa: E402
                    compute_metrics, gpu_peak_mb, gpu_peak_reset, ipcw_brier_ibs, km_at,
                    load_config, load_dataset, marginal_G, nested_budgets, output_dir,
                    read_manifest, setup_logging, split_seed, stratified_split, surv_struct,
                    truncate, validation_split, write_json)
from models import MODELS, make_model  # noqa: E402

NEURAL = ("deepsurv", "logistic_hazard")
# A task counts as complete only if predictions are valid and these metrics are finite.
REQUIRED_METRICS = ("IBS", "IPA", "Brier_tstar", "Uno_C_tau", "tdAUC_tstar", "O_E", "cal_slope")


# --------------------------------------------------------------------------
# Paths, splits and plan
# --------------------------------------------------------------------------

class Ctx:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fs = cfg["full_study"]
        self.out = output_dir(cfg)
        self.root = self.out / self.fs.get("output_subdir", "full")
        for d in ("splits", "tasks", "locks", "logs", "tables", "figures"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.design = json.loads((self.out / "frozen_design.json").read_text())
        self.manifest = read_manifest(self.out)
        self._data, self._cens = {}, {}

    def data(self, key):
        if key not in self._data:
            self._data[key] = load_dataset(self.cfg, key, self.manifest)
        return self._data[key]

    def split_path(self, key, rep):
        return self.root / "splits" / f"{key}_rep{rep:02d}.json"

    def task_path(self, key, rep, budget, model):
        return self.root / "tasks" / key / f"rep{rep:02d}" / f"{budget}__{model}.json"


def make_split(ctx, key, rep):
    """Deterministic outer split + nested budgets + validation partitions (as in the pilot)."""
    cfg, d = ctx.cfg, ctx.design["datasets"][key]
    X, t, e, strata, _, info = ctx.data(key)
    if bool(d.get("cohort_stratified")) != info["cohort_stratified"]:
        raise RuntimeError(f"{key}: cohort stratification differs from the frozen design.")
    seed = split_seed(cfg, key, rep)
    rng = np.random.default_rng(seed)
    test, pool = stratified_split(strata, cfg["test_fraction"], rng)
    ladder = [b if b == "full" else int(b) for b in d["budget_ladder_feasible"]]
    subsets, skipped = nested_budgets(pool, e, ladder, rng)
    val = {lab: validation_split(np.arange(len(tr)), e[tr], cfg["validation_fraction"], seed + 7)[1]
           for lab, tr in subsets.items()}
    tau = float(d["tau"])
    t_admin = tau * (1 + 1e-6)
    # Fixed integration window [grid_start, tau]: grid_start is the frozen, development-only
    # value (1% quantile of observed times in the split-0 development pool); it never depends
    # on test outcomes and is identical for every repetition.
    grid = np.linspace(float(d["grid_start"]), tau, int(cfg["grid_points"]))
    support = {"min_test_time": float(t[test].min()),
               "grid_start_below_min_test_time": bool(grid[0] < t[test].min()),
               "test_followup_beyond_tau": int(np.sum(t[test] > tau)),
               "pool_followup_beyond_tau": int(np.sum(t[pool] > tau))}
    support["ok"] = support["test_followup_beyond_tau"] > 0 and support["pool_followup_beyond_tau"] > 0
    rec = {"dataset": key, "rep": rep, "seed": seed, "test": test.tolist(), "pool": pool.tolist(),
           "budgets": {k: v.tolist() for k, v in subsets.items()}, "skipped": skipped,
           "validation_positions": {k: v.tolist() for k, v in val.items()},
           "grid": grid.tolist(), "horizons": d["horizons"], "t_star": float(d["primary_horizon"]),
           "tau": tau, "t_admin": t_admin, "n_pool": int(len(pool)), "n_test": int(len(test)),
           "analysis_rule": {k: info[k] for k in ("n_file", "n_excluded", "excluded_csv_rows")},
           "cohort_stratified": info["cohort_stratified"], "evaluation_support": support}
    rec["hash"] = hashlib.sha256(json.dumps(
        {k: rec[k] for k in ("test", "pool", "budgets", "validation_positions", "grid", "t_star", "tau")},
        sort_keys=True).encode()).hexdigest()
    pilot = ctx.out / "pilot" / "splits" / f"{key}_rep{rep}.json"
    if pilot.exists():
        rec["matches_pilot_test_set"] = json.loads(pilot.read_text())["test"] == rec["test"]
    return rec


def load_split(ctx, key, rep):
    path = ctx.split_path(key, rep)
    if not path.exists():
        raise RuntimeError(f"{path} missing: run --prepare first.")
    return json.loads(path.read_text())


def task_list(ctx, n_reps, kind=None):
    tasks = []
    for rep in range(n_reps):
        for key in ctx.cfg["datasets"]:
            labels = ctx.design["datasets"][key]["budget_ladder_feasible"]
            for lab in labels:
                for m in MODELS:
                    if kind == "gpu" and m not in NEURAL or kind == "cpu" and m in NEURAL:
                        continue
                    tasks.append((key, rep, lab, m))
    return tasks


def fits_per_task(ctx, model, rep):
    n_cfg = len(ctx.fs["tuning"]["grids"][model])
    boot = ctx.fs["stability"]["bootstrap"] if rep < ctx.fs["stability"]["repetitions"] else 0
    return n_cfg + (0 if model in NEURAL else 1) + boot


def prepare(ctx, n_reps, log):
    if n_reps > int(ctx.fs["max_repetitions"]):
        raise SystemExit(f"--n-reps {n_reps} exceeds max_repetitions {ctx.fs['max_repetitions']}.")
    for key in ctx.cfg["datasets"]:
        for rep in range(n_reps):
            rec = make_split(ctx, key, rep)
            path = ctx.split_path(key, rep)
            if path.exists():
                old = json.loads(path.read_text())
                if old["hash"] != rec["hash"]:
                    raise SystemExit(f"{path}: existing split differs from the deterministic split - "
                                     "configuration or data changed; refusing to continue.")
                continue
            write_json(rec, path)
            log.info("split written: %s (pilot match: %s)", path.name, rec.get("matches_pilot_test_set"))
    # Exact fit counts and runtime projection from the measured pilot timings.
    per_fit = ctx.out / "feasibility_per_fit.csv"
    pf = pd.read_csv(per_fit) if per_fit.exists() else None
    res = ctx.fs["resources"]
    import torch
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    rows = []
    for key, rep, lab, m in task_list(ctx, n_reps):
        done = ctx.task_path(key, rep, lab, m).exists()
        sec = np.nan
        if pf is not None:
            hit = pf[(pf.dataset == key) & (pf.model == m) & (pf.budget.astype(str) == lab)]
            if len(hit):
                h = hit.iloc[0]
                n_cfg = len(ctx.fs["tuning"]["grids"][m])
                boot = fits_per_task(ctx, m, rep) - n_cfg - (0 if m in NEURAL else 1)
                sec = n_cfg * h.seconds_tuning_fit + (0 if m in NEURAL else h.seconds_final_fit) \
                    + boot * h.seconds_final_fit
        rows.append({"dataset": key, "rep": rep, "budget": lab, "model": m, "done": done,
                     "fits": fits_per_task(ctx, m, rep), "projected_seconds": sec,
                     "resource": "gpu" if m in NEURAL else "cpu"})
    plan = pd.DataFrame(rows)
    plan.to_csv(ctx.root / f"plan_{n_reps}reps.csv", index=False)
    todo = plan[~plan.done]
    cpu_w = int(res["cpu_workers"])
    gpu_w = max(1, n_gpu * int(res["gpu_workers_per_gpu"])) if n_gpu else 1
    cpu_s = todo.loc[todo.resource == "cpu", "projected_seconds"].sum()
    gpu_s = todo.loc[todo.resource == "gpu", "projected_seconds"].sum()
    safety = float(ctx.cfg["reduced_design"].get("safety_factor", 1.5))
    longest = todo["projected_seconds"].max() if len(todo) else 0
    summary = {"n_reps": n_reps, "tasks_total": int(len(plan)), "tasks_remaining": int(len(todo)),
               "fits_total": int(plan.fits.sum()), "fits_remaining": int(todo.fits.sum()),
               "cpu_task_hours_serial": cpu_s / 3600, "gpu_task_hours_serial": gpu_s / 3600,
               "cpu_workers": cpu_w, "gpu_workers": gpu_w, "gpus_visible": n_gpu,
               "projected_wall_hours": safety * max(cpu_s / cpu_w, gpu_s / gpu_w, longest) / 3600,
               "projection_basis": "pilot timings (outputs/feasibility_per_fit.csv), x safety factor "
                                   f"{safety}; the longest single task bounds the wall time from below"}
    write_json(summary, ctx.root / f"plan_{n_reps}reps.json")
    log.info("PLAN %s", json.dumps(summary, indent=1))


# --------------------------------------------------------------------------
# One task
# --------------------------------------------------------------------------

def _params(ctx, model, cfg_override):
    p = dict(ctx.cfg["models"][model])
    p.update(cfg_override)
    return p


def _fit(model_name, params, seed, device, n_jobs, X, t, e, fit_pos, val_pos):
    m = make_model(model_name, params, seed, device, n_jobs)
    if m.needs_validation:
        m.fit(X[fit_pos], t[fit_pos], e[fit_pos], X_val=X[val_pos], time_val=t[val_pos],
              event_val=e[val_pos])
    else:
        m.fit(X, t, e)
    return m


def _val_grid(t_fit, t_val, lower, tau, n=30):
    upper = min(tau, np.nextafter(t_val.max(), -np.inf), np.nextafter(t_fit.max(), -np.inf))
    lower = max(lower, t_val.min())
    return np.linspace(lower, upper, n) if upper > lower else None


def censoring_G(ctx, key, rep, split, X, t, e, test_t_trunc, grid_all):
    """Cox censoring model fitted on the development pool (cached per dataset/rep)."""
    ck = (key, rep)
    if ck not in ctx._cens:
        cc = ctx.fs["conditional_ipcw"]
        pool = np.array(split["pool"])
        pre = Preprocessor(ctx.cfg["datasets"][key].get("categorical", [])).fit(X.iloc[pool])
        Zp = pre.transform(X.iloc[pool]).astype(float)
        test = np.array(split["test"])
        Zt = pre.transform(X.iloc[test]).astype(float)
        mem = cohort_membership(ctx.cfg, key) if cc.get("include_cohort") else None
        if mem is not None:
            info = ctx.data(key)[5]
            rows = np.setdiff1d(np.arange(info["n_file"]), info["excluded_csv_rows"])
            ind = (mem.loc[rows].to_numpy() == "test").astype(float)
            Zp, Zt = np.column_stack([Zp, ind[pool]]), np.column_stack([Zt, ind[test]])
        cm = CoxCensoringModel(cc.get("penalizer", 0.01)).fit(Zp, t[pool], e[pool])
        ctx._cens[ck] = (cm, Zt, mem is not None)
    cm, Zt, with_cohort = ctx._cens[ck]
    G_y, G_g = cm.predict(Zt, test_t_trunc, grid_all)
    return G_y, G_g, cm.concordance_, with_cohort


def run_task(ctx, key, rep, lab, model_name, device, log):
    cfg, fs = ctx.cfg, ctx.fs
    split = load_split(ctx, key, rep)
    X, t, e, _, _, _ = ctx.data(key)
    seed = int(split["seed"])
    tr = np.array(split["budgets"][lab])
    test, pool = np.array(split["test"]), np.array(split["pool"])
    val_pos = np.array(split["validation_positions"][lab])
    fit_pos = np.setdiff1d(np.arange(len(tr)), val_pos)
    grid, t_star, tau, t_admin = (np.array(split["grid"]), split["t_star"], split["tau"],
                                  split["t_admin"])
    horizons = np.array(split["horizons"], float)
    n_jobs = THREADS if model_name == "rsf" else 1
    cat = cfg["datasets"][key].get("categorical", [])
    # Tuning: preprocessing estimated on the fit partition only (validation rows excluded).
    pre_fit = Preprocessor(cat).fit(X.iloc[tr[fit_pos]])
    Xtr, Xte = pre_fit.transform(X.iloc[tr]), pre_fit.transform(X.iloc[test])
    ttr, etr = t[tr], e[tr]
    rec = {"dataset": key, "rep": rep, "budget": lab, "model": model_name, "device": device,
           "seed": seed, "split_hash": split["hash"], "n_train": int(len(tr)),
           "events_train": int(etr.sum()), "n_fit": int(len(fit_pos)), "n_val": int(len(val_pos)),
           "n_test": int(len(test)), "events_test": int(e[test].sum()), "n_pool": int(len(pool)),
           "t_star": t_star, "tau": tau, "warnings": [],
           "evaluation_support": split["evaluation_support"]}
    gpu = device.startswith("cuda") and model_name in NEURAL
    mem_ok = gpu_peak_reset(device) if gpu else False

    # ---- tuning on the training-derived validation partition --------------------------
    vt_f, ve_f = ttr[fit_pos], etr[fit_pos]
    vgrid = _val_grid(vt_f, ttr[val_pos], grid[0], tau)
    tuning, best, best_model = [], None, None
    w0, c0 = time.perf_counter(), time.process_time()
    for i, override in enumerate(fs["tuning"]["grids"][model_name]):
        row = {"config_index": i, "params": override}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                f0 = time.perf_counter()
                Xfit = Xtr[fit_pos]
                m = _fit(model_name, _params(ctx, model_name, override), seed, device, n_jobs,
                         Xtr if model_name in NEURAL else Xfit, ttr if model_name in NEURAL else vt_f,
                         etr if model_name in NEURAL else ve_f, fit_pos, val_pos)
                row["fit_wall_s"] = time.perf_counter() - f0
                if vgrid is not None:
                    from sksurv.metrics import integrated_brier_score
                    adm = vgrid[-1] * (1 + 1e-9)
                    ft, fe = truncate(vt_f, ve_f, adm)
                    vt, ve = truncate(ttr[val_pos], etr[val_pos], adm)
                    S_val = np.clip(m.predict_surv(Xtr[val_pos], vgrid), 0, 1)
                    row["val_IBS"] = float(integrated_brier_score(surv_struct(ft, fe),
                                                                  surv_struct(vt, ve), S_val, vgrid))
                else:
                    row["val_IBS"] = None
                row.update({f"info_{k}": v for k, v in getattr(m, "info", {}).items()})
                row["status"] = "ok"
                rec["warnings"] += sorted({f"{w.category.__name__}: {str(w.message)[:120]}" for w in caught})
        except Exception as exc:  # noqa: BLE001
            row.update({"status": "failed", "error": repr(exc)[:300]})
            m = None
        tuning.append(row)
        if m is not None and row["status"] == "ok":
            score = row["val_IBS"] if row["val_IBS"] is not None else np.inf
            if best is None or score < best[1]:
                best, best_model = (i, score), m
    rec["tuning_wall_s"] = time.perf_counter() - w0
    rec["tuning_cpu_s"] = time.process_time() - c0
    rec["tuning"] = tuning
    if best is None:
        raise RuntimeError("all tuning configurations failed: " +
    repr([(r["config_index"], r.get("error")) for r in tuning]))
    sel = best[0] if np.isfinite(best[1]) else 0
    if not np.isfinite(best[1]):
        rec["warnings"].append("validation IBS unavailable; first configuration used")
        best_model = None
    rec["selected_config"] = sel
    rec["selected_params"] = fs["tuning"]["grids"][model_name][sel]
    params = _params(ctx, model_name, rec["selected_params"])

    # ---- final model ------------------------------------------------------------------
    f0, c0 = time.perf_counter(), time.process_time()
    if model_name in NEURAL:
        # Neural: keep the selected early-stopped run and its fit-partition preprocessing.
        final = best_model if best_model is not None else \
            _fit(model_name, params, seed, device, n_jobs, Xtr, ttr, etr, fit_pos, val_pos)
        Xte_final = Xte
        rec["preprocessing"] = {"tuning": "fit partition", "final": "fit partition (selected run)"}
    else:
        # Classical: refit on the whole training subset with a newly fitted preprocessor.
        pre_full = Preprocessor(cat).fit(X.iloc[tr])
        Xte_final = pre_full.transform(X.iloc[test])
        final = _fit(model_name, params, seed, device, n_jobs, pre_full.transform(X.iloc[tr]),
                     ttr, etr, fit_pos, val_pos)
        rec["preprocessing"] = {"tuning": "fit partition", "final": "whole training subset"}
    rec["final_fit_wall_s"] = time.perf_counter() - f0
    rec["final_fit_cpu_s"] = time.process_time() - c0
    rec["final_info"] = getattr(final, "info", {})
    p0 = time.perf_counter()
    S_all = final.predict_surv(Xte_final, np.concatenate([grid, horizons]))
    rec["predict_wall_s"] = time.perf_counter() - p0
    S, S_h = S_all[:, :len(grid)], S_all[:, len(grid):]
    S_star = S_h[:, list(horizons).index(t_star)]
    checks = check_curves(S_all, len(test), np.concatenate([grid, horizons]))
    checks["nonincreasing"] = check_curves(S, len(test), grid)["nonincreasing"]
    order = np.random.default_rng(1).permutation(len(test))
    checks["alignment_max_abs_diff"] = float(np.max(np.abs(final.predict_surv(Xte_final[order], grid) - S[order])))
    checks["aligned"] = checks["alignment_max_abs_diff"] < 1e-6
    rec["checks"] = checks
    rec["valid"] = all(checks[k] for k in ("shape_ok", "finite", "in_unit_interval", "nonincreasing", "aligned"))

    # ---- evaluation (marginal IPCW; censoring and KM reference from the development pool) --
    m0 = time.perf_counter()
    pt, pe = truncate(t[pool], e[pool], t_admin)
    tt_, te_ = truncate(t[test], e[test], t_admin)
    km_grid = km_at(pt, pe, grid)
    S, S_star = np.clip(S, 0, 1), np.clip(S_star, 0, 1)
    rec["metrics"] = compute_metrics(pt, pe, tt_, te_, S, grid, S_star, t_star, tau, km_grid)
    for h, col in zip(horizons, S_h.T):
        if h != t_star:
            from sksurv.metrics import brier_score
            rec["metrics"][f"Brier_{h:g}"] = float(brier_score(surv_struct(pt, pe), surv_struct(tt_, te_),
                                                               np.clip(col, 0, 1)[:, None], [h])[1][0])

    # ---- conditional-IPCW sensitivity (Cox censoring model on the development pool) -------
    if key in cfg.get("ipcw", {}).get("conditional_ipcw_sensitivity", []):
        cc = fs["conditional_ipcw"]
        g_all = np.append(grid, t_star)
        G_y, G_g, c_cens, with_cohort = censoring_G(ctx, key, rep, split, X, t, e, tt_, g_all)
        Gy_m, Gg_m = marginal_G(pt, pe, tt_, g_all)
        out = {"censoring_model_C": c_cens, "cohort_in_censoring_model": with_cohort}
        for name, (gy, gg) in {"cond": (G_y, G_g), "marg_check": (Gy_m, Gg_m)}.items():
            _, ibs, fl = ipcw_brier_ibs(tt_, te_, S, grid, gy, gg[..., :-1], cc["min_G"])
            _, ibs_km, _ = ipcw_brier_ibs(tt_, te_, np.tile(km_grid, (len(test), 1)), grid, gy,
                                          gg[..., :-1], cc["min_G"])
            b_star = ipcw_brier_ibs(tt_, te_, S_star[:, None], np.array([t_star]), gy, gg[..., -1:],
                                    cc["min_G"])[0][0]
            out.update({f"IBS_{name}": ibs, f"IBS_KM_{name}": ibs_km, f"IPA_{name}": 1 - ibs / ibs_km,
                        f"Brier_tstar_{name}": float(b_star), f"share_G_floored_{name}": fl})
        rec["metrics_conditional_ipcw"] = out
    rec["metrics_wall_s"] = time.perf_counter() - m0

    # ---- prediction stability on the common held-out patients -------------------------
    arrays = {"test_index": test, "horizons": horizons, "F_horizons": (1 - S_h).astype(np.float32)}
    if rep < int(fs["stability"]["repetitions"]):
        s0 = time.perf_counter()
        B = int(fs["stability"]["bootstrap"])
        F0 = 1 - S_star
        Fb = np.full((B, len(test)), np.nan, dtype=np.float32)
        for b in range(B):
            rng = np.random.default_rng(seed + 10_000 + b)
            try:
                # Each refit repeats the development procedure, including preprocessing,
                # estimated on its own bootstrap development rows; test rows keep their order.
                if model_name in NEURAL:
                    fp = rng.choice(fit_pos, len(fit_pos), replace=True)
                    pre_b = Preprocessor(cat).fit(X.iloc[tr[fp]].reset_index(drop=True))
                    mb = _fit(model_name, params, seed + b + 1, device, n_jobs, pre_b.transform(X.iloc[tr]),
                              ttr, etr, fp, val_pos)
                else:
                    bi = rng.choice(len(tr), len(tr), replace=True)
                    Xb = X.iloc[tr[bi]].reset_index(drop=True)
                    pre_b = Preprocessor(cat).fit(Xb)
                    mb = _fit(model_name, params, seed + b + 1, device, n_jobs, pre_b.transform(Xb),
                              ttr[bi], etr[bi], None, None)
                Fb[b] = 1 - np.clip(mb.predict_surv(pre_b.transform(X.iloc[test]), np.array([t_star]))[:, 0], 0, 1)
                if not np.isfinite(Fb[b]).all():
                    rec["warnings"].append(f"bootstrap {b}: non-finite predictions (refit excluded)")
            except Exception as exc:  # noqa: BLE001
                rec["warnings"].append(f"bootstrap {b} failed: {exc!r}"[:200])
        ok = ~np.isnan(Fb).any(axis=1)
        c = 1 - km_at(pt, pe, t_star)[0]               # development-pool marginal risk at t*
        D = np.abs(Fb[ok] - F0[None, :])
        rec["stability"] = {
            "bootstrap_ok": int(ok.sum()), "bootstrap_requested": B, "threshold_risk": float(c),
            "instability_index": float(D.mean()) if ok.any() else None,       # Riley & Collins
            "mean_patient_sd": float(Fb[ok].std(axis=0, ddof=1).mean()) if ok.sum() > 1 else None,
            "p90_patient_mad": float(np.quantile(D.mean(axis=0), 0.9)) if ok.any() else None,
            "classification_instability": float(((Fb[ok] > c) != (F0 > c)[None, :]).mean()) if ok.any() else None,
        }
        rec["stability_wall_s"] = time.perf_counter() - s0
        arrays.update({"F_tstar_original": F0.astype(np.float32), "F_tstar_bootstrap": Fb})
    rec["gpu_peak_mem_mb"] = gpu_peak_mb(device, mem_ok)
    rec["warnings"] = sorted(set(rec["warnings"]))[:50]
    required = list(REQUIRED_METRICS)
    if "metrics_conditional_ipcw" in rec:
        required += ["IBS_cond", "IPA_cond", "Brier_tstar_cond"]
    allm = {**rec["metrics"], **rec.get("metrics_conditional_ipcw", {})}
    bad = [k for k in required if not np.isfinite(allm.get(k) if allm.get(k) is not None else np.nan)]
    # Constant risk makes calibration slope non-estimable.
    if "cal_slope" in bad and np.ptp(np.asarray(S_star, dtype=float)) <= 1e-12:
        bad.remove("cal_slope")
        rec["metrics"]["cal_slope"] = None
        rec["calibration_slope_estimable"] = False
        rec["warnings"].append("Calibration slope not estimable: constant risk")
    else:
        rec["calibration_slope_estimable"] = "cal_slope" not in bad
    rec["nonfinite_required_metrics"] = bad
    rec["calibration_curve_estimable"] = bool(np.isfinite(rec["metrics"].get("ICI", np.nan)))
    reasons = ([] if rec["valid"] else ["invalid predictions"]) + \
        ([f"non-finite required metrics: {bad}"] if bad else []) + \
        ([] if split["evaluation_support"]["ok"] else ["no follow-up beyond tau in test or pool"])
    rec["status"] = "ok" if not reasons else "invalid"
    rec["invalid_reasons"] = reasons
    return rec, arrays


# --------------------------------------------------------------------------
# Worker loop with lock files and atomic checkpoints
# --------------------------------------------------------------------------

def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def claim(lock):
    host = socket.gethostname()
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "host": host, "t": time.time()}).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                info = json.loads(Path(lock).read_text())
            except Exception:  # noqa: BLE001
                return False
            if info.get("host") == host and not _pid_alive(int(info.get("pid", -1))):
                Path(lock).unlink(missing_ok=True)     # stale lock from a dead process
                continue
            return False
    return False


def atomic_write(path, rec, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path.with_suffix(".npz"))
    tmp = path.with_suffix(".json.tmp")
    write_json(rec, tmp)
    os.replace(tmp, path)


def worker(ctx, n_reps, kind, device, wid, retry_failed, log):
    if kind == "gpu":
        import torch
        torch.set_num_threads(THREADS)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            log.warning("No GPU visible; neural tasks run on CPU.")
            device = "cpu"
    tasks = task_list(ctx, n_reps, kind)
    # Largest training sets first for better load balance.
    size = {(k, r, b): len(load_split(ctx, k, r)["budgets"][b]) for k, r, b, _ in tasks}
    tasks.sort(key=lambda x: (-size[x[:3]], x[1], x[0], x[3]))
    done = 0
    for key, rep, lab, m in tasks:
        path = ctx.task_path(key, rep, lab, m)
        split_hash = load_split(ctx, key, rep)["hash"]
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("status") == "ok" and old.get("split_hash") == split_hash:
                continue
            if old.get("status") in ("failed", "invalid") and not retry_failed:
                continue
        lock = ctx.root / "locks" / f"{key}_rep{rep:02d}_{lab}__{m}.lock"
        if not claim(lock):
            continue
        try:
            if path.exists():
                old = json.loads(path.read_text())
                if old.get("status") == "ok" and old.get("split_hash") == split_hash:
                    continue
            t0 = time.perf_counter()
            log.info("[%s] start %s rep%02d budget=%s %s", wid, key, rep, lab, m)
            rec, arrays = run_task(ctx, key, rep, lab, m, device if m in NEURAL else "cpu", log)
            rec.update({"task_wall_s": time.perf_counter() - t0, "worker": wid})
            atomic_write(path, rec, arrays)
            done += 1
            mt = rec["metrics"]
            (log.info if rec["status"] == "ok" else log.error)(
                "[%s] %s %s rep%02d budget=%s %s cfg=%s IPA=%.4f C=%.4f %.0fs %s", wid,
                "done" if rec["status"] == "ok" else "INVALID", key, rep, lab, m, rec["selected_config"],
                mt["IPA"], mt["Uno_C_tau"], rec["task_wall_s"], rec["invalid_reasons"])
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] FAILED %s rep%02d budget=%s %s: %r\n%s", wid, key, rep, lab, m, exc,
                      traceback.format_exc())
            atomic_write(path, {"dataset": key, "rep": rep, "budget": lab, "model": m,
                                "status": "failed", "error": repr(exc)[:1000], "split_hash": split_hash,
                                "traceback": traceback.format_exc()[-4000:], "worker": wid}, None)
        finally:
            Path(lock).unlink(missing_ok=True)
    log.info("[%s] finished: %d tasks completed in this session", wid, done)


# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------

def collect(ctx, n_reps, log):
    rows, tune, fails = [], [], []
    for key, rep, lab, m in task_list(ctx, n_reps):
        path = ctx.task_path(key, rep, lab, m)
        if not path.exists():
            fails.append({"dataset": key, "rep": rep, "budget": lab, "model": m, "status": "missing"})
            continue
        r = json.loads(path.read_text())
        status = r.get("status")
        if status == "ok" and r.get("split_hash") != load_split(ctx, key, rep)["hash"]:
            status = "stale (split changed)"
        if status != "ok":
            fails.append({"dataset": key, "rep": rep, "budget": lab, "model": m, "status": status,
                          "error": r.get("error") or "; ".join(r.get("invalid_reasons", []))})
            continue
        flat = {k: r[k] for k in ("dataset", "rep", "budget", "model", "device", "n_train", "events_train",
                                  "n_fit", "n_val", "n_test", "events_test", "n_pool", "t_star", "tau",
                                  "selected_config", "valid", "tuning_wall_s", "tuning_cpu_s",
                                  "final_fit_wall_s", "final_fit_cpu_s", "predict_wall_s",
                                  "metrics_wall_s", "task_wall_s", "gpu_peak_mem_mb")}
        flat["selected_params"] = json.dumps(r["selected_params"])
        flat["stability_wall_s"] = r.get("stability_wall_s")
        flat["calibration_curve_estimable"] = r.get("calibration_curve_estimable")
        flat["n_warnings"] = len(r.get("warnings", []))
        flat.update(r["metrics"])
        flat.update(r.get("metrics_conditional_ipcw", {}))
        flat.update({f"stab_{k}": v for k, v in (r.get("stability") or {}).items()})
        flat.update({f"check_{k}": v for k, v in r["checks"].items()})
        rows.append(flat)
        for tr in r["tuning"]:
            tune.append({"dataset": key, "rep": rep, "budget": lab, "model": m,
                         "config_index": tr["config_index"], "params": json.dumps(tr["params"]),
                         "val_IBS": tr.get("val_IBS"), "fit_wall_s": tr.get("fit_wall_s"),
                         "status": tr["status"], "selected": tr["config_index"] == r["selected_config"],
                         "epochs_run": tr.get("info_epochs_run"), "best_epoch": tr.get("info_best_epoch")})
    res = pd.DataFrame(rows)
    res.to_csv(ctx.root / "results_tasks.csv", index=False)
    pd.DataFrame(tune).to_csv(ctx.root / "tuning_records.csv", index=False)
    pd.DataFrame(fails, columns=["dataset", "rep", "budget", "model", "status", "error"]).to_csv(
        ctx.root / "incomplete_tasks.csv", index=False)
    total = len(task_list(ctx, n_reps))
    by_status = pd.Series([f["status"] for f in fails]).value_counts().to_dict() if fails else {}
    summary = {"n_reps": n_reps, "tasks_expected": total, "tasks_ok": len(rows),
               "tasks_not_ok": len(fails), "not_ok_by_status": by_status,
               "complete": len(rows) == total}
    write_json(summary, ctx.root / f"collect_summary_{n_reps}reps.json")
    log.info("COLLECT %s", summary)
    if fails:
        log.warning("%d tasks missing, failed, invalid or stale - see incomplete_tasks.csv; rerun the workers to retry.",
                    len(fails))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full_study_config.yaml")
    ap.add_argument("--n-reps", type=int, required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--worker", choices=["cpu", "gpu"])
    ap.add_argument("--worker-id", default="w0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-retry-failed", action="store_true")
    args = ap.parse_args()
    ctx = Ctx(load_config(args.config))
    log = setup_logging(f"{'prepare' if args.prepare else 'collect' if args.collect else args.worker_id}",
                        ctx.root / "logs")
    if args.prepare:
        prepare(ctx, args.n_reps, log)
    elif args.collect:
        collect(ctx, args.n_reps, log)
    elif args.worker:
        worker(ctx, args.n_reps, args.worker, args.device, args.worker_id, not args.no_retry_failed, log)
    else:
        ap.error("choose --prepare, --worker or --collect")


if __name__ == "__main__":
    main()
