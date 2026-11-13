#!/usr/bin/env python3
"""Controlled end-to-end pilot: one outer split, pilot budgets, five fixed-config models.

For each dataset: stratified test split -> nested event-budget training subsets ->
train-only preprocessing -> fit -> predict on a common grid -> validity checks ->
censoring-aware metrics (censoring and KM reference from the development pool only).

    python run_pilot.py --config pilot_config.yaml                 # all datasets, cuda:0
    python run_pilot.py --datasets metabric gbsg --device cuda:1    # second GPU process
"""
import argparse
import os
import sys

import yaml

# Thread limits must be set before numpy/torch are imported.
_cfg_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "pilot_config.yaml"
with open(_cfg_path) as _fh:
    _threads = str(yaml.safe_load(_fh).get("cpu_threads", 4))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _threads
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import resource  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (Preprocessor, check_curves, compute_metrics, gpu_peak_mb,  # noqa: E402
                    gpu_peak_reset, km_at, load_config,
                    load_dataset, nested_budgets, output_dir, read_manifest, setup_logging,
                    split_seed,
                    stratified_split, truncate, validation_split, write_json)
from models import MODELS, make_model  # noqa: E402


class GpuSampler:
    """Samples GPU utilisation (pynvml, if installed) while a neural model trains."""

    def __init__(self, device):
        self.index = int(device.split(":")[1]) if device.startswith("cuda:") else None
        self.samples, self._stop = [], threading.Event()

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)

            def loop():
                while not self._stop.is_set():
                    self.samples.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    time.sleep(0.2)
            if self.index is not None:
                self._thread = threading.Thread(target=loop, daemon=True)
                self._thread.start()
        except Exception:  # noqa: BLE001
            self.index = None
        return self

    def __exit__(self, *exc):
        self._stop.set()

    def summary(self):
        if not self.samples:
            return {"gpu_util_mean": None, "gpu_util_max": None}
        return {"gpu_util_mean": float(np.mean(self.samples)), "gpu_util_max": float(np.max(self.samples))}


def resolve_device(requested, log):
    """'auto' -> cuda:0 if a GPU is visible, else cpu. Only one GPU is used per process."""
    import torch
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("Device auto-detection: %d GPU(s) visible -> using %s", torch.cuda.device_count(),
                 requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; neural models run on CPU.")
        return "cpu"
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return requested


def fit_predict(name, cfg, seed, device, Xtr, ttr, etr, val, Xte, grid, t_star):
    """Fit one model and predict; returns survival on grid and at t_star, timing and info."""
    model = make_model(name, cfg["models"][name], seed, device, cfg.get("tree_n_jobs", 1))
    kw = {}
    if model.needs_validation:
        fit_rows, val_rows = val
        kw = dict(X_val=Xtr[val_rows], time_val=ttr[val_rows], event_val=etr[val_rows])
        Xfit, tfit, efit = Xtr[fit_rows], ttr[fit_rows], etr[fit_rows]
    else:
        Xfit, tfit, efit = Xtr, ttr, etr
    gpu = device.startswith("cuda") and model.needs_validation
    # Initialise CUDA before resetting peak-memory stats and never let
    # memory monitoring abort training.
    mem_ok = gpu_peak_reset(device) if gpu else False
    with warnings.catch_warnings(record=True) as caught, GpuSampler(device if gpu else "cpu") as gs:
        warnings.simplefilter("always")
        w0, c0 = time.perf_counter(), time.process_time()
        model.fit(Xfit, tfit, efit, **kw)
        w1, c1 = time.perf_counter(), time.process_time()
        S = model.predict_surv(Xte, np.append(grid, t_star))
        w2, c2 = time.perf_counter(), time.process_time()
    timing = {"fit_wall_s": w1 - w0, "fit_cpu_s": c1 - c0, "predict_wall_s": w2 - w1,
              "predict_cpu_s": c2 - c1,
              "peak_rss_mb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "gpu_peak_mem_mb": gpu_peak_mb(device, mem_ok),
              **gs.summary()}
    warn_msgs = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})
    return model, S[:, :-1], S[:, -1], timing, warn_msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot_config.yaml")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    pdir = out / "pilot"
    (pdir / "splits").mkdir(parents=True, exist_ok=True)
    (pdir / "predictions").mkdir(exist_ok=True)
    tag = ("_".join(args.datasets) if args.datasets else "all") + \
        ("__" + "_".join(args.models) if args.models else "")
    log = setup_logging(f"pilot_{tag}", pdir)
    design_path = out / "frozen_design.json"
    if not design_path.exists():
        sys.exit("frozen_design.json not found: run freeze_design.py first.")
    design = json.loads(design_path.read_text())
    manifest = read_manifest(out)
    device = resolve_device(args.device or cfg.get("device", "cpu"), log)
    models = args.models or list(MODELS)
    datasets = args.datasets or list(cfg["datasets"])
    log.info("Pilot start: datasets=%s models=%s device=%s threads=%s", datasets, models,
             device, os.environ["OMP_NUM_THREADS"])

    rows, det_rows, leakage = [], [], {}
    for rep in range(int(cfg["outer_repetitions"])):
        for key in datasets:
            d = design["datasets"].get(key, {})
            if "error" in d or not d:
                log.error("%s: no frozen design (%s); skipped.", key, d.get("error"))
                continue
            seed = split_seed(cfg, key, rep)
            rng = np.random.default_rng(seed)
            X, t, e, strata, _, load_info = load_dataset(cfg, key, manifest)
            if load_info["n_excluded"]:
                log.info("%s: analysis rule excluded %d record(s) (%s); CSV unchanged.", key,
                         load_info["n_excluded"], load_info["excluded_reason"])
            if bool(d.get("cohort_stratified")) != load_info["cohort_stratified"]:
                raise RuntimeError(f"{key}: cohort stratification differs from the frozen design; "
                                   "re-run freeze_design.py --force.")
            test, pool = stratified_split(strata, cfg["test_fraction"], rng)
            subsets, skipped = nested_budgets(pool, e, cfg["pilot_budgets"], rng)
            t_star, tau = float(d["primary_horizon"]), float(d["tau"])
            t_admin = tau * (1 + 1e-6)
            pt, pe = truncate(t[pool], e[pool], t_admin)
            tt_, te = truncate(t[test], e[test], t_admin)
            grid = np.linspace(max(d["grid_start"], float(tt_.min())), tau, int(cfg["grid_points"]))
            km_grid = km_at(pt, pe, grid)
            split_rec = {"dataset": key, "rep": rep, "seed": seed, "test": test, "pool": pool,
                         "budgets": subsets, "skipped": skipped, "grid": grid,
                         "analysis_rule": load_info,
                         "t_star": t_star, "tau": tau, "validation": {}}
            for label, reason in skipped.items():
                log.warning("%s rep%d budget %s skipped: %s", key, rep, label, reason)
                rows.append({"dataset": key, "rep": rep, "budget": label, "status": "skipped",
                             "error": reason})
            lk = {"test_pool_overlap": int(np.intersect1d(test, pool).size), "budgets": {}}
            for label, tr in subsets.items():
                val = validation_split(np.arange(len(tr)), e[tr], cfg["validation_fraction"], seed + 7)
                split_rec["validation"][label] = tr[val[1]]
                pre = Preprocessor(cfg["datasets"][key].get("categorical", [])).fit(X.iloc[tr])
                Xtr, Xte = pre.transform(X.iloc[tr]), pre.transform(X.iloc[test])
                lk["budgets"][label] = {
                    "train_test_overlap": int(np.intersect1d(tr, test).size),
                    "train_subset_of_pool": bool(np.isin(tr, pool).all()),
                    "validation_subset_of_train": bool(np.isin(tr[val[1]], tr).all()),
                    "validation_test_overlap": int(np.intersect1d(tr[val[1]], test).size),
                    "preprocessor_fit_rows": pre.n_fit_, "train_rows": len(tr),
                    "dropped_constant_columns": pre.dropped_constant_}
                for name in models:
                    base = {"dataset": key, "rep": rep, "budget": label, "model": name,
                            "n_train": len(tr), "events_train": int(e[tr].sum()),
                            "n_test": len(test), "events_test": int(e[test].sum()),
                            "t_star": t_star, "tau": tau, "device": device if name in
                            ("deepsurv", "logistic_hazard") else "cpu"}
                    try:
                        model, S, S_star, timing, warns = fit_predict(
                            name, cfg, seed, device, Xtr, t[tr], e[tr], val, Xte, grid, t_star)
                        checks = check_curves(np.column_stack([S, S_star]), len(test), np.append(grid, t_star))
                        checks["nonincreasing"] = check_curves(S, len(test), grid)["nonincreasing"]
                        # Alignment: predictions must follow patients when rows are permuted.
                        order = np.random.default_rng(1).permutation(len(test))
                        S_perm = model.predict_surv(Xte[order], grid)
                        checks["alignment_max_abs_diff"] = float(np.max(np.abs(S_perm - S[order])))
                        checks["aligned"] = checks["alignment_max_abs_diff"] < 1e-6
                        valid = all(checks[k] for k in ("shape_ok", "finite", "in_unit_interval",
                                                        "nonincreasing", "aligned"))
                        m0 = time.perf_counter()
                        metrics = compute_metrics(pt, pe, tt_, te, np.clip(S, 0, 1), grid,
                                                  np.clip(S_star, 0, 1), t_star, tau, km_grid)
                        timing["metrics_wall_s"] = time.perf_counter() - m0
                        np.savez_compressed(pdir / "predictions" / f"{key}_rep{rep}_{label}_{name}.npz",
                                            test_index=test, grid=grid, S=S.astype(np.float32),
                                            S_tstar=S_star.astype(np.float32))
                        rows.append({**base, "status": "ok" if valid else "invalid_predictions",
                                     **timing, **checks, **metrics,
                                     **{f"info_{k}": v for k, v in model.info.items()},
                                     "warnings": " | ".join(warns)[:2000]})
                        for w in warns:
                            log.warning("%s rep%d %s %s: %s", key, rep, label, name, w)
                        log.info("%s rep%d budget=%s %-15s ok=%s fit=%.1fs IPA=%.4f C=%.4f slope=%s",
                                 key, rep, label, name, valid, timing["fit_wall_s"], metrics["IPA"],
                                 metrics["Uno_C_tau"], metrics.get("cal_slope"))
                        if cfg.get("determinism_check") and rep == 0 and label == next(iter(subsets)):
                            _, S2, _, _, _ = fit_predict(name, cfg, seed, device, Xtr, t[tr], e[tr],
                                                         val, Xte, grid, t_star)
                            det_rows.append({"dataset": key, "budget": label, "model": name,
                                             "max_abs_diff_same_seed": float(np.max(np.abs(S2 - S)))})
                    except Exception as exc:  # noqa: BLE001
                        log.error("%s rep%d %s %s FAILED: %r\n%s", key, rep, label, name, exc,
                                  traceback.format_exc())
                        rows.append({**base, "status": "failed", "error": repr(exc)[:500]})
            leakage[f"{key}_rep{rep}"] = lk
            write_json(split_rec, pdir / "splits" / f"{key}_rep{rep}.json")
            pd.DataFrame(rows).to_csv(pdir / f"pilot_metrics_{tag}.csv", index=False)

    pd.DataFrame(rows).to_csv(pdir / f"pilot_metrics_{tag}.csv", index=False)
    pd.DataFrame(det_rows).to_csv(pdir / f"determinism_{tag}.csv", index=False)
    write_json(leakage, pdir / f"leakage_checks_{tag}.json")
    log.info("Pilot finished. %d rows written to %s", len(rows), pdir / f"pilot_metrics_{tag}.csv")


if __name__ == "__main__":
    main()
