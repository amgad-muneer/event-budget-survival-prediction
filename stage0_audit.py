#!/usr/bin/env python3
"""Stage 0 audit for the event-budget survival benchmarking study.

Verifies the computing environment, inventories and checksums the dataset
files, audits each dataset's schema and survival outcome, and computes the
Kaplan-Meier / reverse-Kaplan-Meier follow-up support needed to choose
evaluation horizons and event budgets.

The data directory is opened read-only and is never written to. All outputs
go to a new, timestamped directory that must lie outside the data directory.

Usage:
    python stage0_audit.py --data-dir data --out-root outputs

Optional, once units/columns are confirmed by a human:
    --override gbsg:duration:event      (dataset:time_col:event_col)
    --units support=days --units metabric=months
"""
import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_KEYS = ("support", "metabric", "gbsg")
DATA_EXTS = {".csv", ".tsv", ".feather", ".parquet", ".h5", ".hdf5", ".hdf",
             ".pkl", ".pickle", ".json", ".txt", ".md", ".zip", ".gz", ".npz"}
LOADABLE_EXTS = {".csv", ".tsv", ".feather", ".parquet", ".h5", ".hdf5", ".hdf"}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "scikit-survival",
            "lifelines", "pycox", "torchtuples", "torch", "h5py", "tables",
            "pyarrow", "numba", "matplotlib"]
TIME_NAMES = ["duration", "time", "t", "futime", "survival_time", "os_months",
              "rfstime", "time_to_event", "days", "months"]
EVENT_NAMES = ["event", "e", "status", "death", "dead", "fstat", "label",
               "event_observed", "delta"]
BUDGET_LADDER = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
DEV_FRACTION = 0.70          # planned development-pool share per outer split
TEST_FRACTION = 1.0 - DEV_FRACTION
INNER_FOLDS = 5
CLINICAL_ROUNDS = {"days": [30, 90, 180, 365, 730, 1095, 1825],
                   "months": [6, 12, 24, 36, 60, 120, 180, 240]}
ADMISSIBLE_S = (0.2, 0.9)    # KM event-free survival range at a horizon
ADMISSIBLE_G = 0.3           # minimum censoring survival G(t-) at a horizon
TAU_G = 0.2                  # minimum G(t-) for the integration limit
# Published DeepSurv split sizes for the pooled GBSG data (Katzman et al. 2018):
GBSG_PUBLISHED_SPLIT = {"train (Rotterdam)": 1546, "test (GBSG)": 686}

WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print("WARNING:", msg, flush=True)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def run_cmd(cmd, timeout=120):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"returncode": out.returncode, "stdout": out.stdout.strip(),
                "stderr": out.stderr.strip()}
    except Exception as exc:  # noqa: BLE001
        return {"returncode": None, "error": repr(exc)}


def read_first_line(path):
    try:
        return Path(path).read_text().strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return None


def meminfo():
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, val = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                info[key + "_GiB"] = round(int(val.split()[0]) / 1024 ** 2, 2)
    except Exception:  # noqa: BLE001
        pass
    info["cgroup_memory_max"] = read_first_line("/sys/fs/cgroup/memory.max")
    return info


def environment_report():
    env = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "cpu_count_os": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": read_first_line("/sys/fs/cgroup/cpu.max"),
        "memory": meminfo(),
        "env_threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "CUDA_VISIBLE_DEVICES")},
        "packages": {},
        "imports": {},
    }
    for pkg in PACKAGES:
        try:
            env["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            env["packages"][pkg] = None

    # Import checks for the classes the study depends on.
    checks = {
        "sksurv": "from sksurv.linear_model import CoxPHSurvivalAnalysis\n"
                  "from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis\n"
                  "from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score",
        "pycox": "from pycox.models import CoxPH, LogisticHazard\nimport torchtuples",
        "lifelines": "from lifelines import KaplanMeierFitter, CoxPHFitter",
    }
    for name, code in checks.items():
        try:
            exec(code, {})  # noqa: S102 - fixed, local import statements
            env["imports"][name] = "ok"
        except Exception as exc:  # noqa: BLE001
            env["imports"][name] = f"FAILED: {exc!r}"
            warn(f"Import check failed for {name}: {exc!r}")

    gpu = {"torch_import": None}
    try:
        torch = importlib.import_module("torch")
        gpu["torch_import"] = "ok"
        gpu["torch_cuda_build"] = torch.version.cuda
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        gpu["devices"] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                gpu["devices"].append({
                    "index": i, "name": props.name,
                    "total_GiB": round(total / 1024 ** 3, 2),
                    "free_GiB": round(free / 1024 ** 3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
            # Smoke test: a tiny matmul on each device.
            for i in range(torch.cuda.device_count()):
                x = torch.randn(256, 256, device=f"cuda:{i}")
                float((x @ x).sum().item())
            gpu["smoke_test"] = "ok"
    except Exception as exc:  # noqa: BLE001
        gpu["error"] = repr(exc)
        warn(f"GPU/torch check failed: {exc!r}")
    gpu["nvidia_smi"] = run_cmd(["nvidia-smi", "--query-gpu=index,name,driver_version,"
                                 "memory.total,memory.used,memory.free",
                                 "--format=csv,noheader"])
    env["gpu"] = gpu
    env["pip_check"] = run_cmd([sys.executable, "-m", "pip", "check"])
    if "No module named pip" in env["pip_check"].get("stderr", ""):
        env["pip_check"]["note"] = "pip not installed in this interpreter; conflicts not checked"
    elif env["pip_check"].get("returncode") not in (0, None):
        warn("`pip check` reports dependency conflicts (see environment.json).")
    return env


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def inventory(data_dir, exclude=(), max_depth=3):
    """Data files under data_dir, skipping hidden folders and the project/output folders."""
    rows = []
    base_depth = len(data_dir.parts)
    exclude = [Path(e).resolve() for e in exclude]
    for root, dirs, files in os.walk(data_dir):
        root_p = Path(root)
        if len(root_p.parts) - base_depth >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".")
                   and not any((root_p / d).resolve() == e or e in (root_p / d).resolve().parents
                               or (root_p / d).resolve() in e.parents for e in exclude)]
        for f in sorted(files):
            p = root_p / f
            if p.suffix.lower() not in DATA_EXTS:
                continue
            st = p.stat()
            rows.append({
                "relpath": str(p.relative_to(data_dir)),
                "size_bytes": st.st_size,
                "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
                "sha256": sha256(p),
                "dataset_key": next((k for k in DATASET_KEYS if k in f.lower()), None),
            })
    return rows


# --------------------------------------------------------------------------
# Loading (read-only)
# --------------------------------------------------------------------------

def describe_h5(path):
    import h5py
    tree = []
    with h5py.File(path, "r") as fh:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                tree.append({"name": name, "kind": "dataset", "shape": list(obj.shape),
                             "dtype": str(obj.dtype)})
            else:
                tree.append({"name": name, "kind": "group", "attrs": list(obj.attrs.keys())})
        fh.visititems(visit)
    return tree


def load_h5(path):
    """Return (dataframe or None, structure, loader note)."""
    import h5py
    structure = describe_h5(path)
    with h5py.File(path, "r") as fh:
        top = list(fh.keys())
        deepsurv_like = top and all(
            isinstance(fh[g], h5py.Group) and {"x", "t", "e"} <= set(fh[g].keys()) for g in top)
        if deepsurv_like:
            parts = []
            for g in top:
                x = np.asarray(fh[g]["x"])
                part = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
                part["duration"] = np.asarray(fh[g]["t"]).ravel()
                part["event"] = np.asarray(fh[g]["e"]).ravel()
                part["_h5_group"] = g
                parts.append(part)
            return (pd.concat(parts, ignore_index=True), structure,
                    "DeepSurv-style h5 (groups with x/t/e); '_h5_group' records the stored split")
    try:
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
        if len(keys) == 1:
            return pd.read_hdf(path, keys[0], mode="r"), structure, f"pandas HDFStore key {keys[0]}"
        return None, structure, f"pandas HDFStore with keys {keys}; not auto-loaded"
    except Exception as exc:  # noqa: BLE001
        return None, structure, f"unrecognised h5 layout ({exc!r}); structure only"


def load_table(path):
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if ext == ".tsv" else ","), None, "csv"
    if ext == ".feather":
        return pd.read_feather(path), None, "feather"
    if ext == ".parquet":
        return pd.read_parquet(path), None, "parquet"
    return load_h5(path)


# --------------------------------------------------------------------------
# Survival utilities (validated against lifelines when available)
# --------------------------------------------------------------------------

def km_table(time, event):
    """Kaplan-Meier estimate. Returns times, S(t) after each time, n at risk, d, c."""
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    order = np.argsort(time, kind="mergesort")
    t, e = time[order], event[order]
    uniq, idx_start, counts = np.unique(t, return_index=True, return_counts=True)
    d = np.add.reduceat(e, idx_start)
    c = counts - d
    n_at_risk = len(t) - np.concatenate([[0], np.cumsum(counts)[:-1]])
    surv = np.cumprod(1.0 - d / n_at_risk)
    return pd.DataFrame({"time": uniq, "n_at_risk": n_at_risk, "events": d,
                         "censored": c, "surv": surv})


def step_eval(tab, t, left=False):
    """Evaluate a KM step function at t (right-continuous; left=True gives S(t-))."""
    times = tab["time"].to_numpy()
    surv = tab["surv"].to_numpy()
    idx = np.searchsorted(times, t, side="left" if left else "right") - 1
    return np.where(idx >= 0, surv[np.clip(idx, 0, None)], 1.0)


def validate_km(time, event):
    try:
        from lifelines import KaplanMeierFitter
    except Exception:  # noqa: BLE001
        return {"lifelines_check": "lifelines unavailable"}
    tab = km_table(time, event)
    kmf = KaplanMeierFitter().fit(time, event)
    ref = kmf.survival_function_at_times(tab["time"]).to_numpy()
    rtab = km_table(time, 1 - np.asarray(event))
    rkmf = KaplanMeierFitter().fit(time, 1 - np.asarray(event))
    rref = rkmf.survival_function_at_times(rtab["time"]).to_numpy()
    return {"max_abs_diff_km": float(np.max(np.abs(ref - tab["surv"]))),
            "max_abs_diff_reverse_km": float(np.max(np.abs(rref - rtab["surv"])))}


def km_median(tab):
    below = tab[tab["surv"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else None


# --------------------------------------------------------------------------
# Dataset audit
# --------------------------------------------------------------------------

def detect_outcome(df, override=None):
    cols = {c.lower(): c for c in df.columns}
    if override:
        t, e = override
        if t in df.columns and e in df.columns:
            return t, e, "user override"
        warn(f"Override columns {override} not found; falling back to detection.")
    t = next((cols[n] for n in TIME_NAMES if n in cols), None)
    e = next((cols[n] for n in EVENT_NAMES if n in cols), None)
    how = "matched by column name"
    if t is None or e is None:
        how = "NOT FOUND - set --override after inspecting the columns"
    return t, e, how


def robust_extremes(x, z=6.0):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    return int(np.sum(np.abs(x - med) / mad > z)) if mad > 0 else 0


def univariate_c(time, event, x):
    try:
        from lifelines.utils import concordance_index
    except Exception:  # noqa: BLE001
        return None
    ok = np.isfinite(x)
    if ok.sum() < 10 or np.nanstd(x[ok]) == 0:
        return None
    c = concordance_index(time[ok], x[ok], event[ok])
    return float(max(c, 1 - c))


def event_auc(event, x):
    from scipy.stats import mannwhitneyu
    ok = np.isfinite(x)
    a, b = x[ok & (event == 1)], x[ok & (event == 0)]
    if len(a) == 0 or len(b) == 0 or np.std(x[ok]) == 0:
        return None
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    auc = u / (len(a) * len(b))
    return float(max(auc, 1 - auc))


def predictor_profile(df, predictors, time, event):
    rows = []
    n = len(df)
    for col in predictors:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        row = {"column": col, "dtype": str(s.dtype), "n_missing": int(s.isna().sum()),
               "n_unique": int(s.nunique(dropna=True))}
        if numeric:
            x = s.to_numpy(dtype=float)
            fin = x[np.isfinite(x)]
            row.update({
                "min": float(np.min(fin)) if fin.size else None,
                "max": float(np.max(fin)) if fin.size else None,
                "mean": float(np.mean(fin)) if fin.size else None,
                "sd": float(np.std(fin, ddof=1)) if fin.size > 1 else None,
                "skew": float(pd.Series(fin).skew()) if fin.size > 2 else None,
                "integer_valued": bool(fin.size and np.all(np.equal(np.mod(fin, 1), 0))),
                "n_robust_z_gt6": robust_extremes(x),
                "univariate_C_abs": univariate_c(time, event, x),
                "event_AUC_abs": event_auc(event, x),
                "spearman_with_time": float(pd.Series(x).corr(pd.Series(time), method="spearman")),
            })
            row["type_hint_unverified"] = ("binary" if row["n_unique"] <= 2 else
                                           "low-cardinality (<=10 levels)" if row["n_unique"] <= 10 else
                                           "continuous/count")
        else:
            row["type_hint_unverified"] = "non-numeric"
        row["id_like"] = bool(row["n_unique"] == n and (not numeric or row.get("integer_valued"))
                              or re.search(r"(^id$|_id$|^pid|patient|subject)", str(col).lower()))
        flags = []
        if row["id_like"]:
            flags.append("possible identifier")
        if (row.get("univariate_C_abs") or 0) > 0.80:
            flags.append("very strong univariate C (>0.80): check for post-outcome information")
        if (row.get("event_AUC_abs") or 0) > 0.90:
            flags.append("near-separation of event status (AUC>0.90)")
        if abs(row.get("spearman_with_time") or 0) > 0.95:
            flags.append("almost identical ranking to follow-up time")
        row["flags"] = flags
        rows.append(row)
    return rows


def horizon_table(km, rkm, time, event, units=None):
    t_events = time[event == 1]
    cands = {}
    for q in (0.10, 0.25, 0.33, 0.50, 0.67, 0.75, 0.90):
        cands[f"event-time q{int(q * 100)}"] = float(np.quantile(t_events, q))
    unit_sets = [units] if units in CLINICAL_ROUNDS else list(CLINICAL_ROUNDS)
    for u in unit_sets:
        for r in CLINICAL_ROUNDS[u]:
            if r < time.max():
                tag = f"{r} {u}" + ("" if units == u else " (units UNVERIFIED)")
                cands[tag] = float(r)
    g_left_events = step_eval(rkm, t_events, left=True)
    rows = []
    for label, t in sorted(cands.items(), key=lambda kv: kv[1]):
        s = float(step_eval(km, t))
        g = float(step_eval(rkm, t, left=True))
        at_risk = int(np.sum(time >= t))
        ev_before = int(np.sum((time <= t) & (event == 1)))
        w_max = float(1.0 / np.min(g_left_events[t_events <= t])) if ev_before else None
        rows.append({
            "candidate": label, "time": t, "KM_S": s, "reverse_KM_G_left": g,
            "n_at_risk": at_risk, "events_up_to_t": ev_before,
            "expected_test_at_risk": round(TEST_FRACTION * at_risk, 1),
            "max_ipcw_weight_events": w_max, "ipcw_weight_survivors": (1.0 / g) if g > 0 else None,
            "admissible_horizon": bool(ADMISSIBLE_S[0] <= s <= ADMISSIBLE_S[1] and g >= ADMISSIBLE_G),
            "admissible_tau": bool(g >= TAU_G and TEST_FRACTION * at_risk >= 30),
        })
    return rows


def propose_horizons(rows):
    """Unit-free proposal: admissible event-time quantile candidates nearest q25/q50/q75."""
    q = [r for r in rows if r["candidate"].startswith("event-time q") and r["admissible_horizon"]]
    pick = {}
    for target in ("q25", "q50", "q75"):
        match = [r for r in q if r["candidate"].endswith(target)]
        if match:
            pick[target] = match[0]["time"]
    taus = [r["time"] for r in rows if r["admissible_tau"]]
    return {"status": "PROPOSED - NOT FROZEN; round to clinical values once units are confirmed",
            "horizons_by_event_quantile": pick,
            "max_admissible_tau": max(taus) if taus else None}


def budget_table(n, n_events, n_predictors):
    dev_events = int(np.floor(DEV_FRACTION * n_events))
    frac = n_events / n
    rows = []
    for e in [b for b in BUDGET_LADDER if b < dev_events] + [dev_events]:
        rows.append({"budget_events": e, "label": "full pool" if e == dev_events else str(e),
                     "expected_n": int(np.ceil(e / frac)),
                     "events_per_raw_predictor": round(e / max(n_predictors, 1), 1),
                     "events_per_inner_fold": round(e / INNER_FOLDS, 1),
                     "inner_fold_warning": e / INNER_FOLDS < 5})
    return {"expected_dev_pool_events": dev_events, "event_fraction": frac, "budgets": rows}


def censoring_diagnostic(df, predictors, time, event):
    """Cox model for the censoring hazard. A diagnostic, not a test of independence."""
    try:
        from lifelines import CoxPHFitter
    except Exception:  # noqa: BLE001
        return {"status": "lifelines unavailable"}
    num = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].astype(float)
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    X = (X - X.mean()) / X.std()
    X["_T"], X["_C"] = time, 1 - event
    if X["_C"].sum() < 10:
        return {"status": "too few censored observations"}
    try:
        cph = CoxPHFitter(penalizer=0.01).fit(X, duration_col="_T", event_col="_C")
        lr = cph.log_likelihood_ratio_test()
        coefs = cph.params_.abs().sort_values(ascending=False)
        return {
            "status": "ok", "n_censored": int(X["_C"].sum()),
            "censoring_model_C": float(cph.concordance_index_),
            "lr_test_stat": float(lr.test_statistic), "lr_test_df": int(lr.degrees_freedom),
            "lr_test_p": float(lr.p_value),
            "largest_abs_coefs": {k: float(cph.params_[k]) for k in coefs.index[:5]},
            "interpretation": ("Censoring-model C near 0.5 supports marginal IPCW; C >= 0.60 "
                               "suggests covariate-dependent censoring and motivates a "
                               "prespecified Cox-IPCW sensitivity analysis. Significance alone "
                               "is not informative at these sample sizes."),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": f"failed: {exc!r}"}


def gbsg_cohort_check(df, time, event):
    out = {"published_deepsurv_split": GBSG_PUBLISHED_SPLIT}
    if "_h5_group" in df.columns:
        groups = {}
        for g, sub in df.groupby("_h5_group"):
            ti, ei = sub["duration"].to_numpy(float), sub["event"].to_numpy(int)
            groups[g] = {"n": len(sub), "events": int(ei.sum()), "max_time": float(ti.max()),
                         "median_followup_reverse_km": km_median(km_table(ti, 1 - ei))}
        sizes = sorted(v["n"] for v in groups.values())
        out["h5_groups"] = groups
        out["matches_published_sizes"] = sizes == sorted(GBSG_PUBLISHED_SPLIT.values())
        out["conclusion"] = (
            "Stored h5 split sizes match the published Rotterdam/GBSG split; group membership "
            "is a provenance-consistent cohort marker (cite Katzman et al. 2018 for the mapping)."
            if out["matches_published_sizes"] else
            "Stored h5 groups do NOT match the published split sizes; do not treat them as cohorts.")
    else:
        matches = []
        for c in df.columns:
            if c in (time, event):
                continue
            vc = df[c].value_counts(dropna=False)
            if len(vc) == 2 and sorted(vc.to_list()) == sorted(GBSG_PUBLISHED_SPLIT.values()):
                matches.append(c)
        out["columns_with_matching_two_level_counts"] = matches
        out["conclusion"] = (
            "Flat file without a stored split. Cohort membership is NOT recoverable from this file "
            "unless a matching column is independently documented; no identifier was created."
            + (f" Columns with coincidentally matching counts (unverified): {matches}" if matches else ""))
    return out


def audit_dataset(key, path, df, loader_note, structure, override, units):
    rep = {"dataset": key, "file": str(path), "loader": loader_note, "h5_structure": structure}
    rep["shape"] = list(df.shape)
    rep["columns"] = [str(c) for c in df.columns]
    rep["dtypes"] = {str(c): str(t) for c, t in df.dtypes.items()}
    tcol, ecol, how = detect_outcome(df, override)
    rep["time_col"], rep["event_col"], rep["outcome_detection"] = tcol, ecol, how
    if tcol is None or ecol is None:
        warn(f"{key}: outcome columns not identified; downstream audit skipped.")
        return rep, None

    time = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    ev_raw = df[ecol]
    rep["event_values"] = {str(k): int(v) for k, v in ev_raw.value_counts(dropna=False).items()}
    if not set(pd.unique(ev_raw.dropna())) <= {0, 1, True, False, 0.0, 1.0}:
        warn(f"{key}: event column is not coded 0/1 ({rep['event_values']}); not recoded.")
        return rep, None
    event = ev_raw.astype(float).to_numpy()

    invalid = ~np.isfinite(time) | (time <= 0) | ~np.isfinite(event)
    rep["time_summary"] = {
        "min": float(np.nanmin(time)), "max": float(np.nanmax(time)),
        "quantiles": {f"q{int(q * 100)}": float(np.nanquantile(time, q))
                      for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
        "n_nonpositive": int(np.sum(time <= 0)), "n_nan_or_inf": int(np.sum(~np.isfinite(time))),
        "integer_valued_fraction": float(np.mean(np.mod(time[np.isfinite(time)], 1) == 0)),
        "n_distinct": int(pd.Series(time).nunique()),
        "units": units or "UNVERIFIED - confirm from provenance documentation",
    }
    if invalid.any():
        warn(f"{key}: {int(invalid.sum())} rows with invalid duration/event; excluded from "
             "survival summaries only (data untouched).")
    time, event = time[~invalid], event[~invalid].astype(int)
    sub = df.loc[~invalid].reset_index(drop=True)

    rep["n"] = int(len(time))
    rep["n_events"] = int(event.sum())
    rep["n_censored"] = int(len(event) - event.sum())
    rep["event_fraction"] = float(event.mean())
    rep["missing_by_column"] = {str(c): int(v) for c, v in df.isna().sum().items() if v}
    predictors = [c for c in df.columns if c not in (tcol, ecol, "_h5_group")]
    rep["n_predictors_raw"] = len(predictors)
    rep["duplicates"] = {
        "full_rows": int(df.duplicated().sum()),
        "predictor_profiles": int(df[predictors].duplicated().sum()),
        "unique_patient_count": "not establishable: no patient identifier column"
        if not any(re.search(r"(^id$|_id$|^pid|patient|subject)", str(c).lower())
                   for c in df.columns) else "identifier-like column present; verify",
    }
    rep["predictors"] = predictor_profile(sub, predictors, time, event)
    for p in rep["predictors"]:
        for f in p["flags"]:
            warn(f"{key}: column {p['column']}: {f}")

    km, rkm = km_table(time, event), km_table(time, 1 - event)
    rep["km_validation"] = validate_km(time, event)
    rep["followup"] = {
        "max_time": float(time.max()), "max_event_time": float(time[event == 1].max()),
        "median_followup_reverse_km": km_median(rkm), "median_survival_km": km_median(km),
        "share_censored_in_last_decile_of_time_range":
            float(np.mean(time[event == 0] >= time.min() + 0.9 * (time.max() - time.min())))
            if (event == 0).any() else None,
    }
    rep["horizon_candidates"] = horizon_table(km, rkm, time, event, units)
    rep["horizon_proposal"] = propose_horizons(rep["horizon_candidates"])
    rep["budgets"] = budget_table(len(time), int(event.sum()), len(predictors))
    rep["censoring_diagnostic"] = censoring_diagnostic(sub, predictors, time, event)
    if key == "gbsg":
        rep["gbsg_cohort"] = gbsg_cohort_check(sub, tcol, ecol)
    return rep, (km, rkm, time, event)


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def plot_followup(curves, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    keys = list(curves)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        km, rkm, proposal = curves[key]
        ax.step(km["time"], km["surv"], where="post", label="KM event-free S(t)")
        ax.step(rkm["time"], rkm["surv"], where="post", label="Reverse KM G(t)", ls="--")
        for t in proposal["horizons_by_event_quantile"].values():
            ax.axvline(t, color="grey", lw=0.8, ls=":")
        ax.axhline(ADMISSIBLE_G, color="red", lw=0.6, ls=":")
        ax.set_title(key)
        ax.set_xlabel("time (units per audit report)")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("probability")
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    path = out_dir / "stage0_followup_diagnostic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def markdown_report(env, inv, reports, out_dir):
    L = ["# Stage 0 audit report", "", f"Generated: {env['timestamp_utc']}", ""]
    L += ["## Environment", "",
          f"- Python: `{env['python'].split()[0]}` ({env['python_executable']})",
          f"- Platform: `{env['platform']}`",
          f"- CPUs (os / affinity / cgroup): {env['cpu_count_os']} / {env['cpu_affinity']} / "
          f"{env['cgroup_cpu_max']}",
          f"- Memory: {env['memory']}",
          f"- CUDA available: {env['gpu'].get('cuda_available')} "
          f"(torch CUDA build {env['gpu'].get('torch_cuda_build')})"]
    for d in env["gpu"].get("devices", []):
        L.append(f"  - GPU {d['index']}: {d['name']}, total {d['total_GiB']} GiB, "
                 f"free {d['free_GiB']} GiB, CC {d['compute_capability']}")
    L += ["", "| package | version |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in env["packages"].items()]
    L += ["", f"Import checks: {env['imports']}",
          f"`pip check` return code: {env['pip_check'].get('returncode')}", ""]
    L += ["## Inventory (SHA-256)", "", "| file | bytes | dataset | sha256 |", "|---|---|---|---|"]
    L += [f"| {r['relpath']} | {r['size_bytes']} | {r['dataset_key']} | `{r['sha256'][:16]}...` |"
          for r in inv]
    for rep in reports:
        L += ["", f"## {rep['dataset']} - `{rep['file']}`", "",
              f"- Loader: {rep['loader']}", f"- Shape: {rep['shape']}",
              f"- Columns: {rep['columns']}",
              f"- Outcome: time=`{rep['time_col']}`, event=`{rep['event_col']}` "
              f"({rep['outcome_detection']})"]
        if "n" not in rep:
            continue
        L += [f"- n={rep['n']}, events={rep['n_events']}, censored={rep['n_censored']} "
              f"(event fraction {rep['event_fraction']:.3f}); event coding {rep['event_values']}",
              f"- Time: {rep['time_summary']}",
              f"- Missing: {rep['missing_by_column'] or 'none'}",
              f"- Duplicates: {rep['duplicates']}",
              f"- KM self-check vs lifelines: {rep['km_validation']}",
              f"- Follow-up: {rep['followup']}", "",
              "| predictor | type hint (unverified) | unique | missing | min | max | skew | "
              "|z|>6 | univ. C | flags |", "|---|---|---|---|---|---|---|---|---|---|"]
        for p in rep["predictors"]:
            f = lambda v: "" if v is None else (f"{v:.3g}" if isinstance(v, float) else v)  # noqa: E731
            L.append(f"| {p['column']} | {p.get('type_hint_unverified')} | {p['n_unique']} | "
                     f"{p['n_missing']} | {f(p.get('min'))} | {f(p.get('max'))} | "
                     f"{f(p.get('skew'))} | {p.get('n_robust_z_gt6', '')} | "
                     f"{f(p.get('univariate_C_abs'))} | {'; '.join(p['flags'])} |")
        L += ["", "Horizon candidates (admissible: S in [0.2,0.9] and G(t-) >= 0.3):", "",
              "| candidate | time | S(t) | G(t-) | at risk | exp. test at risk | max IPCW wt "
              "| horizon ok | tau ok |", "|---|---|---|---|---|---|---|---|---|"]
        for h in rep["horizon_candidates"]:
            w = h["max_ipcw_weight_events"]
            L.append(f"| {h['candidate']} | {h['time']:.4g} | {h['KM_S']:.3f} | "
                     f"{h['reverse_KM_G_left']:.3f} | {h['n_at_risk']} | "
                     f"{h['expected_test_at_risk']} | {'' if w is None else f'{w:.2f}'} | "
                     f"{h['admissible_horizon']} | {h['admissible_tau']} |")
        L += ["", f"Proposal: {rep['horizon_proposal']}", "",
              f"Budgets (dev pool = {DEV_FRACTION:.0%}; expected dev events "
              f"{rep['budgets']['expected_dev_pool_events']}):", "",
              "| budget | expected n | events/raw predictor | events/inner fold | warn |",
              "|---|---|---|---|---|"]
        L += [f"| {b['label']} | {b['expected_n']} | {b['events_per_raw_predictor']} | "
              f"{b['events_per_inner_fold']} | {b['inner_fold_warning']} |"
              for b in rep["budgets"]["budgets"]]
        L += ["", f"Censoring diagnostic: {rep['censoring_diagnostic']}"]
        if "gbsg_cohort" in rep:
            L += ["", f"GBSG cohort check: {rep['gbsg_cohort']}"]
    L += ["", "## Warnings", ""] + [f"- {w}" for w in WARNINGS or ["none"]]
    L += ["", "## Requires human confirmation before freezing", "",
          "- Time units and endpoint definition for each dataset (from provenance documents).",
          "- Clinical identity of anonymised predictors (needed only for any domain-based transform).",
          "- Final horizons (rounded clinical values) and event budgets.",
          "- Dataset licences / usage terms."]
    path = out_dir / "stage0_report.md"
    path.write_text("\n".join(L))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parent / "outputs"))
    ap.add_argument("--override", action="append", default=[],
                    help="dataset:time_col:event_col (after inspecting a first run)")
    ap.add_argument("--units", action="append", default=[],
                    help="dataset=days|months (only once confirmed from documentation)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not data_dir.is_dir():
        sys.exit(f"Data directory not found: {data_dir}")
    if out_root == data_dir or out_root in data_dir.parents:
        sys.exit("Refusing to run: outputs may not be written directly into (or above) the data "
                 "directory. Use a project subfolder, e.g. --out-root outputs.")
    out_dir = out_root / ("stage0_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out_dir.mkdir(parents=True, exist_ok=False)

    overrides = {o.split(":")[0].lower(): tuple(o.split(":")[1:3]) for o in args.override}
    units = {u.split("=")[0].lower(): u.split("=")[1].lower() for u in args.units}

    print("Environment check...", flush=True)
    env = environment_report()
    env["data_dir_writable_by_this_user"] = os.access(data_dir, os.W_OK)
    if env["data_dir_writable_by_this_user"]:
        warn("The data directory is writable by this user; this script never writes there, "
             "but consider `chmod -R a-w` on it for protection.")
    (out_dir / "environment.json").write_text(json.dumps(to_jsonable(env), indent=2))

    print("Inventory and checksums...", flush=True)
    project_dir = Path(__file__).resolve().parent
    inv = inventory(data_dir, exclude=(out_root, project_dir))
    if data_dir in out_root.parents:
        print(f"Note: outputs go to {out_root}, a project subfolder of the data directory; "
              "the dataset files themselves are only read.", flush=True)
    (out_dir / "checksums.sha256").write_text(
        "".join(f"{r['sha256']}  {r['relpath']}\n" for r in inv))

    reports, curves = [], {}
    for key in DATASET_KEYS:
        files = [r for r in inv if r["dataset_key"] == key
                 and Path(r["relpath"]).suffix.lower() in LOADABLE_EXTS]
        if not files:
            warn(f"No loadable file found for {key}.")
            continue
        if len(files) > 1:
            warn(f"{key}: {len(files)} candidate files; each is audited separately - "
                 "confirm which one the study uses.")
        for r in files:
            path = data_dir / r["relpath"]
            print(f"Auditing {key}: {r['relpath']}", flush=True)
            try:
                df, structure, note = load_table(path)
                if df is None:
                    reports.append({"dataset": key, "file": str(path), "loader": note,
                                    "h5_structure": structure, "shape": None, "columns": [],
                                    "time_col": None, "event_col": None,
                                    "outcome_detection": "not loaded"})
                    warn(f"{key}: {r['relpath']} not loaded ({note}).")
                    continue
                rep, cur = audit_dataset(key, path, df, note, structure,
                                         overrides.get(key), units.get(key))
                reports.append(rep)
                if cur is not None and key not in curves:
                    curves[key] = (cur[0], cur[1], rep["horizon_proposal"])
                    cur[0].to_csv(out_dir / f"{key}_km.csv", index=False)
                    cur[1].to_csv(out_dir / f"{key}_reverse_km.csv", index=False)
            except Exception as exc:  # noqa: BLE001
                warn(f"{key}: audit failed for {r['relpath']}: {exc!r}")
                (out_dir / f"{key}_error.txt").write_text(traceback.format_exc())

    if curves:
        plot_followup(curves, out_dir)

    # Confirm the data files were not changed during the run.
    after = {r["relpath"]: sha256(data_dir / r["relpath"]) for r in inv}
    changed = [k for k, v in after.items() if v != next(r["sha256"] for r in inv if r["relpath"] == k)]
    if changed:
        warn(f"Checksums changed during the run (external modification?): {changed}")

    (out_dir / "stage0_audit.json").write_text(json.dumps(to_jsonable(
        {"environment": env, "inventory": inv, "datasets": reports, "warnings": WARNINGS,
         "checksums_unchanged": not changed}), indent=2))
    # Project-level manifests (latest audit), consumed by the pilot scripts.
    (out_root / "environment_manifest.json").write_text(json.dumps(to_jsonable(env), indent=2))
    (out_root / "dataset_manifest.json").write_text(json.dumps(to_jsonable(
        {"stage0_dir": str(out_dir), "data_dir": str(data_dir), "inventory": inv,
         "datasets": reports, "checksums_unchanged": not changed, "warnings": WARNINGS,
         "unresolved_metadata": ["time units", "endpoint definitions", "predictor identities",
                                 "licences/usage terms"]}), indent=2))
    report = markdown_report(env, inv, reports, out_dir)
    print(f"\nDone. Report: {report}\nAll outputs: {out_dir}")


if __name__ == "__main__":
    main()
