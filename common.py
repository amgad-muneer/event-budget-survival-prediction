"""Shared utilities: configuration, logging, data loading, splitting,
preprocessing, survival-curve handling and censoring-aware metrics."""
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
from stage0_audit import load_table  # noqa: E402

EPS = 1e-8


# --------------------------------------------------------------------------
# Configuration and logging
# --------------------------------------------------------------------------

def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    """Load YAML; a `base_config` key inherits (deep-merges) another config file."""
    import yaml
    path = Path(path).resolve()
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if cfg.get("base_config"):
        base = load_config(path.parent / cfg.pop("base_config"))
        base.pop("_config_path", None)
        cfg = _merge(base, cfg)
    cfg["_config_path"] = str(path)
    return cfg


# --------------------------------------------------------------------------
# GPU memory monitoring (optional; never allowed to break training)
# --------------------------------------------------------------------------

def gpu_peak_reset(device):
    """Initialise CUDA on `device` and reset its peak-memory counter. Returns success.

    Calling reset_peak_memory_stats before CUDA is initialised fails on some
    driver/torch combinations (found in the real-data pilot), so initialise first and
    treat any remaining error as 'not measured'."""
    try:
        import torch
        torch.cuda.init()
        torch.cuda.set_device(torch.device(device))
        torch.cuda.reset_peak_memory_stats(device)
        return True
    except Exception:  # noqa: BLE001
        return False


def gpu_peak_mb(device, enabled=True):
    if not enabled:
        return None
    try:
        import torch
        return float(torch.cuda.max_memory_allocated(device) / 2 ** 20)
    except Exception:  # noqa: BLE001
        return None


def output_dir(cfg):
    out = Path(cfg["output_dir"])
    if not out.is_absolute():
        out = PROJECT_DIR / out
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir
    data_dir = data_dir.resolve()
    if out.resolve() == data_dir or out.resolve() in data_dir.parents:
        sys.exit("Refusing to run: output_dir may not be the data directory or one of its parents.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_logging(name, out_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler, level in ((logging.FileHandler(out_dir / f"{name}.log"), logging.INFO),
                           (logging.FileHandler(out_dir / "errors.log"), logging.WARNING),
                           (logging.StreamHandler(sys.stdout), logging.INFO)):
        handler.setLevel(level)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def write_json(obj, path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=default))


def read_manifest(out_dir):
    path = out_dir / "dataset_manifest.json"
    if not path.exists():
        sys.exit(f"{path} not found: run stage0_audit.py with --out-root {out_dir} first.")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def split_seed(cfg, key, rep):
    """Seed of the outer split for (dataset, repetition); shared by freeze_design and the pilot."""
    return int(cfg["seed"]) + 1000 * rep + sum(map(ord, key))


def cohort_membership(cfg, key):
    """Verified cohort membership per original CSV row (GBSG), or None.

    Used only if recover_gbsg_cohort.py marked the match complete and verified."""
    ccfg = cfg["datasets"][key].get("cohort_membership")
    if not ccfg:
        return None
    path = output_dir(cfg) / ccfg
    report = path.with_name("gbsg_cohort_report.json")
    if not (path.exists() and report.exists()):
        return None
    if not json.loads(report.read_text()).get("verified_complete"):
        return None
    return pd.read_csv(path).set_index("csv_row")["h5_partition"]


def load_dataset(cfg, key, manifest):
    """Load one dataset exactly as audited; the CSV file itself is only read.

    Analysis rule (documented in DATA_PROVENANCE.md and the config): records with a
    non-positive or missing duration, or an event code outside {0, 1}, are excluded from the
    analysis set. A record censored at time 0 is never observed at risk, so excluding it leaves
    the Kaplan-Meier estimate and the partial likelihood unchanged.

    Returns X (DataFrame), time, event, strata, audit record, info dict."""
    dcfg = cfg["datasets"][key]
    reps = [r for r in manifest["datasets"] if r["dataset"] == key and r.get("n")]
    if dcfg.get("file", "auto") != "auto":
        reps = [r for r in reps if Path(r["file"]).name == dcfg["file"]]
    if len(reps) != 1:
        raise ValueError(f"{key}: expected exactly one audited file, found {len(reps)}; "
                         "set datasets.{key}.file in the config.")
    rep = reps[0]
    df, _, _ = load_table(Path(rep["file"]))
    tcol = rep["time_col"] if dcfg.get("time_col", "auto") == "auto" else dcfg["time_col"]
    ecol = rep["event_col"] if dcfg.get("event_col", "auto") == "auto" else dcfg["event_col"]
    time = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    event = pd.to_numeric(df[ecol], errors="coerce").to_numpy(float)
    keep = np.isfinite(time) & (time > 0) & np.isin(event, (0, 1))
    if cfg.get("analysis_rules", {}).get("nonpositive_duration", "exclude") != "exclude":
        raise ValueError("Only the 'exclude' rule for non-positive durations is implemented.")
    info = {"n_file": int(len(df)), "n_excluded": int((~keep).sum()),
            "excluded_csv_rows": np.flatnonzero(~keep).tolist(),
            "excluded_reason": "duration <= 0, missing, or event code not in {0,1}",
            "excluded_events": int(np.nansum(event[~keep] == 1)),
            "cohort_stratified": False}
    strata = event[keep].astype(int).astype(str)
    member = cohort_membership(cfg, key)
    if member is not None:
        rows = np.flatnonzero(keep)
        if not np.isin(rows, member.index).all():
            raise ValueError(f"{key}: cohort membership file does not cover all analysis rows.")
        strata = np.array([f"{g}|{s}" for g, s in zip(member.loc[rows].to_numpy(), strata)])
        info["cohort_stratified"] = True
    df = df.loc[keep].reset_index(drop=True)
    time, event = time[keep], event[keep].astype(int)
    if "_h5_group" in df.columns:          # stored DeepSurv split (GBSG cohorts)
        strata = np.array([f"{g}|{s}" for g, s in zip(df["_h5_group"].astype(str), strata)])
        info["cohort_stratified"] = True
    predictors = [c for c in df.columns if c not in (tcol, ecol, "_h5_group")]
    return df[predictors], time, event, strata, rep, info


def stratified_split(strata, test_fraction, rng):
    test = []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        test.extend(idx[: int(round(test_fraction * len(idx)))])
    test = np.sort(np.array(test, dtype=int))
    dev = np.setdiff1d(np.arange(len(strata)), test)
    return test, dev


def nested_budgets(dev_idx, event, budgets, rng):
    """Nested event-budget subsets that keep the pool's event fraction.

    Returns {label: indices} and {label: reason} for skipped budgets."""
    ev = dev_idx[event[dev_idx] == 1].copy()
    ce = dev_idx[event[dev_idx] == 0].copy()
    rng.shuffle(ev)
    rng.shuffle(ce)
    ratio = len(ce) / len(ev)
    out, skipped = {}, {}
    for b in budgets:
        if b == "full":
            out["full"] = np.sort(dev_idx)
            continue
        b = int(b)
        if b >= len(ev):
            skipped[str(b)] = f"budget {b} >= pool events {len(ev)}"
            continue
        out[str(b)] = np.sort(np.concatenate([ev[:b], ce[: int(round(b * ratio))]]))
    return out, skipped


def validation_split(train_idx, event, fraction, seed):
    """Training-derived validation partition (stratified by event), for neural stopping."""
    rng = np.random.default_rng(seed)
    val = []
    for v in (0, 1):
        idx = train_idx[event[train_idx] == v].copy()
        rng.shuffle(idx)
        val.extend(idx[: max(1, int(round(fraction * len(idx))))])
    val = np.sort(np.array(val, dtype=int))
    return np.setdiff1d(train_idx, val), val


class Preprocessor:
    """Median imputation, one-hot for declared categoricals, z-scoring; fit on training rows only."""

    def __init__(self, categorical=()):
        self.categorical = list(categorical)

    def fit(self, X):
        X = X.copy()
        self.numeric_ = [c for c in X.columns if c not in self.categorical]
        num = X[self.numeric_].apply(pd.to_numeric, errors="coerce")
        self.medians_ = num.median()
        num = num.fillna(self.medians_)
        self.levels_ = {c: sorted(X[c].dropna().unique().tolist()) for c in self.categorical}
        full = self._assemble(num, X)
        self.mean_, self.sd_ = full.mean(), full.std(ddof=0)
        self.kept_ = [c for c in full.columns if self.sd_[c] > 0]
        self.dropped_constant_ = [c for c in full.columns if self.sd_[c] <= 0]
        self.n_fit_ = len(X)
        return self

    def _assemble(self, num, X):
        parts = [num]
        for c in self.categorical:
            for lev in self.levels_[c][1:]:
                parts.append(pd.Series((X[c] == lev).astype(float), name=f"{c}=={lev}", index=X.index))
        return pd.concat(parts, axis=1)

    def transform(self, X):
        num = X[self.numeric_].apply(pd.to_numeric, errors="coerce").fillna(self.medians_)
        full = self._assemble(num, X)[self.kept_]
        return ((full - self.mean_[self.kept_]) / self.sd_[self.kept_]).to_numpy(np.float32)


# --------------------------------------------------------------------------
# Survival curves
# --------------------------------------------------------------------------

def eval_step_curves(times, surv, grid):
    """Evaluate right-continuous survival step curves surv (n x len(times)) on grid.

    S(t) = 1 before the first time point; the last value is carried forward."""
    times = np.asarray(times, float)
    idx = np.searchsorted(times, grid, side="right") - 1
    out = np.ones((surv.shape[0], len(grid)))
    ok = idx >= 0
    out[:, ok] = surv[:, idx[ok]]
    return out


def km_curve(time, event):
    order = np.argsort(time, kind="mergesort")
    t, e = time[order], event[order]
    uniq, start, counts = np.unique(t, return_index=True, return_counts=True)
    d = np.add.reduceat(e, start)
    at_risk = len(t) - np.concatenate([[0], np.cumsum(counts)[:-1]])
    return uniq, np.cumprod(1.0 - d / at_risk)


def km_at(time, event, grid):
    t, s = km_curve(time, event)
    return eval_step_curves(t, s[None, :], np.atleast_1d(grid))[0]


def truncate(time, event, t_admin):
    """Administrative censoring at t_admin. Leaves all estimands at t <= tau < t_admin unchanged."""
    return np.minimum(time, t_admin), (event.astype(bool) & (time <= t_admin)).astype(int)


def check_curves(S, n_expected, grid):
    inc = np.diff(S, axis=1)
    return {
        "shape_ok": S.shape == (n_expected, len(grid)),
        "finite": bool(np.isfinite(S).all()),
        "in_unit_interval": bool((S >= -EPS).all() and (S <= 1 + EPS).all()),
        "nonincreasing": bool(inc.size == 0 or inc.max() <= 1e-7),
        "max_increase": float(inc.max()) if inc.size else 0.0,
        "min_S": float(np.nanmin(S)), "max_S": float(np.nanmax(S)),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def surv_struct(time, event):
    from sksurv.util import Surv
    return Surv.from_arrays(event.astype(bool), time.astype(float))


def rcs3(x, knots):
    """Restricted cubic spline with 3 knots (Harrell): returns the single nonlinear term."""
    k1, k2, k3 = knots
    p = lambda u: np.maximum(u, 0) ** 3  # noqa: E731
    term = p(x - k1) - p(x - k2) * (k3 - k1) / (k3 - k2) + p(x - k3) * (k2 - k1) / (k3 - k2)
    return term / (k3 - k1) ** 2


def calibration_at(time, event, F_hat, t_star):
    """Calibration at t_star on evaluation data (Crowson et al. 2016; Austin et al. 2020).

    O/E uses the evaluation-set Kaplan-Meier estimate; slope is the Cox coefficient of
    cloglog(F_hat); ICI/E50/E90 use a 3-knot restricted cubic spline of cloglog(F_hat)."""
    from lifelines import CoxPHFitter
    # Calibration AT t*: follow-up is censored at t*, so no event information beyond the
    # horizon enters the calibration models (the KM estimate at t* is unaffected).
    time, event = truncate(time, event, t_star)
    F = np.clip(F_hat, 1e-6, 1 - 1e-6)
    lp = np.log(-np.log(1 - F))
    observed = 1 - km_at(time, event, t_star)[0]
    res = {"observed_risk": float(observed), "expected_risk": float(F.mean()),
           "O_E": float(observed / F.mean())}
    df = pd.DataFrame({"T": time, "E": event, "lp": lp})
    for penalizer in (0.0, 1e-4):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res["cal_slope"] = float(CoxPHFitter(penalizer=penalizer)
                                         .fit(df, "T", "E").params_["lp"])
                knots = np.quantile(lp, [0.1, 0.5, 0.9])
                if np.unique(knots).size < 3:
                    raise ValueError("degenerate spline knots (too few distinct predictions)")
                df["sp"] = rcs3(lp, knots)
                cph = CoxPHFitter(penalizer=penalizer).fit(df, "T", "E")
                obs = 1 - cph.predict_survival_function(df[["lp", "sp"]], times=[t_star]).to_numpy()[0]
                df = df.drop(columns="sp")
            diff = np.abs(obs - F)
            res.update({"ICI": float(diff.mean()), "E50": float(np.median(diff)),
                        "E90": float(np.quantile(diff, 0.9)), "cal_penalizer": penalizer})
            break
        except Exception as exc:  # noqa: BLE001
            res["cal_error"] = repr(exc)
            df = df.drop(columns="sp", errors="ignore")
    return res


def compute_metrics(pool_t, pool_e, test_t, test_e, S_grid, grid, S_star, t_star, tau,
                    km_ref_grid):
    """Censoring-aware metrics. Censoring distribution and KM reference come from the
    development pool only; evaluation outcomes come from the test set."""
    from sksurv.metrics import (brier_score, concordance_index_ipcw, cumulative_dynamic_auc,
                                integrated_brier_score)
    tr, te = surv_struct(pool_t, pool_e), surv_struct(test_t, test_e)
    n = len(test_t)
    risk = 1.0 - S_star
    m = {}
    # IBS over the fixed window [grid[0], tau] with sksurv's marginal censoring estimator.
    # The shared estimator equals sksurv's integrated_brier_score exactly (validate_metrics.py)
    # but, unlike sksurv, does not require grid[0] >= the smallest test time, so the
    # prespecified window is never altered by test outcomes.
    G_y, G_g = marginal_G(pool_t, pool_e, test_t, grid)
    m["IBS"] = ipcw_brier_ibs(test_t, test_e, S_grid, grid, G_y, G_g)[1]
    m["IBS_KM_pool"] = ipcw_brier_ibs(test_t, test_e, np.tile(km_ref_grid, (n, 1)), grid, G_y, G_g)[1]
    try:
        m["IBS_sksurv_absdiff"] = abs(float(integrated_brier_score(tr, te, S_grid, grid)) - m["IBS"])
    except ValueError:                                   # grid[0] below the smallest test time
        m["IBS_sksurv_absdiff"] = None
    m["IPA"] = 1.0 - m["IBS"] / m["IBS_KM_pool"]
    m["Brier_tstar"] = float(brier_score(tr, te, S_star[:, None], [t_star])[1][0])
    m["Uno_C_tau"] = float(concordance_index_ipcw(tr, te, risk, tau=tau)[0])
    m["tdAUC_tstar"] = float(cumulative_dynamic_auc(tr, te, risk, [t_star])[0][0])
    m.update(calibration_at(test_t, test_e, 1.0 - S_star, t_star))
    return m


# --------------------------------------------------------------------------
# IPCW Brier / IBS with an arbitrary censoring model (marginal or conditional)
# --------------------------------------------------------------------------

def ipcw_brier_ibs(test_t, test_e, S_grid, grid, G_at_y, G_at_grid, min_G=0.0):
    """IPCW Brier score on `grid` and its integral (same estimator and trapezoidal rule as
    sksurv). G_at_y: censoring survival at each patient's own time, shape (n,); G_at_grid:
    shape (n, len(grid)) for conditional models or (len(grid),) for marginal ones.
    Weights are stabilised by flooring G at `min_G`. Returns (brier(grid), IBS, share floored)."""
    G_y = np.maximum(np.asarray(G_at_y, float), min_G)
    G_g = np.asarray(G_at_grid, float)
    G_g = np.maximum(np.broadcast_to(G_g, (len(test_t), len(grid))), min_G)
    case = (test_t[:, None] <= grid[None, :]) & (test_e[:, None] == 1)
    ctrl = test_t[:, None] > grid[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(case, S_grid ** 2 / G_y[:, None], 0.0) + \
            np.where(ctrl, (1 - S_grid) ** 2 / G_g, 0.0)
    brier = b.mean(axis=0)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    ibs = float(trapz(brier, grid) / (grid[-1] - grid[0])) if len(grid) > 1 else float("nan")
    floored = float(np.mean(np.concatenate([np.asarray(G_at_y) < min_G,
                                            np.ravel(np.broadcast_to(G_at_grid, G_g.shape)) < min_G])))
    return brier, ibs, floored


def marginal_G(pool_t, pool_e, test_t, grid):
    """sksurv's marginal (Kaplan-Meier) censoring estimator, fitted on the development pool."""
    from sksurv.nonparametric import CensoringDistributionEstimator
    cens = CensoringDistributionEstimator().fit(surv_struct(pool_t, pool_e))
    return cens.predict_proba(test_t), cens.predict_proba(grid)


class CoxCensoringModel:
    """Conditional censoring distribution G(t | x) from a Cox model for the censoring hazard,
    fitted on the development pool only (sensitivity analysis to marginal IPCW)."""

    def __init__(self, penalizer=0.01):
        self.penalizer = penalizer

    def fit(self, Z, time, event):
        from lifelines import CoxPHFitter
        df = pd.DataFrame(Z, columns=[f"z{i}" for i in range(Z.shape[1])])
        df["T"], df["C"] = time, 1 - event
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cph_ = CoxPHFitter(penalizer=self.penalizer).fit(df, "T", "C")
        H = self.cph_.baseline_cumulative_hazard_
        self.h_times_, self.h0_ = H.index.to_numpy(float), H.iloc[:, 0].to_numpy(float)
        self.cols_ = list(df.columns[:-2])
        self.concordance_ = float(self.cph_.concordance_index_)
        return self

    def _H0(self, t):
        idx = np.searchsorted(self.h_times_, t, side="right") - 1
        return np.where(idx >= 0, self.h0_[np.clip(idx, 0, None)], 0.0)

    def predict(self, Z, time, grid):
        """Return G(y_i | x_i) and G(grid | x_i) (right-continuous, as in sksurv)."""
        df = pd.DataFrame(Z, columns=self.cols_)
        ph = self.cph_.predict_partial_hazard(df).to_numpy(float)
        G_y = np.exp(-self._H0(time) * ph)
        G_g = np.exp(-self._H0(grid)[None, :] * ph[:, None])
        return G_y, G_g


def calibration_curve(time, event, F_hat, t_star, points):
    """Smoothed observed risk at t_star versus predicted risk (3-knot spline of
    cloglog(F_hat) in a Cox model fitted on evaluation data; Austin et al. 2020).
    Returns observed risk evaluated at the predicted-risk values `points`."""
    from lifelines import CoxPHFitter
    time, event = truncate(time, event, t_star)      # calibration at t*: censor follow-up at t*
    F = np.clip(F_hat, 1e-6, 1 - 1e-6)
    lp = np.log(-np.log(1 - F))
    knots = np.quantile(lp, [0.1, 0.5, 0.9])
    if np.unique(knots).size < 3:
        return np.full(len(points), np.nan)
    df = pd.DataFrame({"T": time, "E": event, "lp": lp, "sp": rcs3(lp, knots)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter(penalizer=1e-4).fit(df, "T", "E")
    P = np.clip(np.asarray(points, float), 1e-6, 1 - 1e-6)
    lpp = np.log(-np.log(1 - P))
    new = pd.DataFrame({"lp": lpp, "sp": rcs3(lpp, knots)})
    return 1 - cph.predict_survival_function(new, times=[t_star]).to_numpy()[0]
