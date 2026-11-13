#!/usr/bin/env python3
"""Statistical analysis, tables and figures for the full study - from real task results only.

Reads outputs/full/results_tasks.csv (written by run_full_study.py --collect), the split
manifests and the saved predictions. Writes tables (CSV + Markdown) to outputs/full/tables,
figures (PDF + 300-dpi PNG) to outputs/full/figures and outputs/full/ANALYSIS_SUMMARY.md.

Inference strategy (FULL_STUDY_PROTOCOL.md, section 7):
- Every cell: effect estimate (mean over repetitions) and cross-split variability (SD,
  10th-90th percentiles, win rate vs ridge Cox). Descriptive; no p-values.
- Full budget only (random 70/30 splits = the Nadeau-Bengio setting): corrected resampled-t
  interval (factor 1/R + n_test/n_pool, df = R - 1), p-values and Holm adjustment across the
  four comparisons per cell. No formal inference is made at subsampled budgets.
Runs only on a complete, valid set of tasks; otherwise it writes a diagnostic report.

    python analyze_full_study.py --config full_study_config.yaml --n-reps 5
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import calibration_curve, km_curve, load_config, load_dataset, output_dir, read_manifest, truncate

MODEL_ORDER = ["cox_ridge", "rsf", "gbsa", "deepsurv", "logistic_hazard"]
LABEL = {"cox_ridge": "Ridge Cox", "rsf": "RSF", "gbsa": "GBSA", "deepsurv": "DeepSurv",
         "logistic_hazard": "Logistic-Hazard"}
COLOR = {"cox_ridge": "#000000", "rsf": "#E69F00", "gbsa": "#0072B2", "deepsurv": "#009E73",
         "logistic_hazard": "#CC79A7"}          # Okabe-Ito, colour-blind safe
MARKER = {"cox_ridge": "o", "rsf": "s", "gbsa": "D", "deepsurv": "^", "logistic_hazard": "v"}
DS_LABEL = {"support": "SUPPORT", "metabric": "METABRIC", "gbsg": "Rotterdam-GBSG"}
HIGHER_BETTER = {"IPA": True, "Uno_C_tau": True, "tdAUC_tstar": True, "Brier_tstar": False,
                 "IBS": False, "ICI": False, "abs_slope_dev": False, "stab_instability_index": False,
                 "stab_classification_instability": False, "IPA_cond": True}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def corrected(x, ratio, conf=0.95):
    """Mean and Nadeau-Bengio corrected t interval for repetition-level values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    R = len(x)
    if R == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, p=np.nan, R=0)
    m = x.mean()
    if R < 2:
        return dict(mean=m, lo=np.nan, hi=np.nan, p=np.nan, R=R)
    se = np.sqrt((1 / R + ratio) * x.var(ddof=1))
    q = stats.t.ppf(0.5 + conf / 2, R - 1)
    p = 2 * stats.t.sf(abs(m) / se, R - 1) if se > 0 else (0.0 if m != 0 else 1.0)
    return dict(mean=m, lo=m - q * se, hi=m + q * se, p=p, R=R)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    vals = p[ok][idx] * (ok.sum() - np.arange(ok.sum()))
    vals = np.minimum(1, np.maximum.accumulate(vals))
    tmp = np.empty_like(vals)
    tmp[idx] = vals
    out[ok] = tmp
    return out


def describe(x, q=(0.10, 0.90)):
    """Cross-split summary of repetition-level values (descriptive)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(mean=np.nan, sd=np.nan, lo=np.nan, hi=np.nan, R=0)
    return dict(mean=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                lo=np.quantile(x, q[0]), hi=np.quantile(x, q[1]), R=len(x))


def classify_descriptive(d, delta, higher=True):
    """Practical classification from the cross-split distribution (no inference)."""
    if d["R"] < 3:
        return "insufficient repetitions"
    m, lo, hi = (d["mean"], d["lo"], d["hi"]) if higher else (-d["mean"], -d["hi"], -d["lo"])
    if m >= delta and lo > 0:
        return "consistent improvement"
    if m <= -delta and hi < 0:
        return "consistent deficit"
    if lo > -delta and hi < delta:
        return "no practically relevant difference"
    return "variable across splits"


def classify(r, delta, higher=True):
    """Threshold interpretation of an *unadjusted* corrected 95% interval.

    An interval excluding zero does not establish an effect greater than delta.
    This label is descriptive of the interval; Holm significance is reported
    separately and must not be inferred from this function.
    """
    if delta is None:
        return "threshold not specified"
    lo, hi = (r["lo"], r["hi"]) if higher else (-r["hi"], -r["lo"])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "insufficient repetitions"
    if lo > delta:
        return "interval entirely above +threshold (unadjusted)"
    if hi < -delta:
        return "interval entirely below -threshold (unadjusted)"
    if lo > -delta and hi < delta:
        return "interval contained within practical band (unadjusted)"
    if lo > 0:
        return "positive; practical threshold not established"
    if hi < 0:
        return "negative; practical threshold not established"
    return "threshold not established"


def holm_status(p_holm, mean, higher=True, alpha=0.05):
    """Directional zero-effect evidence using the Holm-adjusted p-value."""
    if not np.isfinite(p_holm):
        return "not tested"
    if p_holm >= alpha:
        return "not significant after Holm"
    signed_mean = mean if higher else -mean
    if signed_mean > 0:
        return "positive after Holm"
    if signed_mean < 0:
        return "negative after Holm"
    return "significant but direction undefined"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def budget_order(labels):
    return sorted(set(labels), key=lambda b: (b == "full", int(b) if b != "full" else 0))


def md_table(df, floatfmt=".4f"):
    if df.empty:
        return "_none_"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(format(v, floatfmt) if isinstance(v, float) else str(v)
                                     for v in r) + " |")
    return "\n".join(out)


def save_table(df, name, tdir):
    df.to_csv(tdir / f"{name}.csv", index=False)
    (tdir / f"{name}.md").write_text(md_table(df))


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 100,
                         "savefig.bbox": "tight", "lines.linewidth": 1.2, "lines.markersize": 3.5})
    return plt


def savefig(fig, fdir, name):
    fig.savefig(fdir / f"{name}.pdf")
    fig.savefig(fdir / f"{name}.png", dpi=300)


def model_legend(fig, plt, models):
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARKER[m], label=LABEL[m]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, -0.02))


def xpos(res, key):
    """Mean training events per budget label (x-axis on a log2 scale)."""
    d = res[res.dataset == key].groupby("budget")["events_train"].mean()
    return {b: float(d[b]) for b in d.index}


def set_budget_axis(ax, xp):
    ax.set_xscale("log", base=2)
    labs = budget_order(xp)
    ax.set_xticks([xp[b] for b in labs])
    ax.set_xticklabels([f"full\n({xp[b]:.0f})" if b == "full" else b for b in labs])
    ax.minorticks_off()
    ax.set_xlabel("training events")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full_study_config.yaml")
    ap.add_argument("--n-reps", type=int, required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write a clearly marked PARTIAL analysis to outputs/full/partial_analysis")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fs = cfg["full_study"]
    out = output_dir(cfg)
    root = out / fs.get("output_subdir", "full")
    source_root = root  # source records always live here, including for partial analyses
    res_path = source_root / "results_tasks.csv"
    summ_path = source_root / f"collect_summary_{args.n_reps}reps.json"
    if not (res_path.exists() and summ_path.exists()):
        raise SystemExit(f"Run: python run_full_study.py --config {args.config} --n-reps {args.n_reps} --collect")
    summ = json.loads(summ_path.read_text())
    try:
        res = pd.read_csv(res_path, dtype={"budget": str})
    except pd.errors.EmptyDataError:            # --collect found no valid task yet
        raise SystemExit(f"No valid tasks in {res_path}: run the run_full_study.py workers, "
                         "then --collect, before analysing.")
    res = res[res.rep < args.n_reps].copy()
    # Reporting-only correction for constant-risk calibration predictions.
    # Raw task records and saved predictions are unchanged.
    nonestimable = []
    candidate = res[res["cal_slope"].notna() &
                    res["calibration_curve_estimable"].astype(str).str.lower().eq("false")]

    for idx, row in candidate.iterrows():
        pred_file = (source_root / "tasks" / str(row["dataset"]) /
                     f"rep{int(row['rep']):02d}" /
                     f"{row['budget']}__{row['model']}.npz")
        with np.load(pred_file) as saved:
            horizons = saved["horizons"]
            risk = saved["F_horizons"][:, np.argmin(np.abs(horizons - row["t_star"]))]

        if risk.size and np.isfinite(risk).all() and np.unique(risk).size == 1:
            nonestimable.append(dict(
                dataset=row["dataset"], rep=int(row["rep"]),
                budget=row["budget"], model=row["model"],
                original_slope=float(row["cal_slope"]),
                correction="not estimable: constant predicted risk"))
            res.at[idx, "cal_slope"] = np.nan

    audit_columns = ["dataset", "rep", "budget", "model",
                     "original_slope", "correction"]
    pd.DataFrame(nonestimable, columns=audit_columns).to_csv(
        source_root / "tables" / "T0b_constant_risk_calibration_audit.csv", index=False)

    print(f"Calibration reporting correction: {len(nonestimable)} slopes set to NA")
    design0 = json.loads((out / "frozen_design.json").read_text())
    cells = [(k, b, m) for k in cfg["datasets"] for b in design0["datasets"][k]["budget_ladder_feasible"]
             for m in MODEL_ORDER]
    have = res.groupby(["dataset", "budget", "model"]).rep.nunique().to_dict() if len(res) else {}
    short = [(k, b, m, have.get((k, b, m), 0)) for k, b, m in cells if have.get((k, b, m), 0) != args.n_reps]
    complete = bool(summ.get("complete")) and not short
    if not complete:
        inc = root / "incomplete_tasks.csv"
        incdf = pd.read_csv(inc) if inc.exists() and inc.stat().st_size > 1 else pd.DataFrame()
        D = ["# INCOMPLETE RUN - no findings produced", "",
             f"Expected {summ.get('tasks_expected')} tasks for {args.n_reps} repetitions; valid: "
             f"{summ.get('tasks_ok')}; not ok by status: {summ.get('not_ok_by_status')}.",
             f"Dataset x budget x model cells without {args.n_reps} valid repetitions: {len(short)}.", "",
             "Re-run the `run_full_study.py --worker` processes (failed/invalid tasks are retried), then `--collect` and re-analyse.", "",
             "## Cells with missing repetitions", "",
             md_table(pd.DataFrame(short, columns=["dataset", "budget", "model", "valid_reps"]).head(100)), "",
             "## First incomplete tasks", "", md_table(incdf.head(50)) if len(incdf) else "_none listed_"]
        (root / "INCOMPLETE_RUN_DIAGNOSTIC.md").write_text("\n".join(D))
        print("\n".join(D[:4]))
        if not args.allow_incomplete:
            raise SystemExit(2)
        root = root / "partial_analysis"
        root.mkdir(exist_ok=True)
    tdir, fdir = root / "tables", root / "figures"
    tdir.mkdir(exist_ok=True)
    fdir.mkdir(exist_ok=True)
    if res.empty:
        raise SystemExit("No valid tasks - nothing to analyse.")
    qq = tuple(fs["analysis"].get("descriptive_quantiles", [0.10, 0.90]))
    formal_budget = str(fs["analysis"].get("formal_inference_budget", "full"))
    res["abs_slope_dev"] = (res["cal_slope"] - 1).abs()
    th = cfg["practical_thresholds"]
    conf = float(fs["analysis"].get("confidence", 0.95))
    ref = fs["analysis"].get("reference_model", "cox_ridge")
    datasets = [d for d in cfg["datasets"] if d in set(res.dataset)]
    models = [m for m in MODEL_ORDER if m in set(res.model)]
    design = json.loads((out / "frozen_design.json").read_text())

    # n_test / n_pool ratio per dataset (from the split manifests)
    ratio = {}
    for key in datasets:
        sp = [json.loads(p.read_text()) for p in sorted((root / "splits").glob(f"{key}_rep*.json"))]
        ratio[key] = float(np.mean([s["n_test"] / s["n_pool"] for s in sp]))

    # ---- Table 1: completeness ---------------------------------------------------------
    comp = res.groupby(["dataset", "model"]).agg(
        valid_tasks=("rep", "size"), reps=("rep", "nunique"),
        calibration_curve_not_estimable=("calibration_curve_estimable",
                                         lambda v: int((~v.astype(bool)).sum())),
        calibration_slope_not_estimable=("cal_slope", lambda v: int(v.isna().sum())))
    save_table(comp.reset_index(), "T0_completeness", tdir)

    # ---- Table: per-model learning-curve summaries -----------------------------------------
    metrics = ["IPA", "IBS", "Uno_C_tau", "tdAUC_tstar", "Brier_tstar", "cal_slope", "ICI", "O_E"]
    rows = []
    for (key, b, m), g in res.groupby(["dataset", "budget", "model"]):
        r = {"dataset": key, "budget": b, "model": m, "R": g.rep.nunique(),
             "events_train": g.events_train.mean(), "n_train": g.n_train.mean(),
             "n_fit_mean": g.n_fit.mean(), "n_val_mean": g.n_val.mean()}
        for mt in metrics:                       # descriptive: mean, SD and P10-P90 across splits
            c = describe(g[mt], qq)
            r.update({f"{mt}_mean": c["mean"], f"{mt}_sd": c["sd"], f"{mt}_lo": c["lo"], f"{mt}_hi": c["hi"]})
        rows.append(r)
    lc = pd.DataFrame(rows)
    lc["bo"] = lc.budget.map(lambda b: budget_order(lc.budget).index(b))
    lc = lc.sort_values(["dataset", "bo", "model"]).drop(columns="bo")
    save_table(lc, "T1_model_performance_by_budget", tdir)

    # ---- Paired differences vs ridge Cox -----------------------------------------------
    pair_metrics = {"IPA": th["delta_IPA"], "Uno_C_tau": th["delta_C"], "tdAUC_tstar": th["delta_C"],
                    "Brier_tstar": None, "ICI": th["delta_ICI"], "abs_slope_dev": None}
    prow = []
    for (key, b), g in res.groupby(["dataset", "budget"]):
        for mt, delta in pair_metrics.items():
            wide = g.pivot_table(index="rep", columns="model", values=mt)
            if ref not in wide:
                continue
            cell = []
            for m in [x for x in models if x != ref and x in wide]:
                d = (wide[m] - wide[ref]).dropna()
                hb = HIGHER_BETTER.get(mt, True)
                ds_ = describe(d, qq)
                win = float(np.mean(d > 0) if hb else np.mean(d < 0)) if len(d) else np.nan
                row = {"dataset": key, "budget": b, "metric": mt, "model": m, "R": ds_["R"],
                       "diff_mean": ds_["mean"], "diff_sd": ds_["sd"], "p10": ds_["lo"], "p90": ds_["hi"],
                       "win_rate_vs_cox": win, "threshold": delta,
                       "inference": "formal" if b == formal_budget else "descriptive",
                       "nb_lo": np.nan, "nb_hi": np.nan, "p": np.nan}
                if b == formal_budget:
                    c = corrected(d, ratio[key], conf)
                    row.update({"nb_lo": c["lo"], "nb_hi": c["hi"], "p": c["p"],
                                "classification": classify(c, delta, hb) if delta is not None else ""})
                else:
                    row["classification"] = classify_descriptive(ds_, delta, hb) if delta is not None else ""
                cell.append(row)
            ph = holm([x["p"] for x in cell])     # NaN (no test) at subsampled budgets
            for x, p in zip(cell, ph):
                x["p_holm"] = p
                x["holm_status"] = (holm_status(p, x["diff_mean"],
                                               HIGHER_BETTER.get(mt, True))
                                    if b == formal_budget else "not tested (descriptive)")
            prow += cell
    pairs = pd.DataFrame(prow)
    save_table(pairs, "T2_paired_differences_vs_ridge_cox", tdir)

    # ---- Stability ---------------------------------------------------------------------
    st = res.dropna(subset=["stab_instability_index"]) if "stab_instability_index" in res else pd.DataFrame()
    srow = []
    for (key, b, m), g in (st.groupby(["dataset", "budget", "model"]) if len(st) else []):
        r = {"dataset": key, "budget": b, "model": m, "R": len(g)}
        for mt in ("stab_instability_index", "stab_mean_patient_sd", "stab_p90_patient_mad",
                   "stab_classification_instability"):
            c = describe(g[mt], qq)
            r.update({f"{mt}_mean": c["mean"], f"{mt}_sd": c["sd"], f"{mt}_lo": c["lo"], f"{mt}_hi": c["hi"]})
        srow.append(r)
    stab = pd.DataFrame(srow)
    save_table(stab, "T3_prediction_stability", tdir)

    # ---- Computational efficiency ------------------------------------------------------
    res["train_wall_s"] = res["tuning_wall_s"] + res["final_fit_wall_s"]
    eff = res.groupby(["dataset", "budget", "model"]).agg(
        R=("rep", "nunique"), median_tuning_plus_final_s=("train_wall_s", "median"),
        median_final_fit_s=("final_fit_wall_s", "median"), median_predict_s=("predict_wall_s", "median"),
        median_stability_s=("stability_wall_s", "median"), max_gpu_peak_mem_mb=("gpu_peak_mem_mb", "max"),
        device=("device", "first")).reset_index()
    save_table(eff, "T4_computational_cost", tdir)
    if "stab_bootstrap_ok" in res:
        bsd = res.dropna(subset=["stab_bootstrap_ok"]).copy()
        bsd["refits_failed"] = bsd["stab_bootstrap_requested"] - bsd["stab_bootstrap_ok"]
        bs = bsd.groupby(["dataset", "budget", "model"]).agg(
            stability_tasks=("stab_bootstrap_ok", "size"),
            refits_requested=("stab_bootstrap_requested", "sum"),
            refits_failed=("refits_failed", "sum"),
            tasks_with_failed_refits=("refits_failed", lambda v: int((v > 0).sum())),
            tasks_without_any_valid_refit=("stab_bootstrap_ok", lambda v: int((v == 0).sum()))).reset_index()
        save_table(bs, "T3b_stability_refit_failures", tdir)
    tot = res.groupby("model").agg(total_task_hours=("task_wall_s", lambda s: s.sum() / 3600)).reset_index()
    save_table(tot, "T4b_total_compute_by_model", tdir)

    # ---- Tuning selections -------------------------------------------------------------
    tune = pd.read_csv(source_root / "tuning_records.csv", dtype={"budget": str})
    sel = tune[tune.selected].groupby(["dataset", "budget", "model", "config_index"]).size() \
        .rename("times_selected").reset_index()
    save_table(sel, "T5_tuning_selection_frequency", tdir)
    tf = tune.groupby(["model"]).agg(configs_failed=("status", lambda s: int((s != "ok").sum())),
                                     configs_total=("status", "size")).reset_index()
    save_table(tf, "T5b_tuning_failures", tdir)

    # ---- Conditional-IPCW sensitivity --------------------------------------------------
    sens = pd.DataFrame()
    if "IPA_cond" in res:
        cs = res.dropna(subset=["IPA_cond"])
        srows = []
        for (key, b), g in cs.groupby(["dataset", "budget"]):
            for mt, cond in (("IPA", "IPA_cond"), ("Brier_tstar", "Brier_tstar_cond")):
                wm = g.pivot_table(index="rep", columns="model", values=mt)
                wc = g.pivot_table(index="rep", columns="model", values=cond)
                for m in [x for x in models if x != ref and x in wm]:
                    hb = HIGHER_BETTER[mt]
                    delta = th["delta_IPA"] if mt == "IPA" else None
                    out_r = {"dataset": key, "budget": b, "metric": mt, "model": m,
                             "inference": "formal" if b == formal_budget else "descriptive"}
                    for tag, w in (("marginal", wm), ("conditional", wc)):
                        dd = (w[m] - w[ref]).dropna()
                        if b == formal_budget:
                            c = corrected(dd, ratio[key], conf)
                            cls = classify(c, delta, hb) if delta else ""
                            rng_ = f"NB [{c['lo']:.4f}, {c['hi']:.4f}]"
                        else:
                            c = describe(dd, qq)
                            cls = classify_descriptive(c, delta, hb) if delta else ""
                            rng_ = f"P10-P90 [{c['lo']:.4f}, {c['hi']:.4f}]"
                        out_r.update({f"diff_{tag}": c["mean"], f"range_{tag}": rng_, f"class_{tag}": cls})
                    srows.append(out_r)
        sens = pd.DataFrame(srows)
        if len(sens):
            sens["classification_agrees"] = sens.class_marginal == sens.class_conditional
        integ = float((cs["IBS_marg_check"] - cs["IBS"]).abs().max())
        cens_c = cs.groupby("dataset")["censoring_model_C"].mean().to_dict()
        save_table(sens, "T6_conditional_ipcw_sensitivity", tdir)
    else:
        integ, cens_c = None, {}

    # ---- Figures -------------------------------------------------------------------------
    plt = setup_mpl()
    ncol = len(datasets)
    W = 7.0

    # Fig 1: follow-up and horizons (all analysis records; descriptive)
    manifest = read_manifest(out)
    fig, axes = plt.subplots(1, ncol, figsize=(8.6, 2.7), squeeze=False)
    for ax, key in zip(axes[0], datasets):
        _, t, e, _, _, _ = load_dataset(cfg, key, manifest)
        tk, sk = km_curve(t, e)
        tg, sg = km_curve(t, 1 - e)
        ax.step(np.r_[0, tk], np.r_[1, sk], where="post", color="k", label="event-free survival (KM)")
        ax.step(np.r_[0, tg], np.r_[1, sg], where="post", color="0.55", ls="--", label="censoring survival (reverse KM)")
        d = design["datasets"][key]
        for h in d["horizons"]:
            ax.axvline(h, color="#D55E00", lw=1.4 if h == d["primary_horizon"] else 0.7,
                       ls="-" if h == d["primary_horizon"] else ":")
        ax.set_title(f"{DS_LABEL.get(key, key)}\n(n={len(t):,}, events={int(e.sum()):,})",
                     fontsize=8, pad=5)
        ax.set_xlabel(f"time ({d.get('units', '')})")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("probability")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.00), fontsize=7)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    savefig(fig, fdir, "Fig1_followup_and_horizons")
    plt.close(fig)

    def curve_panel(ax, key, mt, label):
        xp = xpos(res, key)
        sub = lc[lc.dataset == key]
        for m in models:
            s = sub[sub.model == m]
            s = s.assign(x=s.budget.map(xp)).sort_values("x")
            ax.plot(s.x, s[f"{mt}_mean"], color=COLOR[m], marker=MARKER[m])
            ax.fill_between(s.x, s[f"{mt}_lo"], s[f"{mt}_hi"], color=COLOR[m], alpha=0.12, lw=0)
        set_budget_axis(ax, xp)
        ax.set_ylabel(label)

    # Fig 2: learning curves
    fig, axes = plt.subplots(2, ncol, figsize=(W, 4.4), squeeze=False)
    for j, key in enumerate(datasets):
        curve_panel(axes[0][j], key, "IPA", "IPA (1 - IBS/IBS$_{KM}$)")
        axes[0][j].axhline(0, color="0.6", lw=0.6)
        axes[0][j].set_title(DS_LABEL.get(key, key))
        curve_panel(axes[1][j], key, "Uno_C_tau", r"Uno's C ($\tau$)")
    model_legend(fig, plt, models)
    fig.text(0.5, 0.035, "X-axis: available training events; neural fits reserve a validation partition for early stopping",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    savefig(fig, fdir, "Fig2_learning_curves")
    plt.close(fig)

    # Fig 3: paired differences vs ridge Cox
    fig, axes = plt.subplots(2, ncol, figsize=(W, 4.4), squeeze=False)
    for i, (mt, delta, lab) in enumerate((("IPA", th["delta_IPA"], r"$\Delta$IPA vs ridge Cox"),
                                          ("Uno_C_tau", th["delta_C"], r"$\Delta$C vs ridge Cox"))):
        for j, key in enumerate(datasets):
            ax = axes[i][j]
            xp = xpos(res, key)
            sub = pairs[(pairs.dataset == key) & (pairs.metric == mt)] if len(pairs) else pairs
            others = [m for m in models if m != ref]
            for k, m in enumerate(others):
                s = sub[sub.model == m]
                xs = np.array([xp[b] for b in s.budget]) * 2 ** ((k - (len(others) - 1) / 2) * 0.06)
                ax.errorbar(xs, s.diff_mean, yerr=[s.diff_mean - s.p10, s.p90 - s.diff_mean],
                            fmt=MARKER[m], color=COLOR[m], ms=3.5, lw=0.9, capsize=1.5)
            ax.axhspan(-delta, delta, color="0.9", lw=0)
            ax.axhline(0, color="0.4", lw=0.6)
            set_budget_axis(ax, xp)
            ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(DS_LABEL.get(key, key))
    model_legend(fig, plt, [m for m in models if m != ref])
    fig.text(0.5, 0.035, "Error bars: descriptive P10–P90 across splits, including full budget; not 95% CIs",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    savefig(fig, fdir, "Fig3_paired_differences")
    plt.close(fig)

    # Fig 4: calibration summaries
    fig, axes = plt.subplots(2, ncol, figsize=(W, 4.4), squeeze=False)
    for j, key in enumerate(datasets):
        curve_panel(axes[0][j], key, "cal_slope", "calibration slope")
        axes[0][j].axhspan(*th["slope_band"], color="0.9", lw=0)
        axes[0][j].axhline(1, color="0.4", lw=0.6)
        axes[0][j].set_title(DS_LABEL.get(key, key))
        curve_panel(axes[1][j], key, "ICI", "ICI at t*")
    model_legend(fig, plt, models)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, fdir, "Fig4_calibration_summary")
    plt.close(fig)

    # Fig 4b: smoothed calibration curves at t* (mean over repetitions)
    cb = [str(b) for b in fs["analysis"].get("calibration_curve_budgets", [50, "full"])]
    pts = np.linspace(0.02, 0.98, 49)
    fig, axes = plt.subplots(len(cb), ncol, figsize=(W, 2.4 * len(cb)), squeeze=False)
    curve_support = []
    for j, key in enumerate(datasets):
        _, t, e, _, _, _ = load_dataset(cfg, key, manifest)
        for i, b in enumerate(cb):
            ax = axes[i][j]
            ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls="--")
            for m in models:
                curves = []
                for p in sorted((source_root / "tasks" / key).glob(f"rep*/{b}__{m}.npz")):
                    z = np.load(p)
                    sp = json.loads((source_root / "splits" / f"{key}_{p.parent.name}.json").read_text())
                    hz = list(z["horizons"])
                    F = z["F_horizons"][:, hz.index(sp["t_star"])].astype(float)
                    tt_, te_ = truncate(t[z["test_index"]], e[z["test_index"]], sp["t_admin"])
                    try:
                        c = calibration_curve(tt_, te_, F, sp["t_star"], pts)
                        lo, hi = np.quantile(F, [0.05, 0.95])
                        c[(pts < lo) | (pts > hi)] = np.nan
                        curves.append(c)
                    except Exception:  # noqa: BLE001
                        continue
                # A risk-grid point without contributing splits is not estimable.
                # Require at least 25% of the available curves (minimum three)
                # for a displayed segment; do not extrapolate into unsupported tails.
                required = max(3, int(np.ceil(0.25 * len(curves))))
                n_supported = 0
                lo_count = 0
                hi_count = 0
                if curves:
                    arr = np.asarray(curves, dtype=float)
                    counts = np.isfinite(arr).sum(axis=0)
                    numer = np.nansum(arr, axis=0)
                    means = np.divide(numer, counts,
                                      out=np.full(pts.shape, np.nan), where=counts > 0)
                    display = counts >= required
                    means[~display] = np.nan
                    n_supported = int(display.sum())
                    if n_supported:
                        lo_count = int(counts[display].min())
                        hi_count = int(counts[display].max())
                        ax.plot(pts, means, color=COLOR[m])
                curve_support.append(dict(dataset=key, budget=b, model=m,
                                          curves_available=len(curves),
                                          min_required_repetitions=required,
                                          grid_points_displayed=n_supported,
                                          min_contributing_repetitions=lo_count,
                                          max_contributing_repetitions=hi_count))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{DS_LABEL.get(key, key)}, budget {b}")
            ax.set_xlabel("predicted risk at t*")
            ax.set_ylabel("observed risk at t*")
    model_legend(fig, plt, models)
    fig.text(0.5, 0.035, "Curves shown only where ≥25% of available splits (minimum 3) contribute; no extrapolation",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_table(pd.DataFrame(curve_support), "T7_calibration_curve_support", tdir)
    savefig(fig, fdir, "Fig4b_calibration_curves")
    plt.close(fig)

    # Fig 5: prediction stability
    if len(stab):
        fig, axes = plt.subplots(2, ncol, figsize=(W, 4.4), squeeze=False)
        for j, key in enumerate(datasets):
            xp = xpos(res, key)
            for i, (mt, lab) in enumerate((("stab_instability_index", "instability index (mean |ΔF(t*)|)"),
                                           ("stab_classification_instability", "classification instability"))):
                ax = axes[i][j]
                sub = stab[stab.dataset == key]
                for m in models:
                    s = sub[sub.model == m].assign(x=lambda d: d.budget.map(xp)).sort_values("x")
                    ax.plot(s.x, s[f"{mt}_mean"], color=COLOR[m], marker=MARKER[m])
                    ax.fill_between(s.x, s[f"{mt}_lo"], s[f"{mt}_hi"], color=COLOR[m], alpha=0.12, lw=0)
                set_budget_axis(ax, xp)
                ax.set_ylabel(lab)
                if i == 0:
                    ax.set_title(DS_LABEL.get(key, key))
        model_legend(fig, plt, models)
        fig.text(0.5, 0.035, "Stability estimated from first five outer splits; 10 bootstrap refits per split",
                 ha="center", fontsize=6.5)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        savefig(fig, fdir, "Fig5_prediction_stability")
        plt.close(fig)

        # Fig 5b: prediction instability plots (repetition 0, smallest budget >= 100)
        fig, axes = plt.subplots(ncol, len(models), figsize=(W, 1.5 * ncol + 0.4), squeeze=False)
        for i, key in enumerate(datasets):
            labs = budget_order(res[res.dataset == key].budget)
            b = next((x for x in labs if x != "full" and int(x) >= 100), labs[0])
            for j, m in enumerate(models):
                ax = axes[i][j]
                p = source_root / "tasks" / key / "rep00" / f"{b}__{m}.npz"
                if p.exists():
                    z = np.load(p)
                    if "F_tstar_bootstrap" in z:
                        F0, Fb = z["F_tstar_original"], z["F_tstar_bootstrap"]
                        sel_i = np.random.default_rng(0).choice(len(F0), min(300, len(F0)), replace=False)
                        for row in Fb:
                            ax.scatter(F0[sel_i], row[sel_i], s=1, color=COLOR[m], alpha=0.25, lw=0)
                ax.plot([0, 1], [0, 1], color="0.5", lw=0.6)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                if i == 0:
                    ax.set_title(LABEL[m])
                if j == 0:
                    ax.set_ylabel(f"{DS_LABEL.get(key, key)}\nbudget {b}\nbootstrap F(t*)")
                if i == ncol - 1:
                    ax.set_xlabel("original F(t*)")
        fig.tight_layout()
        savefig(fig, fdir, "Fig5b_prediction_instability_plots")
        plt.close(fig)

    # Fig 6: computational cost and cost-benefit at the full budget
    fig, axes = plt.subplots(2, ncol, figsize=(W, 4.4), squeeze=False)
    for j, key in enumerate(datasets):
        xp = xpos(res, key)
        ax = axes[0][j]
        sub = eff[eff.dataset == key]
        for m in models:
            s = sub[sub.model == m].assign(x=lambda d: d.budget.map(xp)).sort_values("x")
            ax.plot(s.x, s.median_tuning_plus_final_s, color=COLOR[m], marker=MARKER[m])
        ax.set_yscale("log")
        set_budget_axis(ax, xp)
        ax.set_ylabel("tuning + final fit (s, median)")
        ax.set_title(DS_LABEL.get(key, key))
        ax = axes[1][j]
        full = pairs[(pairs.dataset == key) & (pairs.metric == "IPA") & (pairs.budget == "full")] if len(pairs) else pairs
        for m in models:
            tsec = sub[(sub.model == m) & (sub.budget == "full")].median_tuning_plus_final_s
            if tsec.empty:
                continue
            tsec = float(tsec.iloc[0])
            if m == ref:
                ax.scatter(tsec, [0], color=COLOR[m], marker=MARKER[m])
                continue
            r = full[full.model == m]
            if len(r):
                ax.errorbar([tsec], r.diff_mean.to_numpy(), yerr=[(r.diff_mean - r.nb_lo).to_numpy(),
                            (r.nb_hi - r.diff_mean).to_numpy()], fmt=MARKER[m], color=COLOR[m], capsize=1.5)
        ax.axhspan(-th["delta_IPA"], th["delta_IPA"], color="0.9", lw=0)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("tuning + final fit (s), full budget")
        ax.set_ylabel(r"$\Delta$IPA vs ridge Cox")
    model_legend(fig, plt, models)
    fig.text(0.5, 0.035, "CPU/GPU wall times are not hardware-normalized; error bars: unadjusted 95% corrected CIs",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    savefig(fig, fdir, "Fig6_computational_efficiency")
    plt.close(fig)

    # Fig S1: conditional vs marginal IPCW
    if len(sens):
        cs = res.dropna(subset=["IPA_cond"])
        keys = sorted(set(cs.dataset), key=datasets.index)
        fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 2.4), squeeze=False)
        for ax, key in zip(axes[0], keys):
            s = cs[cs.dataset == key]
            for m in models:
                q = s[s.model == m]
                ax.scatter(q.IPA, q.IPA_cond, s=5, color=COLOR[m], marker=MARKER[m], alpha=0.6, lw=0)
            lim = [min(s.IPA.min(), s.IPA_cond.min()), max(s.IPA.max(), s.IPA_cond.max())]
            ax.plot(lim, lim, color="0.5", lw=0.6)
            ax.set_xlabel("IPA, marginal IPCW")
            ax.set_ylabel("IPA, conditional IPCW")
            ax.set_title(DS_LABEL.get(key, key))
        model_legend(fig, plt, models)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        savefig(fig, fdir, "FigS1_conditional_ipcw_sensitivity")
        plt.close(fig)

    # ---- Summary file (facts only; interpretation is left to the authors) ------------------
    n_formal = int((pairs.inference == "formal").sum()) if len(pairs) else 0
    L = (["# PARTIAL ANALYSIS - INCOMPLETE RUN, NOT FINDINGS", "",
          "See ../INCOMPLETE_RUN_DIAGNOSTIC.md.", ""] if not complete else []) + [
         "# Full-study analysis summary", "",
         f"Complete run: {complete}. Valid repetitions per dataset: "
         f"{res.groupby('dataset').rep.nunique().to_dict()}; valid tasks analysed: {len(res)}.",
         f"Subsampled budgets: descriptive only - mean paired difference, SD, P{int(qq[0]*100)}-P{int(qq[1]*100)} "
         "across splits and win rate; no p-values.",
         f"Full budget: UNADJUSTED corrected resampled-t {int(conf * 100)}% intervals "
         f"(Nadeau-Bengio; factor 1/R + n_test/n_pool = "
         f"{ {k: round(v, 3) for k, v in ratio.items()} }); Holm-adjusted P VALUES within "
         f"each dataset x metric cell (4 comparisons); formal comparisons reported: {n_formal}.",
         "Interval-based practical-threshold statements are UNADJUSTED; a positive effect "
         "with an interval crossing +delta does not establish superiority beyond delta.",
         "Statistical evidence after Holm is reported separately in holm_status. "
         "Descriptive P10-P90 ranges are not confidence intervals.",
         "NEURAL SAMPLE-SIZE NOTE: DeepSurv and Logistic-Hazard retain early-stopped "
         "fits trained on the fitting partition; classical models are refitted on the "
         "entire training subset. Event budget is therefore available events, not "
         "identical effective fitting-event count. The comparisons assess the "
         "specified training pipelines, not model architecture under equalized "
         "effective sample sizes.",
         "STABILITY NOTE: Bootstrap refit stability is evaluated on the first five "
         "outer splits only (10 bootstrap refits per split), not all 20.",
         "CALIBRATION NOTE: Slopes are not estimable for constant predictions; retain "
         "other valid metrics and explicitly report non-estimability.",
         f"Practical thresholds (fixed in advance): {th}.", "",
         "## Paired differences vs ridge Cox (IPA and Uno's C)", "",
         md_table(pairs[pairs.metric.isin(["IPA", "Uno_C_tau"])][
             ["dataset", "budget", "metric", "model", "R", "diff_mean", "p10", "p90", "win_rate_vs_cox",
              "inference", "nb_lo", "nb_hi", "p_holm", "holm_status",
              "classification"]]) if len(pairs) else "_none_", ""]
    if len(sens):
        L += ["## Conditional-IPCW sensitivity", "",
              f"Censoring-model C (mean): {cens_c}; integrity check max |IBS_marg_check - IBS| = {integ:.2e}.",
              f"Classification agreement (marginal vs conditional): "
              f"{sens.classification_agrees[sens.metric == 'IPA'].mean():.1%} of IPA comparisons.", ""]
    L += ["## Files", "", "Tables: " + ", ".join(sorted(p.name for p in tdir.glob("*.csv"))),
          "", "Figures: " + ", ".join(sorted(p.name for p in fdir.glob("*.pdf")))]
    (root / "ANALYSIS_SUMMARY.md").write_text("\n".join(L))
    print("\n".join(L[:6]))
    print(f"Tables: {tdir}\nFigures: {fdir}")


if __name__ == "__main__":
    main()
