#!/usr/bin/env python3
"""Validate the censoring-aware metric pipeline on synthetic data with known truth.

Exponential proportional-hazards event times, independent censoring. Censored-data
metrics for the true model must agree with complete-data (uncensored) values, and
known miscalibration must be detected. Writes metric_validation.csv; exits non-zero
if any check fails.

    python validate_metrics.py --config pilot_config.yaml
"""
import argparse

import numpy as np
import pandas as pd

from common import (CoxCensoringModel, compute_metrics, ipcw_brier_ibs, km_at, load_config,
                    marginal_G, output_dir, setup_logging, truncate)


def simulate(n, rng, beta=(0.8, -0.5, 0.3), lam0=0.1, cens_rate=0.06, admin=25.0,
             cens_dep=0.0):
    X = rng.normal(size=(n, len(beta)))
    lp = X @ np.asarray(beta)
    T = rng.exponential(1.0 / (lam0 * np.exp(lp)))
    C = np.minimum(rng.exponential(1.0 / (cens_rate * np.exp(cens_dep * X[:, 0]))), admin)
    return X, lp, T, np.minimum(T, C), (T <= C).astype(int)


def true_surv(lp, grid, lam0=0.1, scale=1.0):
    return np.exp(-lam0 * np.exp(scale * lp)[:, None] * np.asarray(grid)[None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot_config.yaml")
    ap.add_argument("--n", type=int, default=6000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    log = setup_logging("validate_metrics", out)
    rng = np.random.default_rng(cfg["seed"])

    from sksurv.metrics import concordance_index_censored
    from scipy.stats import mannwhitneyu

    _, _, _, pool_t, pool_e = simulate(args.n, rng)
    _, lp, T, test_t, test_e = simulate(args.n, rng)
    t_star, tau = float(np.quantile(T, 0.5)), float(np.quantile(T, 0.75))
    grid = np.linspace(np.quantile(test_t, 0.01), tau, 50)
    t_admin = tau * (1 + 1e-6)
    pt, pe = truncate(pool_t, pool_e, t_admin)
    tt_, te = truncate(test_t, test_e, t_admin)
    km_grid = km_at(pt, pe, grid)
    km_star = km_at(pt, pe, t_star)[0]

    def metrics(S_grid, S_star, a=(pt, pe, tt_, te)):
        return compute_metrics(a[0], a[1], a[2], a[3], S_grid, grid, S_star, t_star, tau, km_grid)

    S_true, S_true_star = true_surv(lp, grid), true_surv(lp, [t_star])[:, 0]
    S_dist, S_dist_star = true_surv(lp, grid, scale=2.0), true_surv(lp, [t_star], scale=2.0)[:, 0]
    perm = rng.permutation(len(lp))
    m_true = metrics(S_true, S_true_star)
    m_dist = metrics(S_dist, S_dist_star)
    m_noise = metrics(S_true[perm], S_true_star[perm])
    m_km = metrics(np.tile(km_grid, (len(lp), 1)), np.full(len(lp), km_star))
    m_untrunc = compute_metrics(pool_t, pool_e, test_t, test_e, S_true, grid, S_true_star,
                                t_star, tau, km_grid)
    m_mono = metrics(S_true, S_true_star ** 3)  # monotone transform of risk

    brier_complete = float(np.mean(((T > t_star).astype(float) - S_true_star) ** 2))
    risk = 1 - S_true_star
    c_complete = concordance_index_censored(T <= tau, np.minimum(T, tau), risk)[0]
    cases, controls = risk[T <= t_star], risk[T > t_star]
    auc_complete = mannwhitneyu(cases, controls).statistic / (len(cases) * len(controls))

    checks = [
        ("IPCW Brier(t*) of true model ~ complete-data Brier", m_true["Brier_tstar"],
         brier_complete, abs(m_true["Brier_tstar"] - brier_complete) < 0.01),
        ("IBS ordering: true < KM < shuffled", m_true["IBS"], m_noise["IBS"],
         m_true["IBS"] < m_km["IBS"] < m_noise["IBS"]),
        ("IPA of the KM reference = 0", m_km["IPA"], 0.0, abs(m_km["IPA"]) < 1e-10),
        ("IPA of true model > 0", m_true["IPA"], 0.0, m_true["IPA"] > 0),
        ("Uno C(tau) ~ complete-data C truncated at tau", m_true["Uno_C_tau"], c_complete,
         abs(m_true["Uno_C_tau"] - c_complete) < 0.015),
        ("td-AUC(t*) ~ complete-data AUC(t*)", m_true["tdAUC_tstar"], auc_complete,
         abs(m_true["tdAUC_tstar"] - auc_complete) < 0.015),
        # Tolerance 1e-4: sksurv treats risks within tied_tol=1e-8 as ties, and the cubic
        # transform moves a few near-ties across that tolerance.
        ("Uno C invariant to monotone risk transform", m_mono["Uno_C_tau"], m_true["Uno_C_tau"],
         abs(m_mono["Uno_C_tau"] - m_true["Uno_C_tau"]) < 1e-4),
        ("Truncation at tau leaves IBS unchanged", m_untrunc["IBS"], m_true["IBS"],
         abs(m_untrunc["IBS"] - m_true["IBS"]) < 1e-9),
        ("Truncation at tau leaves Uno C unchanged", m_untrunc["Uno_C_tau"], m_true["Uno_C_tau"],
         abs(m_untrunc["Uno_C_tau"] - m_true["Uno_C_tau"]) < 1e-9),
        ("Calibration slope of true model ~ 1", m_true.get("cal_slope"), 1.0,
         0.9 <= m_true.get("cal_slope", np.nan) <= 1.1),
        ("Calibration slope detects doubled linear predictor (~0.5)", m_dist.get("cal_slope"), 0.5,
         0.4 <= m_dist.get("cal_slope", np.nan) <= 0.6),
        ("ICI of true model small", m_true.get("ICI"), 0.02, m_true.get("ICI", np.nan) < 0.02),
        ("ICI larger for miscalibrated model", m_dist.get("ICI"), m_true.get("ICI"),
         m_dist.get("ICI", 0) > m_true.get("ICI", 1)),
        ("O/E of true model ~ 1", m_true["O_E"], 1.0, 0.95 <= m_true["O_E"] <= 1.05),
    ]

    # Shared IPCW implementation (used for the conditional-IPCW sensitivity analysis) must
    # reproduce sksurv exactly when given sksurv's marginal censoring estimator.
    Gy, Gg = marginal_G(pt, pe, tt_, grid)
    _, ibs_own, _ = ipcw_brier_ibs(tt_, te, S_true, grid, Gy, Gg)
    checks.append(("Shared IPCW-IBS implementation equals sksurv IBS", ibs_own, m_true["IBS"],
                   abs(ibs_own - m_true["IBS"]) < 1e-9))

    # Covariate-dependent censoring: marginal IPCW is biased; the Cox censoring model
    # (fitted on the development pool) should move the estimate towards the complete-data value.
    Xp2, _, _, pt2, pe2 = simulate(args.n, rng, cens_dep=1.0)
    Xt2, lp2, T2, tt2, te2 = simulate(args.n, rng, cens_dep=1.0)
    a2 = (*truncate(pt2, pe2, t_admin), *truncate(tt2, te2, t_admin))
    S2 = true_surv(lp2, [t_star])[:, 0]
    m_dep = compute_metrics(*a2, true_surv(lp2, grid), grid, S2, t_star, tau,
                            km_at(a2[0], a2[1], grid))
    brier_dep = float(np.mean(((T2 > t_star).astype(float) - S2) ** 2))
    cm = CoxCensoringModel().fit(Xp2, pt2, pe2)
    Gy2, Gg2 = cm.predict(Xt2, a2[2], np.array([t_star]))
    b_cond = float(ipcw_brier_ibs(a2[2], a2[3], S2[:, None], np.array([t_star]), Gy2, Gg2)[0][0])
    checks.append(("Conditional IPCW reduces Brier(t*) bias under covariate-dependent censoring",
                   b_cond, brier_dep, abs(b_cond - brier_dep) < abs(m_dep["Brier_tstar"] - brier_dep)))

    rows = [{"check": c, "value": v, "reference": r, "pass": bool(p)} for c, v, r, p in checks]
    rows.append({"check": "INFO: marginal-IPCW Brier(t*) under covariate-dependent "
                          "censoring (true model)", "value": m_dep["Brier_tstar"],
                 "reference": brier_dep, "pass": None})
    df = pd.DataFrame(rows)
    df.to_csv(out / "metric_validation.csv", index=False)
    for r in rows:
        log.info("%-75s value=%s ref=%s pass=%s", r["check"], r["value"], r["reference"], r["pass"])
    failed = [r["check"] for r in rows if r["pass"] is False]
    if failed:
        log.error("Metric validation FAILED: %s", failed)
        raise SystemExit(1)
    log.info("All metric validation checks passed.")


if __name__ == "__main__":
    main()
