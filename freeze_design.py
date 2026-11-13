#!/usr/bin/env python3
"""Freeze evaluation horizons, tau and event budgets before any model is fitted.

Horizons come from external clinical conventions in pilot_config.yaml (with units verified
in DATA_PROVENANCE.md). Their statistical support is checked on DEVELOPMENT data only:
the development pool of outer split 0, which is reproduced exactly as in run_pilot.py.
Test outcomes are never used. If a dataset has `horizons: auto`, the fallback rule uses
event-time quantiles of the development pool (again never the test set).

Writes outputs/frozen_design.json; refuses to overwrite unless --force, and then keeps
the previous file with a timestamp.

    python freeze_design.py --config pilot_config.yaml
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil

import numpy as np

from common import (km_at, load_config, load_dataset, output_dir, read_manifest, split_seed,
                    stratified_split, write_json)

S_RANGE, MIN_G, TAU_G, MIN_TEST_AT_RISK = (0.2, 0.9), 0.3, 0.2, 30


def support_at(t, time, event, test_ratio):
    """KM S(t), reverse-KM G(t-) and expected test patients at risk, from development data."""
    s = float(km_at(time, event, t)[0])
    g = float(km_at(time, 1 - event, np.nextafter(t, -np.inf))[0])
    at_risk = int(np.sum(time >= t))
    return {"time": float(t), "S_dev": s, "G_dev_left": g, "n_at_risk_dev": at_risk,
            "expected_test_at_risk": round(test_ratio * at_risk, 1),
            "horizon_ok": bool(S_RANGE[0] <= s <= S_RANGE[1] and g >= MIN_G),
            "tau_ok": bool(g >= TAU_G and test_ratio * at_risk >= MIN_TEST_AT_RISK)}


def freeze_dataset(cfg, key, manifest):
    dcfg = cfg["datasets"][key]
    _, t, e, strata, rep, info = load_dataset(cfg, key, manifest)
    rng = np.random.default_rng(split_seed(cfg, key, 0))
    test, pool = stratified_split(strata, cfg["test_fraction"], rng)
    tp, ep = t[pool], e[pool]
    ratio = len(test) / len(pool)
    out = {"file": rep["file"], "time_col": rep["time_col"], "event_col": rep["event_col"],
           "units": dcfg.get("units", "UNVERIFIED"), "endpoint": dcfg.get("endpoint", "UNVERIFIED"),
           "analysis_rule_exclusions": {k: info[k] for k in ("n_file", "n_excluded", "excluded_csv_rows")},
           "cohort_stratified": info["cohort_stratified"],
           "support_basis": "development pool of outer split 0 (test outcomes not used)",
           "notes": []}
    if dcfg.get("horizons", "auto") == "auto":
        q = cfg["horizon_rule"]["quantiles"]
        hz = [float(np.quantile(tp[ep == 1], p)) for p in q]
        primary = hz[cfg["horizon_rule"]["primary_index"]]
        out["horizon_source"] = f"fallback rule: development-pool event-time quantiles {q}"
    else:
        hz = [float(h) for h in dcfg["horizons"]]
        primary = float(dcfg["primary"])
        out["horizon_source"] = dcfg.get("horizon_source", "config (external clinical values)")
    checks = {str(h): support_at(h, tp, ep, ratio) for h in hz}
    if not checks[str(primary)]["horizon_ok"] and dcfg.get("fallback_primary") is not None:
        fb = float(dcfg["fallback_primary"])
        out["notes"].append(f"primary {primary:g} not supported in development data; prespecified "
                            f"fallback {fb:g} used")
        primary = fb
        checks.setdefault(str(fb), support_at(fb, tp, ep, ratio))
    for h, c in checks.items():
        if not c["horizon_ok"]:
            out["notes"].append(f"horizon {h} not supported (S={c['S_dev']:.3f}, G={c['G_dev_left']:.3f})")
    tau = float(dcfg.get("tau", max(hz)))
    if not support_at(tau, tp, ep, ratio)["tau_ok"]:
        ok = [h for h in sorted(set(hz) | {primary}) if support_at(h, tp, ep, ratio)["tau_ok"] and h >= primary]
        out["notes"].append(f"tau {tau:g} not supported in development data; capped at "
                            f"{ok[-1] if ok else primary:g}")
        tau = ok[-1] if ok else primary
    out.update({"horizons": hz, "primary_horizon": primary, "tau": tau, "support_checks": checks,
                "grid_start": float(np.quantile(tp, 0.01)),
                "expected_pool_events": int(ep.sum()),
                "budget_ladder_feasible": [str(b) for b in cfg["reduced_design"]["budget_ladder"]
                                           if b == "full" or int(b) < int(ep.sum())]})
    out["admissible"] = checks[str(primary)]["horizon_ok"] and support_at(tau, tp, ep, ratio)["tau_ok"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot_config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    target = out / "frozen_design.json"
    if target.exists():
        if not args.force:
            raise SystemExit(f"{target} exists (design already frozen). Use --force to re-freeze.")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy(target, out / f"frozen_design_superseded_{stamp}.json")
    manifest = read_manifest(out)
    design = {"frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "manifest_sha256": hashlib.sha256(
                  (out / "dataset_manifest.json").read_bytes()).hexdigest(),
              "admissibility_rule": {"S_range": S_RANGE, "min_G_left": MIN_G, "tau_min_G": TAU_G,
                                     "tau_min_expected_test_at_risk": MIN_TEST_AT_RISK},
              "datasets": {}}
    for key in cfg["datasets"]:
        try:
            design["datasets"][key] = freeze_dataset(cfg, key, manifest)
        except Exception as exc:  # noqa: BLE001
            design["datasets"][key] = {"error": repr(exc)}
    write_json(design, target)
    print(json.dumps(design, indent=2))


if __name__ == "__main__":
    main()
