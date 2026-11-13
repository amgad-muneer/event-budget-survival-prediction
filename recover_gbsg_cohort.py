#!/usr/bin/env python3
"""Recover Rotterdam / GBSG cohort membership for gbsg.csv by matching it against the
original DeepSurv HDF5 file (train = Rotterdam, n=1546; test = GBSG, n=686).

Matching is by record VALUES (all 7 covariates, duration and event, compared at float32
precision rounded to 4 decimals), never by row position. A CSV row is assigned a cohort only
if every HDF5 record with identical values lies in the same partition and the record counts
agree. Membership is written only if the match is complete and verified; otherwise the
pilot stratifies by event status alone and nothing is invented.

Get the HDF5 file (public, read-only use):
    mkdir -p external
    curl -L -o external/gbsg_cancer_train_test.h5 \
      https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/gbsg/gbsg_cancer_train_test.h5

    python recover_gbsg_cohort.py --config pilot_config.yaml
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROJECT_DIR, load_config, output_dir, read_manifest, write_json
from stage0_audit import load_table

EXPECTED_SHA256 = "bd455f4395322a8b23040c15cedbef00ad51a447d02c95f1f1d686a6b196484b"
EXPECTED = {"train": {"n": 1546, "events": 968}, "test": {"n": 686, "events": 299}}
LABELS = {"train": "Rotterdam (DeepSurv train partition)", "test": "GBSG (DeepSurv test partition)"}


def keys(X, t, e):
    """Value keys at float32 precision, rounded to 4 decimals."""
    M = np.column_stack([X, t, e]).astype(np.float32).astype(np.float64).round(4)
    return ["|".join(f"{v:.4f}" for v in row) for row in M]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot_config.yaml")
    ap.add_argument("--h5", default=str(PROJECT_DIR / "external" / "gbsg_cancer_train_test.h5"))
    args = ap.parse_args()
    import h5py
    cfg = load_config(args.config)
    out = output_dir(cfg) / "gbsg_cohort"
    out.mkdir(parents=True, exist_ok=True)
    h5 = Path(args.h5)
    if not h5.exists():
        raise SystemExit(f"{h5} not found - download it first (see the docstring / README).")
    sha = hashlib.sha256(h5.read_bytes()).hexdigest()

    rep = next(r for r in read_manifest(output_dir(cfg))["datasets"] if r["dataset"] == "gbsg" and r.get("n"))
    df, _, _ = load_table(Path(rep["file"]))
    tcol, ecol = rep["time_col"], rep["event_col"]
    xcols = [c for c in df.columns if c not in (tcol, ecol)]
    csv_keys = keys(df[xcols].to_numpy(float), df[tcol].to_numpy(float), df[ecol].to_numpy(float))

    h5_keys, parts = [], []
    part_stats = {}
    with h5py.File(h5, "r") as f:
        for g in f:
            x, t, e = f[g]["x"][:], f[g]["t"][:].ravel(), f[g]["e"][:].ravel()
            if x.shape[1] != len(xcols):
                raise SystemExit(f"HDF5 has {x.shape[1]} covariates, CSV has {len(xcols)}; not matched.")
            h5_keys += keys(x, t, e)
            parts += [g] * len(t)
            part_stats[g] = {"n": int(len(t)), "events": int(e.sum()),
                             "t_min": float(t.min()), "t_max": float(t.max())}

    h5_tab = pd.DataFrame({"key": h5_keys, "part": parts}).groupby("key")["part"].agg(list)
    csv_count = pd.Series(csv_keys).value_counts()
    rows = []
    for i, k in enumerate(csv_keys):
        if k not in h5_tab.index:
            status, part = "unmatched", None
        else:
            hp = h5_tab[k]
            if len(hp) != csv_count[k]:
                status, part = "count_mismatch", None
            elif len(set(hp)) == 1:
                status, part = "matched_unique_partition", hp[0]
            else:
                status, part = "ambiguous_identical_records_in_both_partitions", None
        rows.append({"csv_row": i, "h5_partition": part, "cohort_label": LABELS.get(part),
                     "match_status": status})
    mem = pd.DataFrame(rows)
    assigned = mem["h5_partition"].value_counts().to_dict()
    events_by_part = {p: int(df.loc[mem["h5_partition"] == p, ecol].sum()) for p in ("train", "test")}
    multiset_equal = sorted(csv_keys) == sorted(h5_keys)
    positional = list(mem["h5_partition"]) == ["train"] * EXPECTED["train"]["n"] + ["test"] * EXPECTED["test"]["n"]
    checks = {
        "h5_sha256_matches_published_file": sha == EXPECTED_SHA256,
        "csv_and_h5_record_multisets_identical": multiset_equal,
        "all_rows_assigned": bool(mem["h5_partition"].notna().all()),
        "partition_sizes_match_1546_686": assigned.get("train") == 1546 and assigned.get("test") == 686,
        "partition_event_counts_match_968_299": events_by_part == {"train": 968, "test": 299},
    }
    report = {"h5_file": str(h5), "h5_sha256": sha, "expected_sha256": EXPECTED_SHA256,
              "csv_file": rep["file"], "h5_partitions": part_stats,
              "status_counts": mem["match_status"].value_counts().to_dict(),
              "assigned_counts": assigned, "events_by_assigned_partition": events_by_part,
              "checks": checks, "verified_complete": all(checks.values()),
              "csv_order_equals_train_then_test": positional,
              "label_mapping": LABELS,
              "label_basis": ("DeepSurv (Katzman et al. 2018) trained on Rotterdam and tested on GBSG; "
                              "partition sizes 1546/686 and the 686-patient partition's 299 events "
                              "and 0.263-87.36 month range are consistent with the original GBSG trial data."),
              "use": "outer-split stratification only; never a predictor"}
    if report["verified_complete"]:
        mem[["csv_row", "h5_partition", "cohort_label", "match_status"]].to_csv(
            out / "gbsg_cohort_membership.csv", index=False)
    else:
        mem.to_csv(out / "gbsg_cohort_match_diagnostics.csv", index=False)
        stale = out / "gbsg_cohort_membership.csv"
        if stale.exists():
            stale.unlink()
    write_json(report, out / "gbsg_cohort_report.json")
    print(pd.Series(checks).to_string())
    print("verified_complete:", report["verified_complete"],
          "| CSV order equals train-then-test:", positional)


if __name__ == "__main__":
    main()
