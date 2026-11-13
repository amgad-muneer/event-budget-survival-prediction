#!/usr/bin/env python3
"""Download the public DeepSurv benchmark files and build the three analysis CSVs.

The study uses the processed SUPPORT, METABRIC and Rotterdam-GBSG cohorts distributed with
DeepSurv (Katzman et al., 2018, BMC Med Res Methodol 18:24). This script

1. downloads the three original HDF5 files into ``external/`` and verifies their SHA-256
   checksums against the values recorded in DATA_PROVENANCE.md;
2. concatenates each file's ``train`` and ``test`` partitions (in that order, as PyCox does)
   and writes ``data/support.csv``, ``data/metabric.csv`` and ``data/gbsg.csv`` with the
   covariates ``x0, x1, ...`` plus ``duration`` and ``event``;
3. checks the resulting record and event counts against DATA_PROVENANCE.md.

No data files are stored in this repository. Please respect the licences and terms of the
original sources when using or redistributing the downloaded files.

Usage:
    python download_data.py [--data-dir data] [--external-dir external]
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/"
HERE = Path(__file__).resolve().parent

# dataset key -> (path in the DeepSurv repository, sha256, expected records, expected events)
FILES = {
    "support": ("support/support_train_test.h5",
                "e398f49930e4efc4ecd675a72ac547f0d1ae25ccf13f6a1024cded12ea88a83e", 8873, 6036),
    "metabric": ("metabric/metabric_IHC4_clinical_train_test.h5",
                 "e959131973c906c7e8b1e662c81160914768156a6da145a9bfed9b9ee8bf3e72", 1904, 1103),
    "gbsg": ("gbsg/gbsg_cancer_train_test.h5",
             "bd455f4395322a8b23040c15cedbef00ad51a447d02c95f1f1d686a6b196484b", 2232, 1267),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {BASE + rel}")
    urllib.request.urlretrieve(BASE + rel, dest)


def build_csv(h5_path):
    import h5py
    parts = []
    with h5py.File(h5_path, "r") as fh:
        for group in ("train", "test"):  # same order as pycox.datasets.from_deepsurv
            x = np.asarray(fh[group]["x"])
            df = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
            df["duration"] = np.asarray(fh[group]["t"]).ravel()
            df["event"] = np.asarray(fh[group]["e"]).ravel().astype(int)
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--external-dir", default=str(HERE / "external"))
    args = ap.parse_args()
    data_dir, ext_dir = Path(args.data_dir), Path(args.external_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for key, (rel, digest, n_expected, ev_expected) in FILES.items():
        print(key)
        h5 = ext_dir / Path(rel).name
        if not h5.exists():
            fetch(rel, h5)
        if sha256(h5) != digest:
            sys.exit(f"{h5}: SHA-256 mismatch (expected {digest}). Delete the file and retry, or "
                     "check DATA_PROVENANCE.md; the upstream file may have changed.")
        df = build_csv(h5)
        if len(df) != n_expected or int(df["event"].sum()) != ev_expected:
            sys.exit(f"{key}: got {len(df)} records / {int(df['event'].sum())} events, "
                     f"expected {n_expected} / {ev_expected}.")
        out = data_dir / f"{key}.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out}  ({len(df)} records, {int(df['event'].sum())} events; checksum OK)")
    print("Done. Next: python stage0_audit.py --data-dir data --out-root outputs")


if __name__ == "__main__":
    main()
