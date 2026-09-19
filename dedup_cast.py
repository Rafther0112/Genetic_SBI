"""
dedup_cast.py  (v2)
-------------------
The CAST mESC table has 188 columns but only 94 distinct cells: each cell appears twice,
once under a p1 name (shared with C57) and once under the matching p2 name. This script
proves that and writes a corrected CAST table with the 94 distinct cells, in the C57
column order.

v2 fixes a bug in v1: the CSV contains NaN entries, and casting NaN to int64 produces
huge negative sentinels, which then fail the expression filter of run_realdata.py (that
is why the deduplicated CAST run kept only 1,335 genes instead of ~7,000). Values are now
taken from the ORIGINAL .npz produced by prep_larsson.py whenever it is available, so the
corrected file goes through exactly the same preprocessing as the published one; the CSV
is used only to recover the column names. If no .npz is given, NaNs are filled with zero
and the count is reported.

Usage:
    python dedup_cast.py \
        --c57  txburst/data/SS3_c57_UMIs_mESC.csv \
        --cast txburst/data/SS3_cast_UMIs_mESC.csv \
        --cast_npz txburst/data/SS3_cast_UMIs_mESC.npz
"""
import argparse
import hashlib
import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path, index_col=0)
    return df.T if df.shape[0] < df.shape[1] else df          # genes x cells


def col_hash(M, names):
    return {n: hashlib.md5(np.ascontiguousarray(
        np.nan_to_num(M[:, j])).tobytes()).hexdigest() for j, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c57", required=True)
    ap.add_argument("--cast", required=True)
    ap.add_argument("--cast_npz", default=None,
                    help="original .npz from prep_larsson.py; its values are used verbatim")
    ap.add_argument("--suffix", default="_dedup")
    a = ap.parse_args()

    c57, cast = load(a.c57), load(a.cast)
    names = list(cast.columns)
    print(f"[in] C57  {c57.shape[0]} genes x {c57.shape[1]} cells")
    print(f"[in] CAST {cast.shape[0]} genes x {cast.shape[1]} cells (csv)")
    assert list(c57.index) == list(cast.index), "gene order differs between alleles"

    n_nan = int(np.isnan(cast.values.astype(float)).sum())
    print(f"[csv] NaN entries in the CAST table: {n_nan}")

    if a.cast_npz:
        M = np.load(a.cast_npz)["counts"]
        if M.shape[0] != cast.shape[0] and M.shape[1] == cast.shape[0]:
            M = M.T
        assert M.shape == cast.shape, f"npz {M.shape} does not match csv {cast.shape}"
        csv_ok = np.nan_to_num(cast.values.astype(float)) == M.astype(float)
        print(f"[npz] values taken from {a.cast_npz}; agreement with NaN-filled csv: "
              f"{100*csv_ok.mean():.4f}% of entries")
    else:
        M = np.nan_to_num(cast.values.astype(float), nan=0.0)
        print("[npz] no --cast_npz given: NaNs filled with zero")
    M = np.rint(M).astype(np.int64)
    assert M.min() >= 0, "negative counts after cast: NaN handling still wrong"

    h = col_hash(M, names)
    groups = {}
    for j, nm in enumerate(names):
        groups.setdefault(h[nm], []).append(nm)
    shared = [c for c in c57.columns if c in set(names)]
    paired = sum(1 for cs in groups.values()
                 if any(c in set(c57.columns) for c in cs)
                 and any(c not in set(c57.columns) for c in cs))
    print(f"\n[cast] {len(groups)} distinct column contents among {len(names)} columns")
    print(f"[cast] names shared with C57: {len(shared)}; duplicate groups pairing a shared "
          f"name with an extra name: {paired}")
    if paired == len(shared):
        print("  -> every extra column copies a cell that C57 also has; 94 cells in both alleles.")

    idx = [names.index(c) for c in shared]
    out = M[:, idx]
    base = a.cast[:-4] if a.cast.endswith(".csv") else a.cast
    np.savez(base + a.suffix + ".npz", counts=out)
    pd.DataFrame(out, index=cast.index, columns=shared).to_csv(base + a.suffix + ".csv")
    print(f"\n[out] {out.shape[0]} genes x {out.shape[1]} cells, dtype {out.dtype}, "
          f"min {out.min()}, max {out.max()}")
    print(f"[written] {base + a.suffix}.npz and .csv")

    print("\n[filter] genes passing run_realdata.load_counts "
          "(mean > 0.05 AND nonzero in >= 20 cells):")
    for name, A in [("C57  (94)", np.rint(np.nan_to_num(c57.values.astype(float))).astype(np.int64)),
                    ("CAST (188)", M), ("CAST (94)", out)]:
        keep = (A.mean(1) > 0.05) & (A.shape[1] - (A == 0).sum(1) >= 20)
        print(f"  {name:11s} {int(keep.sum()):6d} genes")
    print("  These are the numbers run_realdata.py will report. If CAST (94) is far below")
    print("  C57 (94), the alleles are not comparable and the two runs must be restricted")
    print("  to their common gene set.")


if __name__ == "__main__":
    main()