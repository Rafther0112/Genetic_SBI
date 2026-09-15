"""
dedup_cast.py
-------------
The CAST mESC table has 188 columns but only 94 distinct cells: 94 columns are exact
copies of 94 others, and only 94 of the 188 names are shared with the C57 table. This
script proves that, shows which name is paired with which, and writes a corrected CAST
table restricted to the 94 cells shared with C57, in the same column order, so the two
alleles are literally paired measurements of the same cells.

It does NOT overwrite anything: outputs go to new files with the suffix _dedup.

Usage:
    python dedup_cast.py \
        --c57 txburst/data/SS3_c57_UMIs_mESC.csv \
        --cast txburst/data/SS3_cast_UMIs_mESC.csv
"""
import argparse
import hashlib
import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path, index_col=0)
    return df.T if df.shape[0] < df.shape[1] else df          # genes x cells


def col_hash(df):
    return {c: hashlib.md5(np.ascontiguousarray(
        np.nan_to_num(df[c].values)).tobytes()).hexdigest() for c in df.columns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c57", required=True)
    ap.add_argument("--cast", required=True)
    ap.add_argument("--suffix", default="_dedup")
    a = ap.parse_args()

    c57, cast = load(a.c57), load(a.cast)
    print(f"[in] C57  {c57.shape[0]} genes x {c57.shape[1]} cells")
    print(f"[in] CAST {cast.shape[0]} genes x {cast.shape[1]} cells")
    assert list(c57.index) == list(cast.index), "gene order differs between alleles"

    h = col_hash(cast)
    groups = {}
    for c, v in h.items():
        groups.setdefault(v, []).append(c)
    dup_groups = {v: cs for v, cs in groups.items() if len(cs) > 1}
    print(f"\n[cast] {len(groups)} distinct column contents among {cast.shape[1]} columns; "
          f"{len(dup_groups)} contents appear more than once")
    shared = [c for c in c57.columns if c in set(cast.columns)]
    print(f"[cast] names shared with C57: {len(shared)}; names only in CAST: "
          f"{cast.shape[1] - len(shared)}")

    print("\n  first five duplicate pairs (name in C57  <->  extra CAST name):")
    shown = 0
    for v, cs in dup_groups.items():
        in_c57 = [c for c in cs if c in set(c57.columns)]
        extra = [c for c in cs if c not in set(c57.columns)]
        if in_c57 and extra and shown < 5:
            print(f"    {in_c57[0]:28s} <-> {extra[0]}")
            shown += 1
    paired = sum(1 for v, cs in dup_groups.items()
                 if any(c in set(c57.columns) for c in cs)
                 and any(c not in set(c57.columns) for c in cs))
    print(f"  duplicate groups pairing a shared name with an extra name: {paired}")
    if paired == len(shared):
        print("  -> every extra CAST column is a copy of a cell that C57 also has:")
        print("     the effective sample is 94 cells in BOTH alleles.")

    out = cast[shared]
    print(f"\n[out] corrected CAST: {out.shape[0]} genes x {out.shape[1]} cells, "
          f"column order matched to C57")
    dup_left = out.T.duplicated().sum()
    print(f"      duplicated columns remaining: {dup_left}")

    base = a.cast[:-4] if a.cast.endswith(".csv") else a.cast
    out.to_csv(base + a.suffix + ".csv")
    np.savez(base + a.suffix + ".npz", counts=out.values.astype(np.int64))
    c57o = c57[shared]
    print(f"[written] {base + a.suffix}.csv and .npz")

    print("\n[sanity] summaries are invariant under exact duplication, so these should match:")
    for name, df in [("CAST 188 cols", cast), ("CAST 94 cols", out)]:
        M = np.nan_to_num(df.values).astype(float)
        m = M.mean(1); v = M.var(1)
        f = np.divide(v, m, out=np.zeros_like(m), where=m > 1e-9)
        z = (M == 0).mean(1)
        print(f"  {name:14s} mean Fano {f.mean():.4f}   mean zero fraction {z.mean():.4f}")
    print("  (identical values confirm the duplication is exact; what changes is n_cells,")
    print("   which is what the estimator is conditioned on.)")

    print("\n[gene filter] genes passing two common filters, per allele:")
    for name, df in [("C57 (94)", c57o), ("CAST (188)", cast), ("CAST (94)", out)]:
        M = np.nan_to_num(df.values)
        n_cells_expr = (M > 0).sum(1)
        print(f"  {name:12s} mean UMI > 0.05: {(M.mean(1) > 0.05).sum():5d} | "
              f">0 in >=10 cells (absolute): {(n_cells_expr >= 10).sum():5d} | "
              f">0 in >=10% of cells: {(n_cells_expr >= 0.1*M.shape[1]).sum():5d}")
    print("  If your paper's gene counts (6,430 C57 vs 7,637 CAST) came from an ABSOLUTE")
    print("  cell-count filter, duplication is why CAST kept more genes.")


if __name__ == "__main__":
    main()
