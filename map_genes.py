"""
map_genes.py
------------
Turns the row indices that run_realdata.py prints into gene names. Those indices refer to
the matrix AFTER the expression filter, so this script reapplies the same filter to the
original CSV (whose index holds the gene identifiers) and reads off the names.

Usage:
    python map_genes.py --csv txburst/data/SS3_c57_UMIs_mESC.csv \
        --idx 3459 4667 4215 5697 2684 2117 6279 4725 4663 3239
    python map_genes.py --csv txburst/data/SS3_cast_UMIs_mESC_dedup.csv \
        --idx 5166 2711 2495 4627 5197 888 1976 2382 6026 3823
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.csv, index_col=0)
    if df.shape[0] < df.shape[1]:
        df = df.T
    M = np.rint(np.nan_to_num(df.values.astype(float))).astype(int)
    keep = (M.mean(1) > 0.05) & (M.shape[1] - (M == 0).sum(1) >= 20)   # same as load_counts
    names = df.index[keep]
    print(f"[filter] {keep.sum()} genes kept of {len(df)} ({df.shape[1]} cells)")
    print(f"{'row':>6s}  {'gene':<20s} {'mean':>7s} {'Fano':>7s} {'zero frac':>9s}")
    kept = M[keep]
    for i in a.idx:
        if i >= len(names):
            print(f"{i:6d}  OUT OF RANGE (only {len(names)} genes after the filter)")
            continue
        c = kept[i].astype(float)
        f = c.var() / c.mean() if c.mean() > 0 else float("nan")
        print(f"{i:6d}  {str(names[i]):<20s} {c.mean():7.2f} {f:7.2f} {(c == 0).mean():9.2f}")
    print("\nPaste this to Claude for Table 6. If the gene column shows Ensembl IDs, the")
    print("symbols are in txburst/data/mouse_gene_annotation.csv.")


if __name__ == "__main__":
    main()
