"""
check_larsson.py
----------------
Real-data sanity check before rerunning Section 4.8. For the two allelic UMI tables it
reports shape, duplicated cell columns (identical names or identical content), whether
the cell identifiers of the two alleles match, and how many genes pass the expression
filter. Settles the "94 cells for C57 vs 188 for CAST" question.

No torch needed.   Usage:
    python check_larsson.py txburst/data/SS3_c57_UMIs_mESC.csv txburst/data/SS3_cast_UMIs_mESC.csv
"""
import sys
import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path, index_col=0)
    if df.shape[0] < df.shape[1]:
        df = df.T
    return df            # genes x cells


def describe(name, df):
    M = np.nan_to_num(df.values)
    dup_names = df.columns[df.columns.duplicated()].tolist()
    dup_content = int(pd.DataFrame(M.T).duplicated().sum())
    expressed = int((M.mean(1) > 0.05).sum())
    print(f"\n[{name}] {df.shape[0]} genes x {df.shape[1]} cells")
    print(f"  duplicated cell names   : {len(dup_names)}")
    print(f"  duplicated cell columns : {dup_content} (identical counts across all genes)")
    print(f"  genes with mean UMI > 0.05: {expressed}")
    print(f"  first cell ids: {list(df.columns[:4])}")
    return set(map(str, df.columns))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ca, cb = describe(sys.argv[1], a), describe(sys.argv[2], b)
    print(f"\n[cells] shared ids: {len(ca & cb)} | only first: {len(ca - cb)} | only second: {len(cb - ca)}")
    print("If the alleles come from the same cells, the shared count should equal both totals.")


if __name__ == "__main__":
    main()
