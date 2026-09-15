"""
prep_larsson.py
---------------
Convert a Larsson-2019 allelic UMI CSV (genes x cells, with gene/cell headers) into
the counts matrix that run_realdata.py expects (.npz with key 'counts', genes x cells).

Usage:
    python prep_larsson.py txburst/data/SS3_c57_UMIs_mESC.csv
    # -> writes SS3_c57_UMIs_mESC.npz
"""
import sys
import numpy as np
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path, index_col=0)                 # index = genes, columns = cells
M = np.rint(np.nan_to_num(df.values)).astype(np.int64)
print(f"loaded {M.shape[0]} rows x {M.shape[1]} cols")

# genes x cells: genes are the tens-of-thousands dimension, cells the hundreds
if M.shape[0] < M.shape[1]:
    M = M.T
    print(f"transposed -> {M.shape[0]} genes x {M.shape[1]} cells")

# sanity: integer, non-negative
M = np.clip(M, 0, None)
expressed = (M.mean(1) > 0.05).sum()
print(f"expressed genes (mean UMI > 0.05): {expressed}")
print(f"per-gene observed Fano (median over expressed): "
      f"{np.median([g.var()/g.mean() for g in M if g.mean() > 0.05]):.2f}")

out = path.rsplit('.', 1)[0] + ".npz"
np.savez(out, counts=M)
print(f"saved {out}  ({M.shape[0]} genes x {M.shape[1]} cells)")
