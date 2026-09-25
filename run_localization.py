"""
E4  --  ¿Se puede saber ANTES de entrenar si vale la pena asignar?

El paper dice que la asignacion solo paga si el error del sustituto esta
localizado, y lo comprueba con la curva de calibracion por bin, que exige
entrenar los estimadores. Este script obtiene la misma distincion sin entrenar
ningun NPE:

  1. Simula pares (theta, x) con el simulador exacto y con el sustituto, sobre
     los MISMOS theta.
  2. Entrena un clasificador exacto-vs-sustituto sobre (theta, resumenes).
     Su logit estima el log de la razon de verosimilitudes
     p_exacto(x|theta) / p_sustituto(x|theta).
  3. Por bin de Fano reporta:
       - AUC              : cuan distinguibles son (magnitud de la discrepancia)
       - nESS             : tamaño efectivo de muestra de los pesos de importancia
       - uncovered        : fraccion de datos exactos que el sustituto casi nunca
                            produce (clasificador > 0.99), es decir masa no cubierta
  4. Devuelve el INDICE DE LOCALIZACION
       L = (max_b uncovered_b - min_b uncovered_b) / max(max_b uncovered_b, eps)
     L cercano a 0  -> discrepancia pareja: la asignacion no puede pagar
     L cercano a 1  -> discrepancia concentrada: la asignacion puede pagar

Validado en el telegrafo: LNA da uncovered 0.27-0.74 (L bajo, y en efecto ninguna
asignacion gana) y NB da 0.00-0.11 (L alto). Control de cordura: exacto contra
exacto da AUC 0.5, nESS 0.99 y uncovered 0.

Uso:
  python run_localization.py --system telegraph --exact cme --surrogate lna
  python run_localization.py --system telegraph --exact cme --surrogate nb
  python run_localization.py --system threestate --exact fsp --surrogate nb
  python run_localization.py --system telegraph --exact cme --surrogate cme   # control
"""
import argparse, numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import exp_common as E


def fano_of(theta, system):
    if system == "threestate":
        return np.asarray(E.TS3.analytic_fano3(theta))
    from telegraph import exact_fano
    r = 10.0 ** np.asarray(theta)
    return np.array([exact_fano(*ri) for ri in r])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="telegraph", choices=["telegraph", "threestate"])
    p.add_argument("--exact", default="cme")
    p.add_argument("--surrogate", default="lna")
    p.add_argument("--n", type=int, default=12000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--bins", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results/e4_localization.jsonl")
    a = p.parse_args()

    for seed in a.seeds:
        prior = E.prior_of(a.system)
        import torch; torch.manual_seed(seed)
        th = prior.sample((a.n,)).numpy()
        rng = np.random.default_rng(seed)
        xe = E.simulate(th, a.exact, a.n_cells, rng, a.system)
        xs = E.simulate(th, a.surrogate, a.n_cells, rng, a.system)

        X = np.vstack([np.hstack([th, xe]), np.hstack([th, xs])])
        y = np.r_[np.ones(a.n), np.zeros(a.n)]
        idx = np.random.default_rng(seed).permutation(2 * a.n)
        tr, te = idx[:a.n], idx[a.n:]                       # mitad y mitad
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             early_stopping=True,
                                             random_state=seed).fit(X[tr], y[tr])
        pr = np.clip(clf.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        logw = np.log(pr) - np.log1p(-pr)                   # log p_exacto/p_surrogate
        th_te, y_te = X[te, :th.shape[1]], y[te]
        fano = fano_of(th_te, a.system)
        edges = np.quantile(fano, np.linspace(0, 1, a.bins + 1)); edges[-1] += 1
        b = np.clip(np.digitize(fano, edges) - 1, 0, a.bins - 1)

        rows = []
        for k in range(a.bins):
            m = b == k; ms, me = m & (y_te == 0), m & (y_te == 1)
            lw = logw[ms]; w = np.exp(lw - lw.max())
            ness = float(w.sum() ** 2 / (w ** 2).sum() / max(len(w), 1))
            rows.append(dict(bin=k, fano=float(np.median(fano[m])),
                             auc=float(roc_auc_score(y_te[m], pr[m])),
                             ness=ness, uncovered=float(np.mean(pr[me] > 0.99))))
        u = np.array([r["uncovered"] for r in rows])
        L = float((u.max() - u.min()) / max(u.max(), 1e-6))
        E.append_row(a.out, dict(system=a.system, exact=a.exact,
                                 surrogate=a.surrogate, seed=seed,
                                 localization_index=L, per_bin=rows,
                                 verdict="asignacion puede pagar" if L > 0.5
                                 else "asignacion no deberia pagar"))


if __name__ == "__main__":
    import os; os.makedirs("results", exist_ok=True); main()
