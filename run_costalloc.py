"""
run_costalloc.py  (E5b)
=======================
Asignacion consciente del costo, a presupuesto fijo EN SEGUNDOS.

Motivo: el costo del SSA depende de theta (el paper reporta un factor 622 sobre
el prior, correlacion 0.98 en log-log con k_syn * p_on). Con costos heterogeneos
aparece una segunda razon para colocar, independiente de la localizacion de la
discrepancia: a igual presupuesto en segundos, gastar uniformemente en theta no
es optimo, porque unas regiones cuestan mucho mas que otras.

Esquemas comparados, todos al mismo presupuesto B en segundos:
  uniform      : elige thetas al azar hasta agotar B
  discrepancy  : prioriza por masa no cubierta (indice de localizacion por region)
  cheap-first  : prioriza por costo bajo, ignorando la discrepancia
  ratio        : prioriza por  masa_no_cubierta(theta) / costo(theta)   <-- la regla
  exact-only   : uniform pero sin rellenar con el sustituto

Prediccion: 'ratio' gana incluso cuando la discrepancia NO esta localizada, que
es el caso donde hoy no hay ninguna ganancia que mostrar.

Uso:
  python run_costalloc.py --budget 600 1800 --seeds 0 1 2 --surrogate lna
  (budget en segundos de simulacion exacta; conviene empezar pequeño)
"""
import argparse, os, json, numpy as np, torch
import exp_common as E
from exp_common import train_npe, evaluate, shared_test_set, append_row


def cost_model(theta, costs, system):
    """segundos por simulacion exacta en cada theta, segun el ajuste del SSA."""
    fit = costs[system].get("ssa_fit")
    r = 10.0 ** np.asarray(theta)
    if system == "telegraph":
        load = r[:, 2] * r[:, 0] / (r[:, 0] + r[:, 1])
    else:
        load = np.asarray(E.TS3.expected_counts3(theta))
    if fit is None:            # sin ajuste: costo constante
        return np.full(len(theta), costs[system]["cme"]["sec_per_sim"])
    return np.exp(fit["intercept"]) * load ** fit["slope"]


def uncovered_by_theta(theta, xe, xs, seed=0):
    """masa no cubierta local: prob. de que el clasificador marque 'solo exacto'."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.vstack([np.hstack([theta, xe]), np.hstack([theta, xs])])
    y = np.r_[np.ones(len(theta)), np.zeros(len(theta))]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         random_state=seed).fit(X, y)
    p = clf.predict_proba(np.hstack([theta, xe]))[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def pick(order, cost, budget):
    """toma thetas en el orden dado hasta agotar el presupuesto en segundos."""
    c = np.cumsum(cost[order])
    k = int(np.searchsorted(c, budget))
    return order[:k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="telegraph")
    p.add_argument("--exact", default="ssa")
    p.add_argument("--surrogate", default="lna")
    p.add_argument("--pool", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--budget", type=float, nargs="+", default=[600, 1800])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--costs", default="results/costs.json")
    p.add_argument("--out", default="results/e5b_costalloc.jsonl")
    a = p.parse_args()
    costs = json.load(open(a.costs))

    for seed in a.seeds:
        th_e, x_e = E.cached_sims(a.exact, a.pool, a.n_cells, seed, system=a.system)
        th_s, x_s = E.cached_sims(a.surrogate, a.pool, a.n_cells, seed, system=a.system)
        tt, xt = shared_test_set(a.n_test, a.n_cells, seed, system=a.system,
                                 exact=a.exact)
        th = th_e.numpy()
        c = cost_model(th, costs, a.system)
        u = uncovered_by_theta(th, x_e.numpy(), x_s.numpy(), seed)
        rng = np.random.default_rng(seed)

        orders = {"uniform": rng.permutation(a.pool),
                  "discrepancy": np.argsort(-u),
                  "cheap-first": np.argsort(c),
                  "ratio": np.argsort(-(u / c))}
        for B in a.budget:
            for name, order in orders.items():
                ex = pick(order, c, B)
                if len(ex) < 200:
                    print(f"  presupuesto {B}s alcanza solo {len(ex)} exactas, saltando")
                    continue
                mask = np.zeros(a.pool, bool); mask[ex] = True
                for arm, (th_tr, x_tr) in {
                        "mix": (torch.cat([th_e[mask], th_s[~mask]]),
                                torch.cat([x_e[mask], x_s[~mask]])),
                        "exact-only": (th_e[mask], x_e[mask])}.items():
                    if arm == "exact-only" and name != "uniform":
                        continue          # solo-exacto se reporta una vez
                    de, _ = train_npe(th_tr, x_tr, system=a.system, seed=seed)
                    r = evaluate(de, tt, xt)
                    append_row(a.out, dict(seed=seed, budget_s=B, scheme=name,
                                           arm=arm, n_exact=int(len(ex)),
                                           mean_cost=float(c[ex].mean()),
                                           mean_uncovered=float(u[ex].mean()),
                                           cov=r["cov"], gap=abs(r["cov"] - 0.9),
                                           bias=r["bias"]))


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True); main()
