"""
run_cost_table.py  (E5a, parte 1)
=================================
Mide el costo de cada simulador en UNA misma maquina, que es lo que convierte el
eje del presupuesto de "numero de simulaciones" a "segundos".

Produce results/costs.json con:
  * c_sim   : segundos por simulacion (snapshot de n_cells celulas), por simulador
  * ssa_fit : ajuste log-log del costo del SSA contra k_syn * p_on, que es la
              dependencia en theta que hace posible la asignacion consciente del
              costo (E5b). El paper ya reporta un factor 622 a lo largo del prior.

Uso:
  python run_cost_table.py --system telegraph --sims cme fsp ssa cle lna nb hybrid
  python run_cost_table.py --system threestate --sims fsp ssa cle nb hybrid
"""
import argparse, json, os, time, numpy as np
import warnings; warnings.filterwarnings("ignore")
import exp_common as E


def time_sim(theta, sim, n_cells, system, reps=3):
    """Segundos por simulacion, mediana de reps repeticiones."""
    ts = []
    for r in range(reps):
        rng = np.random.default_rng(100 + r)
        t0 = time.perf_counter()
        E.simulate(theta, sim, n_cells, rng, system)
        ts.append((time.perf_counter() - t0) / len(theta))
    return float(np.median(ts)), float(np.std(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="telegraph", choices=["telegraph", "threestate"])
    p.add_argument("--sims", nargs="+", default=["cme", "cle", "lna", "nb"])
    p.add_argument("--n", type=int, default=400, help="thetas para el cronometraje")
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--ssa_n", type=int, default=120, help="thetas para el ajuste de costo del SSA")
    p.add_argument("--out", default="results/costs.json")
    a = p.parse_args()

    import torch; torch.manual_seed(0)
    prior = E.prior_of(a.system)
    theta = prior.sample((a.n,)).numpy()

    costs = {}
    for s in a.sims:
        try:
            c, sd = time_sim(theta, s, a.n_cells, a.system)
            costs[s] = dict(sec_per_sim=c, std=sd, n_cells=a.n_cells)
            print(f"  {s:8s} {c*1e3:9.3f} ms/sim   ({c*8000/60:6.2f} min por 8000)")
        except Exception as e:
            print(f"  {s:8s} no disponible: {e}")

    # dependencia del costo del SSA en theta: se cronometra theta por theta
    if "ssa" in a.sims:
        th = prior.sample((a.ssa_n,)).numpy()
        per = []
        for i in range(a.ssa_n):
            rng = np.random.default_rng(7)
            t0 = time.perf_counter()
            E.simulate(th[i:i+1], "ssa", a.n_cells, rng, a.system)
            per.append(time.perf_counter() - t0)
        per = np.array(per)
        r = 10.0 ** th
        if a.system == "telegraph":
            kon, koff, ksyn = r[:, 0], r[:, 1], r[:, 2]
            load = ksyn * kon / (kon + koff)          # k_syn * p_on
        else:
            load = np.asarray(E.TS3.expected_counts3(th))   # k_syn * pi_ACTIVE
        b, loga = np.polyfit(np.log(load), np.log(per), 1)
        rho = np.corrcoef(np.log(load), np.log(per))[0, 1]
        costs["ssa_fit"] = dict(slope=float(b), intercept=float(loga),
                                logcorr=float(rho),
                                span=float(per.max() / per.min()),
                                note="sec_per_sim(theta) = exp(intercept) * load**slope")
        print(f"  SSA: costo ~ load^{b:.2f}, corr log-log {rho:.3f}, "
              f"rango {per.max()/per.min():.0f}x sobre el prior")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    key = a.system
    all_costs = {}
    if os.path.exists(a.out):
        all_costs = json.load(open(a.out))
    all_costs[key] = costs
    json.dump(all_costs, open(a.out, "w"), indent=2)
    print("guardado en", a.out)


if __name__ == "__main__":
    main()
