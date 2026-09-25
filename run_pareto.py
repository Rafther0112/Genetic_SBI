"""
run_pareto.py  (E5a, parte 2)
=============================
Reponderar el eje del presupuesto SIN entrenar nada mas.

Un estimador ya entrenado no depende de los costos, asi que las corridas de E1,
E2 y E3 se pueden releer contra un presupuesto en segundos:

    C(n_e, n_s) = n_e * c_e + n_s * c_s

Definiendo rho = c_e / c_s (cuantas veces mas caro es el exacto), el costo en
unidades de simulacion exacta es  C/c_e = n_e + n_s / rho.

Dos salidas:
  1. FRENTE DE PARETO: calibracion contra presupuesto, con una curva por brazo,
     para varios valores de rho (el medido y algunos hipoteticos).
  2. RAZON DE EQUILIBRIO rho*: el valor minimo de rho para el cual algun brazo
     que usa el sustituto supera a entrenar solo con simulaciones exactas al
     mismo presupuesto. Si el rho medido del sistema es menor que rho*, el
     sustituto no compensa ahi. Es el numero que la segunda direccion de trabajo
     futuro de Krouglova et al. (2026) pide y que su supuesto de costo
     despreciable para la baja fidelidad da por descontado.

Uso:
  python run_pareto.py --runs results/v5/e1_twostage.jsonl results/v5/e2_crossover.jsonl \
                       --costs results/costs.json --system telegraph --surrogate lna
"""
import argparse, json, os, numpy as np
from collections import defaultdict


def load_runs(paths, n_sims):
    """Devuelve filas con (arm, n_exact, n_surrogate, gap) promediadas por semilla."""
    rows = []
    for p in paths:
        for line in open(p):
            if not line.strip(): continue
            r = json.loads(line)
            arm = r.get("arm") or r.get("scheme")
            if arm is None: continue
            f = r.get("f", 0.0)
            ne = r.get("n_exact", int(round(f * n_sims)))
            if arm in ("exact-only",):
                ns = 0
            elif arm in ("surrogate",):
                ne, ns = 0, n_sims
            elif arm == "two-stage":
                ns = n_sims                      # preentrena con el conjunto completo
            else:                                 # mixture y esquemas de colocacion
                ns = n_sims - ne
            rows.append(dict(arm=arm, seed=r["seed"], n_e=ne, n_s=ns,
                             gap=abs(r["cov"] - 0.9), cov=r["cov"]))
    agg = defaultdict(list)
    for r in rows: agg[(r["arm"], r["n_e"], r["n_s"])].append(r["gap"])
    return [dict(arm=k[0], n_e=k[1], n_s=k[2], gap=float(np.mean(v)),
                 sd=float(np.std(v)), n=len(v)) for k, v in agg.items()]


def exact_curve(rows):
    """Interpola la calidad de solo-exacto como funcion de log n_e."""
    pts = sorted([(r["n_e"], r["gap"]) for r in rows if r["arm"] == "exact-only"])
    if len(pts) < 2:
        raise SystemExit("hacen falta al menos dos puntos de exact-only (corran E2)")
    x = np.log(np.array([p[0] for p in pts], float)); y = np.array([p[1] for p in pts])
    return lambda ne: float(np.interp(np.log(max(ne, 1.0)), x, y))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--costs", default="results/costs.json")
    p.add_argument("--system", default="telegraph")
    p.add_argument("--exact", default="cme")
    p.add_argument("--surrogate", default="lna")
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--plot", default="results/pareto.png")
    a = p.parse_args()

    rows = load_runs(a.runs, a.n_sims)
    g_exact = exact_curve(rows)

    rho_meas = None
    if os.path.exists(a.costs):
        C = json.load(open(a.costs)).get(a.system, {})
        if a.exact in C and a.surrogate in C:
            rho_meas = C[a.exact]["sec_per_sim"] / C[a.surrogate]["sec_per_sim"]
            print(f"razon de costos medida rho = c_{a.exact}/c_{a.surrogate} = {rho_meas:.2f}")
            if rho_meas < 1:
                print("  (el simulador 'exacto' es MAS BARATO que el sustituto en este sistema)")

    # rho* : minimo rho para el que algun brazo con sustituto gana a igual presupuesto
    grid = np.concatenate([np.linspace(0.1, 10, 100), np.geomspace(10, 1e4, 100)])
    print("\nbrazo               n_e     n_s    gap    rho* (equilibrio)")
    best_rho = np.inf
    for r in sorted(rows, key=lambda r: (r["arm"], r["n_e"])):
        if r["arm"] in ("exact-only",) or r["n_s"] == 0: continue
        rho_star = None
        for rho in grid:
            n_eq = r["n_e"] + r["n_s"] / rho          # presupuesto en unidades de exacta
            if r["gap"] < g_exact(n_eq):              # gana a solo-exacto al mismo costo
                rho_star = rho; break
        s = f"{rho_star:7.2f}" if rho_star else "  nunca"
        print(f"{r['arm']:18s} {r['n_e']:6d} {r['n_s']:7d}  {r['gap']:.3f}  {s}")
        if rho_star: best_rho = min(best_rho, rho_star)

    if np.isfinite(best_rho):
        print(f"\nrho* global = {best_rho:.2f}: el sustituto compensa solo si el exacto "
              f"es al menos {best_rho:.1f}x mas caro que el.")
        if rho_meas is not None:
            v = "SI" if rho_meas >= best_rho else "NO"
            print(f"En este sistema rho medido = {rho_meas:.2f}, luego {v} compensa.")
    else:
        print("\nNingun brazo con sustituto gana a solo-exacto a ningun rho evaluado.")

    # figura: calibracion contra presupuesto, para el rho medido y dos hipoteticos
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rhos = [r for r in [rho_meas, 10.0, 100.0] if r]
    fig, axes = plt.subplots(1, len(rhos), figsize=(4.2 * len(rhos), 3.4), squeeze=False)
    for ax, rho in zip(axes[0], rhos):
        for arm in sorted({r["arm"] for r in rows}):
            pts = sorted([(r["n_e"] + r["n_s"] / rho, r["gap"])
                          for r in rows if r["arm"] == arm])
            if not pts: continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=arm, ms=4)
        ax.set_xscale("log"); ax.set_xlabel("presupuesto (simulaciones exactas equivalentes)")
        ax.set_ylabel("|cobertura - 0.90|"); ax.set_title(f"rho = {rho:.1f}")
        ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(a.plot, dpi=130, bbox_inches="tight")
    print("figura en", a.plot)


if __name__ == "__main__":
    main()
