"""
E3  --  El otro signo de la ley de localizacion.

En el telegrafo el LNA esta mal calibrado en TODOS los sextiles, asi que ninguna
asignacion le gana al azar. En el promotor de tres estados el NB si esta
localizado (cobertura 0.93 en el sextil menos rafaga y 0.68 en el penultimo,
Sec. 5.7 del paper). La prediccion es que ALLI la asignacion guiada por Fano si
debe superar a la uniforme. Si sale, C2 deja de ser un negativo y pasa a ser una
ley con los dos signos.

Esquemas (identicos a los del paper, sobre los theta de entrenamiento, donde el
Fano analitico es conocido):
   uniform, exact-only, fano-hard, fano-soft(a), inverse

Uso:
  python run_alloc3state.py --f 0.25 0.50 --seeds 0 1 2 --surrogate nb
  python run_alloc3state.py --f 0.25 0.50 --seeds 0 1 2 --surrogate cle   # control
"""
import argparse, numpy as np, torch
import exp_common as E
from exp_common import cached_sims, train_npe, evaluate, shared_test_set, append_row

SYS = "threestate"


def schemes(fano, k, rng):
    """Indices que reciben simulacion exacta, por esquema."""
    n = len(fano); order = np.argsort(-fano)              # mayor Fano primero
    out = {"uniform": rng.choice(n, k, replace=False),
           "fano-hard": order[:k],
           "inverse": order[::-1][:k]}
    rank = np.empty(n); rank[order] = np.linspace(1, 0, n)   # 1 = mas bursty
    for a in (2, 6):
        w = 1 + a * rank; w = w / w.sum()
        out[f"fano-soft({a})"] = rng.choice(n, k, replace=False, p=w)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--surrogate", default="nb", choices=["nb", "cle", "hybrid"])
    p.add_argument("--param", type=int, default=4, help="indice de k_syn en theta")
    p.add_argument("--f", type=float, nargs="+", default=[0.25, 0.5])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results/e3_alloc3state.jsonl")
    a = p.parse_args()
    if E.TS3 is None:
        raise SystemExit("Falta threestate.py (simulate_batch3, analytic_fano3, make_prior3)")

    for seed in a.seeds:
        th_e, x_e = cached_sims("fsp", a.n_sims, a.n_cells, seed, system=SYS)
        th_s, x_s = cached_sims(a.surrogate, a.n_sims, a.n_cells, seed, system=SYS)
        tt, xt = shared_test_set(a.n_test, a.n_cells, seed, system=SYS, exact="fsp")
        fano = np.asarray(E.TS3.analytic_fano3(th_e.numpy()))
        rng = np.random.default_rng(seed)

        # sustituto puro, como referencia
        de, _ = train_npe(th_s, x_s, system=SYS, seed=seed)
        r = evaluate(de, tt, xt, a.n_post, param=a.param)
        append_row(a.out, dict(seed=seed, f=0.0, scheme="surrogate",
                               surrogate=a.surrogate, cov=r["cov"], bias=r["bias"]))

        for f in a.f:
            k = int(round(f * a.n_sims))
            sel = schemes(fano, k, np.random.default_rng(seed))
            # solo exacto: el mismo subconjunto uniforme, sin el sustituto
            ex = sel["uniform"]
            de, _ = train_npe(th_e[ex], x_e[ex], system=SYS, seed=seed)
            r = evaluate(de, tt, xt, a.n_post, param=a.param)
            append_row(a.out, dict(seed=seed, f=f, scheme="exact-only",
                                   surrogate=a.surrogate, cov=r["cov"],
                                   gap=abs(r["cov"] - 0.9), bias=r["bias"]))
            for name, ex in sel.items():
                mask = np.zeros(a.n_sims, bool); mask[ex] = True
                th = torch.cat([th_e[mask], th_s[~mask]])
                x = torch.cat([x_e[mask], x_s[~mask]])
                de, _ = train_npe(th, x, system=SYS, seed=seed)
                r = evaluate(de, tt, xt, a.n_post, param=a.param)
                append_row(a.out, dict(seed=seed, f=f, scheme=name,
                                       surrogate=a.surrogate, cov=r["cov"],
                                       gap=abs(r["cov"] - 0.9), bias=r["bias"],
                                       mean_fano_exact=float(fano[ex].mean())))


if __name__ == "__main__":
    import os; os.makedirs("results", exist_ok=True); main()
