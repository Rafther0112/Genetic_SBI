"""
E3  --  Colocacion guiada frente a uniforme, en los dos sistemas.

Generaliza run_alloc3state.py: sirve para el telegrafo (exacto = cme) y para el
promotor de tres estados (exacto = fsp), con el parametro de evaluacion elegible.

La prediccion registrada en docs/prereg_E3.md, hecha ANTES de correr esto y solo
con el diagnostico de E4 (sin entrenar ningun NPE), es:

  telegraph/nb    sev 0.366  G50 0.86  -> la colocacion guiada DEBE ganar
  threestate/nb   sev 0.426  G50 0.69  -> la colocacion guiada DEBE ganar
  threestate/cle  sev 0.998  G50 0.00  -> ningun esquema debe ganar
  telegraph/lna   sev 0.936  G50 0.04  -> ningun esquema debe ganar

Esquemas (sobre los theta de entrenamiento, donde el Fano analitico se conoce):
  uniform, exact-only, fano-hard, fano-soft(2), fano-soft(6), inverse
y opcionalmente oracle, que usa el error real del sustituto y que un usuario no
podria calcular: es la cota superior de lo que cualquier esquema podria lograr.

Uso:
  python run_alloc.py --system telegraph  --exact cme --surrogate nb  --param 2
  python run_alloc.py --system threestate --exact fsp --surrogate nb  --param 4
  python run_alloc.py --system threestate --exact fsp --surrogate cle --param 4
  (--param es el indice de k_syn en theta: 2 en el telegrafo, 4 en tres estados)
"""
import argparse, os, numpy as np, torch
import exp_common as E
from exp_common import cached_sims, train_npe, evaluate, shared_test_set, append_row


def fano_of(theta, system):
    if system == "threestate":
        return np.asarray(E.TS3.analytic_fano3(theta))
    from telegraph import exact_fano
    r = 10.0 ** np.asarray(theta)
    return np.array([exact_fano(*ri) for ri in r])


def schemes(fano, k, rng, oracle_err=None):
    """Indices de los theta que reciben simulacion exacta, por esquema."""
    n = len(fano); order = np.argsort(-fano)
    out = {"uniform": rng.choice(n, k, replace=False),
           "fano-hard": order[:k],
           "inverse": order[::-1][:k]}
    rank = np.empty(n); rank[order] = np.linspace(1, 0, n)
    for a in (2, 6):
        w = 1 + a * rank; w = w / w.sum()
        out[f"fano-soft({a})"] = rng.choice(n, k, replace=False, p=w)
    if oracle_err is not None:
        out["oracle"] = np.argsort(-np.asarray(oracle_err))[:k]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="telegraph", choices=["telegraph", "threestate"])
    p.add_argument("--exact", default="cme")
    p.add_argument("--surrogate", default="nb")
    p.add_argument("--param", type=int, default=2,
                   help="indice de k_syn en theta: 2 (telegrafo) o 4 (tres estados)")
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--f", type=float, nargs="+", default=[0.25, 0.5])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--oracle", action="store_true",
                   help="agrega el esquema oraculo (cota superior, no implementable)")
    p.add_argument("--out", default="results/e3_alloc.jsonl")
    a = p.parse_args()

    tag = f"{a.system}/{a.surrogate}"
    for seed in a.seeds:
        th_e, x_e = cached_sims(a.exact, a.n_sims, a.n_cells, seed, system=a.system)
        th_s, x_s = cached_sims(a.surrogate, a.n_sims, a.n_cells, seed, system=a.system)
        assert torch.allclose(th_e, th_s), "los theta deben coincidir entre simuladores"
        tt, xt = shared_test_set(a.n_test, a.n_cells, seed, system=a.system,
                                 exact=a.exact)
        fano = fano_of(th_e.numpy(), a.system)

        # referencia: sustituto puro (presupuesto exacto = 0)
        de, _ = train_npe(th_s, x_s, system=a.system, seed=seed)
        r0 = evaluate(de, tt, xt, a.n_post, param=a.param)
        append_row(a.out, dict(tag=tag, seed=seed, f=0.0, scheme="surrogate",
                               system=a.system, surrogate=a.surrogate,
                               param=a.param, cov=r0["cov"], gap=abs(r0["cov"] - 0.9),
                               bias=r0["bias"]))

        # error del sustituto en cada theta de entrenamiento, para el oraculo
        oracle_err = None
        if a.oracle:
            oracle_err = np.empty(a.n_sims)
            for i in range(a.n_sims):
                s = E.flow_sample(de, x_e[i], 200).numpy()
                oracle_err[i] = abs(s[:, a.param].mean() - th_e[i, a.param].item())

        for f in a.f:
            k = int(round(f * a.n_sims))
            sel = schemes(fano, k, np.random.default_rng(seed), oracle_err)

            ex = sel["uniform"]
            de, _ = train_npe(th_e[ex], x_e[ex], system=a.system, seed=seed)
            r = evaluate(de, tt, xt, a.n_post, param=a.param)
            append_row(a.out, dict(tag=tag, seed=seed, f=f, scheme="exact-only",
                                   system=a.system, surrogate=a.surrogate,
                                   param=a.param, cov=r["cov"],
                                   gap=abs(r["cov"] - 0.9), bias=r["bias"]))

            for name, ex in sel.items():
                mask = np.zeros(a.n_sims, bool); mask[ex] = True
                th = torch.cat([th_e[mask], th_s[~mask]])
                x = torch.cat([x_e[mask], x_s[~mask]])
                de, _ = train_npe(th, x, system=a.system, seed=seed)
                r = evaluate(de, tt, xt, a.n_post, param=a.param)
                append_row(a.out, dict(tag=tag, seed=seed, f=f, scheme=name,
                                       system=a.system, surrogate=a.surrogate,
                                       param=a.param, cov=r["cov"],
                                       gap=abs(r["cov"] - 0.9), bias=r["bias"],
                                       mean_fano_exact=float(fano[ex].mean())))


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True); main()
