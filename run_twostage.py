"""
E1  --  ¿El sesgo viene del simulador barato o de mezclarlo en un solo objetivo?

A presupuesto exacto igual (f * N simulaciones exactas) compara:
   exact-only   : entrena solo con las f*N exactas
   mixture      : una etapa, f*N exactas + (1-f)*N del sustituto (lo del paper)
   two-stage    : preentrena con N del sustituto y afina con las f*N exactas
                  (esquema de Krouglova et al., ICLR 2026)
   surrogate    : referencia, solo sustituto

Es el experimento que responde la objecion previsible: "nadie mezcla en un solo
objetivo, MF-NPE usa dos etapas". La Proposicion 1 no aplica a two-stage, asi que
su resultado decide si el mensaje del paper es "el sustituto no sirve" o
"el sustituto no debe entrar al mismo objetivo".

Uso:
  python run_twostage.py --f 0.25 0.50 0.75 --seeds 0 1 2 3 4 --lr 1e-4
  python run_twostage.py --f 0.25 --seeds 0 --lr 3e-4 1e-4 3e-5    # barrido de lr
"""
import argparse, numpy as np, torch
from exp_common import (cached_sims, train_npe, finetune, evaluate,
                        shared_test_set, append_row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--surrogate", default="lna", choices=["lna", "cle", "nb"])
    p.add_argument("--f", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--lr", type=float, nargs="+", default=[1e-4])
    p.add_argument("--out", default="results/e1_twostage.jsonl")
    a = p.parse_args()

    for seed in a.seeds:
        th_e, x_e = cached_sims("cme", a.n_sims, a.n_cells, seed)
        th_s, x_s = cached_sims(a.surrogate, a.n_sims, a.n_cells, seed)
        assert torch.allclose(th_e, th_s), "los theta deben coincidir entre simuladores"
        tt, xt = shared_test_set(a.n_test, a.n_cells, seed)

        # referencia: solo sustituto, presupuesto exacto = 0
        de_s, _ = train_npe(th_s, x_s, seed=seed)
        r = evaluate(de_s, tt, xt, a.n_post)
        append_row(a.out, dict(seed=seed, f=0.0, arm="surrogate", lr=None,
                               cov=r["cov"], bias=r["bias"], post_sd=r["post_sd"]))

        for f in a.f:
            k = int(round(f * a.n_sims))
            g = torch.Generator().manual_seed(1000 + seed)
            idx = torch.randperm(a.n_sims, generator=g)
            ex, sur = idx[:k], idx[k:]          # los exactos y los del sustituto

            # 1) solo exacto
            de, _ = train_npe(th_e[ex], x_e[ex], seed=seed)
            r = evaluate(de, tt, xt, a.n_post)
            append_row(a.out, dict(seed=seed, f=f, arm="exact-only", lr=None,
                                   n_exact=k, cov=r["cov"], bias=r["bias"],
                                   post_sd=r["post_sd"]))

            # 2) mezcla de una etapa
            th_m = torch.cat([th_e[ex], th_s[sur]]); x_m = torch.cat([x_e[ex], x_s[sur]])
            de, _ = train_npe(th_m, x_m, seed=seed)
            r = evaluate(de, tt, xt, a.n_post)
            append_row(a.out, dict(seed=seed, f=f, arm="mixture", lr=None,
                                   n_exact=k, cov=r["cov"], bias=r["bias"],
                                   post_sd=r["post_sd"]))

            # 3) dos etapas: preentrenar con TODO el sustituto, afinar con las exactas
            for lr in a.lr:
                de_pre, _ = train_npe(th_s, x_s, seed=seed)
                de_ft = finetune(de_pre, th_e[ex], x_e[ex], lr=lr, seed=seed)
                r = evaluate(de_ft, tt, xt, a.n_post)
                append_row(a.out, dict(seed=seed, f=f, arm="two-stage", lr=lr,
                                       n_exact=k, cov=r["cov"], bias=r["bias"],
                                       post_sd=r["post_sd"]))


if __name__ == "__main__":
    import os; os.makedirs("results", exist_ok=True); main()
