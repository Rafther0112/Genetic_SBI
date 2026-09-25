"""
exp_common.py
=============
Capa comun para los experimentos E1-E4 del paper.

ADAPTACION AL REPO (unico punto a revisar):
  Este modulo espera encontrar, en el repo del paper,
    * make_prior()                                      -> BoxUniform sobre log10 rates
    * simulate_batch(theta, sim, n_cells, rng)          -> (N, d_x) resumenes
    * los simuladores 'cme', 'lna', 'cle', 'nb'
  Si los nombres difieren, editar SOLO el bloque IMPORTS y la funcion simulate().
  Para el sistema de tres estados se espera un modulo threestate.py con
    * make_prior3(), simulate_batch3(theta, sim, n_cells, rng)  con sim en
      {'fsp','cle','nb','hybrid'}
    * analytic_fano3(theta)  (ec. 5 del paper; moment formula via Q)
  Si el modulo no existe, run_alloc3state.py aborta con un mensaje claro.

Todo lo demas (entrenamiento, afinado, muestreo, metricas) es autocontenido.
"""
import os, time, json, numpy as np, torch
import warnings; warnings.filterwarnings("ignore")

# ----------------------------- IMPORTS DEL REPO -----------------------------
from sbi_experiment import make_prior, simulate_batch          # <-- editar si cambian
try:
    import ts3_adapter as TS3
except Exception:
    TS3 = None
from sbi.inference import NPE

CACHE = os.environ.get("SBI_CACHE", "cache"); os.makedirs(CACHE, exist_ok=True)
PARAMS = ["k_on", "k_off", "k_syn"]


# ----------------------------- SIMULACION -----------------------------------
def simulate(theta, sim, n_cells, rng, system="telegraph"):
    if system == "telegraph":
        if sim == "hybrid":
            # el hibrido del telegrafo vive en run_hybrid.py, no en sbi_experiment
            import run_hybrid as H
            return H.simulate(theta, "hybrid", n_cells, rng)
        return simulate_batch(theta, sim, n_cells, rng)
    if TS3 is None:
        raise RuntimeError("Falta el adaptador de tres estados")
    return TS3.simulate_batch3(theta, sim, n_cells, rng)


def prior_of(system="telegraph"):
    return make_prior() if system == "telegraph" else TS3.make_prior3()


def cached_sims(sim, n_sims, n_cells, seed, system="telegraph"):
    """Conjunto (theta, x) cacheado. Un mismo seed reutiliza los MISMOS theta
    entre simuladores, de modo que exacto y sustituto sean comparables."""
    path = f"{CACHE}/{system}_{sim}_{n_sims}_{n_cells}_{seed}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return (torch.as_tensor(d["theta"], dtype=torch.float32),
                torch.as_tensor(d["x"], dtype=torch.float32))
    prior = prior_of(system)
    torch.manual_seed(seed)
    theta = prior.sample((n_sims,))
    rng = np.random.default_rng(seed)
    t0 = time.time()
    x = simulate(theta.numpy(), sim, n_cells, rng, system)
    print(f"[sim] {system}/{sim} n={n_sims} en {time.time()-t0:.0f}s")
    np.savez(path, theta=theta.numpy(), x=x)
    return theta, torch.as_tensor(x, dtype=torch.float32)


# ----------------------------- NPE ------------------------------------------
def train_npe(theta, x, system="telegraph", seed=0, **kw):
    torch.manual_seed(seed); np.random.seed(seed)
    inf = NPE(prior=prior_of(system))
    inf.append_simulations(theta, x)
    de = inf.train(**kw)
    return de, inf


def finetune(de, theta, x, lr=1e-4, epochs=200, batch=200, val_frac=0.1,
             patience=20, seed=0, verbose=False):
    """Segunda etapa: continua el entrenamiento del mismo estimador de densidad
    sobre pares nuevos (aqui, exactos). Es el esquema preentrenar-y-afinar de
    Krouglova et al. (2026), sin mezclar ambos simuladores en un solo objetivo."""
    torch.manual_seed(seed)
    n = len(theta); idx = torch.randperm(n)
    nv = max(1, int(val_frac * n)); vi, ti = idx[:nv], idx[nv:]
    opt = torch.optim.Adam(de.parameters(), lr=lr)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        de.train()
        perm = ti[torch.randperm(len(ti))]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = -de.log_prob(theta[b], condition=x[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        de.eval()
        with torch.no_grad():
            v = -de.log_prob(theta[vi], condition=x[vi]).mean().item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in de.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
        if verbose and ep % 10 == 0: print(f"   ep{ep} val={v:.3f}")
    if best_state is not None: de.load_state_dict(best_state)
    return de


def flow_sample(de, x_row, n_post=1000):
    with torch.no_grad():
        s = de.sample((n_post,), condition=x_row.unsqueeze(0))
    return s.reshape(n_post, -1)


# ----------------------------- METRICAS -------------------------------------
def evaluate(de, theta_test, x_test, n_post=1000, param=0):
    """Cobertura del 90%, sesgo firmado y contraccion, para un parametro."""
    tt = theta_test.numpy()
    cov = np.empty(len(tt)); bias = np.empty(len(tt)); sd = np.empty(len(tt))
    for i in range(len(tt)):
        s = flow_sample(de, x_test[i], n_post).numpy()
        u = (s[:, param] < tt[i, param]).mean()
        cov[i] = float(0.05 <= u <= 0.95)
        bias[i] = s[:, param].mean() - tt[i, param]
        sd[i] = s[:, param].std()
    return dict(cov=cov.mean(), bias=bias.mean(), post_sd=sd.mean(),
                cov_raw=cov, bias_raw=bias)


def shared_test_set(n_test, n_cells, seed, system="telegraph", exact="cme"):
    prior = prior_of(system); torch.manual_seed(10_000 + seed)
    tt = prior.sample((n_test,))
    rng = np.random.default_rng(10_000 + seed)
    x = torch.as_tensor(simulate(tt.numpy(), exact, n_cells, rng, system),
                        dtype=torch.float32)
    return tt, x


def append_row(path, row):
    with open(path, "a") as f: f.write(json.dumps(row) + "\n")
    print("  ->", json.dumps(row))
