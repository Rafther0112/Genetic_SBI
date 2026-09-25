"""
ts3_adapter.py  (v4)
====================
Puente entre threestate.py del repo y la interfaz que usan los scripts E3/E4/E5.

Firmas reales del repo (verificadas):
    fsp_stationary_pmf(theta, M=None)
    fsp_mean_fano(theta, M=None)                  -> (mean, fano)
    prom_stationary(k01, k10, k12, k21)           -> pi (3,)
    exact_mean(theta)
    sample_cle_3state_batched(k01,k10,k12,k21,k_syn, rng, t_max=25.0, dt=0.02)
    sample_hybrid_3state_batched(k01,k10,k12,k21,k_syn, rng, t_max=25.0, dt=0.02)
    sample_nb_3state(theta, n_cells, rng, M=None)
    sample_ssa(theta, n_cells, rng, t_max=40.0)

Calibracion automatica comprobada en el repo:
    el modulo espera TASAS (no log10) y el estado transcribente es el indice 2.

Cambio de v3 a v4: fsp_stationary_pmf devuelve (2, M): una fila con la
rejilla de conteos y otra con las probabilidades, no tres estados. Se detecta
la disposicion comparando con la media analitica.

Cambio de v2 a v3: fsp_stationary_pmf puede devolver la distribucion CONJUNTA
sobre (estado del promotor, numero de moleculas). Aplanarla, como hacia v2,
producia conteos disparatados (media 17x mayor y cero ceros). Ahora se
marginaliza sobre el estado del promotor y se verifica contra la media analitica.

Instalacion:
    poner este archivo junto a threestate.py
    en exp_common.py:  import threestate as TS3   ->   import ts3_adapter as TS3
Verificacion:
    python ts3_adapter.py
"""
import numpy as np
import torch
from sbi.utils import BoxUniform
import threestate as T
from telegraph import summary_stats

# ------------------------------------------------------------------ REVISAR
# Caja del prior sobre log10 de (k01, k10, k12, k21, k_syn).
# Copiar la del script con el que se genero la figura del tres-estados del paper.
LOW3 = torch.tensor([-2.0, -1.0, -2.0, -1.0, 0.0])
HIGH3 = torch.tensor([1.3, 1.3, 1.3, 1.3, 2.3])

_STATE = {"active": None, "log10": None}     # se detectan solos en _calibrate()


def make_prior3():
    return BoxUniform(low=LOW3, high=HIGH3)


# ------------------------------------------------------------------ momentos
def _Q(r):
    k01, k10, k12, k21 = r[:4]
    return np.array([[-k01, k01, 0.0],
                     [k10, -(k10 + k12), k12],
                     [0.0, k21, -k21]])


def _calibrate():
    """Detecta (a) si el repo espera tasas o log10 y (b) cual estado del
    promotor es el transcribente, usando exact_mean(theta) = k_syn * pi_A."""
    if _STATE["active"] is not None:
        return
    th_log = np.array([[-0.5, 0.0, -0.3, 0.2, 1.5],
                       [0.3, -0.2, 0.1, 0.5, 2.0],
                       [-1.2, 0.4, -0.8, 0.0, 1.0]])
    for log10 in (False, True):
        try:
            act, ok = [], []
            for t in th_log:
                r = 10.0 ** t
                m = float(np.asarray(T.exact_mean(t if log10 else r)).ravel()[0])
                pi = np.asarray(T.prom_stationary(*r[:4])).ravel()
                j = int(np.argmin(np.abs(m - r[4] * pi)))
                ok.append(abs(m - r[4] * pi[j]) / max(m, 1e-12) < 1e-6)
                act.append(j)
            if all(ok) and len(set(act)) == 1:
                _STATE["log10"], _STATE["active"] = log10, act[0]
                return
        except Exception:
            continue
    raise RuntimeError("no pude calibrar: revisar exact_mean / prom_stationary")


def _arg(theta_log):
    """theta en log10 -> lo que espera el repo (tasas o log10)."""
    _calibrate()
    t = np.asarray(theta_log, float)
    return t if _STATE["log10"] else 10.0 ** t


def _mean_fano_one(theta_log):
    """Ec. (5) del paper: momentos cerrados, sin FSP."""
    _calibrate()
    r = 10.0 ** np.asarray(theta_log, float)
    A = _STATE["active"]
    pi = np.asarray(T.prom_stationary(*r[:4])).ravel()
    piA = pi[A]
    M = np.linalg.inv(np.eye(3) - _Q(r))
    return r[4] * piA, 1.0 + r[4] * (piA * M[A, A] - piA ** 2) / piA


def analytic_fano3(theta):
    theta = np.atleast_2d(np.asarray(theta, float))
    return np.array([_mean_fano_one(t)[1] for t in theta])


def expected_counts3(theta):
    theta = np.atleast_2d(np.asarray(theta, float))
    return np.array([_mean_fano_one(t)[0] for t in theta])


# ------------------------------------------------------------------ pmf exacta
def _pmf_counts(arg, mean_ref, M=None):
    """Devuelve (valores, probabilidades) de la pmf MARGINAL sobre conteos.

    fsp_stationary_pmf devuelve un arreglo cuya disposicion no es fija: puede ser
    un vector de probabilidades, una conjunta (estado, m), o dos filas con la
    rejilla de conteos y sus probabilidades. Se enumeran las lecturas plausibles
    y se elige la que reproduce la media analitica. Si ninguna lo hace, se aborta
    mostrando el diagnostico de cada candidata, en vez de devolver conteos
    silenciosamente incorrectos.
    """
    p = np.asarray(T.fsp_stationary_pmf(arg, M), dtype=float)
    cands = []                                     # (nombre, valores, probs)
    if p.ndim == 1:
        cands.append(("vector de probabilidades", np.arange(len(p)), p))
    elif p.ndim == 2:
        n0, n1 = p.shape
        if n0 == 2:                                # dos filas: rejilla y pmf
            cands.append(("fila 0 = rejilla, fila 1 = pmf", p[0], p[1]))
            cands.append(("fila 1 = rejilla, fila 0 = pmf", p[1], p[0]))
        if n1 == 2:                                # dos columnas
            cands.append(("col 0 = rejilla, col 1 = pmf", p[:, 0], p[:, 1]))
            cands.append(("col 1 = rejilla, col 0 = pmf", p[:, 1], p[:, 0]))
        cands.append(("marginal sumando eje 0", np.arange(n1), p.sum(0)))
        cands.append(("marginal sumando eje 1", np.arange(n0), p.sum(1)))
    else:
        raise RuntimeError(f"fsp_stationary_pmf devolvio ndim={p.ndim}")

    tol = mean_ref + 0.05                          # tolerancia con piso, la media puede ser ~0
    best, err, how, diag = None, np.inf, None, []
    for name, v, q in cands:
        v = np.asarray(v, float).ravel(); q = np.asarray(q, float).ravel()
        if len(v) != len(q) or np.any(q < -1e-9) or q.sum() <= 0:
            diag.append(f"{name}: descartada"); continue
        qn = np.clip(q, 0, None); qn = qn / qn.sum()
        m = float((v * qn).sum())
        e = abs(m - mean_ref) / tol
        diag.append(f"{name}: media={m:.4f} (err rel {e:.3f})")
        if e < err:
            best, err, how = (v, qn), e, name
    if best is None or err > 0.05:
        raise RuntimeError(
            f"no recupero la pmf marginal de conteos (forma {np.shape(p)}, "
            f"media analitica {mean_ref:.4f}).\n   " + "\n   ".join(diag))
    _pmf_counts.how = how
    return best


_pmf_counts.how = None


# ------------------------------------------------------------------ muestreo
def _counts_batched(fn, theta, n_cells, rng, **kw):
    """Una sola llamada para las N*n_cells trayectorias (cle, hybrid)."""
    r = 10.0 ** np.asarray(theta, float)
    a = [np.repeat(r[:, j], n_cells) for j in range(5)]
    c = np.asarray(fn(a[0], a[1], a[2], a[3], a[4], rng, **kw))
    return c.reshape(len(theta), n_cells)


def simulate_batch3(theta, sim, n_cells, rng, features="full", M=None):
    """(N,5) log10 -> (N, 6) resumenes. sim en {fsp, ssa, cle, hybrid, nb}."""
    theta = np.atleast_2d(np.asarray(theta, float))
    if sim in ("cle", "hybrid"):
        fn = T.sample_cle_3state_batched if sim == "cle" else T.sample_hybrid_3state_batched
        counts = _counts_batched(fn, theta, n_cells, rng)
    else:
        counts = np.empty((len(theta), n_cells), dtype=float)
        for i, t in enumerate(theta):
            arg = _arg(t)
            if sim in ("fsp", "exact"):
                vals, probs = _pmf_counts(arg, _mean_fano_one(t)[0], M)
                counts[i] = rng.choice(vals, size=n_cells, p=probs)
            elif sim == "nb":
                counts[i] = np.asarray(T.sample_nb_3state(arg, n_cells, rng, M)).ravel()
            elif sim == "ssa":
                counts[i] = np.asarray(T.sample_ssa(arg, n_cells, rng)).ravel()
            else:
                raise ValueError(f"simulador desconocido: {sim}")
    return np.stack([summary_stats(c, features=features) for c in counts])


# ------------------------------------------------------------------ chequeos
if __name__ == "__main__":
    import time
    _calibrate()
    print(f"calibracion: el repo espera "
          f"{'log10' if _STATE['log10'] else 'tasas'}; "
          f"estado transcribente = indice {_STATE['active']}")

    th = make_prior3().sample((6,)).numpy()

    print("\n1) ec. (5) contra fsp_mean_fano:")
    for t in th:
        ma, fa = _mean_fano_one(t)
        out = np.asarray(T.fsp_mean_fano(_arg(t))).ravel()
        mf, ff = out[0], out[1]
        e = abs(fa - ff) / max(abs(ff), 1e-12)
        print(f"   mean {ma:9.3f} vs {mf:9.3f} | fano {fa:9.3f} vs {ff:9.3f} | "
              f"err {e:.1e} {'OK' if e < 1e-3 else '<-- REVISAR'}")

    print("\n2) muestreo (3 thetas, 200 celulas):")
    rng = np.random.default_rng(0)
    for s in ["fsp", "cle", "hybrid", "nb", "ssa"]:
        try:
            t0 = time.time()
            x = simulate_batch3(th[:3], s, 200, rng)
            print(f"   {s:7s} ok en {time.time()-t0:6.2f}s  resumenes[0] = {np.round(x[0], 3)}")
        except Exception as e:
            print(f"   {s:7s} FALLA: {type(e).__name__}: {e}")
    if _pmf_counts.how:
        print(f"   (pmf del FSP recuperada por: {_pmf_counts.how})")

    print("\n3) coherencia: media empirica de cada simulador contra la analitica")
    print("   (fsp y ssa son exactos y deben coincidir con la analitica; el resto")
    print("    puede desviarse, eso es la misspecificacion que el paper mide)")
    for t in th[:4]:
        ref = _mean_fano_one(t)[0]
        fila = []
        for s in ["fsp", "ssa", "hybrid", "nb", "cle"]:
            x = simulate_batch3(t[None, :], s, 2000, np.random.default_rng(1))
            fila.append(f"{s}={np.expm1(x[0, 0]):8.2f}")
        print(f"   analitica={ref:9.3f} | " + "  ".join(fila))

    print("\n4) fraccion de ceros (fsp y ssa deben coincidir):")
    for t in th[:4]:
        fila = []
        for s in ["fsp", "ssa", "hybrid", "nb", "cle"]:
            x = simulate_batch3(t[None, :], s, 2000, np.random.default_rng(2))
            fila.append(f"{s}={x[0, 3]:.3f}")
        print("   " + "  ".join(fila))