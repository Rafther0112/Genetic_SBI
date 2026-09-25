"""
plot_diagnostic.py
==================
Figura del diagnostico previo (dos paneles, ancho de la caja de texto).

  A  Por region del prior, dos curvas por sustituto: la discrepancia con el
     simulador exacto (linea solida) y la masa no cubierta, la fraccion de datos
     exactos que el sustituto practicamente nunca produce (punteada). Solo el
     telegrafo, para que el estilo de linea signifique una sola cosa. Los dos
     modos de falla quedan a la vista: la CLE y el LNA entran saturados y con
     masa no cubierta alta en todas las regiones (soporte), mientras el NB sube
     desde cero con masa no cubierta casi nula (forma).
  B  Predicho contra medido: un punto por par sustituto-sistema. El color da el
     sustituto y la forma el sistema, asi que no hace falta rotular cada punto.

Decisiones de diseno tomadas tras mirar la version anterior:
  * el tamano del marcador ya no codifica la masa no cubierta: los marcadores
    grandes tapaban la propia linea;
  * las curvas se rotulan junto a su extremo derecho en vez de con una leyenda
    encima de los datos;
  * el hibrido no aparece en A porque solo existe en el sistema de tres estados,
    y dibujarlo pegado a cero solo anadia una entrada de leyenda invisible.

Las coberturas medidas vienen de las tablas del paper: editar MEASURED si cambian.

Uso:
  python plot_diagnostic.py --e4 results/e4_localization.jsonl
"""
import argparse, json, os, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from figstyle import (apply, SIM, GREY, RULE, FIT, finish, panel_label, note,
                      save, square, log_minor)

MEASURED = {("telegraph", "nb"): 0.91, ("telegraph", "lna"): 0.51,
            ("telegraph", "cle"): 0.18, ("threestate", "nb"): 0.79,
            ("threestate", "cle"): 0.23, ("threestate", "hybrid"): 0.93}
NAME = {"cle": "CLE", "lna": "LNA", "nb": "NB", "hybrid": "hybrid"}
SYSNAME = {"telegraph": "telegraph", "threestate": "three-state"}
MK = {"telegraph": "o", "threestate": "s"}
EXACT = ("cme", "fsp")


def load(fn):
    d = defaultdict(list)
    for l in open(fn):
        if l.strip():
            r = json.loads(l)
            d[(r["system"], r["surrogate"])].append(r["per_bin"])
    return d


def tv(bins):
    return np.clip(2 * np.array([b["auc"] for b in bins]) - 1, 0, 1)


def mean_over(runs, key):
    return np.mean([[b[key] for b in run] for run in runs], axis=0)


def panel_bins(ax, D, system="telegraph"):
    xmax = 0
    for (sysname, sur), runs in sorted(D.items()):
        if sysname != system or sur in EXACT:
            continue
        fano = mean_over(runs, "fano"); xmax = max(xmax, fano[-1])
        t = np.mean([tv(r) for r in runs], axis=0)
        u = mean_over(runs, "uncovered")
        ax.plot(fano, t, "-", color=SIM[sur], lw=1.4, zorder=3)
        ax.plot(fano, u, ":", color=SIM[sur], lw=1.4, zorder=3)
        ax.plot(fano, t, "o", color=SIM[sur], ms=2.6, zorder=4)
    ctrl = [k for k in D if k[0] == system and k[1] in EXACT]
    if ctrl:
        fano = mean_over(D[ctrl[0]], "fano")
        ax.plot(fano, np.mean([tv(r) for r in D[ctrl[0]]], axis=0), "-",
                color=GREY, lw=1.1, zorder=2)


    ax.set_xscale("log"); ax.set_ylim(-0.04, 1.08); ax.set_xlim(right=xmax * 1.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("burstiness of the region\n(analytic Fano factor, log scale)")
    ax.set_ylabel("fraction")
    panel_label(ax, "A", "Discrepancy vs. burstiness")
    log_minor(ax, "x"); square(ax); finish(ax, grid="y")


def panel_pred(ax, D):
    ax.axhline(0.90, ls=(0, (4, 2)), color=GREY, lw=1, zorder=1)
    ax.annotate("nominal", xy=(1.07, 0.90), xytext=(0, 3),
                textcoords="offset points", color=GREY, fontsize=6.5, ha="right")
    for (sysname, sur), runs in sorted(D.items()):
        if (sysname, sur) not in MEASURED:
            continue
        vals = [tv(r).mean() for r in runs]
        x, xs = float(np.mean(vals)), float(np.std(vals))
        ax.errorbar(x, MEASURED[(sysname, sur)], xerr=xs, fmt=MK[sysname],
                    color=SIM[sur], mfc="white", mew=1.4, ms=5.5, capsize=2,
                    elinewidth=0.9, zorder=3)
    ax.set_xlim(-0.06, 1.12); ax.set_ylim(0.05, 1.02)
    ax.set_xlabel("predicted before training:\nprior-averaged discrepancy")
    ax.set_ylabel("measured after training:\n$90\\%$ interval coverage")

    panel_label(ax, "B", "Predicted vs. measured")
    square(ax); finish(ax)


def legend_outside(fig, D, system):
    """Una sola leyenda para los dos paneles, fuera del area de datos.

    Dentro de los ejes, las entradas se leen como anotaciones sobre los datos;
    a la derecha se leen como lo que son, la clave de la figura.
    """
    sur = [s for (sy, s) in D if sy == system and s not in EXACT]
    h = [Line2D([], [], color=SIM[s], marker="o", ls="-", lw=1.4, ms=3.4,
                label=NAME[s]) for s in ["cle", "lna", "nb"] if s in sur]
    h.append(Line2D([], [], color=SIM["hybrid"], marker="o", ls="", mfc="white",
                    mew=1.4, ms=5, label=NAME["hybrid"] + " (B only)"))
    h.append(Line2D([], [], color=GREY, ls="-", lw=1.1, label="exact vs exact"))
    h.append(Line2D([], [], color="none", label=" "))
    h.append(Line2D([], [], color=FIT, ls="-", lw=1.3, label="discrepancy (A)"))
    h.append(Line2D([], [], color=FIT, ls=":", lw=1.3, label="uncovered mass (A)"))
    h.append(Line2D([], [], color="none", label=" "))
    h += [Line2D([], [], color=FIT, marker=MK[s], ls="", mfc="white", mew=1.3,
                 ms=5, label=SYSNAME[s] + " (B)") for s in ["telegraph", "threestate"]]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.775, 0.52),
               frameon=False, handlelength=1.5, labelspacing=0.32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e4", default="results/e4_localization.jsonl")
    p.add_argument("--system", default="telegraph")
    p.add_argument("--out", default="figures/diagnostic.pdf")
    a = p.parse_args()
    apply()
    D = load(a.e4)
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.5))
    panel_bins(ax[0], D, a.system)
    panel_pred(ax[1], D)
    fig.tight_layout(pad=0.4, w_pad=1.3, rect=(0, 0, 0.76, 1))
    legend_outside(fig, D, a.system)
    save(fig, a.out)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True); main()