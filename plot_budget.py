"""
plot_budget.py
==============
Dos figuras, cada una disenada al ancho real de la caja de texto (5.5 in), para que
no haya que reescalarlas: escalar una figura reduce tambien su tipografia.

  budget.pdf     A  ley del sesgo (log-log, con el ajuste)
                 B  distancia a la nominal contra el presupuesto exacto
  placement.pdf     asimetria de la colocacion, un panel ancho y bajo

Uso:
  python plot_budget.py --e1 results/e1_twostage.jsonl --e2 results/e2_crossover.jsonl \
                        --e3 results/e3_alloc.jsonl --tag telegraph/nb
"""
import argparse, json, os, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedLocator, FuncFormatter, MultipleLocator,
                               NullFormatter)
from figstyle import (apply, ARM, SCHEME, GREY, RULE, FIT, finish, panel_label,
                      note, save, square, log_minor)

NOMINAL = 0.90
fmt = FuncFormatter(lambda v, _: f"{v:g}")


def load(*fns):
    r = []
    for fn in fns:
        r += [json.loads(l) for l in open(fn) if l.strip()]
    return r


def agg(rows, keyf, field, keep=lambda r: True):
    d = defaultdict(list)
    for r in rows:
        if keep(r) and r.get(field) is not None:
            d[keyf(r)].append(r[field])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in d.items()}


def panel_scaling(ax, rows):
    m = agg(rows, lambda r: r["f"], "bias", lambda r: r.get("arm") == "mixture")
    f = np.array(sorted(m))
    b = np.array([abs(m[k][0]) for k in f]); e = np.array([m[k][1] for k in f])
    x = 1 - f
    k, loga = np.polyfit(np.log(x), np.log(b), 1)
    r = np.corrcoef(np.log(x), np.log(b))[0, 1]

    xs = np.geomspace(0.22, 1.02, 60)
    ax.plot(xs, np.exp(loga) * xs ** k, color=FIT, lw=1.0, zorder=2)
    ax.errorbar(x, b, yerr=e, fmt="o", color=ARM["mixture"], mfc="white",
                mew=1.2, ms=4, capsize=2, elinewidth=0.9, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.22, 1.12); ax.set_ylim(0.045, 0.62)
    ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 0.75, 1.0]))
    ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.4]))
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(fmt); a.set_minor_formatter(NullFormatter())
    ax.set_xlabel("surrogate share $1-f$   (log scale)")
    ax.set_ylabel("bias in $k_{\\mathrm{on}}$ ($\\log_{10}$ units, log scale)")
    ax.text(0.97, 0.06, f"$|b|={np.exp(loga):.2f}\\,(1-f)^{{{k:.2f}}}$,  $r={r:.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=FIT)
    panel_label(ax, "A", "Bias vs. surrogate share")
    log_minor(ax); square(ax); finish(ax, grid="both")


def panel_arms(ax, rows):
    for arm, lab in [("exact-only", "exact only"),
                     ("mixture", "mixture"),
                     ("two-stage", "two-stage")]:
        m = agg(rows, lambda r: r["f"], "cov", lambda r, a=arm: r.get("arm") == a)
        if not m:
            continue
        f = np.array(sorted(m))
        g = np.array([abs(m[k][0] - NOMINAL) for k in f])
        sd = np.array([m[k][1] for k in f])
        few = np.array([m[k][2] < 3 for k in f])
        ax.fill_between(f, np.maximum(g - sd, 0), g + sd, color=ARM[arm],
                        alpha=0.12, lw=0, zorder=1)
        ax.plot(f, g, "-", color=ARM[arm], lw=1.3, zorder=3, label=lab)
        ax.plot(f[~few], g[~few], "o", color=ARM[arm], mfc="white", mew=1.2, ms=3.6, zorder=4)
        ax.plot(f[few], g[few], "o", color=ARM[arm], ms=3.6, zorder=4)

    sm = agg(rows, lambda r: "s", "cov", lambda r: r.get("arm") == "surrogate")
    ax.set_xscale("log")
    ax.set_xlim(0.017, 0.95); ax.set_ylim(0, 0.235)
    ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.25, 0.5, 0.75]))
    ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("exact budget, fraction $f$ of $8{,}000$   (log scale)")
    ax.set_ylabel("$|\\,$coverage $-\\,0.90\\,|$")
    if sm:
        ax.annotate(f"surrogate alone: {abs(sm['s'][0]-NOMINAL):.2f}", xy=(0.97, 0.93),
                    xycoords="axes fraction", ha="right", va="top", fontsize=7,
                    color=GREY)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88))
    panel_label(ax, "B", "Calibration vs. exact budget")
    log_minor(ax, "x"); square(ax); finish(ax)


def fig_placement(rows, tag, f_show, out):
    base = {(r["seed"], r["f"]): r["cov"] for r in rows
            if r.get("scheme") == "uniform" and r.get("tag") == tag}
    d = defaultdict(list)
    for r in rows:
        if r.get("tag") != tag or r.get("scheme") in (None, "uniform", "surrogate"):
            continue
        b0 = base.get((r["seed"], r["f"]))
        if b0 is None or abs(r["f"] - f_show) > 1e-9:
            continue
        d[r["scheme"]].append(abs(b0 - NOMINAL) - abs(r["cov"] - NOMINAL))

    order = [k for k in ["fano-soft(2)", "fano-soft(6)", "fano-hard", "exact-only",
                         "inverse"] if k in d]
    # Etiquetas cortas: las largas empujan el eje a la derecha y descuadran el
    # titulo. El detalle de cada esquema va en el texto.
    pretty = {"fano-soft(2)": "Fano, $a=2$",
              "fano-soft(6)": "Fano, $a=6$",
              "fano-hard": "top Fano only",
              "exact-only": "no surrogate",
              "inverse": "reversed"}
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ys = np.arange(len(order))[::-1]
    for y, k in zip(ys, order):
        v = np.array(d[k]); col = SCHEME.get(k, FIT)
        ax.errorbar(v.mean(), y, xerr=v.std(), fmt="o", color=col,
                    mfc="white" if len(v) > 2 else col, mew=1.3, ms=5,
                    capsize=2.5, elinewidth=0.9, zorder=3)
    ax.axvline(0, color=RULE, lw=0.9, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels([pretty[k] for k in order])
    ax.set_ylim(-0.55, len(order) - 0.45)
    lo = min(np.mean(d[k]) - np.std(d[k]) for k in order)
    hi = max(np.mean(d[k]) + np.std(d[k]) for k in order)
    pad = 0.10 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    # las flechas viven DENTRO de la etiqueta del eje: como anotaciones sueltas
    # chocaban con los numeros de las marcas
    ax.set_xlabel("worse $\\leftarrow$   closer to nominal than random placement"
                  "   $\\rightarrow$ better")
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    finish(ax, grid="x")
    fig.tight_layout(pad=0.4)
    # titulo centrado sobre la figura: anclado al eje quedaria descuadrado, porque
    # las etiquetas de categoria desplazan el eje a la derecha
    fig.text(0.5, 0.985, "Guided vs. random placement", ha="center", va="top",
             fontsize=matplotlib.rcParams["axes.titlesize"])
    fig.subplots_adjust(top=0.86)
    save(fig, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e1", default="results/e1_twostage.jsonl")
    p.add_argument("--e2", default="results/e2_crossover.jsonl")
    p.add_argument("--e3", default="results/e3_alloc.jsonl")
    p.add_argument("--tag", default="telegraph/nb")
    p.add_argument("--f", type=float, default=0.25)
    p.add_argument("--out", default="figures/budget.pdf")
    p.add_argument("--out_place", default="figures/placement.pdf")
    a = p.parse_args()
    apply()
    r12 = load(a.e1, a.e2)
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.9))
    panel_scaling(ax[0], r12)
    panel_arms(ax[1], r12)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    save(fig, a.out)
    fig_placement(load(a.e3), a.tag, a.f, a.out_place)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True); main()