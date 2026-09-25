"""
figstyle.py
===========
Estilo unico para todas las figuras del paper. Importar y llamar apply() antes
de dibujar. La idea es que el lector aprenda el codigo de color una sola vez:
cada simulador conserva su color en las cuatro figuras.

    from figstyle import apply, SIM, GREY, finish, panel_label
    apply()

Paleta (verde, azul, naranja, morado, mas dos neutros):
    verde   exacto y sus controles: lo que funciona
    azul    NB, el sustituto de conteo
    morado  LNA
    naranja CLE
    gris    baselines sin informacion (azar, uniforme, prior)
    rojo    solo para el control invertido, que existe para mostrar dano

Tipografia serif para que las figuras no desentonen con el cuerpo del paper,
que va en Times. Si el sistema no tiene STIX, matplotlib cae en DejaVu Serif
sin romper nada.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- paleta
SIM = {
    "exact":     "#14695A",   # verde profundo
    "cme":       "#14695A",
    "fsp":       "#14695A",
    "hybrid":    "#4FA98B",   # verde claro: control que repara
    "nb":        "#2F6DB5",   # azul
    "lna":       "#7A4BA8",   # morado
    "cle":       "#D98324",   # naranja
}
# Los esquemas de colocacion NO reutilizan los colores de simulador: alli el azul
# es el NB y el morado el LNA, y repetirlos aqui rompe el codigo de color. Un solo
# acento verde en dos tonos para lo guiado, gris para el azar y rojo para el control
# que existe para mostrar dano.
SCHEME = {
    "uniform":      "#8C8C8C",
    "exact-only":   "#14695A",
    "fano-soft(2)": "#14695A",
    "fano-soft(6)": "#4FA98B",
    "fano-hard":    "#D98324",
    "inverse":      "#B03A2E",
}
ARM = {"exact-only": "#14695A", "mixture": "#D98324",
       "two-stage": "#2F6DB5", "surrogate": "#8C8C8C"}
GREY, RULE, FIT = "#8C8C8C", "#5A5A5A", "#333333"

LABEL = {"cme": "CME", "fsp": "FSP", "nb": "NB", "lna": "LNA", "cle": "CLE",
         "hybrid": "hybrid", "exact": "exact"}


def apply(base=8):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1.5,
        "axes.linewidth": 0.7,
        "axes.labelpad": 3.5,
        "axes.titlepad": 6,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,
        "figure.figsize": (5.5, 2.6),
        "figure.dpi": 160,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # fuentes incrustadas, no trazos
    })


def finish(ax, grid="y"):
    """Quita el marco superior y derecho y pone una rejilla discreta."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#4A4A4A")
    if grid:
        ax.grid(axis=grid, color="#D8D8D8", lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
    ax.tick_params(colors="#4A4A4A")


# "descriptive" (por defecto): la letra y QUE se grafica, p.ej. "Bias vs surrogate
# share". "letter": solo la letra. La conclusion va en el caption en los dos casos:
# un titulo que afirma un resultado obliga al revisor a leer dos veces la misma
# afirmacion, una en la figura y otra en el pie.
TITLES = "descriptive"


def panel_label(ax, letter, text=None, dy=1.02):
    """Etiqueta de panel fuera del area de datos."""
    ax.set_title("")
    txt = f"$\\bf{{{letter}}}$" if (TITLES == "letter" or not text) \
        else f"$\\bf{{{letter}}}$   {text}"
    ax.text(0.0, dy, txt, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=mpl.rcParams["axes.titlesize"])


def square(ax, aspect=1.0):
    """Caja cuadrada. Importa donde la pendiente se lee a ojo, como en un
    ajuste log-log: una caja achatada distorsiona la pendiente aparente."""
    ax.set_box_aspect(aspect)


def log_minor(ax, which="both"):
    """Marcas menores en los ejes logaritmicos, para que la escala se vea."""
    from matplotlib.ticker import LogLocator, NullFormatter
    for name in (("x", "y") if which == "both" else (which,)):
        axis = getattr(ax, name + "axis")
        if getattr(ax, "get_" + name + "scale")() == "log":
            axis.set_minor_locator(LogLocator(subs=np.arange(2, 10) * 0.1))
            axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=1.8, color="#7A7A7A")


def note(ax, text, xy=(0.98, 0.04), **kw):
    """Anotacion discreta dentro del panel (unidades, n, advertencias)."""
    ax.text(*xy, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=mpl.rcParams["legend.fontsize"] - 0.5, color="#4A4A4A", **kw)


def save(fig, path):
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print("guardada", path)
