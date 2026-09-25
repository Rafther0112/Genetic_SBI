"""
summarize.py  (v3)
==================
Agrega los .jsonl de E1-E5 e imprime las tablas listas para el tex.

CORRECCION DE v2 -> v3
----------------------
Las diferencias pareadas se calculaban con la clave (semilla, f). Cuando un
mismo archivo contiene varias configuraciones (telegraph/nb, threestate/nb,
threestate/cle), el uniform de una configuracion quedaba emparejado con los
esquemas de otra, y los resultados eran basura: p.ej. inverse aparecia ganando
cuando en crudo era claramente peor. Ahora la clave incluye ademas el sistema,
el simulador sustituto y el parametro evaluado.

Tambien: todas las comparaciones se hacen en |cov - 0.90| y en |sesgo|, y se
reporta n, porque con una o dos semillas no se puede afirmar nada.

Uso:
  python summarize.py results/e3_alloc.jsonl
  python summarize.py results/e1_twostage.jsonl results/e2_crossover.jsonl
"""
import sys, json, numpy as np
from collections import defaultdict


def load(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def unit_of(r):
    """Identifica la corrida: todo menos el esquema/brazo comparado."""
    return (r.get("tag") or r.get("system", ""), r.get("surrogate", ""),
            r.get("param", ""), r.get("seed"), r.get("f"), r.get("budget_s"))


def armfield_of(rows):
    return "scheme" if any("scheme" in r for r in rows) else "arm"


def groupfields_of(rows, arm):
    g = []
    if any("tag" in r for r in rows): g.append("tag")
    elif any("system" in r for r in rows): g.append("system")
    if any("budget_s" in r for r in rows): g.append("budget_s")
    else: g.append("f")
    g.append(arm)
    if any(r.get("lr") for r in rows): g.append("lr")
    return g


def _agg(rows, keys, field):
    D = defaultdict(list)
    for r in rows:
        if r.get(field) is not None:
            D[tuple(r.get(k) for k in keys)].append(r[field])
    return D


def _print(D, title, fmt="{:7.3f}"):
    if not D: return
    print(f"\n  {title}")
    for k in sorted(D, key=lambda t: tuple(str(x) for x in t)):
        v = np.array(D[k], float)
        print(f"    {' | '.join(str(x) for x in k):40s} " + fmt.format(v.mean()) +
              f" +/- {v.std():.3f}  (n={len(v)})")


def _paired(rows, base_arm, arm, keys, field, center=None):
    """Diferencia pareada contra el brazo de referencia, dentro de la MISMA
    corrida (sistema, sustituto, parametro, semilla, presupuesto).
    Positivo = el esquema es mejor que la referencia."""
    dev = (lambda v: abs(v - center)) if center is not None else abs
    base = {}
    for r in rows:
        if r.get(arm) == base_arm and r.get(field) is not None:
            base[unit_of(r)] = r[field]
    P = defaultdict(list)
    for r in rows:
        if r.get(arm) == base_arm or r.get(field) is None:
            continue
        b = base.get(unit_of(r))
        if b is None:
            continue
        P[tuple(r.get(k) for k in keys)].append(dev(b) - dev(r[field]))
    return P


def _print_paired(P, title, note=""):
    if not P: return
    print(f"\n  {title}")
    if note: print(f"  {note}")
    for k in sorted(P, key=lambda t: tuple(str(x) for x in t)):
        v = np.array(P[k], float)
        if len(v) < 2:
            sig = "1 semilla, no concluyente"
        elif v.mean() - v.std() > 0:
            sig = "GANA"
        elif v.mean() + v.std() < 0:
            sig = "pierde"
        else:
            sig = "empata"
        print(f"    {' | '.join(str(x) for x in k):40s} {v.mean():+.3f} "
              f"+/- {v.std():.3f}  (n={len(v)})  {sig}")


def main(fn):
    rows = load(fn)
    if not rows:
        print(f"\n=== {fn}: vacio ==="); return
    arm = armfield_of(rows)
    keys = groupfields_of(rows, arm)
    base = "uniform" if any(r.get(arm) == "uniform" for r in rows) else "exact-only"
    print(f"\n=== {fn}  (referencia: {base}) ===")

    _print(_agg(rows, keys, "cov"), "cobertura 90%")
    _print(_agg(rows, keys, "bias"), "sesgo firmado (log10)")
    _print(_agg(rows, keys, "post_sd"), "anchura de la posterior (sd, log10)")

    _print_paired(_paired(rows, base, arm, keys, "cov", center=0.9),
                  f"CALIBRACION: diferencia pareada contra {base} en |cov-0.90|",
                  "(positivo = mas cerca de la nominal; emparejado dentro de la misma corrida)")
    _print_paired(_paired(rows, base, arm, keys, "bias"),
                  f"PRECISION: diferencia pareada contra {base} en |sesgo|",
                  "(positivo = menos sesgado)")
    _print_paired(_paired(rows, base, arm, keys, "post_sd"),
                  f"ANCHURA: diferencia pareada contra {base} en sd de la posterior",
                  "(positivo = mas estrecha; leer junto a las dos tablas anteriores)")


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        main(fn)