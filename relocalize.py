"""
relocalize.py  (v2)
===================
Diagnostico de localizacion a partir de los .jsonl de E4. No resimula nada.

POR QUE CAMBIO RESPECTO A v1
----------------------------
v1 media la severidad con la MASA NO CUBIERTA (fraccion de datos exactos que el
sustituto casi nunca produce). Eso capta el desajuste de SOPORTE, que es como
fallan los sustitutos gaussianos: no pueden generar la inflacion de ceros.

Pero es ciego al desajuste de FORMA. El NB es una familia de conteo con soporte
completo: produce los mismos valores que el exacto, con pesos distintos. En el
tres-estados v1 le asigno severidad 0.004 ("el sustituto ya sirve") cuando el
paper reporta que el NPE entrenado con el sale con cobertura 0.79, cayendo de
0.93 a 0.68 con la burstiness. El dano existe y la masa no cubierta no lo ve.

La cantidad que si lo ve ya estaba guardada: el AUC del clasificador, que estima
la distancia en variacion total,  TV_b ~ 2*AUC_b - 1.

DIAGNOSTICO (tres numeros)
--------------------------
  severidad   sev = media_b TV_b
      cuanto difiere el sustituto del exacto. Si es baja, el sustituto sirve tal
      cual y no hay nada que repartir.
  ganancia    G(f) = 1 - (media de la fraccion (1-f) de bins con menor TV) / sev
      que parte de ese dano puede quitar la colocacion optima con presupuesto f.
      G = 0 si TV es plano (colocar no sirve), G -> 1 si esta concentrado.
  tipo        soporte (masa no cubierta alta) o forma (soporte completo, pesos
      distintos). No cambia el veredicto, explica el mecanismo.

  colocar paga  <=>  sev alta  Y  G alta.

Uso:
  python relocalize.py results/e4_localization.jsonl
  python relocalize.py results/e4_localization.jsonl results/v5/e4_localization.jsonl
  python relocalize.py --bins results/e4_localization.jsonl     # detalle por bin
"""
import sys, json, numpy as np
from collections import defaultdict

SEV_MIN, G_MIN, UNC_MIN = 0.15, 0.50, 0.10   # umbrales, a declarar en el paper


def tv_of(bins):
    """TV por bin desde el AUC del clasificador, recortada a [0, 1]."""
    return np.clip(2.0 * np.array([b["auc"] for b in bins]) - 1.0, 0.0, 1.0)


def gain(v, f):
    """Fraccion del dano que quita la colocacion optima con presupuesto f."""
    v = np.asarray(v, float)
    keep = max(1, int(round((1 - f) * len(v))))
    return float(1 - np.sort(v)[:keep].mean() / max(v.mean(), 1e-12))


def metrics(bins):
    tv = tv_of(bins)
    u = np.array([b["uncovered"] for b in bins])
    ness = np.array([b["ness"] for b in bins])
    return dict(sev=float(tv.mean()), tv=tv, unc=u, ness=ness,
                g25=gain(tv, 0.25), g50=gain(tv, 0.50),
                unc_mean=float(u.mean()),
                safe=float((tv < 0.15).mean()))


def verdict(sev, g50, unc_mean):
    if sev < SEV_MIN:
        v = "sustituto ya sirve: nada que repartir"
    elif g50 < G_MIN:
        v = "error disperso: colocar NO paga"
    else:
        v = "error concentrado: colocar PAGA"
    tipo = "soporte" if unc_mean > UNC_MIN else "forma"
    return v, tipo


def main(paths, show_bins=False):
    D = defaultdict(list)
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            D[(r["system"], r["exact"], r["surrogate"])].append(r["per_bin"])

    print(f"{'sistema':11s} {'sust':7s} {'sev(TV)':>8s} {'G(.25)':>7s} {'G(.50)':>7s} "
          f"{'no cub':>7s} {'safe':>5s} {'tipo':>8s}   veredicto")
    for k in sorted(D):
        M = [metrics(b) for b in D[k]]
        sev = np.mean([m["sev"] for m in M]); ssd = np.std([m["sev"] for m in M])
        g25 = np.mean([m["g25"] for m in M]); g50 = np.mean([m["g50"] for m in M])
        unc = np.mean([m["unc_mean"] for m in M]); safe = np.mean([m["safe"] for m in M])
        v, tipo = verdict(sev, g50, unc)
        print(f"{k[0]:11s} {k[2]:7s} {sev:8.3f} {g25:7.2f} {g50:7.2f} {unc:7.3f} "
              f"{safe:5.2f} {tipo:>8s}   {v}   (sd {ssd:.3f}, n={len(M)})")

        if show_bins:
            tv = np.mean([m["tv"] for m in M], axis=0)
            u = np.mean([m["unc"] for m in M], axis=0)
            e = np.mean([m["ness"] for m in M], axis=0)
            f = np.mean([[b["fano"] for b in bb] for bb in D[k]], axis=0)
            for j in range(len(tv)):
                print(f"      bin {j} Fano~{f[j]:8.2f}  TV={tv[j]:.3f}  "
                      f"no cub={u[j]:.3f}  nESS={e[j]:.3f}")

    print(f"\numbrales: sev>={SEV_MIN}, G(.50)>={G_MIN}; tipo 'soporte' si la masa "
          f"no cubierta media supera {UNC_MIN}")
    print("TV por bin estimada como 2*AUC-1; el control exacto-contra-exacto "
          "debe dar sev ~ 0")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args, show_bins="--bins" in sys.argv)