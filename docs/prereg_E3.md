Prediccion registrada el <fecha>, antes de correr E3.
Diagnostico previo (sin entrenar NPE), umbrales sev>=0.15 y G(0.50)>=0.5:
  telegraph/nb   sev 0.366  G50 0.86  -> colocacion guiada DEBE superar a uniform (en k_syn)
  threestate/nb  sev 0.426  G50 0.69  -> colocacion guiada DEBE superar a uniform (en k_syn)
  threestate/cle sev 0.998  G50 0.00  -> ningun esquema debe superar a uniform
  telegraph/lna  sev 0.936  G50 0.04  -> ningun esquema debe superar a uniform (ya verificado)
  threestate/hybrid sev 0.007        -> no aplica, el sustituto ya sirve
Criterio: ganancia pareada por semilla en |cobertura - 0.90| frente a uniform,
con f = 0.25 y 0.50, tres semillas.
