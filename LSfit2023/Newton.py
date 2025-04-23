# Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu. September 20, 2022.

import numpy as np

class Newton():
  def __init__(self, NtIn):
    (C1M, CxM, V0, V1, V2, Vx, S2, BPfrac) = NtIn

    ON2Sh = 18
    a = CxM / ON2Sh / 1660.6 / S2
    a = (a ** (- V0 / Vx)) * (1 - BPfrac)
    b = Vx * ON2Sh
    c = 1 - V0 / Vx
    Vsh = 520 * np.ones( len(C1M) )

    x1, x0, IterNo, IterMax = Vsh, 0, 0, 1000
    while np.max(np.abs(x1 - x0)) > 0.0001 and IterNo < IterMax:
        x0 = x1
        IterNo = IterNo + 1
        f = a * (x0 ** c) - x0 + b
        df = a * c * (x0 ** (c - 1)) - 1
        x1 = x0 - f / df
    Vsh = x1

    self.Rsh_c = (3 * (Vsh + V1) / 4 / np.pi) ** (1 / 3)
    self.Rsh_a = (3 * (Vsh + V2) / 4 / np.pi) ** (1 / 3)
