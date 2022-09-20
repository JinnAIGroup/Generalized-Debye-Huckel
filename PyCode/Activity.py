# Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu. September 20, 2022.

import numpy as np

class Activity():
  def __init__(self, ActIn):
    (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, \
     q1, q2, V0, V1, V2, V3, V4, diS, T) = ActIn

    epsln, e, mol, kB = 8.854187, 1.6022, 6.022045, 1.380649
    kBTe = (kB * T / e) * 0.0001
    S1 = 1000 * e / (4 * np.pi * kBTe * epsln)
    S2 = 0.1 * mol * e / (kBTe * epsln)

    C2M = q1 * C1M
    BornR_c = theta * BornR0[0]
    BornR_a = theta * BornR0[1]

    BPfrac = (V1 * C1M + V2 * C2M + V3 * C3M + V4 * C4M) / S2 / 1660.6
    LAMBDA = (V1 * V1 * C1M + V2 * V2 * C2M + V3 * V3 * C3M + V4 * V4 * C4M) / S2 / 1660.6
    LAMBDA = LAMBDA + (1 - BPfrac) * V0
    LAMBDA = (C1M / S2 / 1660.6) * ((V1 - V2) ** 2) / LAMBDA
    LDebye = np.sqrt( diS * epsln * kBTe * 1.6606 / e / ( ((1 - LAMBDA) * q1 * q1 * C1M - q1 * q2 * C1M) / S2 ) )
    LBjerrum = S1 / diS
    Lcorr = np.sqrt(LBjerrum * LDebye / 48)

    a1 = q1 * q1 * e * e / (8 * np.pi * epsln * diS * kB * T) * (10 ** 7)
    a2 = q2 * q2 * e * e / (8 * np.pi * epsln * diS * kB * T) * (10 ** 7)
    lam = 1 - 4 * (Lcorr ** 2) / (LDebye ** 2)
    lambda1 = (1 - np.sqrt(lam)) / (2 * (Lcorr ** 2))
    lambda2 = (1 + np.sqrt(lam)) / (2 * (Lcorr ** 2))
    d1 = lambda2 * ( (Lcorr ** 2) * lambda1 - 1 )
    d2 = lambda1 * ( (Lcorr ** 2) * lambda2 - 1 )
    THETA_c = (d1 - d2) / ( d1 * (np.sqrt(lambda1) * Rsh_c + 1) - d2 * (np.sqrt(lambda2) * Rsh_c + 1) )
    THETA_a = (d1 - d2) / ( d1 * (np.sqrt(lambda1) * Rsh_a + 1) - d2 * (np.sqrt(lambda2) * Rsh_a + 1) )

    gamma1 = a1 * (1 / BornR_c - 1 / BornR0[0] + (THETA_c - 1) / Rsh_c)
    gamma2 = a2 * (1 / BornR_a - 1 / BornR0[1] + (THETA_a - 1) / Rsh_a)

    self.g_PF = (np.abs(q2) * gamma1 + np.abs(q1) * gamma2) / (np.abs(q1) + np.abs(q2))
