'''
Author: Jinn-Liang Liu, May 5, 2025.

Papers:
P1: Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu,
    Generalized Debye–Hückel model for activity coefficients of electrolytes in water–methanol mixtures,
    Fluid Phase Equilibria 565, 113662 (2023)
P2: Chin-Lung Li, Ren-Chuen Chen, Xiaodong Liang, Jinn-Liang Liu,
    Generalized Debye-Hückel theory of electrolyte thermodynamics: I. Application, 2025.
P3: Chin-Lung Li, Jinn-Liang Liu, Generalized Debye-Hückel equation from Poisson-Bikerman theory,
    SIAM J. Appl. Math. 80, 2003-2023 (2020).

Jinn-Liang Liu's unpublished research notes:
PF0:   A 3D Poisson-Nernst-Planck Solver for Modeling Biological Ion Channels, August 30, 2012
PF1-4: 3D Poisson-Fermi-Nernst-Planck Solvers for Biological Modeling (Part 1-4), August 6, 2013 - November 11, 2022
'''
import numpy as np
import warnings
warnings.filterwarnings("ignore")

ϵ_0, e, mol, kB, V0 = 8.854187, 1.6022, 6.022045, 1.380649, 1.0

class ActF_2():  # Activity Formula 2 [P2(3.5)]
  def __init__(self, ActIn, ActIn_Mix):  # IS: array used by class ActF_2
    (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, IS, C1m, \
     q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T) = ActIn

    (q5, q6, V5, V6, C5M, C6M) = ActIn_Mix[0]
    (q7, q8, V7, V8, C7M, C8M) = ActIn_Mix[1]


    kBTe = (kB * T / e) * 0.0001
    S1 = 1000 * e / (4 * np.pi * kBTe * ϵ_0)
    S2 = 0.1 * mol * e / (kBTe * ϵ_0)
    C2M = -q1 * C1M / q2  # array

    BornR_c = theta * BornR0[0]
    BornR_a = theta * BornR0[1]
    BPfrac = (V1 * C1M + V2 * C2M + V3 * C3M + V4 * C4M + V5 * C5M + V6 * C6M + V7 * C7M + V8 * C8M) / S2 / 1660.5655
    GammaB = 1 - BPfrac  # Bulk Particle fraction
    BICfrac = (q1 * V1 * C1M + q2 * V2 * C2M + q5 * V5 * C5M + q6 * V6 * C6M + q7 * V7 * C7M + q8 * V8 * C8M) / S2 / 1660.5655
    LAMBDA = (V1 * V1 * C1M + V2 * V2 * C2M + V3 * V3 * C3M + V4 * V4 * C4M + V5 * V5 * C5M + V6 * V6 * C6M + V7 * V7 * C7M + V8 * V8 * C8M) / S2 / 1660.5655

    LAMBDA = V0 * GammaB + LAMBDA
    LAMBDA = BICfrac / LAMBDA  # [P2(2.10)]

    LDebye = np.sqrt( ϵ_s_x_I * ϵ_0 * kBTe * 1.6606 / e / ( ( \
               q1 * q1 * C1M - q1 * V1 * LAMBDA * C1M + \
               q2 * q2 * C2M - q2 * V2 * LAMBDA * C2M + \
               q5 * q5 * C5M - q5 * V5 * LAMBDA * C5M + \
               q6 * q6 * C6M - q6 * V6 * LAMBDA * C6M + \
               q7 * q7 * C7M - q7 * V7 * LAMBDA * C7M + \
               q8 * q8 * C8M - q8 * V8 * LAMBDA * C8M
               ) / S2 ) )  # (array)
    LBjerrum = S1 / ϵ_s_x_I  # = 7.14 ~ 17.1 (scalar)

    Lcorr = np.sqrt(LBjerrum * LDebye / 48)  # [P1(5)] array

    lam = 1 - 4 * (Lcorr ** 2) / (LDebye ** 2)  # maybe a negative number
    lambda1 = (1 - np.emath.sqrt(lam)) / (2 * (Lcorr ** 2))  # maybe complex (lam<0), use emath
    lambda2 = (1 + np.emath.sqrt(lam)) / (2 * (Lcorr ** 2))

    ''' Yes: lam < 0
    if len(lam.shape) > 0:
      if any(lam < 0):
        print('WARNING: array lam < 0:', np.around(lam, 3))
    else:
      if lam < 0:
        print('WARNING: scalar lam < 0:', np.around(lam, 3))
        print("LDebye =", np.around(LDebye, 3))
        print('Lcorr:', np.around(Lcorr, 3))
        print('Rsh_c:', np.around(Rsh_c, 3))
          #Rsh_c: [62.781 45.864 29.943 22.329 17.865 12.856 10.115 7.194 5.661 4.716 4.075]
        print('WARNING: complex lambda1, T:', np.around(lambda1, 3), T)
    '''

    T1_c = ( Lcorr * np.sqrt(lambda2) - 1 ) * np.exp(-Rsh_c / Lcorr) - ( Lcorr * np.sqrt(lambda2) + 1 ) * np.exp((Rsh_c - 2 * BornR_c) / Lcorr)
    T1_a = ( Lcorr * np.sqrt(lambda2) - 1 ) * np.exp(-Rsh_a / Lcorr) - ( Lcorr * np.sqrt(lambda2) + 1 ) * np.exp((Rsh_a - 2 * BornR_a) / Lcorr)
    THETA1_c = (Lcorr ** 3) * ( np.sqrt(lambda2) - np.sqrt(lambda1) ) * np.exp(-np.sqrt(lambda1) * Rsh_c) / T1_c
    THETA1_a = (Lcorr ** 3) * ( np.sqrt(lambda2) - np.sqrt(lambda1) ) * np.exp(-np.sqrt(lambda1) * Rsh_a) / T1_a

    T2_c = np.exp(np.sqrt(lambda2) * Rsh_c) / (Lcorr ** 2) * ( np.exp(-Rsh_c / Lcorr) - np.exp((Rsh_c - 2 * BornR_c) / Lcorr) )
    T2_a = np.exp(np.sqrt(lambda2) * Rsh_a) / (Lcorr ** 2) * ( np.exp(-Rsh_a / Lcorr) - np.exp((Rsh_a - 2 * BornR_a) / Lcorr) )
    THETA2_c = -np.exp((np.sqrt(lambda2) - np.sqrt(lambda1)) * Rsh_c) + T2_c * THETA1_c
    THETA2_a = -np.exp((np.sqrt(lambda2) - np.sqrt(lambda1)) * Rsh_a) + T2_a * THETA1_a

    T3A_c = ( np.sqrt(lambda1) * Rsh_c + 1 ) * lambda2 * np.exp(-np.sqrt(lambda1) * Rsh_c)
    T3B_c = ( np.sqrt(lambda2) * Rsh_c + 1 ) * lambda1 * np.exp(-np.sqrt(lambda2) * Rsh_c) * THETA2_c
    T3C_c = 2 * BornR_c / Lcorr * np.exp(-BornR_c / Lcorr) - (Rsh_c / Lcorr + 1) * np.exp(-Rsh_c / Lcorr) \
            - (Rsh_c / Lcorr - 1) * np.exp((Rsh_c - 2 * BornR_c) / Lcorr)
    THETA3_c = (Lcorr ** 2) * (LDebye ** 2) * (T3A_c + T3B_c) + T3C_c * THETA1_c

    T3A_a = ( np.sqrt(lambda1) * Rsh_a + 1 ) * lambda2 * np.exp(-np.sqrt(lambda1) * Rsh_a)
    T3B_a = ( np.sqrt(lambda2) * Rsh_a + 1 ) * lambda1 * np.exp(-np.sqrt(lambda2) * Rsh_a) * THETA2_a
    T3C_a = 2 * BornR_a / Lcorr * np.exp(-BornR_a / Lcorr) - (Rsh_a / Lcorr + 1) * np.exp(-Rsh_a / Lcorr) \
            - (Rsh_a / Lcorr - 1) * np.exp((Rsh_a - 2 * BornR_a) / Lcorr)
    THETA3_a = (Lcorr ** 2) * (LDebye ** 2) * (T3A_a + T3B_a) + T3C_a * THETA1_a

    T4A_c = (Lcorr ** 2) * (LDebye ** 2) * (lambda2 - lambda1) * np.exp(-np.sqrt(lambda1) * Rsh_c)
    T4B_c = 2 * BornR_c / Lcorr * np.exp(-BornR_c / Lcorr) * THETA1_c
    T4C_c = (lambda1 * (LDebye ** 2) - 1) * ( np.exp(-Rsh_c / Lcorr) - np.exp((Rsh_c - 2 * BornR_c) / Lcorr) ) * THETA1_c
    THETA4_c = T4A_c + T4B_c + T4C_c

    T4A_a = (Lcorr ** 2) * (LDebye ** 2) * (lambda2 - lambda1) * np.exp(-np.sqrt(lambda1) * Rsh_a)
    T4B_a = 2 * BornR_a / Lcorr * np.exp(-BornR_a / Lcorr) * THETA1_a
    T4C_a = (lambda1 * (LDebye ** 2) - 1) * ( np.exp(-Rsh_a / Lcorr) - np.exp((Rsh_a - 2 * BornR_a) / Lcorr) ) * THETA1_a
    THETA4_a = T4A_a + T4B_a + T4C_a

    a1 = q1 * q1 * e * e / (8 * np.pi * ϵ_0 * ϵ_s_x_I * kB * T) * (10 ** 7)
    a2 = q2 * q2 * e * e / (8 * np.pi * ϵ_0 * ϵ_s_x_I * kB * T) * (10 ** 7)

    gamma1 = a1 * ( 1 / BornR_c + (THETA4_c / THETA3_c - 1) / Rsh_c \
           - 2 * np.exp(-BornR_c / Lcorr) * THETA1_c / THETA3_c / Lcorr - 1 / BornR0[0] )  # [P2(3.5)] array
    gamma2 = a2 * ( 1 / BornR_a + (THETA4_a / THETA3_a - 1) / Rsh_a \
           - 2 * np.exp(-BornR_a / Lcorr) * THETA1_a / THETA3_a / Lcorr - 1 / BornR0[1] )

    g_PF = (np.abs(q2) * gamma1 + np.abs(q1) * gamma2) / (np.abs(q1) + np.abs(q2))  # [P2(3.6)] array

    '''if len(g_PF.shape) > 0:
      if any(g_PF.imag != 0): print('g_PF.imag != 0 ...')  # Yes g_PF.imag != 0
    else:
      if g_PF.imag != 0: print('g_PF.imag != 0 ...')  # Yes g_PF.imag != 0'''

    Ls    = (GammaB, lambda1, lambda2, LDebye, LBjerrum, Lcorr)
    THETA = (THETA1_c, THETA2_c, THETA3_c, THETA4_c)  # for cation
    #THETA = (THETA1_a, THETA2_a, THETA3_a, THETA4_a)  # for anion

    self.g_PF = g_PF.real  # g_PF is complex with g_PF.imag == 0
    self.Ls, self.THETA = Ls, THETA
