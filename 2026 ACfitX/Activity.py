'''
Author: Jinn-Liang Liu, Nov 24, 2025.

P1: Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu,
    Generalized Debye–Hückel model for activity coefficients of electrolytes in water–methanol mixtures,
    Fluid Phase Equilibria 565, 113662 (2023)
P2: Chin-Lung Li, Ren-Chuen Chen, Xiaodong Liang, Jinn-Liang Liu,
    Generalized Debye-Hückel theory for ion activities in mixed-salt and mixed-solvent electrolyte solutions, 2025.
PF0: Jinn-Liang Liu, A 3D Poisson-Nernst-Planck Solver for Modeling Biological Ion Channels, Unpublished, August 30, 2012.
'''
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Solution Parameters:
#   0: void, 1: cation, 2: anion, 3: H2O, 4: MeOH, x or X: mixing percentage of 4 in [0, 1]
#   5: cation, 6: anion, 7: cation, 8: anion
#   Bulk concentrations in M: C1M (array), C2M (array), C3M (scalar), C4M (scalar)
#   ϵ_s_x (scalar): dielectric constant of mixed solvent, V: volume
#   gamma: mean activity data of target salt CA = ca = 1+2 (array)

ϵ_0, e, mol, kB, V0 = 8.854187, 1.6022, 6.022045, 1.380649, 1.0

class Activity():  # Activity Formula
  def __init__(self, ActIn, ActIn_Mix):
    (theta, BornR0, R_sh, C1M, C3M, C4M, IS, \
     q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T) = ActIn

    (q5, q6, V5, V6, C5M, C6M) = ActIn_Mix[0]
    (q7, q8, V7, V8, C7M, C8M) = ActIn_Mix[1]

    kBTe = (kB * T / e) * 0.0001
    S1 = 1000 * e / (4 * np.pi * kBTe * ϵ_0)  # scaling parameter [PF0(5.10)]
    S2 = 0.1 * mol * e / (kBTe * ϵ_0)  # scaling parameter [PF0(5.12)]
    C2M = -q1 * C1M / q2  # (array)

    BornR_c = theta * BornR0[0]  # [P2(28)]
    BornR_a = theta * BornR0[1]
    BPfrac = (V1 * C1M + V2 * C2M + V3 * C3M + V4 * C4M + V5 * C5M + V6 * C6M + V7 * C7M + V8 * C8M) / S2 / 1660.5655
    GammaB = 1 - BPfrac  # [P2(2.8)] Bulk Particle fraction
    BICfrac = (q1 * V1 * C1M + q2 * V2 * C2M + q5 * V5 * C5M + q6 * V6 * C6M + q7 * V7 * C7M + q8 * V8 * C8M) / S2 / 1660.5655
    LAMBDA = (V1 * V1 * C1M + V2 * V2 * C2M + V3 * V3 * C3M + V4 * V4 * C4M + V5 * V5 * C5M + V6 * V6 * C6M + V7 * V7 * C7M + V8 * V8 * C8M) / S2 / 1660.5655

    LAMBDA = V0 * GammaB + LAMBDA
    LAMBDA = BICfrac / LAMBDA  # [P2(10)]

    LDebye = np.sqrt( ϵ_s_x_I * ϵ_0 * kBTe * 1.6606 / e / ( ( \
               q1 * q1 * C1M - q1 * V1 * LAMBDA * C1M + \
               q2 * q2 * C2M - q2 * V2 * LAMBDA * C2M + \
               q5 * q5 * C5M - q5 * V5 * LAMBDA * C5M + \
               q6 * q6 * C6M - q6 * V6 * LAMBDA * C6M + \
               q7 * q7 * C7M - q7 * V7 * LAMBDA * C7M + \
               q8 * q8 * C8M - q8 * V8 * LAMBDA * C8M
               ) / S2 ) )  # (array)
    LBjerrum = S1 / ϵ_s_x_I  # = 7.14 ~ 17.1 (scalar)

    Lcorr = np.sqrt(LBjerrum * LDebye / 48)  # [P2(25)] (array)

    lam = 1 - 4 * (Lcorr ** 2) / (LDebye ** 2)  # maybe < 0
    lambda1 = (1 - np.emath.sqrt(lam)) / (2 * (Lcorr ** 2))  # maybe complex (lam < 0), use emath
    lambda2 = (1 + np.emath.sqrt(lam)) / (2 * (Lcorr ** 2))

    T1_c = ( Lcorr * np.sqrt(lambda2) - 1 ) * np.exp(-R_sh / Lcorr) - ( Lcorr * np.sqrt(lambda2) + 1 ) * np.exp((R_sh - 2 * BornR_c) / Lcorr)
    T1_a = ( Lcorr * np.sqrt(lambda2) - 1 ) * np.exp(-R_sh / Lcorr) - ( Lcorr * np.sqrt(lambda2) + 1 ) * np.exp((R_sh - 2 * BornR_a) / Lcorr)
    THETA1_c = (Lcorr ** 3) * ( np.sqrt(lambda2) - np.sqrt(lambda1) ) * np.exp(-np.sqrt(lambda1) * R_sh) / T1_c
    THETA1_a = (Lcorr ** 3) * ( np.sqrt(lambda2) - np.sqrt(lambda1) ) * np.exp(-np.sqrt(lambda1) * R_sh) / T1_a

    T2_c = np.exp(np.sqrt(lambda2) * R_sh) / (Lcorr ** 2) * ( np.exp(-R_sh / Lcorr) - np.exp((R_sh - 2 * BornR_c) / Lcorr) )
    T2_a = np.exp(np.sqrt(lambda2) * R_sh) / (Lcorr ** 2) * ( np.exp(-R_sh / Lcorr) - np.exp((R_sh - 2 * BornR_a) / Lcorr) )
    THETA2_c = -np.exp((np.sqrt(lambda2) - np.sqrt(lambda1)) * R_sh) + T2_c * THETA1_c
    THETA2_a = -np.exp((np.sqrt(lambda2) - np.sqrt(lambda1)) * R_sh) + T2_a * THETA1_a

    T3A_c = ( np.sqrt(lambda1) * R_sh + 1 ) * lambda2 * np.exp(-np.sqrt(lambda1) * R_sh)
    T3B_c = ( np.sqrt(lambda2) * R_sh + 1 ) * lambda1 * np.exp(-np.sqrt(lambda2) * R_sh) * THETA2_c
    T3C_c = 2 * BornR_c / Lcorr * np.exp(-BornR_c / Lcorr) - (R_sh / Lcorr + 1) * np.exp(-R_sh / Lcorr) \
            - (R_sh / Lcorr - 1) * np.exp((R_sh - 2 * BornR_c) / Lcorr)
    THETA3_c = (Lcorr ** 2) * (LDebye ** 2) * (T3A_c + T3B_c) + T3C_c * THETA1_c

    T3A_a = ( np.sqrt(lambda1) * R_sh + 1 ) * lambda2 * np.exp(-np.sqrt(lambda1) * R_sh)
    T3B_a = ( np.sqrt(lambda2) * R_sh + 1 ) * lambda1 * np.exp(-np.sqrt(lambda2) * R_sh) * THETA2_a
    T3C_a = 2 * BornR_a / Lcorr * np.exp(-BornR_a / Lcorr) - (R_sh / Lcorr + 1) * np.exp(-R_sh / Lcorr) \
            - (R_sh / Lcorr - 1) * np.exp((R_sh - 2 * BornR_a) / Lcorr)
    THETA3_a = (Lcorr ** 2) * (LDebye ** 2) * (T3A_a + T3B_a) + T3C_a * THETA1_a

    T4A_c = (Lcorr ** 2) * (LDebye ** 2) * (lambda2 - lambda1) * np.exp(-np.sqrt(lambda1) * R_sh)
    T4B_c = 2 * BornR_c / Lcorr * np.exp(-BornR_c / Lcorr) * THETA1_c
    T4C_c = (lambda1 * (LDebye ** 2) * (Lcorr ** 2) / (Lcorr ** 2) - 1) * ( np.exp(-R_sh / Lcorr) - np.exp((R_sh - 2 * BornR_c) / Lcorr) ) * THETA1_c
    THETA4_c = T4A_c + T4B_c + T4C_c

    T4A_a = (Lcorr ** 2) * (LDebye ** 2) * (lambda2 - lambda1) * np.exp(-np.sqrt(lambda1) * R_sh)
    T4B_a = 2 * BornR_a / Lcorr * np.exp(-BornR_a / Lcorr) * THETA1_a
    T4C_a = (lambda1 * (LDebye ** 2) * (Lcorr ** 2) / (Lcorr ** 2) - 1) * ( np.exp(-R_sh / Lcorr) - np.exp((R_sh - 2 * BornR_a) / Lcorr) ) * THETA1_a
    THETA4_a = T4A_a + T4B_a + T4C_a

    a1 = q1 * q1 * e * e / (8 * np.pi * ϵ_0 * kB * T) * (10 ** 7)
    a2 = q2 * q2 * e * e / (8 * np.pi * ϵ_0 * kB * T) * (10 ** 7)

    gamma1_1 = -a1 * ( 1 / ϵ_s_x / BornR0[0] )
    gamma2_1 = -a2 * ( 1 / ϵ_s_x / BornR0[1] )
    gamma1_2 = -a1 * ( -1 / ϵ_s_x_I / BornR_c )
    gamma2_2 = -a2 * ( -1 / ϵ_s_x_I / BornR_a )
    gamma1_3 = -a1 * ( -(THETA4_c / THETA3_c - 1) / ϵ_s_x_I / R_sh )
    gamma2_3 = -a2 * ( -(THETA4_a / THETA3_a - 1) / ϵ_s_x_I / R_sh )
    gamma1_4 = -a1 * ( 2 * np.exp(-BornR_c / Lcorr) * THETA1_c / THETA3_c / ϵ_s_x_I / Lcorr )
    gamma2_4 = -a2 * ( 2 * np.exp(-BornR_a / Lcorr) * THETA1_a / THETA3_a / ϵ_s_x_I / Lcorr )

    gamma1 = gamma1_1 + gamma1_2 + gamma1_3 + gamma1_4  # [P2(18)] (array)
    gamma2 = gamma2_1 + gamma2_2 + gamma2_3 + gamma2_4

    g_PF = ( np.abs(q2) * gamma1 + np.abs(q1) * gamma2 ) / ( np.abs(q1) + np.abs(q2) )

    d_c = 2 * ( 3 * V1 / 4 / np.pi ) ** ( 1 / 3 )
    d_a = 2 * ( 3 * V2 / 4 / np.pi ) ** ( 1 / 3 )
    LDebye_DH = np.sqrt( ϵ_s_x * ϵ_0 * kBTe * 1.6606 / e / ( ( \
               q1 * q1 * C1M + q2 * q2 * C2M + q5 * q5 * C5M + \
               q6 * q6 * C6M + q7 * q7 * C7M + q8 * q8 * C8M
               ) / S2 ) )  # (array)
    gamma1_DH = -a1 * ( 1 / ϵ_s_x / ( LDebye_DH + d_c ) )
    gamma2_DH = -a2 * ( 1 / ϵ_s_x / ( LDebye_DH + d_a ) )

    g_DH = (np.abs(q2) * gamma1_DH + np.abs(q1) * gamma2_DH) / (np.abs(q1) + np.abs(q2))

    Ls    = (GammaB, lambda1, lambda2, LDebye, LBjerrum, Lcorr)
    THETA = (THETA1_c, THETA2_c, THETA3_c, THETA4_c)  # for cation
    THETA_a = (THETA1_a, THETA2_a, THETA3_a, THETA4_a)  # for anion

    self.g_PF = g_PF.real  # g_PF is complex but almost g_PF.imag == 0
    self.Ls, self.THETA, self.THETA_a = Ls, THETA, THETA_a

    self.gamma1 = (gamma1_1, gamma1_2, gamma1_3, gamma1_4, gamma1_DH) # for Fig 5
    self.gamma2 = (gamma2_1, gamma2_2, gamma2_3, gamma2_4, gamma2_DH, g_DH)
