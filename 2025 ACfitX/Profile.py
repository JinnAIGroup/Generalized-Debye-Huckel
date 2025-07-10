'''
Author: Jinn-Liang Liu, July 10, 2025.
For Example 4.5, 4.6
'''
import numpy as np

V0 = 1.0

def Newton2(phi, NtIn, ActIn_Mix):  # [PF3P85(34)] Given phi, find steric
  (GammaB, C1, C2, C3, V1, V2, V3, q1, q2, S2) = NtIn
  (q5, q6, V5, V6, C5, C6) = ActIn_Mix[0]
  (q7, q8, V7, V8, C7, C8) = ActIn_Mix[1]

  n, N, tol, t0 = 1, 100, 0.0001, 1.0

  f0 = GammaB * t0 ** (V0 / V3) \
     + ( V1 * C1 * np.exp(-q1 * phi) * t0 ** (V1 / V3) \
       + V2 * C2 * np.exp(-q2 * phi) * t0 ** (V2 / V3) \
       + V5 * C5 * np.exp(-q5 * phi) * t0 ** (V5 / V3) \
       + V6 * C6 * np.exp(-q6 * phi) * t0 ** (V6 / V3) \
       + V7 * C7 * np.exp(-q7 * phi) * t0 ** (V7 / V3) \
       + V8 * C8 * np.exp(-q8 * phi) * t0 ** (V8 / V3) \
       + V3 * C3 * t0 ) / S2 / 1660.5655 - 1

  while np.abs(f0) > tol and n < N:
    df0 = GammaB * t0 ** (V0 / V3 - 1) * (V0 / V3) \
        + ( V1 * C1 * np.exp(-q1 * phi) * t0 ** (V1 / V3 - 1) * (V1 / V3) \
          + V2 * C2 * np.exp(-q2 * phi) * t0 ** (V2 / V3 - 1) * (V2 / V3) \
          + V5 * C5 * np.exp(-q5 * phi) * t0 ** (V5 / V3 - 1) * (V5 / V3) \
          + V6 * C6 * np.exp(-q6 * phi) * t0 ** (V6 / V3 - 1) * (V6 / V3) \
          + V7 * C7 * np.exp(-q7 * phi) * t0 ** (V7 / V3 - 1) * (V7 / V3) \
          + V8 * C8 * np.exp(-q8 * phi) * t0 ** (V8 / V3 - 1) * (V8 / V3) \
          + V3 * C3 ) / S2 / 1660.5655
    t0 = t0 - f0 / df0
    if t0 < 0: t0 = 1e-10
    f0 = GammaB * t0 ** (V0 / V3) \
       + ( V1 * C1 * np.exp(-q1 * phi) * t0 ** (V1 / V3) \
         + V2 * C2 * np.exp(-q2 * phi) * t0 ** (V2 / V3) \
         + V5 * C5 * np.exp(-q5 * phi) * t0 ** (V5 / V3) \
         + V6 * C6 * np.exp(-q6 * phi) * t0 ** (V6 / V3) \
         + V7 * C7 * np.exp(-q7 * phi) * t0 ** (V7 / V3) \
         + V8 * C8 * np.exp(-q8 * phi) * t0 ** (V8 / V3) \
         + V3 * C3 * t0 ) / S2 / 1660.5655 - 1
    n = n + 1

  #print(' Newton2 n = ', n)
  if n >= N: print(' Warning: Newton2 diverges; np.abs(f0), df0:', np.around(np.abs(f0), 5), np.around(df0, 5))

  ster = np.log(t0) * (V0 / V3)

  return ster


class Profile():  # Given Q, find various function profiles
  def __init__(self, ActIn_Pf, ActIn_Mix, LsP, THETA, p_In):  # all input variables are scalars
    (theta, BornR0, Rsh_c, C1M, C3M, C4M, q1, q2, V1, V2, V3, ϵ_s_x, ϵ_s_x_I, T) = ActIn_Pf
    (q5, q6, V5, V6, C5M, C6M) = ActIn_Mix[0]
    (q7, q8, V7, V8, C7M, C8M) = ActIn_Mix[1]
    (ϵ_s_x, GammaB, lambda1, lambda2, LDebye, Lcorr) = LsP
    (THETA1_c, THETA2_c, THETA3_c, THETA4_c) = THETA
    (pH2O, pH2O_Z, p_La, p_Cl, p_Mg, numPW, numPW_Z, numPWI, numPWI_Z) = p_In

    Q = q1

    ϵ_0, e, mol, kB = 8.854187, 1.6022, 6.022045, 1.380649
    kBTe = (kB * T / e) * 0.0001
    S1 = 1000 * e / (4 * np.pi * kBTe * ϵ_0)
    S2 = 0.1 * mol * e / (kBTe * ϵ_0)
    S3 = 100 * e / ϵ_0 / kB / T  # [PF4Ex3.88(6)]
      # S1, S2, S3 = 560.48 4.24 0.04

    C2M = -q1 * C1M / q2
    BornR_c = theta * BornR0
    NtIn  = (GammaB, C1M, C2M, C3M, V1, V2, V3, q1, q2, S2)
    d1, d2 = lambda1 ** 2, lambda2 ** 2

    LAMBDA = 0
    LDebye_PB = np.sqrt( ϵ_s_x_I * ϵ_0 * kBTe * 1.6606 / e / ( ( \
                q1 * q1 * C1M - q1 * V1 * LAMBDA * C1M + \
                q2 * q2 * C2M - q2 * V2 * LAMBDA * C2M + \
                q5 * q5 * C5M - q5 * V5 * LAMBDA * C5M + \
                q6 * q6 * C6M - q6 * V6 * LAMBDA * C6M + \
                q7 * q7 * C7M - q7 * V7 * LAMBDA * C7M + \
                q8 * q8 * C8M - q8 * V8 * LAMBDA * C8M
                ) / S2 ) )  # (array)

    L  = 10  # = R_s in Angstrom
    rL = np.linspace(0, L, 101)
    DL = L/100.  # for Idx = i
    N  = len(rL)
    eltr, psi, ster, ionC, ionA, ion5, ion6, solv, rho, ditr, numP = \
    np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    eltr_2PF, eltr_PB1, eltr_PB2 = np.zeros(N), np.zeros(N), np.zeros(N)
    rho_2PF, rho_PB1, rho_PB2 = np.zeros(N), np.zeros(N), np.zeros(N)
    eltr_DH = np.zeros(N)  # for Remark 4.4 in 2ndGDH.tex
    a_i, a_j = 1.05, 1.81  # for LaCl3 in Remark 4.4 in 2ndGDH.tex

    A1 = 1 / BornR_c
    A2 = (THETA4_c / THETA3_c - 1) / Rsh_c
    A3 = - 2 * np.exp(-BornR_c / Lcorr) * THETA1_c / THETA3_c / Lcorr
    A = A1 + A2 + A3
      #if A.imag != 0: print('A.imag:', A.imag)  # Yes, but A.imag = -4.35e-19

    for i in range( N ):
      if rL[i] <= BornR_c:
        if rL[i] > BornR_c - DL: IdxBorn = i
        eltr[i] = S1 * Q * A.real / ϵ_s_x_I  # [P2(26a)]
        ditr[i] = 1
        A_hat = 1 / BornR_c - 1 / (LDebye + Rsh_c)
        eltr_2PF[i] = S1 * Q * A_hat / ϵ_s_x_I  # [P2(40a)]
        A_hat = 1 / BornR_c - 1 / (LDebye_PB + Rsh_c)
        eltr_PB1[i] = S1 * Q * A_hat / ϵ_s_x_I
        eltr_PB2[i] = S1 * Q * A_hat / ϵ_s_x
        A_DH = 1 / a_i - 1 / (LDebye + a_i + a_j)  # for Remark 4.4 in 2ndGDH.tex
        eltr_DH[i] = S1 * Q * A_DH / ϵ_s_x

      if rL[i] > Rsh_c:
        if rL[i] < Rsh_c + DL: IdxSh = i - 1
        top = lambda2 * np.exp(-np.sqrt(lambda1) * rL[i]) \
            + lambda1 * np.exp(-np.sqrt(lambda2) * rL[i]) * THETA2_c
        tCr = top / rL[i] / THETA3_c
        Cr = (Lcorr ** 2) * (LDebye ** 2) * tCr  # [P2(26c)]
        eltr[i] = S1 * Q * Cr.real / ϵ_s_x_I
        psi[i]  = S1 * Q * tCr.real / ϵ_s_x_I * ϵ_0  # [P2(27a)]
        ster[i] = Newton2(eltr[i], NtIn, ActIn_Mix)
        ionC[i] = C1M * np.exp(-q1 * eltr[i] + V1 * ster[i] / V0)
        ionA[i] = C2M * np.exp(-q2 * eltr[i] + V2 * ster[i] / V0)
        ion5[i] = C5M * np.exp(-q5 * eltr[i] + V5 * ster[i] / V0)
        ion6[i] = C6M * np.exp(-q6 * eltr[i] + V6 * ster[i] / V0)
        solv[i] = C3M * np.exp(V3 * ster[i] / V0)
        rho[i]  = q1 * ionC[i] + q2 * ionA[i] + q5 * ion5[i] + q6 * ion6[i]  # unit: eM [PF4Ex3.88(7)]

        if Q > 0:
          numP[i] = solv[i] * pH2O_Z + ionC[i] * p_La + ionA[i] * p_Cl + ion5[i] * p_Mg + ion6[i] * p_Cl  # [P2(2.22)]
        else:
          numP[i] = solv[i] * pH2O_Z + ionC[i] * p_La + ionA[i] * p_Cl + ion5[i] * p_Mg + ion6[i] * p_La  # [P2(2.22)]
        X = (ϵ_s_x_I - 1) / (ϵ_s_x_I + 2) * numP[i] / numPWI_Z
        ditr[i] = (2 * X + 1) / (1 - X)

        C_hat = np.exp(-(rL[i] - Rsh_c) / LDebye) / (1 + Rsh_c / LDebye) / rL[i]
        eltr_2PF[i] = S1 * Q * C_hat / ϵ_s_x_I  # [P2(4.5c)]
        ster_2PF = Newton2(eltr_2PF[i], NtIn, ActIn_Mix)
        ionC_2PF = C1M * np.exp(-q1 * eltr_2PF[i] + V1 * ster_2PF / V0)
        ionA_2PF = C2M * np.exp(-q2 * eltr_2PF[i] + V2 * ster_2PF / V0)
        ion5_2PF = C5M * np.exp(-q5 * eltr_2PF[i] + V5 * ster_2PF / V0)
        ion6_2PF = C6M * np.exp(-q6 * eltr_2PF[i] + V6 * ster_2PF / V0)
        rho_2PF[i]  = q1 * ionC_2PF + q2 * ionA_2PF + q5 * ion5_2PF + q6 * ion6_2PF  # unit: eM [PF4Ex3.88(7)]

        C_hat = np.exp(-(rL[i] - Rsh_c) / LDebye_PB) / (1 + Rsh_c / LDebye_PB) / rL[i]
        eltr_PB1[i] = S1 * Q * C_hat / ϵ_s_x_I
        eltr_PB2[i] = S1 * Q * C_hat / ϵ_s_x

        ionC_PB = C1M * np.exp(-q1 * eltr_PB1[i])
        ionA_PB = C2M * np.exp(-q2 * eltr_PB1[i])
        ion5_PB = C5M * np.exp(-q5 * eltr_PB1[i])
        ion6_PB = C6M * np.exp(-q6 * eltr_PB1[i])
        rho_PB1[i]  = q1 * ionC_PB + q2 * ionA_PB + q5 * ion5_PB + q6 * ion6_PB  # unit: eM [PF4Ex3.88(7)]

        ionC_PB = C1M * np.exp(-q1 * eltr_PB2[i])
        ionA_PB = C2M * np.exp(-q2 * eltr_PB2[i])
        ion5_PB = C5M * np.exp(-q5 * eltr_PB2[i])
        ion6_PB = C6M * np.exp(-q6 * eltr_PB2[i])
        rho_PB2[i]  = q1 * ionC_PB + q2 * ionA_PB + q5 * ion5_PB + q6 * ion6_PB  # unit: eM [PF4Ex3.88(7)]

    ster_Sh = ster[IdxSh]
    Gamma_hat = np.exp(ster_Sh) * GammaB  # [P2(9)]
    GammaB_0 = 1 - (V3 * C3M) / S2 / 1660.5655  # C3M at I = 0
    ster_hat = np.log(Gamma_hat/GammaB_0)
    C3M_hat = C3M * np.exp(-ster_hat)  # [P2(7)]
    print(' --- IdxBorn, IdxSh, GammaB, GammaB_0, Gamma_hat:', IdxBorn, IdxSh, np.around(GammaB, 2), np.around(GammaB_0, 2), np.around(Gamma_hat, 2))
    print(' --- ster_Sh, ster_hat, C3M, C3M_hat:', np.around(ster_Sh, 2), np.around(ster_hat, 2), np.around(C3M / S2, 2), np.around(C3M_hat / S2, 2))
    Vh_sh = 4 * np.pi * (Rsh_c ** 3) / 3 - V1  # V1: hard sphere
    N_0 = C3M / S2 / 1660.5655 * Vh_sh
    N_hat = C3M_hat / S2 / 1660.5655 * Vh_sh
    V_void = Vh_sh - N_hat * V3
    print(' --- BornR_c, V^hat_sh, V_void, V_H2O, N_0, N_hat:', np.around(BornR_c, 2), np.around(Vh_sh, 2), np.around(V_void, 2), np.around(V3, 2), np.around(N_0, 2), np.around(N_hat, 2))

    for i in range( N ):
      if rL[i] > BornR_c and rL[i] <= Rsh_c:  # shell domain
        tBr = ( np.exp(-rL[i] / Lcorr) - np.exp((rL[i] - 2 * BornR_c) / Lcorr) ) * THETA1_c / THETA3_c / rL[i]
        Br = 1 / rL[i] + (THETA4_c / THETA3_c - 1) / Rsh_c \
           - 2 * BornR_c * np.exp(-BornR_c / Lcorr) * THETA1_c / THETA3_c / Lcorr / rL[i] + tBr
        tBr = tBr / (Lcorr ** 2)
        eltr[i] = S1 * Q * Br.real / ϵ_s_x_I  # [P2(26b)]
          #if Br.imag != 0: print(' xxx Br.imag != 0 =', Br.imag) # = 5.47e-17
        psi[i]  = S1 * Q * tBr.real / ϵ_s_x_I * ϵ_0  # [PF4Ex3.88(7)]
        ster[i] = ster_hat
        solv[i] = C3M_hat
        X = (ϵ_s_x_I - 1) / (ϵ_s_x_I + 2)
        ditr[i] = (2 * X + 1) / (1 - X)
        B_hat = 1 / rL[i] - 1 / (LDebye + Rsh_c)
        eltr_2PF[i] = S1 * Q * B_hat / ϵ_s_x_I  # [P2(40b)]
        B_hat = 1 / rL[i] - 1 / (LDebye_PB + Rsh_c)
        eltr_PB1[i] = S1 * Q * B_hat / ϵ_s_x_I
        eltr_PB2[i] = S1 * Q * B_hat / ϵ_s_x


    self.rL, self.eltr, self.psi, self.ster, self.ionC, self.ionA, self.solv, self.rho, self.ditr \
    = rL, eltr, psi / 1660.5655, ster, ionC / S2, ionA / S2, solv / S2, rho / S2, ditr

    self.eltr_2PF, self.eltr_PB1, self.eltr_PB2 = eltr_2PF, eltr_PB1, eltr_PB2
    self.rho_2PF, self.rho_PB1, self.rho_PB2 = rho_2PF / S2, rho_PB1 / S2, rho_PB2 / S2
    self.eltr_DH = eltr_DH  # for Remark 4.4 in 2ndGDH.tex

    self.ion5, self.ion6 = ion5 / S2, ion6 / S2

    self.A1, self.A2, self.A3 = A1 * S1 * Q / ϵ_s_x_I, A2.real * S1 * Q / ϵ_s_x_I, A3.real * S1 * Q / ϵ_s_x_I
