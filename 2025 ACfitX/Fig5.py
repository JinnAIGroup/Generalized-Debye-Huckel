'''
Author: Jinn-Liang Liu, Nov 27, 2025.
Figure 5: NaCl+MgCl2 in H2O at T = 25, 100, 200, 300 ◦C.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from Physical import Solvent, Born, m2M
from Data5 import DataFit
from LSfit import LSfit, Activity, LSfitX

# Solution Parameters:
#   0: void, 1: cation, 2: anion, 3: H2O, 4: MeOH, x or X: mixing percentage of 4 in [0, 1]
#   5: cation, 6: anion, 7: cation, 8: anion
#   Bulk concentrations in M: C1M (array), C2M (array), C3M (scalar), C4M (scalar)
#   ϵ_s_x (scalar): dielectric constant of mixed solvent, V: volume
#   gamma: mean activity data of target salt CA = ca = 1+2 (array)

np.set_printoptions(suppress=True)
plt.figure(figsize=(13,8))
a, b, c = 2, 2, 1

Salts = ['NaCl', 'MgCl2']
Z = 0.68

for salt in Salts:
  if salt == 'NaCl':  Ts = [298.15, 373.15, 473.15, 523.15, 573.15]
  if salt == 'MgCl2': Ts = [298.15, 373.15, 423.15, 523.15]
  g_dataT, g_fitT, g_predT, C1m_X_Mix = (), (), (), ()

  for (T, T_i) in zip(Ts, range(len(Ts))):
    # Part 1: Fiting ...

    S2, C3M, C4M, V3, V4, pH2O, _, ϵ_s_x = Solvent(0, T)
    BornR0, q1, q2, p1, p2, V1, V2, mM, DG = Born(salt, ϵ_s_x, 0, T)

    DF = DataFit(salt)
    g_data = DF.lngamma[T_i]

    C1M = m2M(DF.C1m, mM, DG, 0, T) * S2
    C2M = -q1 * C1M / q2

    IS =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)
    numPW = C3M * pH2O
    numPI = C1M * p1 + C2M * p2
    numPWI = numPW + numPI
    fac_pZ = 1 - Z * IS / C3M
    pH2O_Z = fac_pZ * pH2O
    numPW_Z = C3M * pH2O_Z
    numPWI_Z = numPW_Z + numPI
    X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
    ϵ_s_x_I = (2 * X + 1) / (1 - X)

    R_sh = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)

    ActIn_M1 = (0,0,0,0,0,0)
    ActIn_M2 = (0,0,0,0,0,0)
    ActIn_Mix = (ActIn_M1, ActIn_M2)

    LfIn = (g_data, BornR0, R_sh, salt, C1M, C3M, C4M, IS, \
            q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T, ActIn_Mix)
    LfOut = LSfit(LfIn)

    g_fit, alpha = LfOut.g_fit, LfOut.alpha  # fitted results
    theta = 1 + alpha[0] * (IS ** 0.5) + alpha[1] * IS + alpha[2] * (IS ** 1.5) + alpha[3] * (IS ** 2) + alpha[4] * (IS ** 2.5)

    AAD = np.mean(np.abs(g_fit - g_data))
    print(" alpha, AAD% =", np.around(alpha, 5), np.around(AAD*100, 2), salt)

    g_dataT, g_fitT = g_dataT + (g_data, ), g_fitT + (g_fit, )

    if T_i == 0 and salt == 'MgCl2':
      ActIn_M1 = (0,0,0,0,0,0)
      ActIn_M2 = (0,0,0,0,0,0)
      ActIn_Mix = (ActIn_M1, ActIn_M2)
      ActIn = (theta[-1], BornR0, R_sh[-1], C1M[-1], C3M, C4M, IS[-1], \
               q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I[-1], T)
      ActOut = Activity(ActIn, ActIn_Mix)
      gamma1 = ActOut.gamma1
      gamma2 = ActOut.gamma2

      print(" MgCl2 IS[-1] in Fig 5f =", np.around(IS[-1], 2))
      sum1 = gamma1[0] + gamma1[1] + gamma1[2] + gamma1[3]
      sum2 = gamma2[0] + gamma2[1] + gamma2[2] + gamma2[3]
      print(" gamma1_1, gamma1_2, gamma1_3, gamma1_4, sum1, gamma1_DH =", np.around(gamma1[0], 2), \
        np.around(gamma1[1], 2), np.around(gamma1[2], 2), np.around(gamma1[3], 2), np.around(sum1, 2), np.around(gamma1[4], 2) )
      print(" gamma2_1, gamma2_2, gamma2_3, gamma2_4, sum2, gamma2_DH =", np.around(gamma2[0], 2), \
        np.around(gamma2[1], 2), np.around(gamma2[2], 2), np.around(gamma2[3], 2), np.around(sum2, 2), np.around(gamma2[4], 2) )
      print(" g_DH, g_GDH =", np.around(gamma2[5], 2), np.around(g_fitT[T_i][-1], 2) )

    # Part 2: Mix-Salt Predicting ...

    C1M = m2M(DF.C1m_Mix, mM, DG, 0, T) * S2  # new C1m_Mix
    C2M = -q1 * C1M / q2

    if salt == 'NaCl':  salt_1 = 'MgCl2'  # add 1 salt CA to mix, C = 5, A = 6
    if salt == 'MgCl2': salt_1 = 'NaCl'

    _, q5, q6, p5, p6, V5, V6, mM_Mix, D_Mix = Born(salt_1, ϵ_s_x, 0, T)  # for H2O

    C5m = DF.C1m_Mix  # add C5m to C1m
    C5M = m2M(C5m, mM_Mix, D_Mix, 0, T) * S2
    C6M = -q5 * C5M / q6

    IS_X =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2)  # 2-salt array

    ISX0 = IS[0] if IS[0] < IS_X[0] else IS_X[0]
    IS_X = np.linspace(ISX0, IS_X[-1], num=2 * len(IS_X))

    Spline = InterpolatedUnivariateSpline(IS, g_data, k=1)
    g_data_X = Spline(IS_X)

    C1m_X = IS_X * DF.C1m[-1] / IS_X[-1]
    C1M_X = 2 * IS_X / (q1**2 - q2 * q1)

    C1m_Xmix = C1m_X / 2
    C1M = m2M(C1m_Xmix, mM, DG, 0, T) * S2
    C2M = -q1 * C1M / q2
    _, _, _, _, _, _, _, mM_X, D_X = Born(salt_1, ϵ_s_x, 0, T)
    C5M = m2M(C1m_Xmix, mM_X, D_X, 0, T) * S2
    C6M = -q5 * C5M / q6

    IS_X = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2)  # 2-salt inter/eXtrapolation array
    numPW = C3M * pH2O
    numPI = C1M * p1 + C2M * p2 + C5M * p5 + C6M * p6
    numPWI = numPW + numPI
    fac_pZ = 1 - Z * IS_X / C3M
    pH2O_Z = fac_pZ * pH2O
    numPW_Z = C3M * pH2O_Z
    numPWI_Z = numPW_Z + numPI
    X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
    ϵ_s_x_I = (2 * X + 1) / (1 - X)

    R_sh = (1660.5655 / 8 / (C1M + C2M + C5M + C6M) * S2) ** (1/3)

    ActIn_M1 = (q5, q6, V5, V6, C5M, C6M)
    ActIn_M2 = (0,0,0,0,0,0)
    ActIn_Mix = (ActIn_M1, ActIn_M2)

    alphaX = alpha[0]
    LfInX = (g_data_X, BornR0, R_sh, salt, C1M, C3M, C4M, IS_X, \
             q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T, alphaX, ActIn_Mix)

    LfOut = LSfitX(LfInX)
    alpha_X = LfOut.alpha
    print(" alpha_X =", np.around(alpha_X, 5), salt, "+", salt_1)
    theta_X = 1 + alpha_X[0] * (IS_X ** 0.5) + alpha_X[1] * IS_X + alpha_X[2] * (IS_X ** 1.5) + alpha_X[3] * (IS_X ** 2) + alpha_X[4] * (IS_X ** 2.5)

    g_predT = g_predT + (g_data_X, )  # Predicted results
    C1m_X_Mix = C1m_X_Mix + (C1m_X, )

    if T_i == 0 and salt == 'NaCl':
      print(" NaCl IS[-1], IS_X[-1] in Fig 5a, 5a' =", np.around(IS[-1], 2), np.around(IS_X[-1], 2))
      print(" AC of NaCl at 3 m NaCl, 6 m NaCl, 6 m NaCl+MgCl2 =", np.around(g_fitT[T_i][7], 3), np.around(g_fitT[T_i][-1], 3), np.around(g_predT[T_i][-1], 3))

  # Plot Fig 5
  plt.figure(1)
  plt.subplot(a, b, c)
  if salt == Salts[0]: plt.ylabel(r"$\ln\gamma_\pm$", fontsize=12)

  for T_i in range(len(Ts)):
    if T_i == 0:
      mark = 'sb'
      if salt == 'NaCl':
        plt.text(DF.C1m[-1] - 0.1, g_dataT[T_i][-1] - 0.12, '(a)', fontsize=12)
      if salt == 'MgCl2':
        plt.text(DF.C1m[-1] - 0.1, g_dataT[T_i][-1] - 0.5, '(f)', fontsize=12)
    elif T_i == 1:
      mark = '+g'
      if salt == 'NaCl':
        plt.text(DF.C1m[-1] - 0.1, g_dataT[T_i][-1] - 0.12, '(b)', fontsize=12)
      if salt == 'MgCl2':
        plt.text(DF.C1m[-1] - 0.1, g_dataT[T_i][-1] - 0.5, '(g)', fontsize=12)
    elif T_i == 2:
      mark = 'dk'
      if salt == 'NaCl':
        plt.text(DF.C1m[-1] - 0.1, g_dataT[T_i][-1] - 0.14, '(c)', fontsize=12)
      if salt == 'MgCl2':
        plt.text(DF.C1m[-1] - 0.1, g_dataT[T_i][-1] - 0.5, '(h)', fontsize=12)
    elif T_i == 3:
      mark = 'xb'
      if salt == 'NaCl':
        plt.text(DF.C1m[-1] - 0.1, g_dataT[T_i][-1] - 0.14, '(d)', fontsize=12)
      if salt == 'MgCl2':
        plt.text(DF.C1m[-1] - 0.1, g_dataT[T_i][-1] - 0.5, '(i)', fontsize=12)
    elif T_i == 4:
      mark = 'og'
      if salt == 'NaCl':
        plt.text(DF.C1m[-1] - 0.1, g_dataT[T_i][-1] + 0.06, '(e)', fontsize=12)
    else: mark = 'xb'
    plt.plot(DF.C1m, g_dataT[T_i], mark)
    plt.plot(DF.C1m, g_fitT[T_i], '-r')
    if salt == 'NaCl':  title = '(A) NaCl'
    if salt == 'MgCl2': title = '(B) MgCl$_2$'
    plt.title(title, fontsize=12)

  c = c + b

  plt.figure(1)
  plt.subplot(a, b, c)
  plt.xlabel('m (mol/kg)', fontsize=12)
  if salt == Salts[0]: plt.ylabel(r"$\ln\gamma_\pm$", fontsize=12)

  for T_i in range(len(Ts)):
    if T_i == 0:
      if salt == 'NaCl':
        plt.text(C1m_X_Mix[T_i][-1] - 0.2, g_predT[T_i][-1] - 0.18, '(a\')', fontsize=12)
      if salt == 'MgCl2':
        plt.text(C1m_X_Mix[T_i][-1] - 0.2, g_predT[T_i][-1] - 0.4, '(f\')', fontsize=12)
    elif T_i == 1:
      if salt == 'NaCl':
        plt.text(C1m_X_Mix[T_i][-1] - 0.2, g_predT[T_i][-1] - 0.18, '(b\')', fontsize=12)
      if salt == 'MgCl2':
        plt.text(C1m_X_Mix[T_i][-1] - 0.2, g_predT[T_i][-1] - 0.4, '(g\')', fontsize=12)
    elif T_i == 2:
      if salt == 'NaCl':
        plt.text(C1m_X_Mix[T_i][-1] - 0.2, g_predT[T_i][-1] - 0.18, '(c\')', fontsize=12)
      if salt == 'MgCl2':
        plt.text(C1m_X_Mix[T_i][-1] - 0.2, g_predT[T_i][-1] - 0.4, '(h\')', fontsize=12)
    elif T_i == 3:
      if salt == 'NaCl':
        plt.text(C1m_X_Mix[T_i][-1] - 0.2, g_predT[T_i][-1] - 0.18, '(d\')', fontsize=12)
      if salt == 'MgCl2':
        plt.text(C1m_X_Mix[T_i][-1] - 0.2, g_predT[T_i][-1] - 0.4, '(i\')', fontsize=12)
    elif T_i == 4:
      if salt == 'NaCl':
        plt.text(C1m_X_Mix[T_i][-1] - 0.2, g_predT[T_i][-1] + 0.06, '(e\')', fontsize=12)
    plt.plot(C1m_X_Mix[T_i], g_predT[T_i], '-b')
    if salt == 'NaCl':  title = '(C) NaCl+MgCl$_2$'
    if salt == 'MgCl2': title = '(D) MgCl$_2$+NaCl'
    plt.title(title, fontsize=12)

  c = c - b + 1

plt.show()
