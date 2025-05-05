'''
Author: Jinn-Liang Liu, Apr 6, 2025.

For P2 Example 4.3: 1-salt (LiCl), 2-salt (LiCl+MgCl2), 3-salt (LiCl+MgCl2+LaCl3) in H2O.

Papers:
P1: Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu,
    Generalized Debye–Hückel model for activity coefficients of electrolytes in water–methanol mixtures,
    Fluid Phase Equilibria 565, 113662 (2023)
P2: Chin-Lung Li, Ren-Chuen Chen, Xiaodong Liang, Jinn-Liang Liu,
    Generalized Debye-Hückel theory of ion activities in mixed-salt and mixed-solvent electrolyte solutions, 2024.
P3: Chin-Lung Li, Jinn-Liang Liu, Generalized Debye-Hückel equation from Poisson-Bikerman theory,
    SIAM J. Appl. Math. 80, 2003-2023 (2020).

Jinn-Liang Liu's unpublished research notes:
PF0:   A 3D Poisson-Nernst-Planck Solver for Modeling Biological Ion Channels, August 30, 2012
PF1-4: 3D Poisson-Fermi-Nernst-Planck Solvers for Biological Modeling (Part 1-4), August 6, 2013 - November 11, 2022
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from Physical import Solvent, Born, m2M
from Data4_3 import DataFit
from LSfit import LSfit, Activity, LSfitX

# Solution Parameters:
#   0: void, 1: cation, 2: anion, 3: H2O, 4: MeOH, x: mixing percentage of 3 and 4 in [0, 1]
#   5: cation, 6: anion, 7: cation, 8: anion
#   Bulk concentrations in M: C1M (array), C2M (array), C3M (scalar), C4M (scalar)
#   ϵ_s_x (scalar): dielectric constant of mixed solvent [P1(11)], V: volume
#   gamma: mean activity data [P1(16)] of target salt 1+2 (array)

T, Z = 298.15, 0.68  # Z: polarizability factor [P2(2.20)]

S2, C3M, C4M, V3, V4, pH2O, _, ϵ_s_x = Solvent(0, T)  # x=0 for pure H2O

np.set_printoptions(suppress=True)  # set 0.01 not 1e-2
plt.figure(figsize=(13,8))
a, b, c = 1, 3, 1  # subplot(a, b, c): rows, columns, counter

Salts = ['LiCl', 'MgCl2', 'LaCl3']
#Salts = ['LaCl3']

for salt in Salts:
  # Part 1: Fiting ...

  # Born Radius: BornR0 in pure solvent (no salt) [P1(12)(13)]
  BornR0, q1, q2, p1, p2, V1, V2, mM, DG = Born(salt, ϵ_s_x, 0, T)

  DF = DataFit(salt)  # data to fit
  g_data = DF.lngamma  # gamma: mean activity, ln: log

  # Salt molality (m) to Molarity (M): C1m (array) to C1M
  C1M = m2M(DF.C1m, mM, DG, 0, T) * S2  # 0 for H2O
  C2M = -q1 * C1M / q2  # q1 C1M + q2 C2M = 0

  IS = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)  # Ionic Strength (array)
  numPW = C3M * pH2O  # [P2(2.20)]
  numPI = C1M * p1 + C2M * p2
  numPWI = numPW + numPI
  fac_pZ = 1 - Z * IS / C3M
  pH2O_Z = fac_pZ * pH2O
  numPW_Z = C3M * pH2O_Z
  numPWI_Z = numPW_Z + numPI
  X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
  ϵ_s_x_I = (2 * X + 1) / (1 - X)  # [P2(2.20)]

  R_ca = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)  # array [P2(3.4)]
  Rsh_c, Rsh_a = R_ca, R_ca

  # LSfit() returns g_fit as the best fit to g_data with alpha_i [P1(14)] by Least Squares.
  LfIn = (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, DF.C1m, \
          q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T)
  LfOut = LSfit(LfIn)

  g_fit, alpha, theta_all = LfOut.g_fit, LfOut.alpha, LfOut.theta_all  # fitted results

  AAD = np.mean(np.abs(g_fit - g_data))
  print(" alpha, AAD% =", np.around(alpha, 5), np.around(AAD*100, 2), salt)
  theta = 1 + alpha[0] * (IS ** 0.5) + alpha[1] * IS + alpha[2] * (IS ** 1.5) + alpha[3] * (IS ** 2) + alpha[4] * (IS ** 2.5)
  print(' Fig4 No Mix: BornR0[0], BornR_c[-1], Rsh_c[-1], IS[-1]:', np.around(BornR0[0], 2), np.around(theta[-1] * BornR0[0], 2), np.around(Rsh_c[-1], 2), np.around(IS[-1], 2))

  # Part 2: Mix-Salt Predicting ...

  g_pred_Mix, C1m_X_Mix = (), ()  # X: spline eXtrapolation

  for Mix_i in range(2):
    if Mix_i == 0:  # 2-salt
      C1m_Mix = DF.C1m / 2
      if salt == 'LiCl':  salt_1 = 'MgCl2'
      if salt == 'MgCl2': salt_1 = 'LaCl3'
      if salt == 'LaCl3': salt_1 = 'LiCl'

      C1M = m2M(C1m_Mix, mM, DG, 0, T) * S2
      C2M = -q1 * C1M / q2

      _, q5, q6, p5, p6, V5, V6, mM_Mix, D_Mix = Born(salt_1, ϵ_s_x, 0, T)
      C5M = m2M(C1m_Mix, mM_Mix, D_Mix, 0, T) * S2
      C6M = -q5 * C5M / q6
      IS_X = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2)  # array
    else:  # 3-salt
      C1m_Mix = DF.C1m / 3
      if salt == 'LiCl':  salt_1, salt_2 = 'MgCl2', 'LaCl3'
      if salt == 'MgCl2': salt_1, salt_2 = 'LaCl3', 'LiCl'
      if salt == 'LaCl3': salt_1, salt_2 = 'LiCl',  'MgCl2'

      C1M = m2M(C1m_Mix, mM, DG, 0, T) * S2
      C2M = -q1 * C1M / q2

      _, q5, q6, p5, p6, V5, V6, mM_Mix, D_Mix = Born(salt_1, ϵ_s_x, 0, T)
      C5M = m2M(C1m_Mix, mM_Mix, D_Mix, 0, T) * S2
      C6M = -q5 * C5M / q6

      _, q7, q8, p7, p8, V7, V8, mM_Mix, D_Mix = Born(salt_2, ϵ_s_x, 0, T)
      C7M = m2M(C1m_Mix, mM_Mix, D_Mix, 0, T) * S2
      C8M = -q7 * C7M / q8
      IS_X = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2 + C7M * q7 ** 2 + C8M * q8 ** 2)

    ISX0 = IS[0] if IS[0] < IS_X[0] else IS_X[0]
    IS_X = np.linspace(ISX0, IS_X[-1], num=2 * len(IS_X))  # double spline points

    Spline = InterpolatedUnivariateSpline(IS, g_data, k=1)  # k: spline order 1 (linear), 2, ...
    g_data_X = Spline(IS_X)  # inter/eXtrapolation

    C1m_X = IS_X * DF.C1m[-1] / IS_X[-1]  # scale back to 1-salt DF.C1m
      # 2*IS_X = C1M*q1**2 + C2M*q2**2 = C1M*q1**2 - q1*C1M/q2*q2**2 =>
    C1M_X = 2 * IS_X / (q1**2 - q2 * q1)  # 1-salt

    if Mix_i == 0:  # 2-salt inter/eXtrapolation
      C1m_Xmix = C1m_X / 2
      C1M = m2M(C1m_Xmix, mM, DG, 0, T) * S2
      C2M = -q1 * C1M / q2
      _, _, _, _, _, _, _, mM_X, D_X = Born(salt_1, ϵ_s_x, 0, T)
      C5M = m2M(C1m_Xmix, mM_X, D_X, 0, T) * S2
      C6M = -q5 * C5M / q6

      IS_X = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2)  # array
      numPW = C3M * pH2O
      numPI = C1M * p1 + C2M * p2 + C5M * p5 + C6M * p6
      numPWI = numPW + numPI
      fac_pZ = 1 - Z * IS_X / C3M
      pH2O_Z = fac_pZ * pH2O
      numPW_Z = C3M * pH2O_Z
      numPWI_Z = numPW_Z + numPI
      X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
      ϵ_s_x_I = (2 * X + 1) / (1 - X)

      R_ca = (1660.5655 / 8 / (C1M + C2M + C5M + C6M) * S2) ** (1/3)
      Rsh_c, Rsh_a = R_ca, R_ca

      ActIn_M1 = (q5, q6, V5, V6, C5M, C6M)  # for mix-salt 1
      ActIn_M2 = (0,0,0,0,0,0)               # for mix-salt 2
      ActIn_Mix = (ActIn_M1, ActIn_M2)
    else:  # 3-salt inter/eXtrapolation
      C1m_Xmix = C1m_X / 3
      C1M = m2M(C1m_Xmix, mM, DG, 0, T) * S2
      C2M = -q1 * C1M / q2
      _, _, _, _, _, _, _, mM_X, D_X = Born(salt_1, ϵ_s_x, 0, T)
      C5M = m2M(C1m_Xmix, mM_X, D_X, 0, T) * S2
      C6M = -q5 * C5M / q6
      _, _, _, _, _, _, _, mM_X, D_X = Born(salt_2, ϵ_s_x, 0, T)
      C7M = m2M(C1m_Xmix, mM_X, D_X, 0, T) * S2
      C8M = -q7 * C7M / q8

      IS_X = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2 + C7M * q7 ** 2 + C8M * q8 ** 2)
      numPW = C3M * pH2O
      numPI = C1M * p1 + C2M * p2 + C5M * p5 + C6M * p6 + C7M * p7 + C8M * p8
      numPWI = numPW + numPI
      fac_pZ = 1 - Z * IS_X / C3M
      pH2O_Z = fac_pZ * pH2O
      numPW_Z = C3M * pH2O_Z
      numPWI_Z = numPW_Z + numPI
      X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
      ϵ_s_x_I = (2 * X + 1) / (1 - X)
      R_ca = (1660.5655 / 8 / (C1M + C2M + C5M + C6M + C7M + C8M) * S2) ** (1/3)
      Rsh_c, Rsh_a = R_ca, R_ca

      ActIn_M1 = (q5, q6, V5, V6, C5M, C6M)  # for mix-salt 1
      ActIn_M2 = (q7, q8, V7, V8, C7M, C8M)  # for mix-salt 2
      ActIn_Mix = (ActIn_M1, ActIn_M2)

    alphaX = alpha[0]  # add alphaX to LfIn and get LfInX
    LfInX = (g_data_X, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS_X, DF.C1m, \
             q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T, alphaX, ActIn_Mix)

    LfOut = LSfitX(LfInX)  # input: g_data_X and fixed alpha[0]; output: alpha[1], ...
    alpha_X = LfOut.alpha
    if Mix_i == 0:  # 2-salt
      print(" alpha_X =", np.around(alpha_X, 5), salt, "+", salt_1)
    else:  # 3-salt
      print(" alpha_X =", np.around(alpha_X, 5), salt, "+", salt_1, "+", salt_2)
    theta = 1 + alpha_X[0] * (IS_X ** 0.5) + alpha_X[1] * IS_X + alpha_X[2] * (IS_X ** 1.5) + alpha_X[3] * (IS_X ** 2) + alpha_X[4] * (IS_X ** 2.5)

    print('                    Mix: BornR_c[-1], Rsh_c[-1], IS_X[-1]:', np.around(theta[-1] * BornR0[0], 2), np.around(Rsh_c[-1], 2), np.around(IS_X[-1], 2))

    g_pred_Mix = g_pred_Mix + (g_data_X, )
    C1m_X_Mix = C1m_X_Mix + (C1m_X, )

  # Plot predicted results
  plt.figure(1)
  plt.subplot(a, b, c)
  plt.plot(DF.C1m, g_data, 'k.')
  plt.plot(DF.C1m, g_fit, '-r')
  if salt == 'LiCl':
    plt.text(DF.C1m[-1] - 1.1, g_data[-1] - 0.1, '(a1)', fontsize=12)
    title = 'LiCl'
  if salt == 'MgCl2':
    plt.text(DF.C1m[-1] - 1.1, g_data[-1] - 0.2, '(b1)', fontsize=12)
    title = 'MgCl$_2$'
  if salt == 'LaCl3':
    plt.text(DF.C1m[-1] - 0.8, g_data[-1] - 0.25, '(c1)', fontsize=12)
    title = 'LaCl$_3$'
  plt.title(title, fontsize=12)

  plt.xlabel('m (mol/kg)', fontsize=12)
  if salt == Salts[0]: plt.ylabel(r"$\ln\gamma_\pm$", fontsize=12)

  for Mix_i in range(2):
    x = C1m_X_Mix[Mix_i][-1]
    y = g_pred_Mix[Mix_i][-1]
    if Mix_i == 0 and salt == 'LiCl':  plt.text(x - 0.6, y + 0.20, '(a2)', fontsize=12)
    if Mix_i == 0 and salt == 'MgCl2': plt.text(x - 1.2, y - 0.27, '(b2)', fontsize=12)
    if Mix_i == 0 and salt == 'LaCl3': plt.text(x - 0.6, y - 0.05, '(c2)', fontsize=12)
    if Mix_i == 1 and salt == 'LiCl':  plt.text(x - 0.6, y - 1.00, '(a3)', fontsize=12)
    if Mix_i == 1 and salt == 'MgCl2': plt.text(x - 1.0, y - 0.14, '(b3)', fontsize=12)
    if Mix_i == 1 and salt == 'LaCl3': plt.text(x - 0.3, y - 0.28, '(c3)', fontsize=12)

    plt.plot(C1m_X_Mix[Mix_i], g_pred_Mix[Mix_i], '-b')

  c = c + 1

plt.show()
