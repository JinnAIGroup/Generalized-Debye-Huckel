'''
Author: Jinn-Liang Liu, May 5, 2025.

For P2 Example 4.1: LiCl, NaCl, NaBr in (H2O)x+(MeOH)1−x.

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
import matplotlib.pyplot as plt

from Physical import Solvent, Born, m2M
from Data4_1 import DataFit, DataPredict
from LSfit import LSfit, Activity

# Solution Parameters:
#   0: void, 1: cation, 2: anion, 3: H2O, 4: MeOH, x or X: mixing percentage of 3 and 4 in [0, 1]
#   5: cation, 6: anion, 7: cation, 8: anion
#   Bulk concentrations in M: C1M (array), C2M (array), C3M (scalar), C4M (scalar)
#   ϵ_s_x (scalar): dielectric constant of mixed solvent [P1(11)], V: volume
#   gamma: mean activity data [P1(16)] of target salt 1+2 (array)

T, Z = 298.15, 0.68  # Z: polarizability factor [P2(2.22)]

np.set_printoptions(suppress=True)  # set 0.01 not 1e-2
plt.figure(figsize=(13,8))
a, b, c = 2, 3, 1  # subplot(a, b, c): rows, columns, counter for Fit to H2O

Salts = ['NaF', 'NaCl', 'NaBr']

for salt in Salts:
  # Part 1: H2O Fiting ...

  S2, C3M, C4M, V3, V4, pH2O, pMeOH, ϵ_s_x = Solvent(0, T)  # x=0 for pure H2O

  # Born Radius: BornR0 in pure solvent (no salt) [P1(12), (13)]
  BornR0, q1, q2, p1, p2, V1, V2, mM, DG = Born(salt, ϵ_s_x, 0, T)  # for H2O

  DF = DataFit(salt)  # pure-H2O activity data to be fitted
  g_data = np.log(DF.gamma)  # gamma: mean activity [P1(16)]
  #g_data = DF.lngamma  # gamma: mean activity [P1(16)]

  # Salt molality (m) to Molarity (M): C1m (array) to C1M
  C1M = m2M(DF.C1m, mM, DG, 0, T) * S2  # for H2O
  C2M = -q1 * C1M / q2  # q1 C1M + q2 C2M = 0

  IS =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)  # Ionic Strength (array)
  numPW = C3M * pH2O  # [P2(2.22)]
  numPI = C1M * p1 + C2M * p2
  numPWI = numPW + numPI
  fac_pZ = 1 - Z * IS / C3M
  pH2O_Z = fac_pZ * pH2O
  numPW_Z = C3M * pH2O_Z
  numPWI_Z = numPW_Z + numPI
  X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
  ϵ_s_x_I = (2 * X + 1) / (1 - X)  # [P2(2.22)]

  R_ca = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)  # array [P2(3.4)]
  Rsh_c, Rsh_a = R_ca, R_ca

  # LSfit() returns g_fit as the best fit to g_data with alpha_i [P1(14)] by Least Squares.
  LfIn = (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, DF.C1m, \
          q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T)
  LfOut = LSfit(LfIn)

  g_fit, alpha = LfOut.g_fit, LfOut.alpha  # fitted results
  plot_theta = 1 + alpha[0] * (IS ** 0.5) + alpha[1] * IS + alpha[2] * (IS ** 1.5) + alpha[3] * (IS ** 2) + alpha[4] * (IS ** 2.5)

  AAD1 = np.mean(np.abs(g_fit - g_data))
  print(" alpha, AAD1% =", np.around(alpha, 5), np.around(AAD1*100, 2), salt)

  # Part 2: MeOH Fiting for alphaD ...

  DP = DataPredict(salt)  # miXed-solvent activity data to be compared with predicted results
  g_dataX, mixNo, C1mX = DP.g_dataX, DP.mixNo, DP.C1mX

  x, C1m_x, g_dataY = 0.2, C1mX[0], np.log(g_dataX[0])

  _, C3M, C4M, _, _, _, _, ϵ_s_x = Solvent(x, T)
  BornR0, _, _, _, _, _, _, _, _ = Born(salt, ϵ_s_x, x, T)
  C1M = m2M(C1m_x, mM, DG, x, T) * S2
  C2M = -q1 * C1M / q2

  IS =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)
  numPW = C3M * pH2O + C4M * pMeOH  # [P2(2.22)]
  numPI = C1M * p1 + C2M * p2
  numPWI = numPW + numPI
  fac_pZ = 1 - Z * IS / (C3M + C4M)
  pH2O_Z, pMeOH_Z = fac_pZ * pH2O, fac_pZ * pMeOH
  numPW_Z = C3M * pH2O_Z + C4M * pMeOH_Z
  numPWI_Z = numPW_Z + numPI
  X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
  ϵ_s_x_I = (2 * X + 1) / (1 - X)  # [P2(2.22)]

  R_ca = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)  # array [P2(3.4)]
  Rsh_c, Rsh_a = R_ca, R_ca

  LfIn = (g_dataY, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, DF.C1m, \
          q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T)
  LfOut = LSfit(LfIn)

  alphaY = LfOut.alpha
  alphaD = alphaY - alpha

  AAD2 = np.mean(np.abs(LfOut.g_fit - g_dataY))
  print(" alphaY, AAD2% =", np.around(alphaY, 5), np.around(AAD2*100, 2))
  print(" alphaD =", np.around(alphaD, 5))
  g_predX, plot_thetaX = (), ()

  # Part 3: Mix-Solvent Predicting ...

  for i in range(mixNo):
    if   i == 0: x, C1m_x = 0.2, C1mX[0]
    elif i == 1: x, C1m_x = 0.4, C1mX[1]
    elif i == 2: x, C1m_x = 0.6, C1mX[2]
    elif i == 3: x, C1m_x = 0.8, C1mX[3]
    elif i == 4: x, C1m_x = 1.0, C1mX[4]
    else: warnings.warn('Warning: Index i out of range.')

    _, C3M, C4M, _, _, _, _, ϵ_s_x = Solvent(x, T)
    BornR0, _, _, _, _, _, _, _, _ = Born(salt, ϵ_s_x, x, T)
    C1M = m2M(C1m_x, mM, DG, x, T) * S2
    C2M = -q1 * C1M / q2

    IS =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)
    numPW = C3M * pH2O + C4M * pMeOH  # [P2(2.22)]
    numPI = C1M * p1 + C2M * p2
    numPWI = numPW + numPI
    fac_pZ = 1 - Z * IS / (C3M + C4M)
    pH2O_Z, pMeOH_Z = fac_pZ * pH2O, fac_pZ * pMeOH
    numPW_Z = C3M * pH2O_Z + C4M * pMeOH_Z
    numPWI_Z = numPW_Z + numPI
    X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
    ϵ_s_x_I = (2 * X + 1) / (1 - X)  # [P2(2.22)]

    R_ca = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)  # array [P2(3.4)]
    Rsh_c, Rsh_a = R_ca, R_ca

    alpha_x = alpha + x * alphaD
    theta = 1 + alpha_x[0] * (IS ** 0.5) + alpha_x[1] * IS + alpha_x[2] * (IS ** 1.5) + alpha_x[3] * (IS ** 2) + alpha_x[4] * (IS ** 2.5)
    ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, IS, DF.C1m, \
             q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T)

    ActIn_M1 = (0,0,0,0,0,0)  # for mix-salt 1
    ActIn_M2 = (0,0,0,0,0,0)  # for mix-salt 2
    ActIn_Mix = (ActIn_M1, ActIn_M2)

    ActOut = Activity(ActIn, ActIn_Mix)  # Prediction
    g_predX = g_predX + (ActOut.g_PF, )  # Predicted results

    PTX = 1 + alpha_x[0] * (IS ** 0.5) + alpha_x[1] * IS + alpha_x[2] * (IS ** 1.5) + alpha_x[3] * (IS ** 2) + alpha_x[4] * (IS ** 2.5)
    plot_thetaX = plot_thetaX + (PTX, )

  # Plot fitted results [P2F2]
  plt.figure(1)
  plt.subplot(a, b, c)
  plt.plot(DF.C1m, g_data, 'k.')
  plt.plot(DF.C1m, g_fit, '-r')
  N = len(DF.C1m)
  if salt == 'NaF':
    plt.text(DF.C1m[N-1] - 0.09, g_data[N-1] - 0.04, 'x = 0')  # fitted curve
  if salt == 'NaCl':
    plt.text(DF.C1m[N-1] - 0.5, g_data[N-1] - 0.08, 'x = 0')  # fitted curve
  else:
    plt.text(DF.C1m[N-1] - 0.5, g_data[N-1] - 0.12, 'x = 0')  # fitted curve
  if salt == 'NaF':  title = '(A) NaF'
  if salt == 'NaCl': title = '(B) NaCl'
  if salt == 'NaBr': title = '(C) NaBr'
  plt.title(title, fontsize=12)

  if salt == Salts[0]: plt.ylabel(r"$\ln\gamma_\pm$", fontsize=12)

  # Plot predicted results [P2F2]
  for i in range(mixNo):
    if i == 0:
      g_data_x, g_pred_x = np.log(g_dataX[0]), g_predX[0]
      N = len(C1mX[0])
      if salt == 'NaCl':
        plt.text(C1mX[0][N-1] - 0.2, g_data_x[N-1] - 0.08, '0.2')
        plt.plot(C1mX[0], g_pred_x, 'b')
        AAD3 = np.mean(np.abs(g_pred_x - g_data_x))
        print(" AAD3% =", np.around(AAD3*100, 2))
      else:
        plt.text(C1mX[0][N-1] + 0.02, g_data_x[N-1] - 0.02, '0.2')
        plt.plot(C1mX[0], g_pred_x, '-r')
      plt.plot(C1mX[0], g_data_x, 'k.')
    elif i == 1:
      g_data_x, g_pred_x = np.log(g_dataX[1]), g_predX[1]
      N = len(C1mX[1])
      if salt == 'NaCl':
        plt.plot(C1mX[1], g_pred_x, '-r')
      else:
        plt.plot(C1mX[1], g_pred_x, 'b')
        AAD3 = np.mean(np.abs(g_pred_x - g_data_x))
        print(" AAD3% =", np.around(AAD3*100, 2))
      plt.plot(C1mX[1], g_data_x, 'k.')
      plt.text(C1mX[1][N-1] + 0.03, g_data_x[N-1] - 0.02, '0.4')
    elif i == 2:
      g_data_x, g_pred_x = np.log(g_dataX[2]), g_predX[2]
      N = len(C1mX[2])
      plt.text(C1mX[2][N-1] + 0.03, g_data_x[N-1] - 0.02, '0.6')
      plt.plot(C1mX[2], g_data_x, 'k.')
      plt.plot(C1mX[2], g_pred_x, 'b')
      AAD3 = np.mean(np.abs(g_pred_x - g_data_x))
      print(" AAD3% =", np.around(AAD3*100, 2))
    elif i == 3:
      g_data_x, g_pred_x = np.log(g_dataX[3]), g_predX[3]
      N = len(C1mX[3])
      plt.text(C1mX[3][N-1] + 0.03, g_data_x[N-1] - 0.02, '0.8')
      plt.plot(C1mX[3], g_data_x, 'k.')
      plt.plot(C1mX[3], g_pred_x, 'b')
      AAD3 = np.mean(np.abs(g_pred_x - g_data_x))
      print(" AAD3% =", np.around(AAD3*100, 2))
    elif i == 4:
      g_data_x, g_pred_x = np.log(g_dataX[4]), g_predX[4]
      N = len(C1mX[4])
      plt.text(C1mX[4][N-1] + 0.03, g_data_x[N-1] - 0.02, '1.0')
      plt.plot(C1mX[4], g_data_x, 'k.')
      plt.plot(C1mX[4], g_pred_x, 'b')
      AAD3 = np.mean(np.abs(g_pred_x - g_data_x))
      print(" AAD3% =", np.around(AAD3*100, 2))

  c = c + b

  # Plot theta results [P2F2]
  plt.figure(1)
  plt.subplot(a, b, c)
  plt.plot(DF.C1m, plot_theta, '-r')
  plt.xlabel('m (mol/kg)', fontsize=12)

  if salt == Salts[0]: plt.ylabel(r"$\theta$", fontsize=12)

  for i in range(mixNo):
    if i == 0:
      plt.plot(C1mX[0], plot_thetaX[0], '--g')
    elif i == 1:
      plt.plot(C1mX[1], plot_thetaX[1], ':b')
    elif i == 2:
      plt.plot(C1mX[2], plot_thetaX[2], '-.r')
    elif i == 3:
      plt.plot(C1mX[3], plot_thetaX[3], '-+g')
    elif i == 4:
      plt.plot(C1mX[4], plot_thetaX[4], 'xb')

  c = c - b + 1

plt.show()
