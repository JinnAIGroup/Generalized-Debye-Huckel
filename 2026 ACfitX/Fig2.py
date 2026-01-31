'''
Author: Jinn-Liang Liu, Dec 27, 2025.

For P2 Figure 2: LiCl, NaCl, NaBr in (H2O)x+(MeOH)1−x.

P1: Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu,
    Generalized Debye–Hückel model for activity coefficients of electrolytes in water–methanol mixtures,
    Fluid Phase Equilibria 565, 113662 (2023)
P2: Chin-Lung Li, Ren-Chuen Chen, Xiaodong Liang, Jinn-Liang Liu,
    Generalized Debye-Hückel theory for ion activities in mixed-salt and mixed-solvent electrolyte solutions, 2025.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from Physical import Solvent, Born, m2M
from Data2 import DataFit, DataPredict
from LSfit import LSfit, Activity

# Solution Parameters:
#   0: void, 1: cation, 2: anion, 3: H2O, 4: MeOH, x or X: mixing percentage of 4 in [0, 1]
#   5: cation, 6: anion, 7: cation, 8: anion
#   Bulk concentrations in M: C1M (array), C2M (array), C3M (scalar), C4M (scalar)
#   ϵ_s_x (scalar): dielectric constant of mixed solvent, V: volume
#   gamma: mean activity data of target salt CA = ca = 1+2 (array)

np.set_printoptions(suppress=True)  # set 0.01 not 1e-2
plt.figure(figsize=(13,8))
a, b, c = 2, 3, 1  # subplot(a, b, c): rows, columns, counter

Salts = ['NaF', 'NaCl', 'NaBr']
T, Z = 298.15, 0.68  # Z: polarizability factor [P2(25)]

for salt in Salts:
  # Part 1: H2O Fiting ...

  S2, C3M, C4M, V3, V4, pH2O, pMeOH, ϵ_s_x = Solvent(0, T)  # x=0 for pure H2O

  # Born Radius (array): BornR0[0] for c, [1] for a, in pure solvent (no salt) [P2(32)]
  BornR0, q1, q2, p1, p2, V1, V2, mM, DG = Born(salt, ϵ_s_x, 0, T)  # for H2O

  DF = DataFit(salt)  # data to fit
  g_data = np.log(DF.gamma)  # ln(gamma)

  # Salt molality (m) to Molarity (M): C1m (array) to C1M (array)
  C1M = m2M(DF.C1m, mM, DG, 0, T) * S2  # 0 for H2O
  C2M = -q1 * C1M / q2  # q1 C1M + q2 C2M = 0

  IS =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)  # Ionic Strength (array)
  numPW = C3M * pH2O  # Water Polarizability
  numPI = C1M * p1 + C2M * p2  # Ion Polarizability
  numPWI = numPW + numPI
  fac_pZ = 1 - Z * IS / C3M
  pH2O_Z = fac_pZ * pH2O
  numPW_Z = C3M * pH2O_Z
  numPWI_Z = numPW_Z + numPI
  X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
  ϵ_s_x_I = (2 * X + 1) / (1 - X)  # [P2(25)]

  R_sh = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)  # shell Radius (array)

  ActIn_M1 = (0,0,0,0,0,0)  # for mix-salt 1
  ActIn_M2 = (0,0,0,0,0,0)  # for mix-salt 2
  ActIn_Mix = (ActIn_M1, ActIn_M2)

  # LSfit() [P1 Step 1-5] returns the best g_fit to g_data with alpha[i] [P2(32)] by Least Squares.
  LfIn = (g_data, BornR0, R_sh, salt, C1M, C3M, C4M, IS, \
          q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T, ActIn_Mix)
  LfOut = LSfit(LfIn)

  g_fit, alpha = LfOut.g_fit, LfOut.alpha  # fitted results
  plot_theta = 1 + alpha[0] * (IS ** 0.5) + alpha[1] * IS + alpha[2] * (IS ** 1.5) + alpha[3] * (IS ** 2) + alpha[4] * (IS ** 2.5)

  AAD1 = np.mean(np.abs(g_fit - g_data))
  print(" alpha, AAD1% =", np.around(alpha, 5), np.around(AAD1*100, 2), salt)

  # Part 2: MeOH Fiting for alphaD ...

  DP = DataPredict(salt)  # miXed-solvent activity data to be compared with predicted results
  g_dataX, mixNo, C1mX = DP.g_dataX, DP.mixNo, DP.C1mX

  if salt == 'NaF':
    xhat, C1m_xhat, g_data_xhat = 0.2, C1mX[0], np.log(g_dataX[0])
  if salt == 'NaCl':
    xhat, C1m_xhat_1, g_data_xhat_1 = 0.4, C1mX[1], np.log(g_dataX[1])
    # extrapolate C1mX[1] to C1mX[0]
    C1M = m2M(C1mX[1], mM, DG, 0.4, T) * S2
    C2M = -q1 * C1M / q2
    IS = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)
    C1M_X = m2M(C1mX[0], mM, DG, 0.2, T) * S2
    C2M_X = -q1 * C1M_X / q2
    IS_X = 0.5 * (C1M_X * q1 ** 2 + C2M_X * q2 ** 2)
    ISX0 = IS[0] if IS[0] < IS_X[0] else IS_X[0]
    IS_X = np.linspace(ISX0, IS_X[-1], num=len(IS_X))
    Spline = InterpolatedUnivariateSpline(IS, g_data_xhat_1, k=1)  # k: spline order 1 (linear), 2, ...
    g_data_xhat = Spline(IS_X)  # eXtrapolation
    C1m_xhat = IS_X * C1mX[0][-1] / IS_X[-1]  # interpolation
  if salt == 'NaBr':
    xhat, C1m_xhat, g_data_xhat = 0.8, C1mX[3], np.log(g_dataX[3])

  _, C3M, C4M, _, _, _, _, ϵ_s_x = Solvent(xhat, T)
  BornR0, _, _, _, _, _, _, _, _ = Born(salt, ϵ_s_x, xhat, T)
  C1M = m2M(C1m_xhat, mM, DG, xhat, T) * S2
  C2M = -q1 * C1M / q2

  IS =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)
  numPW = C3M * pH2O + C4M * pMeOH
  numPI = C1M * p1 + C2M * p2
  numPWI = numPW + numPI
  fac_pZ = 1 - Z * IS / (C3M + C4M)
  pH2O_Z, pMeOH_Z = fac_pZ * pH2O, fac_pZ * pMeOH
  numPW_Z = C3M * pH2O_Z + C4M * pMeOH_Z
  numPWI_Z = numPW_Z + numPI
  X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
  ϵ_s_x_I = (2 * X + 1) / (1 - X)

  R_sh = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)

  ActIn_M1 = (0,0,0,0,0,0)
  ActIn_M2 = (0,0,0,0,0,0)
  ActIn_Mix = (ActIn_M1, ActIn_M2)

  LfIn = (g_data_xhat, BornR0, R_sh, salt, C1M, C3M, C4M, IS, \
          q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T, ActIn_Mix)
  LfOut = LSfit(LfIn)

  g_fit_xhat, alpha_xhat = LfOut.g_fit, LfOut.alpha  # fitted results
  alphaD = alpha_xhat - alpha  # [P2(33)]

  AAD2 = np.mean(np.abs(g_fit_xhat - g_data_xhat))
  print(" alpha_xhat =", np.around(alpha_xhat, 5))
  print(" alphaD, AAD2% =", np.around(alphaD, 5), np.around(AAD2*100, 2))
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
    numPW = C3M * pH2O + C4M * pMeOH
    numPI = C1M * p1 + C2M * p2
    numPWI = numPW + numPI
    fac_pZ = 1 - Z * IS / (C3M + C4M)
    pH2O_Z, pMeOH_Z = fac_pZ * pH2O, fac_pZ * pMeOH
    numPW_Z = C3M * pH2O_Z + C4M * pMeOH_Z
    numPWI_Z = numPW_Z + numPI
    X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
    ϵ_s_x_I = (2 * X + 1) / (1 - X)

    R_sh = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)

    if x <= xhat:
      alpha_x = alpha + x * alphaD / xhat  # [P2(35)]
    else:
      alpha_x = alpha_xhat + (x - xhat) * alphaD / xhat

    theta = 1 + alpha_x[0] * (IS ** 0.5) + alpha_x[1] * IS + alpha_x[2] * (IS ** 1.5) + alpha_x[3] * (IS ** 2) + alpha_x[4] * (IS ** 2.5)
    ActIn = (theta, BornR0, R_sh, C1M, C3M, C4M, IS, \
             q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)

    ActIn_M1 = (0,0,0,0,0,0)
    ActIn_M2 = (0,0,0,0,0,0)
    ActIn_Mix = (ActIn_M1, ActIn_M2)

    ActOut = Activity(ActIn, ActIn_Mix)  # Prediction
    g_predX = g_predX + (ActOut.g_PF, )  # Predicted results

    PTX = 1 + alpha_x[0] * (IS ** 0.5) + alpha_x[1] * IS + alpha_x[2] * (IS ** 1.5) + alpha_x[3] * (IS ** 2) + alpha_x[4] * (IS ** 2.5)
    plot_thetaX = plot_thetaX + (PTX, )

  # Plot Fig 2
  plt.figure(1)
  plt.subplot(a, b, c)
  plt.plot(DF.C1m, g_data, 'k.')
  plt.plot(DF.C1m, g_fit, '-r')  # fitted at x = 0
  if salt == 'NaCl':
    plt.plot(C1m_xhat_1, g_data_xhat_1, 'k.')
    plt.plot(C1m_xhat[:11], g_fit_xhat[:11], '-r')  # fitted at x = xhat
  else:
    plt.plot(C1m_xhat, g_data_xhat, 'k.')
    plt.plot(C1m_xhat, g_fit_xhat, '-r')

  N = len(DF.C1m)
  if salt == 'NaF':
    plt.text(DF.C1m[N-1] - 0.09, g_data[N-1] - 0.04, 'x = 0')  # fitted curve
  elif salt == 'NaCl':
    plt.text(DF.C1m[N-1] - 0.5, g_data[N-1] - 0.08, 'x = 0')
  elif salt == 'NaBr':
    plt.text(DF.C1m[N-1] - 0.5, g_data[N-1] - 0.12, 'x = 0')

  if salt == 'NaF':  title = '(A) NaF'
  if salt == 'NaCl': title = '(B) NaCl'
  if salt == 'NaBr': title = '(C) NaBr'
  plt.title(title, fontsize=12)

  if salt == Salts[0]: plt.ylabel(r"$\ln\gamma_\pm$", fontsize=12)

  for i in range(mixNo):
    if i == 0:
      g_data_x, g_pred_x = np.log(g_dataX[0]), g_predX[0]
      N = len(C1mX[0])
      if salt == 'NaCl':
        plt.text(C1mX[0][N-1] - 0.2, g_data_x[N-1] - 0.08, '0.2')
      else:
        plt.text(C1mX[0][N-1] + 0.02, g_data_x[N-1] - 0.02, '0.2')
      if salt != 'NaF':
        plt.plot(C1mX[0], g_pred_x, 'b')
        plt.plot(C1mX[0], g_data_x, 'k.')
        AAD3 = np.mean(np.abs(g_pred_x - g_data_x))
        print(" AAD3% =", np.around(AAD3*100, 2))
    elif i == 1:
      g_data_x, g_pred_x = np.log(g_dataX[1]), g_predX[1]
      N = len(C1mX[1])
      plt.text(C1mX[1][N-1] + 0.03, g_data_x[N-1] - 0.02, '0.4')
      if salt != 'NaCl':
        plt.plot(C1mX[1], g_pred_x, 'b')
        plt.plot(C1mX[1], g_data_x, 'k.')
        AAD3 = np.mean(np.abs(g_pred_x - g_data_x))
        print(" AAD3% =", np.around(AAD3*100, 2))
    elif i == 2:
      g_data_x, g_pred_x = np.log(g_dataX[2]), g_predX[2]
      N = len(C1mX[2])
      if salt == 'NaBr':
        plt.text(C1mX[2][N-1] + 0.03, g_data_x[N-1] - 0.06, '0.6')
      else:
        plt.text(C1mX[2][N-1] + 0.03, g_data_x[N-1] - 0.02, '0.6')
      plt.plot(C1mX[2], g_data_x, 'k.')
      plt.plot(C1mX[2], g_pred_x, 'b')
      AAD3 = np.mean(np.abs(g_pred_x - g_data_x))
      print(" AAD3% =", np.around(AAD3*100, 2))
    elif i == 3:
      g_data_x, g_pred_x = np.log(g_dataX[3]), g_predX[3]
      N = len(C1mX[3])
      plt.text(C1mX[3][N-1] + 0.03, g_data_x[N-1] - 0.02, '0.8')
      if salt != 'NaBr':
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
