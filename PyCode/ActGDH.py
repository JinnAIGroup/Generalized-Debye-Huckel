# Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu
# Prediction of activity coefficients in water-methanol mixtures using
# a generalized Debye-Huckel model, arXiv:2209.01892
# September 20, 2022

import numpy as np
import matplotlib.pyplot as plt
import warnings

from Data import DataFit, DataPredict
from Newton import Newton
from LSfit import LSfit
from Activity import Activity

def Solvent(x = None):
  V3 = 4 * np.pi * 1.4 ** 3 / 3
  V4 = 4 * np.pi * 1.915 ** 3 / 3
  Vx = (1 - x) * V3 + x * V4
  C3M = (1 - x) * 55.5
  C4M = x * 24.55
  CxM = (1 - x) * 55.5 + x * 24.55
  diS = (1 - x) * 78.45 + x * 32.66

  return C3M, C4M, CxM, V3, V4, Vx, diS


def Born(salt = None, diS = None, x = None):
  epsln, e, mol = 8.854187, 1.6022, 6.022045

  if salt == 'NaF':
    q1, q2 = 1, -1
    V1 = 4 * np.pi * 0.95 ** 3 / 3
    V2 = 4 * np.pi * 1.36 ** 3 / 3
    mM, D = 41.99, 41.38
    FcatH2O, FcatMeOH = -103.2, -91.9
    FanH2O,  FanMeOH  = -104.4, -109.2
  elif salt == 'NaCl':
    q1, q2 = 1, -1
    V1 = 4 * np.pi * 0.95 ** 3 / 3
    V2 = 4 * np.pi * 1.81 ** 3 / 3
    mM, D = 58.44, 46.62
    FcatH2O, FcatMeOH = -103.2, -91.9
    FanH2O,  FanMeOH  = -74.5,  -81.1
  elif salt == 'NaBr':
    q1, q2 = 1, -1
    V1 = 4 * np.pi * 0.95 ** 3 / 3
    V2 = 4 * np.pi * 1.95 ** 3 / 3
    mM, D = 102.89, 77
    FcatH2O, FcatMeOH = -103.2, -91.9
    FanH2O,  FanMeOH  = -68.3,  -75.1
  else:
    warnings.warn('Warning: Salt not found.')

  Fcat = (1 - x) * FcatH2O + x * FcatMeOH
  Fan  = (1 - x) * FanH2O  + x * FanMeOH
  BornR0c = - q1 * q1 * e * e * mol * 10000 * (1 - 1 / diS) / (Fcat * 4.1868 * 8 * np.pi * epsln)
  BornR0a = - q2 * q2 * e * e * mol * 10000 * (1 - 1 / diS) / (Fan * 4.1868 * 8 * np.pi * epsln)
  BornR0 = np.array([BornR0c, BornR0a])

  return BornR0, q1, q2, V1, V2, mM, D


def m2M(m = None, mM = None, D = None, x = None):
  # 0.9971 = density of water
  # 0.9128 = density of 50-50 water-methanol mixture
  # 0.7866 = density of methanol
  rho_0 = (x - 0.5) * (x - 1) / 0.5 * 0.9971 - x * (x - 1) / 0.25 * 0.9128 + x * (x - 0.5) / 0.5 * 0.7866
  rho_s = rho_0 + D * m / 1000
  M = np.multiply(1000 * m, rho_s) / (1000 + np.multiply(m, mM))

  return M

#--- Physical Constants [Table 1 in the paper]
T, epsln, e, mol, kB, V0 = 298.15, 8.854187, 1.6022, 6.022045, 1.380649, 1.0
kBTe = (kB * T / e) * 0.0001
S2 = 0.1 * mol * e / (kBTe * epsln)

a = 2  # number of rows
b = 3  # number of columns
c = 1  # initialize plot counter

plt.figure(figsize=(10,8))

for salts in range(1, 4):
  #--- Solvent Parameters: diS: dielectric constant [(11)], Vx: volume [(18)]
  #    CxM (scalar): bulk concentration in M [(18)]
  #    1: cation, 2: anion, 3: H2O, 4: MeOH, x: mixing percentage in [0, 1]
  C3M, C4M, CxM, V3, V4, Vx, diS = Solvent(0)
  C3M, C4M, CxM = C3M * S2, C4M * S2, CxM * S2

  if salts == 1:
    salt = 'NaF'
  elif salts == 2:
    salt = 'NaCl'
  elif salts == 3:
    salt = 'NaBr'
  else:
    warnings.warn('Warning: Index salts out of range.')

  #--- Born Radii: BornR0 in pure solvent (no salt) [(12), (13)]
  BornR0, q1, q2, V1, V2, mM, D = Born(salt, diS, 0)

  #--- Activity Data to Fit: C1m (vector): concentration in molality (m), gamma: mean activity [(16)]
  DF = DataFit(salt)
  g_data = np.log(DF.gamma)

  #--- Salt molality (m) to Molarity (M): C1m to C1M
  C1M = m2M(DF.C1m, mM, D, 0) * S2
  C2M = q1 * C1M
  BPfrac = (V1 * C1M + V2 * C2M + V3 * C3M + V4 * C4M) / S2 / 1660.6

  #--- Newton() iteratively solves nonlinear [(18)] for V_sh that yields Rsh_c and Rsh_a.
  NtIn = (C1M, CxM, V0, V1, V2, Vx, S2, BPfrac)
  NtOut = Newton(NtIn)
  Rsh_c, Rsh_a = NtOut.Rsh_c, NtOut.Rsh_a

  #--- LSfit() returns g_fit as the best fit to g_data with alpha(1), (2), (3) [(14)] by Least Squares.
  LfIn = (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, \
          q1, q2, V0, V1, V2, V3, V4, diS, T)
  LfOut = LSfit(LfIn)
  g_fit, alpha, theta = LfOut.g_fit, LfOut.alpha, LfOut.theta

  print(" alpha = ", alpha)

  plot_theta = 1 + alpha[0] * (C1M ** 0.5) + alpha[1] * C1M + alpha[2] * (C1M ** 1.5)

  #--- Experimental data of mixed-solvent solution to predict
  DP = DataPredict(salt)
  mixNo, C1mX, g_dataX, delta_alpha = DP.mixNo, DP.C1mX, DP.g_dataX, DP.delta_alpha

  g_predX, plot_thetaX = (), ()
  for i in range(mixNo):
    if i == 0:
      x, C1m_x = 0.2, C1mX[0]
    elif i == 1:
      x, C1m_x = 0.4, C1mX[1]
    elif i == 2:
      x, C1m_x = 0.6, C1mX[2]
    elif i == 3:
      x, C1m_x = 0.8, C1mX[3]
    elif i == 4:
      x, C1m_x = 1.0, C1mX[4]
    else:
      warnings.warn('Warning: Index i out of range.')

    C3M, C4M, CxM, V3, V4, Vx, diS = Solvent(x)
    C3M, C4M, CxM = C3M * S2, C4M * S2, CxM * S2

    BornR0, q1, q2, V1, V2, mM, D = Born(salt, diS, x)
    C1M = m2M(C1m_x, mM, D, x) * S2
    C2M = q1 * C1M
    BPfrac = (V1 * C1M + V2 * C2M + V3 * C3M + V4 * C4M) / S2 / 1660.6

    NtIn = (C1M, CxM, V0, V1, V2, Vx, S2, BPfrac)
    NtOut = Newton(NtIn)
    Rsh_c, Rsh_a = NtOut.Rsh_c, NtOut.Rsh_a

    alpha_x = alpha + x * delta_alpha
    theta = 1 + alpha_x[0] * (C1M ** 0.5) + alpha_x[1] * C1M + alpha_x[2] * (C1M ** 1.5)
    ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, \
             q1, q2, V0, V1, V2, V3, V4, diS, T)

    ActOut = Activity(ActIn)
    g_predX = g_predX + (ActOut.g_PF, )

    PTX = 1 + alpha_x[0] * (C1M ** 0.5) + alpha_x[1] * C1M + alpha_x[2] * (C1M ** 1.5)
    plot_thetaX = plot_thetaX + (PTX, )

  plt.subplot(a, b, c)
  plt.plot(DF.C1m, g_data, 'k.')
  plt.plot(DF.C1m, g_fit, '-r')
  N = len(DF.C1m)
  plt.text(DF.C1m[N-1] - 0.02, g_data[N-1] - 0.02, 'x=0')
  plt.title(salt, fontsize=12)
  if salts == 1:
    plt.ylabel(r"$\ln\gamma_\pm$", fontsize=12)

  for i in range(mixNo):
    if i == 0:
      g_data_x, g_pred_x = np.log(g_dataX[0]), g_predX[0]
      N = len(C1mX[0])
      plt.text(C1mX[0][N-1] + 0.02, g_data_x[N-1] - 0.02, '0.2')
      plt.plot(C1mX[0], g_data_x, 'k.')
      plt.plot(C1mX[0], g_pred_x, 'b')
    elif i == 1:
      g_data_x, g_pred_x = np.log(g_dataX[1]), g_predX[1]
      N = len(C1mX[1])
      plt.text(C1mX[1][N-1] + 0.02, g_data_x[N-1] - 0.02, '0.4')
      plt.plot(C1mX[1], g_data_x, 'k.')
      plt.plot(C1mX[1], g_pred_x, 'b')
    elif i == 2:
      g_data_x, g_pred_x = np.log(g_dataX[2]), g_predX[2]
      N = len(C1mX[2])
      plt.text(C1mX[2][N-1] + 0.02, g_data_x[N-1] - 0.02, '0.6')
      plt.plot(C1mX[2], g_data_x, 'k.')
      plt.plot(C1mX[2], g_pred_x, 'b')
    elif i == 3:
      g_data_x, g_pred_x = np.log(g_dataX[3]), g_predX[3]
      N = len(C1mX[3])
      plt.text(C1mX[3][N-1] + 0.02, g_data_x[N-1] - 0.02, '0.8')
      plt.plot(C1mX[3], g_data_x, 'k.')
      plt.plot(C1mX[3], g_pred_x, 'b')
    elif i == 4:
      g_data_x, g_pred_x = np.log(g_dataX[4]), g_predX[4]
      N = len(C1mX[4])
      plt.text(C1mX[4][N-1] + 0.02, g_data_x[N-1] - 0.02, '1.0')
      plt.plot(C1mX[4], g_data_x, 'k.')
      plt.plot(C1mX[4], g_pred_x, 'b')
    else:
      warnings.warn('Warning: Index i out of range.')

  c = c + 3

  plt.subplot(a, b, c)
  plt.plot(DF.C1m, plot_theta, '-r')
  plt.xlabel('m (mol/kg)', fontsize=12)
  if salts == 1:
    plt.ylabel(r"$\theta$", fontsize=12)

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
    else:
      warnings.warn('Warning: i out of range.')

  c = c - 3 + 1

plt.show()
