'''
Author: Jinn-Liang Liu, Nov 27, 2025.
Figure 3: NaCl in [NaCl+NaBr] or [NaCl+NaBr+NaF] + [(H2O)x+(MeOH)(1−x)].
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

np.set_printoptions(suppress=True)
plt.figure(figsize=(13,8))
a, b, c = 1, 2, 1

salt = 'NaCl'
T, Z = 298.15, 0.68

# Part 1: H2O Fiting ...

S2, C3M, C4M, V3, V4, pH2O, pMeOH, ϵ_s_x = Solvent(0, T)  # x=0 for pure H2O

BornR0, q1, q2, p1, p2, V1, V2, mM, DG = Born(salt, ϵ_s_x, 0, T)  # for H2O

DF = DataFit(salt)
g_data = np.log(DF.gamma)

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

AAD1 = np.mean(np.abs(g_fit - g_data))
print(" alpha, AAD1% =", np.around(alpha, 5), np.around(AAD1*100, 2), salt)

# Part 2: MeOH Fiting for alphaD ...

DP = DataPredict(salt)
g_dataX, mixNo, C1mX = DP.g_dataX, DP.mixNo, DP.C1mX

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

g_fit_xhat, alpha_xhat = LfOut.g_fit, LfOut.alpha
alphaD = alpha_xhat - alpha

AAD2 = np.mean(np.abs(g_fit_xhat - g_data_xhat))
print(" alpha_xhat =", np.around(alpha_xhat, 5))
print(" alphaD, AAD2% =", np.around(alphaD, 5), np.around(AAD2*100, 2))

# Part 3: Mix-Salt Mix-Solvent Predicting ...

for Mix_i in range(2):
  if Mix_i == 0: C1m_x, g_predX_Mix1 = DP.C1mX0_Mix1, ()
  else:          C1m_x, g_predX_Mix2 = DP.C1mX0_Mix2, ()

  for i in range(6):
    if   i == 0: x = 0.
    elif i == 1: x = 0.2
    elif i == 2: x = 0.4
    elif i == 3: x = 0.6
    elif i == 4: x = 0.8
    elif i == 5: x = 1.0
    else: warnings.warn('Warning: Index i out of range.')

    _, C3M, C4M, _, _, _, _, ϵ_s_x = Solvent(x, T)
    BornR0, _, _, _, _, _, _, _, _ = Born(salt, ϵ_s_x, x, T)
    C1M = m2M(C1m_x, mM, DG, x, T) * S2
    C2M = -q1 * C1M / q2

    if Mix_i == 0:  # 2-salt: NaCl+NaBr
      salt_Mix = 'NaBr'
      _, q5, q6, p5, p6, V5, V6, mM_Mix, D_Mix = Born(salt_Mix, ϵ_s_x, 0, T)
      C5m = DP.C1mX0_Mix1  # add C5m to C1m
      C5M = m2M(C5m, mM_Mix, D_Mix, 0, T) * S2
      C6M = -q5 * C1M / q6

      IS =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2)
      numPW = C3M * pH2O + C4M * pMeOH
      numPI = C1M * p1 + C2M * p2 + C5M * p5 + C6M * p6
      numPWI = numPW + numPI
      fac_pZ = 1 - Z * IS / (C3M + C4M)
      pH2O_Z, pMeOH_Z = fac_pZ * pH2O, fac_pZ * pMeOH
      numPW_Z = C3M * pH2O_Z + C4M * pMeOH_Z
      numPWI_Z = numPW_Z + numPI
      X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
      ϵ_s_x_I = (2 * X + 1) / (1 - X)

      R_sh = (1660.5655 / 8 / (C1M + C2M + C5M + C6M) * S2) ** (1/3)

      ActIn_M1 = (q5, q6, V5, V6, C5M, C6M)
      ActIn_M2 = (0,0,0,0,0,0)
      ActIn_Mix = (ActIn_M1, ActIn_M2)

      if x <= xhat:
        alpha_x = alpha + x * alphaD / xhat
      else:
        alpha_x = alpha_xhat + (x - xhat) * alphaD / xhat

      theta = 1 + alpha_x[0] * (IS ** 0.5) + alpha_x[1] * IS + alpha_x[2] * (IS ** 1.5) + alpha_x[3] * (IS ** 2) + alpha_x[4] * (IS ** 2.5)

      ActIn = (theta, BornR0, R_sh, C1M, C3M, C4M, IS, \
               q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)

      ActOut = Activity(ActIn, ActIn_Mix)  # Prediction
      g_predX_Mix1 = g_predX_Mix1 + (ActOut.g_PF, )  # Predicted results

    else:  # 3-salt: NaCl+NaBr+NaF
      salt_Mix = 'NaBr'
      _, q5, q6, p5, p6, V5, V6, mM_Mix, D_Mix = Born(salt_Mix, ϵ_s_x, 0, T)  # for H2O
      C5m = DP.C1mX0_Mix2  # add C5m to C1m
      C5M = m2M(C5m, mM_Mix, D_Mix, 0, T) * S2
      C6M = -q5 * C1M / q6

      salt_Mix = 'NaF'
      _, q7, q8, p7, p8, V7, V8, mM_Mix, D_Mix = Born(salt_Mix, ϵ_s_x, 0, T)
      C7m = DP.C1mX0_Mix2  # add another C7m to C1m
      C7M = m2M(C7m, mM_Mix, D_Mix, 0, T) * S2
      C8M = -q7 * C1M / q8

      IS =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2 + C7M * q7 ** 2 + C8M * q8 ** 2)  # array
      numPW = C3M * pH2O + C4M * pMeOH
      numPI = C1M * p1 + C2M * p2 + C5M * p5 + C6M * p6 + C7M * p7 + C8M * p8
      numPWI = numPW + numPI
      fac_pZ = 1 - Z * IS / (C3M + C4M)
      pH2O_Z, pMeOH_Z = fac_pZ * pH2O, fac_pZ * pMeOH
      numPW_Z = C3M * pH2O_Z + C4M * pMeOH_Z
      numPWI_Z = numPW_Z + numPI
      X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
      ϵ_s_x_I = (2 * X + 1) / (1 - X)

      R_sh = (1660.5655 / 8 / (C1M + C2M + C5M + C6M + C7M + C8M) * S2) ** (1/3)

      ActIn_M1 = (q5, q6, V5, V6, C5M, C6M)  # for mix-salt 1
      ActIn_M2 = (q7, q8, V7, V8, C7M, C8M)  # for mix-salt 2
      ActIn_Mix = (ActIn_M1, ActIn_M2)

      if x <= xhat:
        alpha_x = alpha + x * alphaD / xhat
      else:
        alpha_x = alpha_xhat + (x - xhat) * alphaD / xhat

      theta = 1 + alpha_x[0] * (IS ** 0.5) + alpha_x[1] * IS + alpha_x[2] * (IS ** 1.5) + alpha_x[3] * (IS ** 2) + alpha_x[4] * (IS ** 2.5)

      ActIn = (theta, BornR0, R_sh, C1M, C3M, C4M, IS, \
               q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)

      ActOut = Activity(ActIn, ActIn_Mix)  # Prediction
      g_predX_Mix2 = g_predX_Mix2 + (ActOut.g_PF, )  # Predicted results

# Plot Fig 3
plt.figure(1)
plt.subplot(a, b, c)
plt.xlabel('m (mol/kg)', fontsize=12)

plt.plot(DF.C1m, g_fit, '-r')
N = len(DF.C1m)
plt.title('(A) NaCl+NaBr', fontsize=12)

plt.ylabel(r"$\ln\gamma_\pm$", fontsize=12)

for i in range(6):
  if i == 0:
    g_pred_x = g_predX_Mix1[0]
    plt.text(C1mX[0][N-1] - 0.35, g_pred_x[N-1] + 0.02, 'x = 0')
  elif i == 1:
    g_pred_x = g_predX_Mix1[1]
    plt.text(C1mX[0][N-1] - 0.2, g_pred_x[N-1] + 0.02, '0.2')
  elif i == 2:
    g_pred_x = g_predX_Mix1[2]
    plt.text(C1mX[0][N-1] - 0.2, g_pred_x[N-1] + 0.02, '0.4')
  elif i == 3:
    g_pred_x = g_predX_Mix1[3]
    plt.text(C1mX[0][N-1] - 0.2, g_pred_x[N-1] + 0.02, '0.6')
  elif i == 4:
    g_pred_x = g_predX_Mix1[4]
    plt.text(C1mX[0][N-1] - 0.2, g_pred_x[N-1] + 0.02, '0.8')
  elif i == 5:
    g_pred_x = g_predX_Mix1[5]
    plt.text(C1mX[0][N-1] - 0.2, g_pred_x[N-1] + 0.02, '1.0')

  plt.plot(C1mX[0], g_pred_x, 'b')

c = c + 1

plt.subplot(a, b, c)
plt.xlabel('m (mol/kg)', fontsize=12)

plt.plot(DF.C1m, g_fit, '-r')
plt.title('(B) NaCl+NaBr+NaF', fontsize=12)

plt.ylabel(r"$\ln\gamma_\pm$", fontsize=12)

for i in range(6):
  if i == 0:
    g_pred_x = g_predX_Mix2[0]
    plt.text(C1mX[0][N-1] - 0.35, g_pred_x[N-1] + 0.02, 'x = 0')
  elif i == 1:
    g_pred_x = g_predX_Mix2[1]
    plt.text(C1mX[0][N-1] - 0.2, g_pred_x[N-1] + 0.02, '0.2')
  elif i == 2:
    g_pred_x = g_predX_Mix2[2]
    plt.text(C1mX[0][N-1] - 0.2, g_pred_x[N-1] + 0.02, '0.4')
  elif i == 3:
    g_pred_x = g_predX_Mix2[3]
    plt.text(C1mX[0][N-1] - 0.2, g_pred_x[N-1] + 0.02, '0.6')
  elif i == 4:
    g_pred_x = g_predX_Mix2[4]
    plt.text(C1mX[0][N-1] - 0.2, g_pred_x[N-1] + 0.02, '0.8')
  elif i == 5:
    g_pred_x = g_predX_Mix2[5]
    plt.text(C1mX[0][N-1] - 0.2, g_pred_x[N-1] + 0.02, '1.0')

  plt.plot(C1mX[0], g_pred_x, 'b')

plt.show()
