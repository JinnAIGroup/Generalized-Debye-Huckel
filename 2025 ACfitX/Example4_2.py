'''
Author: Jinn-Liang Liu, May 12, 2025.
Example 4.2: NaCl in [NaCl+NaBr] or [NaCl+NaBr+NaF] + [(H2O)x+(MeOH)(1−x)].
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

T, Z = 298.15, 0.68
np.set_printoptions(suppress=True)  # set 0.01 not 1e-2
plt.figure(figsize=(13,8))
a, b, c = 1, 2, 1  # subplot(a, b, c): rows, columns, counter

salt = 'NaCl'

# Part 1: H2O Fiting ...

S2, C3M, C4M, V3, V4, pH2O, pMeOH, ϵ_s_x = Solvent(0, T)  # x=0 for pure H2O

# Born Radius: BornR0 in pure solvent (no salt) [P1(12), (13)]
BornR0, q1, q2, p1, p2, V1, V2, mM, DG = Born(salt, ϵ_s_x, 0, T)  # for H2O

DF = DataFit(salt)  # pure-H2O activity data to be fitted
g_data = np.log(DF.gamma)  # gamma: mean activity [P1(16)]

# Salt molality (m) to Molarity (M): C1m (array) to C1M
C1M = m2M(DF.C1m, mM, DG, 0, T) * S2  # for H2O
C2M = -q1 * C1M / q2  # q1 C1M + q2 C2M = 0

IS =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)  # Ionic Strength (array)
numPW = C3M * pH2O
numPI = C1M * p1 + C2M * p2
numPWI = numPW + numPI
fac_pZ = 1 - Z * IS / C3M
pH2O_Z = fac_pZ * pH2O
numPW_Z = C3M * pH2O_Z
numPWI_Z = numPW_Z + numPI
X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
ϵ_s_x_I = (2 * X + 1) / (1 - X)

R_ca = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)
Rsh_c, Rsh_a = R_ca, R_ca

# LSfit() returns g_fit as the best fit to g_data with alpha_1, 2, 3 [P1(14)] by Least Squares.
LfIn = (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, DF.C1m, \
        q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)
LfOut = LSfit(LfIn)

g_fit, alpha = LfOut.g_fit, LfOut.alpha  # fitted results

# Part 2: MeOH Fiting for alphaD ...

DP = DataPredict(salt)  # miXed-solvent activity data to be compared with predicted results
g_dataX, mixNo, C1mX = DP.g_dataX, DP.mixNo, DP.C1mX

x, C1m_x, g_dataY = 0.2, C1mX[0], np.log(g_dataX[0])

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

R_ca = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)
Rsh_c, Rsh_a = R_ca, R_ca

LfIn = (g_dataY, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, DF.C1m, \
        q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)
LfOut = LSfit(LfIn)

alphaY = LfOut.alpha
alphaD = alphaY - alpha

AAD = np.mean(np.abs(g_fit - g_data))
print(" alpha, AAD% =", np.around(alpha, 5), np.around(AAD*100, 2), salt)
print(" alphaD =", np.around(alphaD, 5))

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
      _, q5, q6, p5, p6, V5, V6, mM_Mix, D_Mix = Born(salt_Mix, ϵ_s_x, 0, T)  # for H2O
      C5m = DP.C1mX0_Mix1  # add C5m to C1m
      C5M = m2M(C5m, mM_Mix, D_Mix, 0, T) * S2
      C6M = -q5 * C1M / q6

      IS =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2)  # array
      numPW = C3M * pH2O + C4M * pMeOH
      numPI = C1M * p1 + C2M * p2 + C5M * p5 + C6M * p6
      numPWI = numPW + numPI
      fac_pZ = 1 - Z * IS / (C3M + C4M)
      pH2O_Z, pMeOH_Z = fac_pZ * pH2O, fac_pZ * pMeOH
      numPW_Z = C3M * pH2O_Z + C4M * pMeOH_Z
      numPWI_Z = numPW_Z + numPI
      X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
      ϵ_s_x_I = (2 * X + 1) / (1 - X)

      R_ca = (1660.5655 / 8 / (C1M + C2M + C5M + C6M) * S2) ** (1/3)
      Rsh_c, Rsh_a = R_ca, R_ca

      ActIn_M1 = (q5, q6, V5, V6, C5M, C6M)  # for mix-salt 1
      ActIn_M2 = (0,0,0,0,0,0)          # for mix-salt 2
      ActIn_Mix = (ActIn_M1, ActIn_M2)

      alpha_x = alpha + x * alphaD
      theta = 1 + alpha_x[0] * (IS ** 0.5) + alpha_x[1] * IS + alpha_x[2] * (IS ** 1.5) + alpha_x[3] * (IS ** 2) + alpha_x[4] * (IS ** 2.5)

      ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, IS, DF.C1m, \
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
      _, q7, q8, p7, p8, V7, V8, mM_Mix, D_Mix = Born(salt_Mix, ϵ_s_x, 0, T)  # for H2O
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

      R_ca = (1660.5655 / 8 / (C1M + C2M + C5M + C6M + C7M + C8M) * S2) ** (1/3)
      Rsh_c, Rsh_a = R_ca, R_ca

      ActIn_M1 = (q5, q6, V5, V6, C5M, C6M)  # for mix-salt 1
      ActIn_M2 = (q7, q8, V7, V8, C7M, C8M)  # for mix-salt 2
      ActIn_Mix = (ActIn_M1, ActIn_M2)

      alpha_x = alpha + x * alphaD
      theta = 1 + alpha_x[0] * (IS ** 0.5) + alpha_x[1] * IS + alpha_x[2] * (IS ** 1.5) + alpha_x[3] * (IS ** 2) + alpha_x[4] * (IS ** 2.5)

      ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, IS, DF.C1m, \
               q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)

      ActOut = Activity(ActIn, ActIn_Mix)  # Prediction
      g_predX_Mix2 = g_predX_Mix2 + (ActOut.g_PF, )  # Predicted results

# Plot fitted results
plt.figure(1)
plt.subplot(a, b, c)
plt.xlabel('m (mol/kg)', fontsize=12)

plt.plot(DF.C1m, g_fit, '-r')
N = len(DF.C1m)
plt.title('(A) NaCl+NaBr', fontsize=12)

plt.ylabel(r"$\ln\gamma_\pm$", fontsize=12)

# Plot predicted results
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

# Plot predicted results
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
'''
WARNING: array lam < 0: [ 0.979  0.97   0.953  0.944  0.933  0.918  0.906  0.884  0.851  0.823
  0.789  0.742  0.702  0.635  0.579  0.529  0.444  0.337  0.19   0.067 -0.136 -0.304]
WARNING: array lam < 0: [ 0.979  0.97   0.952  0.944  0.933  0.917  0.905  0.883  0.849  0.822
  0.787  0.739  0.699  0.631  0.574  0.524  0.437  0.328  0.178  0.052 -0.157 -0.332]
'''
