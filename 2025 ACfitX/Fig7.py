'''
Author: Jinn-Liang Liu, Nov 27, 2025.
Figure 7: Comparing GDH, DH on LaCl3+MgCl2+H2O at T = 25 ◦C
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from Physical import Solvent, Born, m2M
from Data6 import DataFit
from LSfit import LSfit, Activity, LSfitX
from Profile import Profile

# Solution Parameters:
#   0: void, 1: cation, 2: anion, 3: H2O, 4: MeOH, x or X: mixing percentage of 4 in [0, 1]
#   5: cation, 6: anion, 7: cation, 8: anion
#   Bulk concentrations in M: C1M (array), C2M (array), C3M (scalar), C4M (scalar)
#   ϵ_s_x (scalar): dielectric constant of mixed solvent, V: volume
#   gamma: mean activity data of target salt CA = ca = 1+2 (array)

np.set_printoptions(suppress=True)
plt.figure(figsize=(13,8))
a, b, c = 1, 1, 1

Salts = ['LaCl3']
Z = 0.68

for salt in Salts:
  if salt == 'LaCl3':  # For Fig 7
    Ts = [298.15]

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
      print(" alpha =", np.around(alpha, 5), salt, T)

      # Part 2: Mix-Salt Predicting ...
      salt_1 = 'MgCl2'
      C1m_Mix = DF.C1m / 2

      C1M = m2M(C1m_Mix, mM, DG, 0, T) * S2
      C2M = -q1 * C1M / q2

      _, q5, q6, p5, p6, V5, V6, mM_Mix, D_Mix = Born(salt_1, ϵ_s_x, 0, T)
      C5M = m2M(C1m_Mix, mM_Mix, D_Mix, 0, T) * S2
      C6M = -q5 * C5M / q6

      IS_X =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2)

      ISX0 = IS[0] if IS[0] < IS_X[0] else IS_X[0]
      IS_X = np.linspace(ISX0, IS_X[-1], num=2 * len(IS_X))

      Spline = InterpolatedUnivariateSpline(IS, g_data, k=1)

      g_data_X = Spline(IS_X)

      C1m_X = IS_X * DF.C1m[-1] / IS_X[-1]

      C1m_Xmix = C1m_X / 2
      C1M = m2M(C1m_Xmix, mM, DG, 0, T) * S2
      C2M = -q1 * C1M / q2
      _, _, _, _, _, _, _, mM_X, D_X = Born(salt_1, ϵ_s_x, 0, T)
      C5M = m2M(C1m_Xmix, mM_X, D_X, 0, T) * S2
      C6M = -q5 * C5M / q6

      IS_X = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2)
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
      theta = 1 + alpha_X[0] * (IS_X ** 0.5) + alpha_X[1] * IS_X + alpha_X[2] * (IS_X ** 1.5) + alpha_X[3] * (IS_X ** 2) + alpha_X[4] * (IS_X ** 2.5)
      print(" RBorn_c[0], R_sh[0], IS[0] =", np.around(BornR0[0] * theta[0], 2), np.around(R_sh[0], 2), np.around(IS[0], 2))  # for Remark 4.4 in 2ndGDH.tex
      print(" RBorn_c[-1], R_sh[-1], IS_X[-1] =", np.around(BornR0[0] * theta[-1], 2), np.around(R_sh[-1], 2), np.around(IS_X[-1], 2))  # for Fig 7A

      ActIn = (theta, BornR0, R_sh, C1M, C3M, C4M, IS_X, \
               q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)

      ActOut = Activity(ActIn, ActIn_Mix)  # Prediction

      # Part 3: Plot Profiles ...
      (GammaB, lambda1, lambda2, LDebye, LBjerrum, Lcorr) = ActOut.Ls
      (THETA1, THETA2, THETA3, THETA4) = ActOut.THETA

      ActInC = (theta[-1], BornR0[0], R_sh[-1], C1M[-1], C3M, C4M, q1, q2, V1, V2, V3, ϵ_s_x, ϵ_s_x_I[-1], T)  # Cut ActIn
      ActIn_M1 = (q5, q6, V5, V6, C5M[-1], C6M[-1])
      ActIn_M2 = (0,0,0,0,0,0)
      ActIn_Mix = (ActIn_M1, ActIn_M2)

      LsP   = (ϵ_s_x, GammaB[-1], lambda1[-1], lambda2[-1], LDebye[-1], Lcorr[-1])
      THETA = (THETA1[-1], THETA2[-1], THETA3[-1], THETA4[-1])
      p_In = (pH2O, pH2O_Z[-1], p1, p2, p5, numPW, numPW_Z[-1], numPWI[-1], numPWI_Z[-1])

      Pf = Profile(ActInC, ActIn_Mix, LsP, THETA, p_In)
      print(" eltr_GDH[0], _DH[0] =", np.around(Pf.eltr[0], 2), np.around(Pf.eltr_DH[0], 2))
      print(" rho_GDH[26], _DH[29] =", np.around(Pf.rho[26], 2), np.around(Pf.rho_DH[29], 2))

# Plot Fig 7
plt.figure(1)
plt.subplot(a, b, c)
plt.plot(Pf.rL, Pf.eltr_DH, 'b--', label='$\phi^{DH}(r)$ in $k_B T / e$')
plt.plot(Pf.rL, Pf.eltr, 'r-', label='$\phi^{GDH}(r)$ in $k_B T / e$')
plt.plot(Pf.rL[29:], Pf.rho_DH[29:] / 5, 'bx', label=r'$\rho^{DH}(r)$/5 in $e$M')
plt.plot(Pf.rL[26:], Pf.rho[26:] / 5, 'r.', label=r'$\rho^{GDH}(r)$/5 in $e$M')
plt.title('Comparison of DH and GDH in LaCl$_3$+MgCl$_2$+H$_2$O', fontsize=12)
plt.xlabel('Distance from the center of La$^{3+}$ in Angstrom', fontsize=12)
plt.ylabel('Electric Potential and Charge Density', fontsize=12)
plt.legend(loc="upper right")

plt.show()
