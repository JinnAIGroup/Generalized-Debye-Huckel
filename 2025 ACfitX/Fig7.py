'''
Author: Jinn-Liang Liu, July 10, 2025.
Example 4.6 Fig 7A: Comparing 4PF, 2PF, PB1, PB2 on LaCl3+MgCl2+H2O at T = 25 ◦C
            Fig 7B: Comparing ϵ_a, ϵ_b, ϵ_s in NaCl+H2O at 25, 300 ◦C
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from Physical import Solvent, Born, m2M
from Data4_5 import DataFit
from LSfit import LSfit, Activity, LSfitX
from Profile import Profile

# Solution Parameters:
#   0: void, 1: cation, 2: anion, 3: H2O, 4: MeOH, x or X: mixing percentage of 3 and 4 in [0, 1]
#   5: cation, 6: anion, 7: cation, 8: anion
#   Bulk concentrations in M: C1M (array), C2M (array), C3M (scalar), C4M (scalar)
#   ϵ_s_x (scalar): dielectric constant of mixed solvent [P2(22)], V: volume
#   gamma: mean activity data [P2(31)] of target salt CA = ca = 1+2 (array)

np.set_printoptions(suppress=True)
plt.figure(figsize=(13,8))
a, b, c = 1, 2, 1

Z = 0.68

Salts = ['LaCl3', 'NaCl']
#Salts = ['LaCl3']  # for Remark 4.4 in 2ndGDH.tex

for salt in Salts:
  if salt == 'LaCl3':  # For Fig 7A
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

      R_ca = (1660.5655 / 8 / (C1M + C2M) * S2) ** (1/3)
      Rsh_c, Rsh_a = R_ca, R_ca

      LfIn = (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, DF.C1m, \
              q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)
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

      R_ca = (1660.5655 / 8 / (C1M + C2M + C5M + C6M) * S2) ** (1/3)
      Rsh_c, Rsh_a = R_ca, R_ca

      ActIn_M1 = (q5, q6, V5, V6, C5M, C6M)  # for mix-salt 1
      ActIn_M2 = (0,0,0,0,0,0)               # for mix-salt 2
      ActIn_Mix = (ActIn_M1, ActIn_M2)

      alphaX = alpha[0]
      LfInX = (g_data_X, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS_X, DF.C1m, \
               q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T, alphaX, ActIn_Mix)

      LfOut = LSfitX(LfInX)
      alpha_X = LfOut.alpha
      print(" alpha_X =", np.around(alpha_X, 5), salt, "+", salt_1)
      theta = 1 + alpha_X[0] * (IS_X ** 0.5) + alpha_X[1] * IS_X + alpha_X[2] * (IS_X ** 1.5) + alpha_X[3] * (IS_X ** 2) + alpha_X[4] * (IS_X ** 2.5)
      print(" RBorn_c[0], Rsh_c[0], IS[0] =", np.around(BornR0[0] * theta[0], 2), np.around(Rsh_c[0], 2), np.around(IS[0], 2))  # for Remark 4.4 in 2ndGDH.tex
      print(" RBorn_c[-1], Rsh_c[-1], IS_X[-1] =", np.around(BornR0[0] * theta[-1], 2), np.around(Rsh_c[-1], 2), np.around(IS_X[-1], 2))  # for Fig 7A

      ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, IS_X, DF.C1m, \
               q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)

      ActOut = Activity(ActIn, ActIn_Mix)  # Prediction

      # Part 3: Plot Profiles ...
      (GammaB, lambda1, lambda2, LDebye, LBjerrum, Lcorr) = ActOut.Ls
      (THETA1, THETA2, THETA3, THETA4) = ActOut.THETA

      ActInC = (theta[-1], BornR0[0], Rsh_c[-1], C1M[-1], C3M, C4M, q1, q2, V1, V2, V3, ϵ_s_x, ϵ_s_x_I[-1], T)  # Cut ActIn
      ActIn_M1 = (q5, q6, V5, V6, C5M[-1], C6M[-1])
      ActIn_M2 = (0,0,0,0,0,0)
      ActIn_Mix = (ActIn_M1, ActIn_M2)

      LsP   = (ϵ_s_x, GammaB[-1], lambda1[-1], lambda2[-1], LDebye[-1], Lcorr[-1])
      THETA = (THETA1[-1], THETA2[-1], THETA3[-1], THETA4[-1])
      p_In = (pH2O, pH2O_Z[-1], p1, p2, p5, numPW, numPW_Z[-1], numPWI[-1], numPWI_Z[-1])
      '''
      # for Remark 4.4 in 2ndGDH.pdf
      LsP   = (ϵ_s_x, GammaB[0], lambda1[0], lambda2[0], LDebye[0], Lcorr[0])
      THETA = (THETA1[0], THETA2[0], THETA3[0], THETA4[0])
      p_In = (pH2O, pH2O_Z[0], p1, p2, p5, numPW, numPW_Z[0], numPWI[0], numPWI_Z[0])
      '''

      Pf = Profile(ActInC, ActIn_Mix, LsP, THETA, p_In)
      print(" eltr_4PF[0], _2PF, _PB1, _PB2, _DH =", np.around(Pf.eltr[0], 2), np.around(Pf.eltr_2PF[0], 2), np.around(Pf.eltr_PB1[0], 2), np.around(Pf.eltr_PB2[0], 2), np.around(Pf.eltr_DH[0], 2))
      print(" rho_4PF[26], _2PF, _PB1, _PB2 =", np.around(Pf.rho[26], 2), np.around(Pf.rho_2PF[26], 2), np.around(Pf.rho_PB1[26], 2), np.around(Pf.rho_PB2[26], 2))
      print(" Ratio rho_PB1 / _2PF =", np.around(Pf.rho_PB1[26]/Pf.rho_2PF[26], 2))

  if salt == 'NaCl':  # For Fig 7B
    Ts = [298.15, 373.15, 473.15, 523.15, 573.15]
    ewTsA = -19.2905 + 29814.5 / np.array(Ts) - 0.019678 * np.array(Ts) + 1.3189 * 1e-4 * np.array(Ts) ** 2 - 3.1144 * 1e-7 * np.array(Ts) ** 3
    tMMB1, tMMB2 = 0.0086, 0.2063  # Table S.3 [Sil23]

    for (T, T_i) in zip(Ts, range(len(Ts))):
      # Part 1: Fiting ...
      S2, C3M, C4M, V3, V4, pH2O, _, ϵ_s_x = Solvent(0, T)  # x=0 for pure H2O
      BornR0, q1, q2, p1, p2, V1, V2, mM, DG = Born(salt, ϵ_s_x, 0, T)  # for H2O

      DF = DataFit(salt)  # pure-H2O activity data to be fitted
      g_data = DF.lngamma[T_i]  # gamma: mean activity

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

      R_c = (1660.5655 / 8 / C1M * S2) ** (1/3)  # one c ion per cubic
      R_a = (1660.5655 / 8 / C2M * S2) ** (1/3)
      Rsh_c, Rsh_a = R_a, R_a  # [PF4P49(14)] (array)

      if T_i == 0:  # Fig 6B at 298.15
        LfIn = (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, DF.C1m, \
                q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)

        LfOut = LSfit(LfIn)
        alpha = LfOut.alpha

        print(" alpha =", np.around(alpha, 5), salt, T)
        theta = alpha[0]

        ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, IS, DF.C1m, \
                 q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)
        ActIn_M1 = (0,0,0,0,0,0)
        ActIn_M2 = (0,0,0,0,0,0)
        ActIn_Mix = (ActIn_M1, ActIn_M2)

        ActOut = Activity(ActIn, ActIn_Mix)  # Prediction

        IS = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)

        eps_c = ϵ_s_x_I

        eps_a = ϵ_s_x - 16.2 * DF.C1m + 3.1 * DF.C1m ** 1.5  # ϵ_b=ϵ(m) Eq2,TableS.2[Sil23] at 298.15

        sum1 = 0.01 * (C1M + C2M)  # Eq14[Sil23]
        sum2 = tMMB1 * C1M / (1 + 0.16 * C1M) + tMMB2 * C2M / (1 + 0.16 * C2M)
        eps_b = ϵ_s_x * (1 + sum1 - sum2)  # Fig 6B at 573.15, Eq14[Sil23]

      elif T_i == 4:  # Fig 6A at 573.15
        LfIn = (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, DF.C1m, \
                q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)
        LfOut = LSfit(LfIn)

        g_fit, alpha = LfOut.g_fit, LfOut.alpha  # fitted results
        print(" alpha =", np.around(alpha, 5), salt, T)

        ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, IS, DF.C1m, \
                 q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)
        ActIn_M1 = (0,0,0,0,0,0)
        ActIn_M2 = (0,0,0,0,0,0)
        ActIn_Mix = (ActIn_M1, ActIn_M2)

        ActOut = Activity(ActIn, ActIn_Mix)  # Prediction

        IS = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2)

        eps_c1 = ϵ_s_x_I

        eps_a1 = ϵ_s_x - 16.2 * DF.C1m + 3.1 * DF.C1m ** 1.5  # ϵ_b=ϵ(m) Eq2,TableS.2[Sil23] at 573.15
        fc = eps_a1 / ϵ_s_x  # Eq2
        eps_a1 = fc * ewTsA[4]  # Fig 6B at 573.15, Eq3[Sil23]
        sum1 = 0.01 * (C1M + C2M)  # Eq14[Sil23]
        sum2 = tMMB1 * C1M / (1 + 0.16 * C1M) + tMMB2 * C2M / (1 + 0.16 * C2M)
        eps_b1 = ϵ_s_x * (1 + sum1 - sum2)  # Fig 6B at 573.15, Eq14[Sil23],  Eq15[Sil23], [PF4Ex3.88]

      else:
        pass


# Plot Fig 7A
plt.figure(1)
plt.subplot(a, b, c)
plt.plot(Pf.rL, Pf.eltr, 'r-', label='$\phi^{4PF}(r)$ in $k_B T / e$')
plt.plot(Pf.rL, Pf.eltr_2PF, 'b--', label='$\phi^{2PF}(r)$ in $k_B T / e$')
plt.plot(Pf.rL, Pf.eltr_PB1, 'g+', label='$\phi^{PB1}(r)$ in $k_B T / e$')
plt.plot(Pf.rL, Pf.eltr_PB2, 'k-', label='$\phi^{PB2}(r)$ in $k_B T / e$')
plt.plot(Pf.rL[26:], Pf.rho[26:] / 5, 'r.', label=r'$\rho^{4PF}(r)$/5 in $e$M')  # we need r'' for \rho
plt.plot(Pf.rL[26:], Pf.rho_2PF[26:] / 5, 'b*', label=r'$\rho^{2PF}(r)$/5 in $e$M')
plt.plot(Pf.rL[26:], Pf.rho_PB1[26:] / 50, 'gs', label=r'$\rho^{PB1}(r)$/50 in $e$M')
plt.plot(Pf.rL[26:], Pf.rho_PB2[26:] / 5, 'kx', label=r'$\rho^{PB2}(r)$/5 in $e$M')
plt.title('(A) Comparison of 4PF, 2PF, PB1, and PB2 Models', fontsize=12)
plt.xlabel('Distance from the center of La$^{3+}$ in Angstrom', fontsize=12)
plt.ylabel('Electric Potential Functions', fontsize=12)
plt.legend(loc="upper right")


c = c + 1

# Plot Fig 7B
plt.figure(1)
plt.subplot(a, b, c)
plt.plot(DF.C1m, eps_a, 'bo',   label='$\epsilon_a$ at 25 $^{\circ}$C')
plt.plot(DF.C1m, eps_b, 'gs',   label='$\epsilon_b$ at 25 $^{\circ}$C')
plt.plot(DF.C1m, eps_c, 'r-',   label='$\epsilon_s(I,T)$ at 25 $^{\circ}$C')
plt.plot(DF.C1m, eps_b1, 'gX',   label='$\epsilon_b$ at 300 $^{\circ}$C')
plt.plot(DF.C1m, eps_c1, 'r--',  label='$\epsilon_s(I,T)$ at 300 $^{\circ}$C')
plt.title('(B) $\epsilon_a$, $\epsilon_b$, $\epsilon_s(I,T)$ in NaCl+H$_2$O at $T=25,$ 300 $^{\circ}$C', fontsize=12)
plt.legend(loc="upper right")
plt.xlabel('m (mol/kg)', fontsize=12)

plt.show()
