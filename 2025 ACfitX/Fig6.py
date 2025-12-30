'''
Author: Jinn-Liang Liu, Nov 27, 2025.
Figure 6A, 6B: Profiles around La in LaCl3+MgCl2+H2O at T = 25 ◦C
Figure 6C, 6D: Profiles around Mg, Cl in MgCl2+LaCl3+H2O at T = 25 ◦C
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
a, b, c = 2, 2, 1

Salts = ['LaCl3', 'MgCl2']
T, Z = 298.15, 0.68
S2, C3M, C4M, V3, V4, pH2O, _, ϵ_s_x = Solvent(0, T)

for salt in Salts:
  # Part 1: Fiting ...
  BornR0, q1, q2, p1, p2, V1, V2, mM, DG = Born(salt, ϵ_s_x, 0, T)
  a_c, a_a = (3 * V1 / 4 / np.pi) ** (1/3),  (3 * V2 / 4 / np.pi) ** (1/3)

  DF = DataFit(salt)
  g_data = DF.lngamma[0]

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
  print(" -------------------- 1-Salt:", salt)
  print(" alpha =", np.around(alpha, 5), salt, T)
  print(' a_c, a_a, BornR0[0], BornR0[1], BornR_c[0], BornR_a[0], R_sh[0]:', np.around(a_c, 2), np.around(a_a, 2), np.around(BornR0[0], 2), np.around(BornR0[1], 2), np.around(theta[0] * BornR0[0], 2), np.around(theta[0] * BornR0[1], 2), np.around(R_sh[0], 2))

  # Part 2: Mix-Salt Predicting ...
  if salt == 'MgCl2': salt_1 = 'LaCl3'
  if salt == 'LaCl3': salt_1 = 'MgCl2'
  C1m_Mix = DF.C1m / 2  # for 2-salt

  C1M = m2M(C1m_Mix, mM, DG, 0, T) * S2
  C2M = -q1 * C1M / q2

  BornR0_56, q5, q6, p5, p6, V5, V6, mM_Mix, D_Mix = Born(salt_1, ϵ_s_x, 0, T)
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
  numPI = C1M * p1 + C2M * p2 + C5M * p5 + C6M * p6
  numPWI = numPW + numPI
  fac_pZ = 1 - Z * IS_X / C3M
  pH2O_Z = fac_pZ * pH2O
  numPW_Z = C3M * pH2O_Z
  numPWI_Z = numPW_Z + numPI
  X = (ϵ_s_x - 1) / (ϵ_s_x + 2) * numPWI_Z / numPWI
  ϵ_s_x_I = (2 * X + 1) / (1 - X)

  R_sh = (1660.5655 / 8 / (C1M + C2M + C5M + C6M) * S2) ** (1/3)

  ActIn_M1 = (q5, q6, V5, V6, C5M, C6M)  # for mix-salt 1
  ActIn_M2 = (0,0,0,0,0,0)               # for mix-salt 2
  ActIn_Mix = (ActIn_M1, ActIn_M2)

  alphaX = alpha[0]
  LfInX = (g_data_X, BornR0, R_sh, salt, C1M, C3M, C4M, IS_X, \
           q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T, alphaX, ActIn_Mix)

  LfOut = LSfitX(LfInX)
  alpha_X = LfOut.alpha
  print(" -------------------- 2-Salt:", salt, "+", salt_1)
  print(" alpha_X =", np.around(alpha_X, 5))
  theta = 1 + alpha_X[0] * (IS_X ** 0.5) + alpha_X[1] * IS_X + alpha_X[2] * (IS_X ** 1.5) + alpha_X[3] * (IS_X ** 2) + alpha_X[4] * (IS_X ** 2.5)

  ActIn = (theta, BornR0, R_sh, C1M, C3M, C4M, IS_X, \
           q1, q2, V1, V2, V3, V4, ϵ_s_x, ϵ_s_x_I, T)

  ActOut = Activity(ActIn, ActIn_Mix)  # Prediction

  # Part 3: Plot Profiles ...
  (GammaB, lambda1, lambda2, LDebye, LBjerrum, Lcorr) = ActOut.Ls
  (THETA1, THETA2, THETA3, THETA4) = ActOut.THETA

  if salt == 'LaCl3':
    print(' ===== La^3+ Profiles in Fig 6A, 6B =====')

    ActIn_Pf = (theta[-1], BornR0[0], R_sh[-1], C1M[-1], C3M, C4M, q1, q2, V1, V2, V3, ϵ_s_x, ϵ_s_x_I[-1], T)  # Cut ActIn
    ActIn_M1 = (q5, q6, V5, V6, C5M[-1], C6M[-1])
    ActIn_M2 = (0,0,0,0,0,0)
    ActIn_Mix = (ActIn_M1, ActIn_M2)

    LsP   = (ϵ_s_x, GammaB[-1], lambda1[-1], lambda2[-1], LDebye[-1], Lcorr[-1])  # Ls for Profile
    THETA = (THETA1[-1], THETA2[-1], THETA3[-1], THETA4[-1])
    p_In = (pH2O, pH2O_Z[-1], p1, p2, p5, numPW, numPW_Z[-1], numPWI[-1], numPWI_Z[-1])
    print(' g_PF[-1], g_data_X[-1], BornR_c[0], BornR_a[0], R_sh[0]:', np.around(ActOut.g_PF[-1], 2), np.around(g_data_X[-1], 2), np.around(theta[0] * BornR0[0], 2), np.around(theta[0] * BornR0[1], 2), np.around(R_sh[0], 2))
    print(' theta[-1], BornR_56_c[-1], BornR_c[-1], BornR_a[-1], R_sh[-1]:', np.around(theta[-1], 2), np.around(theta[-1] * BornR0_56[0], 2), np.around(theta[-1] * BornR0[0], 2), np.around(theta[-1] * BornR0[1], 2), np.around(R_sh[-1], 2))

    Pf = Profile(ActIn_Pf, ActIn_Mix, LsP, THETA, p_In)
    print(' fac_pZ[-1], IS_X[-1], ϵ_s_x, ϵ_s_x_I[-1]:', np.around(fac_pZ[-1], 2), np.around(IS_X[-1], 2), np.around(ϵ_s_x, 2), np.around(ϵ_s_x_I[-1], 2))
    print(' LBjerrum[-1], LDebye[-1], Lcorr[-1]:', np.around(LBjerrum[-1], 2), np.around(LDebye[-1], 2), np.around(Lcorr[-1], 2))
    print(' THETA1[0], THETA2[0], THETA3[0], THETA4[0]:', np.around(THETA1[0], 2), np.around(THETA2[0], 2), np.around(THETA3[0], 2), np.around(THETA4[0], 2))
    print(' THETA1[-1], THETA2[-1], THETA3[-1], THETA4[-1]:', np.around(THETA1[-1], 2), np.around(THETA2[-1], 2), np.around(THETA3[-1], 2), np.around(THETA4[-1], 2))
    print(' THETA1[-1]/THETA3[-1], THETA2[-1]/THETA3[-1], 1/THETA3[-1], THETA4[-1]/THETA3[-1]:', np.around(THETA1[-1]/THETA3[-1], 2), np.around(THETA2[-1]/THETA3[-1], 2), np.around(1/THETA3[-1], 2), np.around(THETA4[-1]/THETA3[-1], 2))
    print(' lambda1[-1], lambda2[-1]:', np.around(lambda1[-1], 2), np.around(lambda2[-1], 2))
    print(' A1, A2, A3:', np.around(Pf.A1, 2), np.around(Pf.A2, 2), np.around(Pf.A3, 2))
    print(' eltr[0], eltr[25], eltr[26]:', np.around(Pf.eltr[0], 2), np.around(Pf.eltr[25], 2), np.around(Pf.eltr[26], 2))

  if salt == 'MgCl2':
    print(' ===== Mg^2+ Profiles in Fig 6C =====')

    ActIn_Pf = (theta[0], BornR0[0], R_sh[0], C1M[0], C3M, C4M, q1, q2, V1, V2, V3, ϵ_s_x, ϵ_s_x_I[-1], T)  # Cut ActIn
    ActIn_M1 = (q5, q6, V5, V6, C5M[0], C6M[0])
    ActIn_M2 = (0,0,0,0,0,0)
    ActIn_Mix = (ActIn_M1, ActIn_M2)

    LsP   = (ϵ_s_x, GammaB[0], lambda1[0], lambda2[0], LDebye[0], Lcorr[0])  # Ls for Profile
    THETA = (THETA1[0], THETA2[0], THETA3[0], THETA4[0])
    p_In = (pH2O, pH2O_Z[0], p1, p2, p5, numPW, numPW_Z[0], numPWI[0], numPWI_Z[0])
    print(' theta[-1], BornR_56_c[-1], BornR_c[-1], BornR_a[-1], R_sh[-1]:', np.around(theta[-1], 2), np.around(theta[-1] * BornR0_56[0], 2), np.around(theta[-1] * BornR0[0], 2), np.around(theta[-1] * BornR0[1], 2), np.around(R_sh[-1], 2))
    print(' theta[0], BornR_56_c[0], BornR_c[0], BornR_a[0], R_sh[0]:', np.around(theta[0], 2), np.around(theta[0] * BornR0_56[0], 2), np.around(theta[0] * BornR0[0], 2), np.around(theta[0] * BornR0[1], 2), np.around(R_sh[0], 2))

    Pf_Mg = Profile(ActIn_Pf, ActIn_Mix, LsP, THETA, p_In)
    print(' fac_pZ[0], IS_X[0], ϵ_s_x, ϵ_s_x_I[0]:', np.around(fac_pZ[0], 2), np.around(IS_X[0], 2), np.around(ϵ_s_x, 2), np.around(ϵ_s_x_I[0], 2))
    print(' LBjerrum[0], LDebye[0], Lcorr[0]:', np.around(LBjerrum[0], 2), np.around(LDebye[0], 2), np.around(Lcorr[0], 2))
    print(' THETA1[0], THETA2[0], THETA3[0], THETA4[0]:', np.around(THETA1[0], 2), np.around(THETA2[0], 2), np.around(THETA3[0], 2), np.around(THETA4[0], 2))
    print(' THETA1[0]/THETA3[0], THETA2[0]/THETA3[0], 1/THETA3[0], THETA4[0]/THETA3[0]:', np.around(THETA1[0]/THETA3[0], 2), np.around(THETA2[0]/THETA3[0], 2), np.around(1/THETA3[0], 2), np.around(THETA4[0]/THETA3[0], 2))
    print(' lambda1[0], lambda2[0]:', np.around(lambda1[0], 2), np.around(lambda2[0], 2))
    print(' A1, A2, A3:', np.around(Pf_Mg.A1, 2), np.around(Pf_Mg.A2, 2), np.around(Pf_Mg.A3, 2))
    print(' eltr[0], eltr[94], eltr[95]:', np.around(Pf_Mg.eltr[0], 2), np.around(Pf_Mg.eltr[25], 2), np.around(Pf_Mg.eltr[26], 2))

    print(' ===== Cl^- Profiles in Fig 6D =====')
    (THETA1, THETA2, THETA3, THETA4) = ActOut.THETA_a

    ActIn_Pf = (theta[0], BornR0[1], R_sh[0], C2M[0], C3M, C4M, q2, q1, V2, V1, V3, ϵ_s_x, ϵ_s_x_I[-1], T)  # Cut ActIn
    ActIn_M1 = (q5, q6, V5, V6, C5M[0], C6M[0])
    ActIn_Mix = (ActIn_M1, ActIn_M2)
    p_In = (pH2O, pH2O_Z[0], p2, p1, p5, numPW, numPW_Z[0], numPWI[0], numPWI_Z[0])

    Pf_Cl = Profile(ActIn_Pf, ActIn_Mix, LsP, THETA, p_In)
    print(' THETA1[0], THETA2[0], THETA3[0], THETA4[0]:', np.around(THETA1[0], 2), np.around(THETA2[0], 2), np.around(THETA3[0], 2), np.around(THETA4[0], 2))
    print(' THETA1[0]/THETA3[0], THETA2[0]/THETA3[0], 1/THETA3[0], THETA4[0]/THETA3[0]:', np.around(THETA1[0]/THETA3[0], 2), np.around(THETA2[0]/THETA3[0], 2), np.around(1/THETA3[0], 2), np.around(THETA4[0]/THETA3[0], 2))
    print(' lambda1[0], lambda2[0]:', np.around(lambda1[0], 2), np.around(lambda2[0], 2))
    print(' A1, A2, A3:', np.around(Pf_Cl.A1, 2), np.around(Pf_Cl.A2, 2), np.around(Pf_Cl.A3, 2))
    print(' eltr[0], eltr[94], eltr[95]:', np.around(Pf_Cl.eltr[0], 2), np.around(Pf_Cl.eltr[25], 2), np.around(Pf_Cl.eltr[26], 2))

# Plot Fig6A  La^3+
plt.figure(1)
plt.subplot(a, b, c)

plt.plot(Pf.rL, Pf.eltr, 'r-', label='$\phi(r)$ in $k_B T / e$')
plt.plot(Pf.rL, Pf.ditr / 10, 'b.', label='$\~{\epsilon}_s(r)$/10')
plt.plot(Pf.rL[23:], Pf.ster[23:] * 5, 'g+', label='$5S(r)$')
plt.title('(A) Potential Profiles of La$^{3+}$ at $I = 70.56$ M', fontsize=12)
plt.xlabel('Distance from the center of La$^{3+}$ in Angstrom', fontsize=12)
plt.ylabel('Potential and Dielectric Functions', fontsize=12)
plt.legend(loc="upper right")

c = c + 1

# Plot Fig6B  La^3+
plt.figure(1)
plt.subplot(a, b, c)

p1, = plt.plot(Pf.rL[23:], Pf.solv[23:] / 10, 'b+', label='$C_{H_2O}(r)$/10 in M')
p2, = plt.plot(Pf.rL[26:], (Pf.ionA[26:] + Pf.ion6[26:]) / 5, 'go', label='$C_{Cl^{-}}(r)$/5 in M')
p3, = plt.plot(Pf.rL[26:], Pf.ionC[26:], 'r-', label='$C_{La^{3+}}(r)$ in M')
p4, = plt.plot(Pf.rL[26:], Pf.ion5[26:], 'k.', label='$C_{Mg^{2+}}(r)$ in M')
p5, = plt.plot(Pf.rL[23:], -Pf.psi[23:], 'g--', label=r'$-\psi(r)$ in $e$M')
plt.title('(B) Concentration Profiles at $I = 70.56$ M', fontsize=12)
plt.xlabel('Distance from the center of La$^{3+}$ in Angstrom', fontsize=12)
plt.ylabel('Density Functions', fontsize=12)
l1 = plt.legend([p1, p2], ['$C_{H_2O}(r)$/10 in M', '$C_{Cl^{-}}(r)$/5 in M'], loc="upper right")
l2 = plt.legend([p3, p4, p5], ['$C_{La^{3+}}(r)$ in M', '$C_{Mg^{2+}}(r)$ in M', r'$-\psi(r)$ in $e$M'], loc=(0.74, 0.42))
plt.gca().add_artist(l1)
c = c + 1

# Plot Fig6C  Mg^2+
plt.figure(1)
plt.subplot(a, b, c)

plt.tight_layout()

plt.plot(Pf_Mg.rL, Pf_Mg.eltr / 4, 'r-', label='$\phi(r)/4$ in $k_B T / e$')
plt.plot(Pf_Mg.rL, Pf_Mg.ditr / 2, 'b.', label='$\~{\epsilon}_s(r)$/2')
plt.plot(Pf_Mg.rL[17:], Pf_Mg.ster[17:] * 5, 'g+', label='$5S(r)$')
plt.title('(C) Potential Profiles of Mg$^{2+}$ at $I = 1.33$ M', fontsize=12)
plt.xlabel('Distance from the center of Mg$^{2+}$ in Angstrom', fontsize=12)
plt.ylabel('Potential and Dielectric Functions', fontsize=12)
plt.legend(loc="center right")

c = c + 1

# Plot Fig6D  Cl^-
plt.figure(1)
plt.subplot(a, b, c)

plt.tight_layout()

plt.plot(Pf_Cl.rL, Pf_Cl.eltr, 'r-', label='$\phi(r)$ in $k_B T / e$')
plt.plot(Pf_Cl.rL, Pf_Cl.ditr / 2, 'b.', label='$\~{\epsilon}_s(r)$/2')
plt.plot(Pf_Cl.rL[26:], Pf_Cl.ster[26:] * 5, 'g+', label='$5S(r)$')
plt.title('(D) Potential Profiles of Cl$^-$ at $I = 1.33$ M', fontsize=12)
plt.xlabel('Distance from the center of Cl$^-$ in Angstrom', fontsize=12)
plt.legend(loc="lower right")

plt.tight_layout()

plt.show()
