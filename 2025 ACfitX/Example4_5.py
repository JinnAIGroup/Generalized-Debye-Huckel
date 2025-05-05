'''
Author: Jinn-Liang Liu, May 5, 2025.
Example 4.5 Fig 6A: Profiles around La in LaCl3+MgCl2+H2O at T = 25 ◦C
            Fig 6B: More Profiles
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from Physical import Solvent, Born, m2M
from Data4_5 import DataFit
from LSfit import LSfit, Activity, LSfitX
from Profile import Profile

# Solution Parameters:
#   0: void, 1: cation, 2: anion, 3: H2O, 4: MeOH, x: mixing percentage of 3 and 4 in [0, 1]
#   5: cation, 6: anion, 7: cation, 8: anion
#   Bulk concentrations in M: C1M (array), C2M (array), C3M (scalar), C4M (scalar)
#   ϵ_s_x (scalar): dielectric constant of mixed solvent [P1(11)], V: volume
#   gamma: mean activity data [P1(16)] of target salt 1+2 (array)

np.set_printoptions(suppress=True)  # set 0.01 not 1e-2
plt.figure(figsize=(13,8))
a, b, c = 1, 2, 1  # subplot(a, b, c): rows, columns, counter

Salts = ['LaCl3']
Ts, Z = [298.15], 0.68

for salt in Salts:
  for (T, T_i) in zip(Ts, range(len(Ts))):
    # Part 1: Fiting ...

    S2, C3M, C4M, V3, V4, pH2O, _, ϵ_s_x = Solvent(0, T)  # x=0 for pure H2O
    BornR0, q1, q2, p1, p2, V1, V2, mM, DG = Born(salt, ϵ_s_x, 0, T)  # for H2O
    a_c, a_a = (3 * V1 / 4 / np.pi) ** (1/3),  (3 * V2 / 4 / np.pi) ** (1/3)

    DF = DataFit(salt)  # for H2O
    g_data = DF.lngamma[T_i]  # gamma: mean activity

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
    V_sh = 4 * np.pi * (R_ca ** 3) / 3 - V1  # shell volumn
    N_H2O_max = V_sh / V3  # max solvation water number (no void)

    LfIn = (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, DF.C1m, \
            q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T)
    LfOut = LSfit(LfIn)

    g_fit, alpha = LfOut.g_fit, LfOut.alpha  # fitted results
    theta = 1 + alpha[0] * (IS ** 0.5) + alpha[1] * IS + alpha[2] * (IS ** 1.5) + alpha[3] * (IS ** 2) + alpha[4] * (IS ** 2.5)
    print(" alpha =", np.around(alpha, 5), salt, T)
    print(' Fig6A No Mix: a_c, BornR0[0], BornR_c[0], Rsh_c[0], N_H2O_max[0]:', np.around(a_c, 2), np.around(BornR0[0], 2), np.around(theta[0] * BornR0[0], 2), np.around(Rsh_c[0], 2), np.around(N_H2O_max[0], 2))

    # Part 2: Mix-Salt Predicting ...

    salt_1 = 'MgCl2'  # for 2-salt, add 1 salt CA to mix, C = 5, A = 6
    C1m_Mix = DF.C1m / 2  # for 2-salt

    C1M = m2M(C1m_Mix, mM, DG, 0, T) * S2  # new C1m_Mix
    C2M = -q1 * C1M / q2

    _, q5, q6, p5, p6, V5, V6, mM_Mix, D_Mix = Born(salt_1, ϵ_s_x, 0, T)  # for H2O
    C5M = m2M(C1m_Mix, mM_Mix, D_Mix, 0, T) * S2
    C6M = -q5 * C5M / q6
      #print(" Mix C1M[-1], C2M[-1] =", np.around(C1M[-1], 2), np.around(C2M[-1], 2), salt)
      #print(" Mix C5M[-1], C6M[-1] =", np.around(C5M[-1], 2), np.around(C6M[-1], 2), salt_1)

    IS_X =  0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2)  # 2-salt array

    ISX0 = IS[0] if IS[0] < IS_X[0] else IS_X[0]
    IS_X = np.linspace(ISX0, IS_X[-1], num=2 * len(IS_X))  # double spline points

    Spline = InterpolatedUnivariateSpline(IS, g_data, k=1)  # k: spline order 1, 2, 3 (cubic)

    g_data_X = Spline(IS_X)  # inter/eXtrapolation

    C1m_X = IS_X * DF.C1m[-1] / IS_X[-1]  # scale back to 1-salt DF.C1m

    C1m_Xmix = C1m_X / 2
    C1M = m2M(C1m_Xmix, mM, DG, 0, T) * S2
    C2M = -q1 * C1M / q2
    _, _, _, _, _, _, _, mM_X, D_X = Born(salt_1, ϵ_s_x, 0, T)
    C5M = m2M(C1m_Xmix, mM_X, D_X, 0, T) * S2
    C6M = -q5 * C5M / q6

    IS_X = 0.5 * (C1M * q1 ** 2 + C2M * q2 ** 2 + C5M * q5 ** 2 + C6M * q6 ** 2)  # 2-salt inter/eXtrapolation array
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
    V_sh = 4 * np.pi * (R_ca ** 3) / 3 - V1
    N_H2O_max = V_sh / V3

    ActIn_M1 = (q5, q6, V5, V6, C5M, C6M)  # for mix-salt 1
    ActIn_M2 = (0,0,0,0,0,0)               # for mix-salt 2
    ActIn_Mix = (ActIn_M1, ActIn_M2)

    alphaX = alpha[0]  # add alphaX to LfIn and get LfInX
    LfInX = (g_data_X, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS_X, DF.C1m, \
             q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T, alphaX, ActIn_Mix)

    LfOut = LSfitX(LfInX)  # input: g_data_X and fixed alpha[0]; output: alpha[1], ...
    alpha_X = LfOut.alpha
    print(" alpha_X =", np.around(alpha_X, 5), salt, "+", salt_1)
    theta = 1 + alpha_X[0] * (IS_X ** 0.5) + alpha_X[1] * IS_X + alpha_X[2] * (IS_X ** 1.5) + alpha_X[3] * (IS_X ** 2) + alpha_X[4] * (IS_X ** 2.5)

    ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, IS_X, DF.C1m, \
             q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T)

    ActOut = Activity(ActIn, ActIn_Mix)  # Prediction
    print(' Fig6A Mix: g_PF[-1], g_data_X[-1], BornR_c[0], Rsh_c[0], N_H2O_max[0]:', np.around(ActOut.g_PF[-1], 2), np.around(g_data_X[-1], 2), np.around(theta[0] * BornR0[0], 2), np.around(Rsh_c[0], 2), np.around(N_H2O_max[0], 2))
    print(' theta[-1], BornR_c[-1], Rsh_c[-1], N_H2O_max[-1]:', np.around(theta[-1], 2), np.around(theta[-1] * BornR0[0], 2), np.around(Rsh_c[-1], 2), np.around(N_H2O_max[-1], 2))

    # Part 3: Plot Profiles ...

    (GammaB, lambda1, lambda2, LDebye, LBjerrum, Lcorr) = ActOut.Ls
    (THETA1, THETA2, THETA3, THETA4) = ActOut.THETA

    ActInC = (theta[-1], BornR0, Rsh_c[-1], C1M[-1], C3M, C4M, q1, q2, V1, V2, V3, ϵ_s_x_I[-1], T)  # Cut ActIn
    ActIn_M1 = (q5, q6, V5, V6, C5M[-1], C6M[-1])
    ActIn_M2 = (0,0,0,0,0,0)
    ActIn_Mix = (ActIn_M1, ActIn_M2)

    LsP   = (ϵ_s_x, GammaB[-1], lambda1[-1], lambda2[-1], LDebye[-1], Lcorr[-1])  # Ls for Profile
    THETA = (THETA1[-1], THETA2[-1], THETA3[-1], THETA4[-1])
    p_In = (pH2O, pH2O_Z[-1], p1, p2, p5, numPW, numPW_Z[-1], numPWI[-1], numPWI_Z[-1])

    Pf = Profile(ActInC, ActIn_Mix, LsP, THETA, p_In)
    print(' eltr[0], fac_pZ[-1], IS_X[-1], ϵ_s_x, ϵ_s_x_I[-1], ditr[16]:', np.around(Pf.eltr[0], 2), np.around(fac_pZ[-1], 2), np.around(IS_X[-1], 2), np.around(ϵ_s_x, 2), np.around(ϵ_s_x_I[-1], 2), np.around(Pf.ditr[16], 2))

# Plot Fig6A
plt.figure(1)
plt.subplot(a, b, c)

plt.plot(Pf.rL, Pf.eltr / 5, 'r-', label='$\phi(r)/5$ in $k_B T / e$')
plt.plot(Pf.rL, Pf.ditr / 5, 'b.', label='$\~{\epsilon}_s(r)$/5')
plt.plot(Pf.rL[12:], Pf.ster[12:] * 5, 'g+', label='$5S(r)$')
plt.title('(A) Potential Profiles in LaCl$_3$+MgCl$_2$+H$_2$O', fontsize=12)
plt.xlabel('Distance from the center of La$^{3+}$ in Angstrom', fontsize=12)
plt.ylabel('Electric, Steric, and Dielectric Functions', fontsize=12)
plt.legend(loc="upper right")
#plt.legend(loc="center right")

c = c + 1

# Plot Fig6B
plt.figure(1)
plt.subplot(a, b, c)

plt.plot(Pf.rL[12:], Pf.solv[12:] / 10, 'b+', label='$C_{H_2O}(r)$/10 in M')
plt.plot(Pf.rL[26:], (Pf.ionA[26:] + Pf.ion6[26:]) / 5, 'go', label='$C_{Cl^{-}}(r)$/5 in M')
plt.plot(Pf.rL[26:], Pf.ionC[26:], 'r-', label='$C_{La^{3+}}(r)$ in M')
plt.plot(Pf.rL[26:], Pf.ion5[26:], 'k.', label='$C_{Mg^{2+}}(r)$ in M')
plt.plot(Pf.rL[12:], -Pf.psi[12:], 'g--', label=r'$-\psi(r)$ in $e$M')
plt.title('(B) Concentration Profiles in LaCl$_3$+MgCl$_2$+H$_2$O', fontsize=12)
plt.xlabel('Distance from the center of La$^{3+}$ in Angstrom', fontsize=12)
plt.ylabel('Water, Ion, and Polarization Density Functions', fontsize=12)
plt.legend(loc="upper right")
#plt.legend(loc="center right")
#plt.legend(loc="lower right")

plt.show()
