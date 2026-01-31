'''
Author: Jinn-Liang Liu, June 26, 2025.

P1: Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu,
    Generalized Debye–Hückel model for activity coefficients of electrolytes in water–methanol mixtures,
    Fluid Phase Equilibria 565, 113662 (2023)
P2: Chin-Lung Li, Ren-Chuen Chen, Xiaodong Liang, Jinn-Liang Liu,
    Generalized Debye-Hückel theory of electrolyte thermodynamics: I. Application, 2025.
P3: Chin-Lung Li, Jinn-Liang Liu, Generalized Debye-Hückel equation from Poisson-Bikerman theory,
    SIAM J. Appl. Math. 80, 2003-2023 (2020).
PF0: Jinn-Liang Liu, A 3D Poisson-Nernst-Planck Solver for Modeling Biological Ion Channels, Unpublished, August 30, 2012.
PF4: Jinn-Liang Liu, 3D Poisson-Nernst-Planck-Fermi Solvers for Biological Modeling (Part 4), Unpublished, June 25, 2025.

[Bar85] Barthel, J., et al. (1985). Vapor pressures of non-aqueous electrolyte
        solutions. Part 1. Alkali metal salts in methanol. Journal of solution chemistry, 14, 621-633.
[Mar91] Marcus, Y. (1991). Thermodynamics of solvation of ions. Part 5.—Gibbs free energy of hydration
        at 298.15 K. Journal of the Chemical Society, Faraday Transactions, 87(18), 2995-2999.
[Pli13] Pliego Jr, J. R., et al. (2013). Absolute single-ion solvation free energy scale
        in methanol determined by the lithium cluster-continuum approach.
        The Journal of Physical Chemistry B, 117(17), 5129-5135.
[Sil23] G.M. Silva, et al. How to account for the concentration dependency of relative permittivity in
        the Debye–Hückel and Born equations. Fluid Phase Equilibria 566 (2023) 113671.
[Val15] M. Valiskó, et al. Unraveling the behavior of the individual ionic activity
        coefficients on the basis of the balance of ion–ion and ion–water interactions, J.
        Phys. Chem. B 119 (4) (2015) 1546–1557.

[Val15]: FcatH2O  Na+: -424 kJ/mol = -101.3 kcal/mol, kJ/mol = 1/4.184 kcal/mol, 4.184: wiki
'''
import numpy as np

c = 0.71  # DG = c ME: by ME (Molecular Weight) if DG (density gradient) [P1T1] not available
# [Bar85TableII] => DG = c ME => find c by TableII
# Verify 'NaCl': DG = c * 58.44 = 41.49 ~= 46.62 => OK apprx.

ϵ_0, e, mol, kB = 8.854187, 1.6022, 6.022045, 1.380649  # Physical Constants [P1T1]

def Solvent(x = None, T = None):  # x = 0: H2O, x = 1: MeOH
  kBTe = (kB * T / e) * 0.0001
  S2 = 0.1 * mol * e / (kBTe * ϵ_0)  # scaling parameter [PF0(5.12)]

  V3 = 4 * np.pi * 1.4 ** 3 / 3  # H2O [P1T1]
  V4 = 4 * np.pi * 1.915 ** 3 / 3  # MeOH

  if   T == 273.15: d_H2O = 55.50  #   0 ◦C [PF4Ex3.88], d: density in M
  elif T == 298.15: d_H2O = 55.34  #  25 ◦C [PF4T3.75]
  elif T == 373.15: d_H2O = 53.18  # 100 ◦C
  elif T == 423.15: d_H2O = 50.90  # 150 ◦C [PF4T3.76]
  elif T == 473.15: d_H2O = 47.85  # 200 ◦C
  elif T == 523.15: d_H2O = 44.30  # 250 ◦C
  elif T == 573.15: d_H2O = 40.24  # 300 ◦C

  ewT1 = -19.2905 + 29814.5 / T - 0.019678 * T + 1.3189 * 1e-4 * T ** 2 - 3.1144 * 1e-7 * T ** 3  # Eq12[Sil23]

  beta1, delta = 3.1306, 8.33  # Eq15[Sil23], [PF4Ex3.88]
  eTf  = beta1 * mol * 10 * (delta ** 2) / (2 * ϵ_0 * kB)
  eT0  = eTf * (55.50 / 273.15)
  ew0  = 87.71  # by hand, 87.98 by Eq12[Sil23]
  ewT2 = ew0 + eTf * (d_H2O / T - 55.50 / 273.15)  # Eq15[Sil23], [PF4Ex3.88]

  C3M = (1 - x) * d_H2O  # in M
  C4M = x * 24.55  # [PF4T3.82A]
  ϵ_s_x = (1 - x) * ewT2 + x * 32.66  # Eq15[Sil23]

  C3M, C4M = C3M * S2, C4M * S2  # unitless
  pH2O, pMeOH = 1.62, 1.32  # polarizability [P2T1]

  return S2, C3M, C4M, V3, V4, pH2O, pMeOH, ϵ_s_x


def Born(salt = None, ϵ_s_x = None, x = None, T = None):
  if salt == 'NaF':
    q1, q2 = 1, -1
    p1, p2 = 0.279, 1.144  # [P2T1]
    V1 = 4 * np.pi * 0.95 ** 3 / 3  # [P1T1]
    V2 = 4 * np.pi * 1.36 ** 3 / 3
    mM, DG = 41.99, 41.38  # [P1T1] mM: Molar mass, [P1T1] DG: Density Gradient
    FcatH2O, FcatMeOH = -101.3, -91.9   # -424 [Val15], [Pli13]=[PF4T3.82B] at T=298.15
    FanH2O,  FanMeOH  = -102.5, -109.2  # -429 [Val15], [Pli13]
  elif salt == 'NaCl':
    q1, q2 = 1, -1
    p1, p2 = 0.279, 3.253
    V1 = 4 * np.pi * 0.95 ** 3 / 3
    V2 = 4 * np.pi * 1.81 ** 3 / 3
    mM, DG = 58.44, 46.62  # [P1T1], [P1T1]
    FcatH2O, FcatMeOH = -101.3, -91.9  # -424 [Val15], [Pli13]=[PF4T3.82B]
    FanH2O,  FanMeOH  = -72.7, -81.1   # -304 [Val15], [Pli13]
  elif salt == 'NaBr':
    q1, q2 = 1, -1
    p1, p2 = 0.279, 4.748
    V1 = 4 * np.pi * 0.95 ** 3 / 3
    V2 = 4 * np.pi * 1.95 ** 3 / 3
    mM, DG = 102.89, 77.13  # [P1T1], [P1T1]
    FcatH2O, FcatMeOH = -101.3, -91.9  # -424 [Val15], [Pli13]=[PF4T3.82B]
    FanH2O,  FanMeOH  = -66.4, -75.1   # -278 [Val15], [Pli13]
  elif salt == 'LiCl':
    q1, q2 = 1, -1
    p1, p2 = 0.029, 3.253
    V1 = 4 * np.pi * 0.6 ** 3 / 3  # [P3P13]
    V2 = 4 * np.pi * 1.81 ** 3 / 3
    mM, DG = 42.39, 32.65  # [PF4T3.78A], [PF4T3.82A]
    FcatH2O, FcatMeOH = -126.4, -118.1  # -529 [Val15], [PF4T3.82B]
    FanH2O,  FanMeOH  = -72.7, -81.1    # -304 [Val15], [Pli13]
  elif salt == 'MgCl2':
    q1, q2 = 2, -1
    p1, p2 = 0.835, 3.253
    V1 = 4 * np.pi * 0.65 ** 3 / 3  # [P3P13]
    V2 = 4 * np.pi * 1.81 ** 3 / 3
    mM, DG = 95.21, 95.21 * c  # [PF4T3.78A],
    FcatH2O, FcatMeOH = -461.52, -(1931.+2.)/4.184  # -1931 [Val15], 2 [Kal00Table5]
      # Verify Na+: 8.2 [Mar88TableI] ~= 7.2 [Kal00Table5]
    FanH2O,  FanMeOH  = -72.7, -81.1  # -304 [Val15], [Pli13]
      # Verify: [Mar88TableI]: -304-13.2 = -317.2/4.184 = -75.81 ~= -81.1
  elif salt == 'CaCl2':
    q1, q2 = 2, -1
    p1, p2 = 0.588, 3.253
    V1 = 4 * np.pi * 0.99 ** 3 / 3  # [P3P13]
    V2 = 4 * np.pi * 1.81 ** 3 / 3
    mM, DG = 110.98, 110.98 * c  # [PF4T3.78A],
    FcatH2O, FcatMeOH = -384.32, -(1608.+11.2)/4.184  # −1608 [Val15], 11.2 [Kal00Table5]
    FanH2O,  FanMeOH  = -72.7, -81.1  # −304 [Val15], [Pli13]
  elif salt == 'LaCl3':
    q1, q2 = 3, -1
    p1, p2 = 4.45, 3.253
    V1 = 4 * np.pi * 1.05 ** 3 / 3  # [P3P13]
    V2 = 4 * np.pi * 1.81 ** 3 / 3
    mM, DG = 110.98, 110.98 * c  # [PF4T3.78A],
    FcatH2O, FcatMeOH = -738.53, -(3090.+16.)/4.184  # -3090 [Mar91], 16=5.4*3 [Kal00Table5] (guess)
    FanH2O,  FanMeOH  = -72.7, -81.1  # [Val15], [Pli13]
  else:
    warnings.warn('Warning: Salt not found.')

  Fcat = (1 - x) * FcatH2O + x * FcatMeOH
  Fan  = (1 - x) * FanH2O  + x * FanMeOH

  ϵ_s_x_298 = (1 - x) * 78.41 + x * 32.66

  BornR0c = - q1 * q1 * e * e * mol * 10000 * (1 - 1 / ϵ_s_x_298) / (Fcat * 4.1868 * 8 * np.pi * ϵ_0)
  BornR0a = - q2 * q2 * e * e * mol * 10000 * (1 - 1 / ϵ_s_x_298) / (Fan * 4.1868 * 8 * np.pi * ϵ_0)
  BornR0 = np.array([BornR0c, BornR0a])  # at T = 298.15

  BornR0 = BornR0 * (298.15 / T) * (1 - 1 / ϵ_s_x) / (1 - 1 / ϵ_s_x_298)  # [P2(4.4))]

  return BornR0, q1, q2, p1, p2, V1, V2, mM, DG


def m2M(m = None, mM = None, DG = None, x = None, T = None):  # molality to Molarity [PF4T3.82A]
    # 0.9971 = density of water [P1T1]
    # 0.9128 = density of 50-50 water-methanol mixture
    # 0.7866 = density of methanol [PF4T3.82A] = 0.7863 [Bar85]
  if   T == 298.15: d_H2O = 0.997  # 25  ◦C [PF4T3.75], d: density in g/cm^3
  elif T == 373.15: d_H2O = 0.958  # 100 ◦C
  elif T == 423.15: d_H2O = 0.917  # 150 ◦C [PF4T3.76]
  elif T == 473.15: d_H2O = 0.862  # 200 ◦C
  elif T == 523.15: d_H2O = 0.798  # 250 ◦C
  elif T == 573.15: d_H2O = 0.725  # 300 ◦C

  rho_0 = (x - 0.5) * (x - 1) / 0.5 * d_H2O - x * (x - 1) / 0.25 * 0.9128 + x * (x - 0.5) / 0.5 * 0.7866
  rho_s = rho_0 + DG * m / 1000

  M = np.multiply(1000 * m, rho_s) / (1000 + np.multiply(m, mM))

  return M
