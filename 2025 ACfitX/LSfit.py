'''
Author: Jinn-Liang Liu, Jan 27, 2025.
'''

import numpy as np

from Activity import ActF_1, ActF_2  # Activity Formula 1, 2

#Activity, AF_idx = ActF_1, 1
Activity, AF_idx = ActF_2, 2
deviate, ErrTol = 0.0001, 0.008
  # alpha = [ 0.01147  0.34639 -0.5708   0.31484 -0.05963] NaF  Example4_1.py
#deviate, ErrTol = 0.0001, 0.003
  # alpha = [ 0.04717  0.20311 -0.3776   0.20943 -0.03961] NaF
  # NG: Fig LaCl3  Example4_3.py
#deviate, ErrTol = 0.0001, 0.001
  # alpha = [ 0.05242  0.18666 -0.36142  0.20436 -0.03948] NaF
  # Error: Fig MgCl2  Example4_3.py alpha = [ 51.82336 -35.5022    8.90877  -0.9709    0.03886] MgCl2



class LSfit():  # input: g_data; output: LS g_fit and alpha[0], [1], [2], [3]
  def __init__(self, LfIn):
    (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, C1m, \
     q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T) = LfIn

    N = len(C1M)
    Nc = int( N * (N - 1) * (N - 2) * (N - 3) * (N - 4) / (24 * 5) )  # total combinations for finding alpha[i]
    theta_all = np.zeros(N)

    ActIn_M1 = (0,0,0,0,0,0)  # for mix-salt 1
    ActIn_M2 = (0,0,0,0,0,0)  # for mix-salt 2
    ActIn_Mix = (ActIn_M1, ActIn_M2)

    # [P1 Step 5.1] Get theta(k) that yields best g_fit(k) to g_data(k) by alternating variation of theta from 1.
    for k in range(N):
      g_fit_k, theta, n = 1.0, 1.0, 1
      while np.abs(g_fit_k - g_data[k]) > ErrTol and theta > 0 and theta < 2:  # tune ErrTol for better self.alpha
        theta = theta + ((-1) ** n) * (deviate * n)
        ActIn = (theta, BornR0, Rsh_c[k], Rsh_a[k], C1M[k], C3M, C4M, IS, C1m, \
                 q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T)

        ActOut = Activity(ActIn, ActIn_Mix)  # Given theta, Activity() returns a mean activity.
        g_fit = ActOut.g_PF
        if (len(g_fit.shape) > 0):
          g_fit_k = g_fit[k]  # g_fit: array by array IS
        else:
          g_fit_k = g_fit  # g_fit: scalar
        n = n + 1

      theta_all[k] = theta

    ALPHA = np.zeros((Nc, 5))
    count = 0

    # [P1 Step 5.2] Given theta_all[i], theta_all[j], theta_all[k],
    # find (alpha[0], ..., [4]) by sloving a 5x5 matrix system for all (i, j, k).
    for i in range(N):  # Fit for H2O
      for j in range(i+1, N):
        for k in range(j+1, N):
          for l in range(k+1, N):
            for m in range(l+1, N):
              ti, tj, tk, tl, tm = theta_all[i], theta_all[j], theta_all[k], theta_all[l], theta_all[m]
              b = np.reshape(np.array([ti - 1, tj - 1, tk - 1, tl - 1, tm - 1]), (5, 1))
              A = [[IS[i] ** 0.5, IS[i], IS[i] ** 1.5, IS[i] ** 2, IS[i] ** 2.5], \
                   [IS[j] ** 0.5, IS[j], IS[j] ** 1.5, IS[j] ** 2, IS[j] ** 2.5], \
                   [IS[k] ** 0.5, IS[k], IS[k] ** 1.5, IS[k] ** 2, IS[k] ** 2.5], \
                   [IS[l] ** 0.5, IS[l], IS[l] ** 1.5, IS[l] ** 2, IS[l] ** 2.5], \
                   [IS[m] ** 0.5, IS[m], IS[m] ** 1.5, IS[m] ** 2, IS[m] ** 2.5] ]

              alpha = np.linalg.solve(A, b)
              ALPHA[count, :] = np.reshape(alpha, (1, 5))
              count = count + 1

    g_fit_all = np.zeros((Nc, N))

    # [P1 Step 5.3]
    for i in range(Nc):
      theta = 1 + ALPHA[i][0] * (IS ** 0.5) + ALPHA[i][1] * IS + ALPHA[i][2] * (IS ** 1.5) \
                + ALPHA[i][3] * (IS ** 2)   + ALPHA[i][4] * (IS ** 2.5)
      ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, IS, C1m, \
               q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T)

      ActOut = Activity(ActIn, ActIn_Mix)
      g_fit_all[i, :] = ActOut.g_PF  # ComplexWarning: Casting complex values to real discards the imaginary part

    # [P1 Step 5.4]
    Errs = np.zeros(Nc)
    g_fit_all[np.isnan(g_fit_all)] = 1000.  # g_fit_all maybe Numpy NaN.

    for i in range(Nc):
      err = np.max( np.abs(g_fit_all[i] - g_data) )
      Errs[i] = err

    # Sort Nc errors in ascending order.
    ascend_errs = sorted(Errs)

    idx = np.argmin(Errs)

    # [P1 Step 5.5] Resume the best 5 alphas, g_fit.
    self.theta_all = theta_all
    self.alpha = ALPHA[idx, :]
    self.g_fit = g_fit_all[idx,:]


class LSfitX():  # for eXtrapolation; input: g_data and alphaX=alpha[0]; output: LS alpha[1], [2]
  def __init__(self, LfInX):  # add alphaX to LfIn and get LfInX for fixing alpha[0]
    (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, IS, C1m, \
     q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T, alphaX, ActIn_Mix) = LfInX

    N = len(C1M)
      #Nc = int(N * (N - 1) * (N - 2) / 6)  # total combinations for finding (alpha[1], [2], [3]) with alpha[0]=alphaX fixed
    Nc = int( N * (N - 1) * (N - 2) * (N - 3) * (N - 4) / (24 * 5) )  # total combinations for finding (alpha[0], [1], [2], [3], [4])
    theta_all = np.zeros(N)

    # [P1 Step 5.1] Get theta(k) that yields best g_fit(k) to g_data(k) by alternating variation of theta from 1.
    for k in range(N):
      g_fit_k, theta, n = 1.0, 1.0, 1
      while np.abs(g_fit_k - g_data[k]) > ErrTol and theta > 0 and theta < 2:  # tune ErrTol for better self.alpha
        theta = theta + ((-1) ** n) * (deviate * n)
        ActIn = (theta, BornR0, Rsh_c[k], Rsh_a[k], C1M[k], C3M, C4M, IS, C1m, \
                 q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T)

        # Given theta, Activity() returns a mean activity.
        ActOut = Activity(ActIn, ActIn_Mix)
        g_fit = ActOut.g_PF
        if (len(g_fit.shape) > 0):
          g_fit_k = g_fit[k]  # g_fit: array by array IS
        else:
          g_fit_k = g_fit  # g_fit: scalar
        n = n + 1

      theta_all[k] = theta

    ALPHA = np.zeros((Nc, 5))
    count = 0

    # Modified [P1 Step 5.2] Given alphaX == alpha[0] fixed, theta_all[i], theta_all[j],
    # find (alphaX, alpha[1],..., [4]) by sloving 4x4 matrix system for all (i, j).
    for i in range(N):  # Fit for H2O
      for j in range(i+1, N):
        for k in range(j+1, N):
          for l in range(k+1, N):
            for m in range(l+1, N):
              ti, tj, tk, tl, tm = theta_all[i], theta_all[j], theta_all[k], theta_all[l], theta_all[m]
              b = np.reshape(np.array([alphaX, tj - 1, tk - 1, tl - 1, tm - 1]), (5, 1))
              A = [[           1,     0,            0,          0,            0], \
                   [IS[j] ** 0.5, IS[j], IS[j] ** 1.5, IS[j] ** 2, IS[j] ** 2.5], \
                   [IS[k] ** 0.5, IS[k], IS[k] ** 1.5, IS[k] ** 2, IS[k] ** 2.5], \
                   [IS[l] ** 0.5, IS[l], IS[l] ** 1.5, IS[l] ** 2, IS[l] ** 2.5], \
                   [IS[m] ** 0.5, IS[m], IS[m] ** 1.5, IS[m] ** 2, IS[m] ** 2.5] ]

              alpha = np.linalg.solve(A, b)
              ALPHA[count, :] = np.reshape(alpha, (1, 5))
              count = count + 1

    g_fit_all = np.zeros((Nc, N))

    # [P1 Step 5.3]
    for i in range(Nc):
      theta = 1 + ALPHA[i][0] * (IS ** 0.5) + ALPHA[i][1] * IS + ALPHA[i][2] * (IS ** 1.5) \
                + ALPHA[i][3] * (IS ** 2)   + ALPHA[i][4] * (IS ** 2.5)

      if any(abs(theta) > 100): theta = theta / 10

      ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, IS, C1m, \
               q1, q2, V1, V2, V3, V4, ϵ_s_x_I, T)

      ActOut = Activity(ActIn, ActIn_Mix)
      g_fit_all[i, :] = ActOut.g_PF  # ComplexWarning: Casting complex values to real discards the imaginary part

    # [P1 Step 5.4]
    Errs = np.zeros(Nc)

    for i in range(Nc):
      err = np.max( np.abs(g_fit_all[i] - g_data) )
      Errs[i] = err

    # Sort Nc errors in ascending order.
    ascend_errs = sorted(Errs)

    idx = np.argmin(Errs)

    # [P1 Step 5.5] Resume the best 3 alphas, g_fit.
    self.alpha = ALPHA[idx, :]
