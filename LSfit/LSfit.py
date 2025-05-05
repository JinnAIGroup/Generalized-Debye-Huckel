# Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu. September 20, 2022.

import numpy as np

from Activity import Activity

class LSfit():
  def __init__(self, LfIn):
    (g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, \
     q1, q2, V0, V1, V2, V3, V4, diS, T) = LfIn

    N = len(C1M)
    Nc = int(N * (N - 1) * (N - 2) / 6)
    theta_all = np.zeros(N)

    # [Step 5.1] Get theta(k) that yields best g_fit(k) to g_data(k) by alternating variation of theta from 1.
    for k in range(N):
      g_fit, theta, n = 1.0, 1.0, 1
      while np.abs(g_fit - g_data[k]) > 0.003 and theta > 0 and theta < 2:
        theta = theta + ((-1) ** n) * (0.0001 * n)
        ActIn = (theta, BornR0, Rsh_c[k], Rsh_a[k], C1M[k], C3M, C4M, \
                 q1, q2, V0, V1, V2, V3, V4, diS, T)

        # Given theta, Activity() returns a mean activity.
        ActOut = Activity(ActIn)
        g_fit = ActOut.g_PF
        n = n + 1

      theta_all[k] = theta

    ALPHA = np.zeros((Nc, 3))
    index_ijk = np.zeros((Nc, 3), dtype=np.uint8)
    count = 0

    # [Step 5.2] Find each (alpha[0], [1], [2]) from each 3-combinations of
    # theta(I_k) by sloving a 3x3 matrix system for all Nc.
    for i in range(N):
      for j in range(i+1, N):
        for k in range(j+1, N):
          ti, tj, tk = theta_all[i], theta_all[j], theta_all[k]
          theta_ijk = np.reshape(np.array([ti, tj, tk]), (3, 1))
          A = [[C1M[i] ** 0.5, C1M[i], C1M[i] ** 1.5], \
               [C1M[j] ** 0.5, C1M[j], C1M[j] ** 1.5], \
               [C1M[k] ** 0.5, C1M[k], C1M[k] ** 1.5] ]

          alpha = np.linalg.solve(A, (theta_ijk - 1))
          index_ijk[count, :] = np.array([i, j, k])
          ALPHA[count, :] = np.reshape(alpha, (1, 3))
          count = count + 1

    g_fit_all = np.zeros((Nc, N))

    # [Step 5.3]
    for i in range(Nc):
      theta = 1 + ALPHA[i][0] * (C1M ** 0.5) + ALPHA[i][1] * C1M + ALPHA[i][2] * (C1M ** 1.5)
      ActIn = (theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, \
               q1, q2, V0, V1, V2, V3, V4, diS, T)

      ActOut = Activity(ActIn)
      g_fit_all[i, :] = ActOut.g_PF

    # [Step 5.4]
    Errs = np.zeros(Nc)

    for i in range(Nc):
      err = 0
      for k in range(N):
        err = err + (g_fit_all[i][k] - g_data[k]) ** 2
      Errs[i] = err

    # Sort Nc errors in ascending order.
    ascend_errs = sorted(Errs)

    if salt == 'NaCl':
      for ii in range(500):
        w = np.where(Errs == ascend_errs[ii])[0]
        idx = w[0]
        if ALPHA[idx][0] > 0.02:
          break
      #print(" ii = ", ii)  # = 236
      # The 236th (not least) error yileds alpha[0] > 0.02 that makes more physical sense.
    else:
      idx = np.argmin(Errs)

    indices = index_ijk[idx, :]
    #print(" indices = ", indices)

    # [Step 5.5] Resume the best 3 thetas, 3 alphas, g_fit.
    self.theta = np.array([ theta_all[indices[0]], theta_all[indices[1]], theta_all[indices[2]] ])
    self.alpha = ALPHA[idx, :]
    self.g_fit = g_fit_all[idx,:]
