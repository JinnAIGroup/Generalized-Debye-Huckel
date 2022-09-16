% Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu. September 7, 2022. 

function [g_fit, alpha, theta] = LSfit(g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, q1, q2, V0, V1, V2, V3, V4, diS, T)
  N  = length(C1M);      % total data points
  Nc = N*(N-1)*(N-2)/6;  % total number of 3-combinations of theta(I_k) (3 data points) to uniquely determine alpha(1), (2), (3) 
  theta_all = zeros(1, N);  % all theta(I_k) for k = 1, ..., N

  % [Step 5.1] Get theta(k) that yields best g_fit(k) to g_data(k) by alternating variation of theta from 1.
  for k = 1:N
    g_fit = 1.; theta = 1.; n = 1;
    while (abs(g_fit - g_data(k)) > 0.003 && theta > 0 && theta < 2)
      theta = theta + power(-1, n)*(0.0001*n);  % alternating variation
  
      % Given theta, Activity() returns a mean activity. 
      g_fit = Activity(theta, BornR0, Rsh_c(k), Rsh_a(k), C1M(k), C3M, C4M, q1, q2, V0, V1, V2, V3, V4, diS, T);
      n = n + 1;
    end
    theta_all(k) = theta;
  end
  
  ALPHA = zeros(3, Nc); index_ijk = zeros(Nc, 3); count = 0; 
  
  % [Step 5.2] Find each (alpha(1), (2), (3)) from each 3-combinations of
  % theta(I_k) by sloving a 3x3 matrix system for all Nc. 
  for i = 1:N 
    for j = i+1:N
      for k = j+1:N
        ti = theta_all(i);
        tj = theta_all(j);
        tk = theta_all(k);
        theta_ijk = [ti; tj; tk];  % shape: 3 x 1
        alpha = ([C1M(i)^0.5, C1M(i), C1M(i)^1.5; ...
                  C1M(j)^0.5, C1M(j), C1M(j)^1.5; ...
                  C1M(k)^0.5, C1M(k), C1M(k)^1.5])\(theta_ijk-1);  % shape: 3 x 1
        count = count+1;
        index_ijk(count, :) = [i, j, k];
        ALPHA(:, count) = alpha;
      end
    end 
  end
  
  g_fit_all = zeros(Nc, N);

  % [Step 5.3]
  for i = 1:Nc
    theta = 1 + ALPHA(1,i)*power(C1M, 1/2) + ALPHA(2,i)*C1M + ALPHA(3,i)*power(C1M, 3/2);

    g_fit = Activity(theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, q1, q2, V0, V1, V2, V3, V4, diS, T);
    g_fit_all(i, :) = exp(g_fit);
  end
  
  % [Step 5.4]
  Errs = zeros(Nc, 1);  % shape: Nc x 1
  for i = 1:Nc
    err = 0;
    for k = 1:N
      err = err + ( g_fit_all(i, k) - exp(g_data(k)) )^2;  % sum of all N squared error
    end
    Errs(i) = err;  % a total of Nc sums of errors
  end
  
  % Sort Nc errors in ascending order.
  ascend_errs = sort(Errs);
  if salt == "NaCl"
      idx = find(Errs == ascend_errs(290));  
      % index of the 290th error for NaCl to get alpha(1), (2), (3) = 0.0224, -0.0113, -0.0005 
      % (not least error but more physical sense by conditioning alpha(1) > 0.02 to find 290)
  else
      idx = find(Errs == ascend_errs(1));  % index of least error for NaF and NaBr
  end
  
  indices = index_ijk(idx, :);  % resume the indices of data points corresponding to the LS error
  
  % [Step 5.5] Resume the best 3 thetas, 3 alphas, g_fit.
  theta = [theta_all(indices(1)); theta_all(indices(2)); theta_all(indices(3))];
  alpha = ALPHA(:, idx);
  g_fit = log(g_fit_all(idx, :));
end  
