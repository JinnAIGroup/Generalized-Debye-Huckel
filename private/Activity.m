% Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu. September 7, 2022. 

function g_PF = Activity(theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, q1, q2, V0, V1, V2, V3, V4, diS, T)
  epsln = 8.854187; e = 1.6022; mol = 6.022045; kB = 1.380649; kBTe = (kB*T/e)*0.0001;
  S1 = 1000*e/(4*pi*kBTe*epsln); S2 = 0.1*mol*e/(kBTe*epsln);  % [(vi)]
  
  C2M = q1*C1M;
  BornR_c = theta.*BornR0(1);  % [(14)]
  BornR_a = theta.*BornR0(2);
   
  BPfrac = (V1*C1M + V2*C2M + V3*C3M + V4*C4M)/S2/1660.6;
  LAMBDA = (V1*V1*C1M + V2*V2*C2M + V3*V3*C3M + V4*V4*C4M)/S2/1660.6;
  LAMBDA = (1-BPfrac)*V0 + ~~LAMBDA;
  LAMBDA = ( C1M/S2/1660.6 )*power(V1 - V2, 2)./LAMBDA;  % [(10)]
      
  LDebye = sqrt( diS*epsln*kBTe*1.6606/e./(((1-LAMBDA)*q1*q1.*C1M - q1*q2*C1M)/S2) );  % Debye Length [(9)]
  LBjerrum = S1/diS;  % Bjerrum Length
  Lcorr = sqrt( LBjerrum*LDebye/48 );  % Correlation Length
  
  a1 = q1*q1*e*e/(8*pi*epsln*diS*kB*T)*power(10,7);
  a2 = q2*q2*e*e/(8*pi*epsln*diS*kB*T)*power(10,7);
  
  lam = 1 - 4*power(Lcorr, 2)./power(LDebye, 2);
  lambda1 = (1 - sqrt(lam))./(2*power(Lcorr, 2));  % [(15)]
  lambda2 = (1 + sqrt(lam))./(2*power(Lcorr, 2));
      
  d1 = lambda2.*( power(Lcorr, 2).*lambda1-1);
  d2 = lambda1.*( power(Lcorr, 2).*lambda2-1);
      
  THETA_c = (d1-d2)./( d1.*( sqrt(lambda1).*Rsh_c+1 ) - d2.*( sqrt(lambda2).*Rsh_c+1 ) );  % [(7)]
  THETA_a = (d1-d2)./( d1.*( sqrt(lambda1).*Rsh_a+1 ) - d2.*( sqrt(lambda2).*Rsh_a+1 ) );
  
  gamma1 = exp( a1*(1./BornR_c - 1./BornR0(1) + (THETA_c-1)./Rsh_c) );  % cation activity [(15)]
  gamma2 = exp( a2*(1./BornR_a - 1./BornR0(2) + (THETA_a-1)./Rsh_a) );  % anion activity
   
  g_PF = (abs(q2)*log(gamma1) + abs(q1)*log(gamma2))/(abs(q1) + abs(q2));  % mean activity in log [(16)]
end