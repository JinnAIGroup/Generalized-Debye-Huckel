% Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu. September 18, 2022. 

function [Rsh_c, Rsh_a] = Newton(C1M, CxM, V0, V1, V2, Vx, S2, BPfrac)
  ON2Sh = 18;  % the occupancy number of solvent (H2O) molecules in 2 solvation (hydration) shells
  a = CxM/ON2Sh/1660.6/S2; a = ( a^(-V0/Vx) )*(1-BPfrac);
  b = Vx*ON2Sh; c = 1 - V0/Vx;
  Vsh = 520*ones(1, length(C1M));
  
  x1 = Vsh; x0 = 0; IterNo = 0; IterMax = 1000;
  while max( abs(x1-x0) ) > 0.0001 && IterNo < IterMax
    x0 = x1;
    IterNo = IterNo + 1;
    f = a.*(x0.^c) - x0 + b;  % [(18), Step 4]
    df = a.*c.*(x0.^(c-1)) - 1;
    x1 = x0 - f./df;  % Newton's linear Eq for approximating f(x) = 0 
  end
  Vsh = x1;
   
  Rsh_c = ( 3*(Vsh+V1)/4/pi ).^(1/3);
  Rsh_a = ( 3*(Vsh+V2)/4/pi ).^(1/3);
end