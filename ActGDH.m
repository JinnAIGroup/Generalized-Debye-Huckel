% Chin-Lung Li, Shu-Yi Chou, Jinn-Liang Liu 
% Prediction of activity coefficients in water-methanol mixtures using
% a generalized Debye-Huckel model, ArXiv
% September 5, 2022
clc
clear

%--- Physical Constants [Table 1 in the paper]
T = 298.15; epsln = 8.854187; e = 1.6022; mol = 6.022045; kB = 1.380649; kBTe = (kB*T/e)*0.0001; % T in K = 25 C
V0 = 1; S2 = 0.1*mol*e/(kBTe*epsln);  % V0: unit volume, S2: scaling factor [(vi)]

ScSz = get(0, 'ScreenSize');
FigPos_gamma = [0.15*ScSz(3) 0.05*ScSz(4) 0.7*ScSz(3) 0.4*ScSz(4)];  % Figure Position for gamma
FigPos_theta = [0.15*ScSz(3) 0.45*ScSz(4) 0.7*ScSz(3) 0.4*ScSz(4)];  % Figure Position for theta

for salts = 1:3
  %--- Solvent Parameters: diS: dielectric constant [(11)], Vx: volume [(18)],
  %    CxM (scalar): bulk concentration in M [(18)], 1: cation, 2: anion, 3: H2O, 4: MeOH [Step 1]
  [C3M, C4M, CxM, V3, V4, Vx, diS] = Solvent(0); % x = 0: solvent = pure H2O
  C3M = C3M*S2; C4M = C4M*S2; CxM = CxM*S2;  % scaled by S2 to unitless

  if salts == 1
      salt = "NaF";  FigPos_SubP = [0.09 0.15 0.26 0.8];  % Figure Position of subplot
  elseif salts == 2
      salt = "NaCl"; FigPos_SubP = [0.4 0.15 0.26 0.8];
  elseif salts == 3
      salt = "NaBr"; FigPos_SubP = [0.71 0.15 0.26 0.8];
  end
  
  %--- Born Radii: BornR0 in pure solvent (no salt) [(12), (13), Step 2]
  [BornR0, q1, q2, V1, V2, mM, D] = Born(salt, diS, 0);

  %--- Activity Data to Fit: C1m (vector): concentration in molality (m), gamma: mean activity [(16)]
  [C1m, gamma] = DataFit(salt); g_data = log(gamma);

  %--- Salt molality (m) to Molarity (M): C1m to C1M [Step 3]
  C1M = m2M(C1m, mM, D, 0).*S2;
  
  C2M = q1*C1M;
  BPfrac = (V1*C1M + V2*C2M + V3*C3M + V4*C4M)/S2/1660.6;  % Bulk Particle fraction [(2)]

  %--- Newton() iteratively solves nonlinear [(18)] for V_sh that yields Rsh_c and Rsh_a [Step 4]. 
  [Rsh_c, Rsh_a] = Newton(C1M, CxM, V0, V1, V2, Vx, S2, BPfrac);
  
  %--- LSfit() returns g_fit as the best fit to g_data with alpha(1), (2), (3) [(14)] by Least Squares [Step 5].
  [g_fit, alpha, theta] = LSfit(g_data, BornR0, Rsh_c, Rsh_a, salt, C1M, C3M, C4M, q1, q2, V0, V1, V2, V3, V4, diS, T);

  %format long  % for all decimals in alpha
  alpha
  
  plot_theta = (1 + (C1M.^0.5)*alpha(1) + C1M*alpha(2) + (C1M.^1.5)*alpha(3));  % for plotting theta [(14)]
  
  figure(1)  % plot data points and fitted curves
  set(gcf, 'Position', FigPos_gamma)
  subplot('Position', FigPos_SubP)
  plot(C1m, g_data, 'k.');  % data points
  hold on
  plot(C1m, g_fit, '-r');  % fitted curve
  text((C1m(end)), g_data(end)-0.02, 'x=0');
  ylabel('ln\gamma_\pm', 'FontSize', 12);
  xlabel('m (mol/kg)', 'FontSize', 12); title(salt);
   
  figure(2)  % plot theta
  set(gcf, 'Position', FigPos_theta)
  subplot('Position', FigPos_SubP)
  plot(C1m, plot_theta, '-r', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'w') % theta
  ylabel('\theta', 'FontSize', 12);
  xlabel('m (mol/kg)', 'FontSize', 12); title(salt);
  
  %--- Experimental data of mixed-solvent solution to predict
  [mixNo, C1mX1, C1mX2, C1mX3, C1mX4, C1mX5, g_dataX1, g_dataX2, g_dataX3, ...
   g_dataX4, g_dataX5, delta_alpha] = DataPredict(salt);

  for i = 1:mixNo
    switch i
      case 1
        x = 0.2; C1m_x = C1mX1; g_data_x = log(g_dataX1);
      case 2
        x = 0.4; C1m_x = C1mX2; g_data_x = log(g_dataX2);
      case 3
        x = 0.6; C1m_x = C1mX3; g_data_x = log(g_dataX3);
      case 4
        x = 0.8; C1m_x = C1mX4; g_data_x = log(g_dataX4);
      case 5
        x = 1.0; C1m_x = C1mX5; g_data_x = log(g_dataX5);
    end
    
    [C3M, C4M, CxM, V3, V4, Vx, diS] = Solvent(x);  % [Step 6]
    C3M = C3M*S2; C4M = C4M*S2; CxM = CxM*S2; 

    [BornR0, q1, q2, V1, V2, mM, D] = Born(salt, diS, x);  % [Step 7]

    C1M = m2M(C1m_x, mM, D, x).*S2;  % [Step 7]
    
    C2M = q1*C1M;
    BPfrac = (V1*C1M + V2*C2M + V3*C3M + V4*C4M)/S2/1660.6;
 
    [Rsh_c, Rsh_a] = Newton(C1M, CxM, V0, V1, V2, Vx, S2, BPfrac);  % [Step 7]

    alpha_x = alpha + x*delta_alpha;
    theta = 1 + alpha_x(1)*power(C1M, 1/2) + alpha_x(2)*C1M + alpha_x(3)*power(C1M, 3/2);

    %--- Given theta, Activity() returns a mean activity [Step 8]. 
    g_pred_x = Activity(theta, BornR0, Rsh_c, Rsh_a, C1M, C3M, C4M, q1, q2, V0, V1, V2, V3, V4, diS, T);

    plot_theta_x = round( 1 + power(C1M,1/2)*alpha_x(1) + C1M*alpha_x(2) + power(C1M,3/2)*alpha_x(3), 5 );
    
    figure(1)  % Plot data points and predicted curves
    hold on
    plot(C1m_x, g_data_x, 'k.');  % data points
    plot(C1m_x, g_pred_x, 'b');  % PF curve
    text((C1m_x(end)), g_data_x(end)-0.02, num2str(x)); % notation of solvent
    
    figure(2)  % Plot theta
    hold on
    if x == 0.2
      plot(C1m_x,plot_theta_x,'--g','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','w');
    elseif x == 0.4
      plot(C1m_x,plot_theta_x,':b','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','w');
    elseif x == 0.6
      plot(C1m_x,plot_theta_x,'-.r','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','w');
    elseif x == 0.8
      plot(C1m_x,plot_theta_x,'-+g','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','w');
    elseif x == 1
      plot(C1m_x,plot_theta_x,'xb','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','w');
    end
  end
end


function [C3M, C4M, CxM, V3, V4, Vx, diS] = Solvent(x)
  V3 = 4*pi*1.4^3/3;    % H2O [(10)]
  V4 = 4*pi*1.915^3/3;  % MeOH
  Vx  = (1-x)*V3 + x*V4;  % [(18)]
  C3M = (1-x)*55.5;
  C4M = x*24.55;
  CxM = (1-x)*55.5  + x*24.55;  % [(18)]
  diS = (1-x)*78.45 + x*32.66;  % [(11)]
end


function [BornR0, q1, q2, V1, V2, mM, D] = Born(salt, diS, x)
  epsln = 8.854187; e = 1.6022; mol = 6.022045;  % [Table 1]
  if salt == "NaCl"   % Free energy: Na+ in H2O = -103.2 MeOH = -91.9; Cl- in H2O = -74.5 MeOH = -81.1
    FcatH2O = -103.2; FcatMeOH = -91.9; FanH2O = -74.5; FanMeOH = -81.1;  % mM: molar mass
    q1 = 1; q2 = -1; V1 = 4*pi*0.95^3/3; V2 = 4*pi*1.81^3/3; mM = 58.44; D = 46.62;  % D: density gradient of solution 
  elseif  salt == "NaBr"
    FcatH2O = -103.2; FcatMeOH = -91.9; FanH2O = -68.3; FanMeOH = -75.1;
    q1 = 1; q2 = -1; V1 = 4*pi*0.95^3/3; V2 = 4*pi*1.95^3/3; mM = 102.89; D = 77;
  elseif  salt == "NaF"
    FcatH2O = -103.2; FcatMeOH = -91.9; FanH2O = -104.4; FanMeOH = -109.2;
    q1 = 1; q2 = -1; V1 = 4*pi*0.95^3/3; V2 = 4*pi*1.36^3/3; mM = 41.99; D = 41.38;
  else
    FcatH2O = -103.2; FcatMeOH = -91.9; FanH2O = -74.5; FanMeOH = -81.1;
    q1 = 1; q2 = -1; V1 = 4*pi*r1^3/3; V2 = 4*pi*r2^3/3; mM = 58.44; D = 46.62;
    warning('Warning: Salt not found.')
  end
  Fcat = (1-x)*FcatH2O + x*FcatMeOH;
  Fan  = (1-x)*FanH2O  + x*FanMeOH;
  BornR0c = -q1*q1*e*e*mol*10000*(1-1/diS)/(Fcat*4.1868*8*pi*epsln);  % Born radius of cation [(12)]
  BornR0a = -q2*q2*e*e*mol*10000*(1-1/diS)/(Fan*4.1868*8*pi*epsln);   % Born radius of anion
  BornR0 = [BornR0c BornR0a];
end


function M = m2M(m, mM, D, x)
  % 0.9971 = density of water
  % 0.9128 = density of 50-50 water-methanol mixture
  % 0.7866 = density of methanol
  rho_0 = (x-0.5)*(x-1)/0.5*0.9971 - x*(x-1)/0.25*0.9128 + x*(x-0.5)/0.5*0.7866;  % Lagrange polynomial [Step 3] 
  rho_s = rho_0 + D*m/1000;  % solution density = solvent density + salt density
  M = 1000*m.*rho_s./(1000 + m.*mM);  % molality to Molarity
end
