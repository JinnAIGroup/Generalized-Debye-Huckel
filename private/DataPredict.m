  % SYC, CLL, JLL, 2022.8.29

function [mixNo, C1mX1, C1mX2, C1mX3, C1mX4, C1mX5, g_dataX1, g_dataX2, g_dataX3, g_dataX4, g_dataX5, delta_alpha] = DataPredict(salt)
  if salt == "NaF"
    mixNo = 5;
    %--- 20%
    C1mX1 = [0.01393 0.02495 0.05895 0.09415 0.1001 0.1804 0.2215 0.2409 0.2750 0.3076 0.3761];
    g_dataX1 = [0.868 0.832 0.766 0.729 0.723 0.677 0.656 0.647 0.634 0.629 0.606];
    %--- 40%
    C1mX2 = [0.008278 0.01205 0.01722 0.02393 0.03855 0.04647 0.05670 0.07006 0.08010 0.09363 0.1001 0.1054 0.1220];
    g_dataX2 = [0.861 0.838 0.810 0.780 0.740 0.724 0.708 0.691 0.679 0.664 0.659 0.655 0.642];
    %--- 60%
    C1mX3 = [0.004008 0.006469 0.008278 0.01000 0.01150 0.01207 0.01524 0.01749 0.02100 0.02190 0.02500 0.02536 0.03001 0.03268 0.03996];
    g_dataX3 = [0.874 0.840 0.822 0.808 0.797 0.789 0.774 0.762 0.745 0.737 0.730 0.732 0.714 0.709 0.690];
    %--- 80%
    C1mX4 = [0.000859 0.001690 0.002886 0.003936 0.005000 0.006500 0.007001 0.007896 0.009542 0.01057 0.01200 0.01305];
    g_dataX4 = [0.923 0.887 0.856 0.835 0.816 0.792 0.785 0.772 0.753 0.745 0.734 0.721];
    %--- 100%
    C1mX5 = [0.000286 0.000585 0.000883 0.002982 0.003411 0.003841 0.004902 0.005964 0.006321 0.006679];
    g_dataX5 = [0.965 0.875 0.862 0.677 0.649 0.625 0.598 0.570 0.547 0.542];
  
    delta_alpha = [0.06; -0.01; 0.005];

  elseif salt== "NaCl"
    mixNo = 4;
    %--- 20%
    C1mX1 = [0.001 0.002 0.005 0.007 0.01 0.015 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.4 0.5 0.7 1 1.5 2 3 4];
    g_dataX1 = [0.958 0.942 0.913 0.899 0.882 0.861 0.844 0.818 0.781 0.755 0.726 0.692 0.669 0.637 0.616 0.602 0.583 0.571 0.569 0.579 0.619 0.677];
    %--- 40%
    C1mX2 = [0.001 0.002 0.005 0.007 0.01 0.015 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.4 0.5 0.7 1 1.5 2];
    g_dataX2 = [0.949 0.930 0.896 0.880 0.861 0.836 0.817 0.788 0.748 0.720 0.690 0.656 0.632 0.600 0.580 0.566 0.549 0.537 0.534 0.543];
    %--- 60%
    C1mX3 = [0.001 0.002 0.005 0.007 0.01 0.015 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.4 0.5 0.7 1];
    g_dataX3 = [0.936 0.912 0.868 0.847 0.823 0.793 0.769 0.732 0.683 0.649 0.612 0.571 0.543 0.508 0.486 0.472 0.458 0.454];
    %--- 80%
    C1mX4 = [0.001 0.002 0.005 0.007 0.01 0.015 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.4 0.5];
    g_dataX4 = [0.914 0.882 0.824 0.798 0.767 0.728 0.699 0.654 0.594 0.554 0.513 0.468 0.439 0.405 0.387 0.378];
    %--- 100% No data.
    C1mX5 = [0, 0];
    g_dataX5 = [0, 0];
  
    delta_alpha = [0.068; -0.0017; -0.0002];

  elseif salt == "NaBr"
    mixNo = 5;
    %--- 20%
    C1mX1 = [0.00925 0.03593 0.06545 0.09519 0.1274 0.1629 0.1829 0.2297 0.3196 0.3951 0.4648 0.6007 0.8087 1.1601 1.4442 1.8381 2.4674 3.0318];
    g_dataX1 = [0.8975 0.8204 0.7858 0.7580 0.7380 0.7234 0.7185 0.7032 0.6860 0.6781 0.6683 0.6570 0.6478 0.6529 0.6612 0.6795 0.7186 0.7562];
    %--- 40%
    C1mX2 = [0.01261 0.03722 0.06562 0.08985 0.1036 0.1314 0.1572 0.1698 0.1894 0.2509 0.3759 0.8245 1.0926 1.3838 1.8882 2.3029 2.6978 2.9879];
    g_dataX2 = [0.8473 0.7791 0.7315 0.6948 0.6839 0.6679 0.6511 0.6478 0.6401 0.6211 0.5885 0.5577 0.5791 0.5810 0.6009 0.6271 0.6466 0.6848];
    %--- 60%
    C1mX3 = [0.01191 0.03642 0.06440 0.08765 0.1094 0.1306 0.1522 0.1621 0.1713 0.1993 0.2632 0.6039 1.0491 1.2994 1.6943 2.1481 2.5350 3.0191];
    g_dataX3 = [0.8299 0.7481 0.6936 0.6719 0.6565 0.6401 0.6282 0.6253 0.6200 0.6107 0.5886 0.5566 0.5377 0.5441 0.5588 0.5890 0.6280 0.6699];
    %--- 80%
    C1mX4 = [0.00806 0.03637 0.06649 0.09696 0.1107 0.1362 0.1596 0.1813 0.1980 0.4324 0.6268 1.2134 1.4151 1.6983 1.8498 2.0334 2.2919 2.5436];
    g_dataX4 = [0.8070 0.6650 0.6069 0.5727 0.5593 0.5438 0.5297 0.5180 0.5107 0.4454 0.4319 0.4567 0.4638 0.4769 0.4873 0.5031 0.5287 0.5470];
    %--- 100%
    C1mX5 = [0.01024 0.04271 0.07211 0.0826 0.1092 0.1214 0.1410 0.1500 0.1685 0.1846 0.2088 0.4246 0.5517 0.8492 1.0300 1.1315 1.3479 1.5131];
    g_dataX5 = [0.7175 0.5661 0.5105 0.4970 0.4721 0.4635 0.4511 0.4461 0.4360 0.4186 0.4079 0.3689 0.3621 0.3652 0.3701 0.3750 0.3914 0.4082];
  
    delta_alpha = [0.027; -0.004; -0.0005];
  end
end