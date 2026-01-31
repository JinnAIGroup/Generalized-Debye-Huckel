'''
Author: Jinn-Liang Liu, Nov 8, 2025.
For Fig4
'''
import warnings
import numpy as np

class DataFit():
  def __init__(self, salt):
    if salt == 'LiCl':
      self.C1m = np.array([0.1, 0.2, 0.5, 0.7, 1, 1.6, 2,	3, 4, 5, 6])
      self.lngamma = np.array([-0.237, -0.280, -0.302, -0.290, -0.255, -0.157, -0.079, 0.146, 0.405, 0.693, 1.004])
    elif salt == 'MgCl2':
      self.C1m = np.array([0.1, 0.2, 0.5, 0.7, 1, 1.6, 2,	3, 4, 5, 6])
      self.lngamma = np.array([-0.633,	-0.716,	-0.716,	-0.623,	-0.512,	-0.234,	0.116, 0.912, 1.805, 2.730, 3.651])
    elif salt == 'LaCl3':
      self.C1m = np.array([0.1, 0.3, 0.5, 0.8, 1, 1.6, 2,	3, 3.9])
      self.lngamma = np.array([-1.121, -1.309, -1.309, -1.121, -0.983, -0.528, -0.147, 0.828, 1.565])
    else:
      warnings.warn('Warning: No data for this salt.')
