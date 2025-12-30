'''
Author: Jinn-Liang Liu, Nov 9, 2025.
For Fig6, Fig7
'''
import warnings
import numpy as np

class DataFit():
  def __init__(self, salt):
    if salt == 'NaCl':
      self.C1m = np.array([0.125, 0.25, 0.5, 0.75, 1, 1.5, 2,	3, 4, 5, 6])
      Ts = [298.15, 373.15, 473.15, 523.15, 573.15]
      self.lngamma = np.zeros((len(Ts), len(self.C1m)), dtype='float32')
      self.lngamma[0] = np.array([-0.246, -0.324, -0.383, -0.407, -0.417, -0.414, -0.392, -0.32,  -0.230, -0.127, -0.007])
      self.lngamma[1] = np.array([-0.286, -0.371, -0.439, -0.46,  -0.473, -0.464, -0.45,  -0.386, -0.312, -0.221, -0.147])
      self.lngamma[2] = np.array([-0.403, -0.537, -0.644, -0.703, -0.731, -0.777, -0.795, -0.793, -0.787, -0.756, -0.73])
      self.lngamma[3] = np.array([-0.511, -0.684, -0.829, -0.917, -0.971, -1.043, -1.089, -1.146, -1.160, -1.174, -1.185])
      self.lngamma[4] = np.array([-0.710, -0.896, -1.101, -1.221, -1.321, -1.439, -1.536, -1.656, -1.717, -1.75,  -1.783])
    elif salt == 'LaCl3':
      self.C1m = np.array([0.1, 0.3, 0.5, 0.8, 1, 1.6, 2,	3, 3.9])
      Ts = [298.15]
      self.lngamma = np.zeros((len(Ts), len(self.C1m)), dtype='float32')
      self.lngamma[0] = np.array([-1.121, -1.309, -1.309, -1.121, -0.983, -0.528, -0.147, 0.828, 1.565])
    elif salt == 'MgCl2':
      self.C1m = np.array([0.1, 0.2, 0.5, 0.7, 1, 1.6, 2,	3, 4])
      Ts = [298.15]
      self.lngamma = np.zeros((len(Ts), len(self.C1m)), dtype='float32')
      self.lngamma[0] = np.array([-0.633,	-0.716,	-0.716,	-0.623,	-0.512,	-0.234,	0.116, 0.912, 1.805])
    else:
      warnings.warn('Warning: No data for this salt.')
