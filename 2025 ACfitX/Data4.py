'''
Author: Jinn-Liang Liu, June 26, 2025.
For Example 4.4. Data from [P3Fig3].
'''
import warnings
import numpy as np

class DataFit():
  def __init__(self, salt):
    self.C1m_Mix = np.array([0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 2.5, 3])  # for mixing

    if salt == 'NaCl':
      self.C1m = np.array([0.125, 0.25, 0.5, 0.75, 1, 1.5, 2,	3, 4, 5, 6])
      Ts = [298.15, 373.15, 473.15, 523.15, 573.15]
      self.lngamma = np.zeros((len(Ts), len(self.C1m)), dtype='float32')
      self.lngamma[0] = np.array([-0.246, -0.324, -0.383, -0.407, -0.417, -0.414, -0.392, -0.32,  -0.230, -0.127, -0.007])
      self.lngamma[1] = np.array([-0.286, -0.371, -0.439, -0.46,  -0.473, -0.464, -0.45,  -0.386, -0.312, -0.221, -0.147])
      self.lngamma[2] = np.array([-0.403, -0.537, -0.644, -0.703, -0.731, -0.777, -0.795, -0.793, -0.787, -0.756, -0.73])
      self.lngamma[3] = np.array([-0.511, -0.684, -0.829, -0.917, -0.971, -1.043, -1.089, -1.146, -1.160, -1.174, -1.185])
      self.lngamma[4] = np.array([-0.710, -0.896, -1.101, -1.221, -1.321, -1.439, -1.536, -1.656, -1.717, -1.75,  -1.783])
    elif salt == 'MgCl2':
      self.C1m = np.array([0.125, 0.25, 0.5, 0.75, 1, 1.5, 2,	3, 4, 5, 6])
      Ts = [298.15, 373.15, 423.15, 523.15]
      self.lngamma = np.zeros((len(Ts), len(self.C1m)), dtype='float32')
      self.lngamma[0] = np.array([-0.633, -0.716, -0.716, -0.623, -0.512, -0.234, 0.116, 0.912, 1.805, 2.730, 3.651])
      self.lngamma[1] = np.array([-0.767, -0.879, -0.963, -0.935, -0.898, -0.699, -0.460, 0.135, 0.842, 1.567, 2.279])
      self.lngamma[2] = np.array([-0.907, -1.042, -1.163, -1.195, -1.167, -1.047, -0.874, -0.456, 0.144, 0.730, 1.372])
      self.lngamma[3] = np.array([-1.484, -1.684, -1.902, -2.033, -2.084, -2.158, -2.102, -1.953, -1.674, -1.321, -1.042])
    else:
      warnings.warn('Warning: No data for this salt.')
