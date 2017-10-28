'''
Computes the sum of the four nearest neighbors in a non-triangular lattice,
and the sum of the six nearest if triangular == True
'''
#pythran export neighborsum(float[])
from parameters import Nz
import numpy as np

def neighborsum(arr):
    result = np.zeros(arr.shape)
    left = np.arange(arr.shape[0]) - 1
    right = np.arange(arr.shape[0]) + 1

    for d in range(arr.ndim-1):
        result += arr.take(left, axis=d, mode='wrap')
        result += arr.take(right, axis=d, mode='wrap')

    return result
