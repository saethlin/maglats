"""
Computes the sum of the four nearest neighbors in a non-triangular lattice,
and the sum of the six nearest if triangular == True
"""
import numpy as np
import numba

@numba.jit(nopython=True)
def neighborsum(arr):
    result = np.zeros(arr.shape)

    # neighbors above
    result[1:] += arr[:-1]
    result[0] += arr[-1]

    # below
    result[:-1] += arr[1:]
    result[-1] += arr[0]

    # left
    result[:, 1:] += arr[:, :-1]
    result[:, 0] += arr[:, -1]

    # right
    result[:, :-1] += arr[:, 1:]
    result[:, -1] += arr[0]

    return result
