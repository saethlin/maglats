"""
Compute the three-component effective H-field at each lattice point
Expects lmbda as an array with same shape as spin
"""
import numpy as np
from parameters import *
from magnet import magnet
from neighborsum import neighborsum
import numba

@numba.jit(nopython=True)
def cross(a, b):
    output = np.empty_like(a)
    output[:, :, 0] = a[:, :, 1] * b[:, :, 2] - a[:, :, 2] * b[:, :, 1]
    output[:, :, 1] = a[:, :, 2] * b[:, :, 0] - a[:, :, 0] * b[:, :, 2]
    output[:, :, 2] = a[:, :, 0] * b[:, :, 1] - a[:, :, 1] * b[:, :, 0]
    return output

@numba.jit(nopython=True)
def effective_field(spin):

    H_eff = (2 * J * neighborsum(spin)) + (2 * D * spin)

    offset = np.array((0.0, 0.0, 0.0))

    if beta != 0.0:
        H_eff -= N * magnet(spin)

    # Landau Damping
    if np.any(lmbda != 0):
        # I don't know why this is done twice, but that's how it was written in
        # the original FORTRAN code
        for repeat in range(2):
            dSdt = cross(spin, H_eff)
            H_eff -= lmbda * dSdt

    dSdt = cross(spin, H_eff)

    return dSdt
