"""
Compute the three-component effective H-field at each lattice point
Expects lmbda as an array with same shape as spin
"""
import numpy as np
from config import *
from magnet import magnet
from neighborsum import neighborsum


def effective_field(spin):

    H_eff = (2 * J * neighborsum(spin)) + (2 * D * spin)

    if beta != 0:
        H_eff -= N * magnet(spin)

    # Landau Damping
    if np.any(lmbda != 0):
        # I don't know why this is done twice, but that's how it was written in
        # the original FORTRAN code
        for repeat in range(2):
            dSdt = np.cross(spin, H_eff)
            H_eff -= lmbda * dSdt

    dSdt = np.cross(spin, H_eff)

    return dSdt
