'''
Returns the energy of the entire spin array
This function is not called heavily during simulation and so doesn't
get any of the numexpr treatment
'''
import numpy as np

from parameters import J,D,N
from neighborsum import neighborsum
from magnet import magnet


def energy(spin):
    # Components of the lattice energy: exchange, anisotropy, demagnetization
    en_exch = -np.sum(J * spin * neighborsum(spin), spin.ndim-1)
    en_anis = -np.sum(D * spin**2, spin.ndim-1)
    en_demag = np.sum(N * spin * magnet(spin), spin.ndim-1)

    return en_exch + en_anis + en_demag
