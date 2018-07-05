"""
Magnetic lattice simulation by Ben Kimock 2015 based work by
Bruce Edward Hubbard, Masayuki Sato, Lars English 02 July 2001
"""

import numpy as np
import os
import tqdm

from parameters import *
from ampsolve import *
from magnet import *
from energy import *
from effective_field import *
from RK4 import *
from neighborsum import *
from skyrmion import *
from matplotlib import pyplot as plt

# Determine name
if afm:
    name = '%.2f' % beta + '_afm'
else:
    name = '%.2f' % beta + '_fer'

if skyrmion:
    name = '_skyrm'


# Open file for saving spin data
if os.path.isfile(name + '.npy'):
    os.remove(name + '.npy')
history = open(name + '.npy', 'ab')

# Allocate the spin array
spin = np.zeros((Nz,) * ndim + (3,))

# Construct the ispin array, which designates the sublattices
ispin = (-1)**(np.arange(Nz))
for dim in range(1, ndim):
    ispin = np.concatenate(([ispin], [-ispin]), 0)
    for n in range(Nz - 2):
        ispin = np.concatenate((ispin, [-ispin[-1]]), 0)

if afm:
    # Determine the ground and max energy
    spin[..., 2] = ispin
    GSE = np.mean(energy(spin))

    spin[..., 2] = 1
    en_max = np.mean(energy(spin)) - GSE
else:
    # Determine the ground state energy
    spin[..., 2] = 1
    GSE = np.mean(energy(spin))

    spin[..., 2] = 0
    spin[..., 0] = 1
    GSE = min(GSE, np.mean(energy(spin)))

    spin[..., 2] = ispin
    en_max = np.mean(energy(spin)) - GSE
    spin[..., 2] = 0
    spin[..., 0] = ispin
    en_max = max(np.mean(energy(spin) - GSE), en_max)


# Place spins in the initial excited configuration
if boundary == 0:
    spin[..., 0] = f
else:
    if ndim == 1:
        spin[..., 0] = f * np.sin(np.pi * (np.arange(Nz)) / (Nz - 1))
    if ndim == 2:
        spin[..., 0] = f * \
            np.prod(np.sin(np.pi * (np.mgrid[:Nz, :Nz]) / (Nz - 1)), 0)
if afm:
    spin[..., 0][ispin == -1] /= -alpha

# Add noise to x component and compute the appropriate z component
if not skyrmion:
    spin[..., 0] += ininl * (np.random.rand(*(Nz,) * ndim) - 0.5) * 2
spin[..., 2] = np.sqrt(1 - spin[..., 0]**2)

# If antiferromagnetic, flip the z-components of each spin
if afm:
    spin[..., 2] *= ispin


# Make a skyrmion
#spin = put_skyrmion(spin,10,50,30,True)
#spin = put_skyrmion(spin,10,50,50,True)
#spin = put_skyrmion(spin,10,50,70,True)


# Save some parameters to the file first
np.save(history, np.array([name, afm, beta, GSE,
                           en_max, 1 + periods * (rkstep // skip + 1)]))
np.save(history, spin)

# Start the simulation
time_step = (2 * np.pi / wafmr) / rkstep
for period in tqdm.tqdm(range(periods)):
    for irkstep in range(rkstep):
        spin = RK4(spin, effective_field, time_step)
        # Only save every skip rk steps
        if irkstep % skip == 0:
            out = spin[..., 0].copy()
            np.save(history, spin)

#plt.imshow(spin[..., 0])
# plt.show()
