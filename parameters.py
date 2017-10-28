from __future__ import division
import numpy as np

#Number of time steps to skip when saving frames for analysis
skip = 1

'''
Lattice properties
'''
#Antiferromagnetic?
afm = False
#Lattice shape parameter
beta = -1/3
#Simulation Length
periods = 500
#Skyrmion?
skyrmion = False

#Amplitude of the noise field
ininl = 0.01

if skyrmion:
    afm = False
    beta = 0
    periods = 100
    ininl = 0

#Number of dimensions for the spin lattice
ndim = 2
#Number of elements along each axis, lattice is always square
Nz = 100
#Boundary modes: 0-periodic 1-closed 2-open	
boundary = 0
#Triangular lattice?
triangular = False

#Interlayer coupling constant- strength of exchange interaction
J = 100.
if afm:
    J *= -1

#Anisotropy vector, we prefer to keep D[0] = D[1] = 0
D = np.array([0, 0, 100.])

#Only simulations I ran were with A = 1,-1
A = -D[2]/J

#Determines strength of the demagnetizing field
M0 = 400

#This should be the frequency of the uniform mode
wafmr = 447.2

'''
RK4 parameters
'''
#Steps in a period of the uniform mode
rkstep = 256

#Damping coefficient with possible structure
dst = np.hypot(*np.mgrid[:Nz,:Nz])
lmbda = np.zeros((Nz,Nz,3))
#lmbda[dst > 35] = 0.1
#lmbda += 0.05

#Applied Field
H_applied = np.array([0,0,0])

#Demagnetization Diagonal Matrix
N = np.array([beta,
              beta,
              -2*beta])
