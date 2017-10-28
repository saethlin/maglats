'''
This file handles a bunch of additional setup- it is seperated from parameters
so that parameters contains only the values that can/should be changed
'''

from parameters import *
import numpy as np
from scipy.optimize import fsolve


# Spin setup- See Lars English's PhD Thesis for source
def alphasolve(alpha, f):
    return (2+A-2/alpha+beta*(1-1/alpha))*np.sqrt(1-alpha**2*f**2)+(
        2+A-2*alpha+beta*(1-alpha))*np.sqrt(1-f**2)


def B(f, alpha, g):
    return -(1/2*(2+beta-beta/alpha)*A+2)*g**2-2*alpha*	np.sqrt(
        (1-f**2)*(1-g**2))+(2+beta)**2/(2*alpha)-beta*(
        beta+2)+1/2*alpha*beta**2


def C(f, alpha, g):
    return -(1/2*(2+beta-beta*alpha)*A+2)*f**2-2/alpha*	np.sqrt(
        (1-f**2)*(1-g**2))+alpha/2*(2+beta)**2-beta*(beta+2)+beta**2/(2*alpha)


def F(f, alpha, g):
    return beta*(2+beta)*(alpha+1/alpha-2)*(beta*(2+beta)*(
        alpha+1/alpha-2)-A*(2+beta)*(alpha**2+1/(alpha**2))*f*g-A*beta*(
        f**2+g**2)-A**2*(f*g)**2+4*(E(f,alpha,g)**2-1))


def E(f, g):
    return g*f-np.sqrt((1-f**2)*(1-g**2))


def lmbdasolve(f):
    alpha = fsolve(alphasolve,0.4,args=f)[0]
    g = -f/alpha
    return (alpha*C(f,alpha,g)+B(f,alpha,g)/alpha)-np.sqrt(
        (alpha*C(f,alpha,g)+B(f,alpha,g)/alpha)**2-F(f,alpha,g))

# Always use the theoretical f_crit if in beta < 0
if afm:
    if ndim == 1:
        f = 0.152
        alpha = fsolve(alphasolve,0.4,args=f)[0]
    elif ndim == 2:
        f = 0.114
        alpha = fsolve(alphasolve,0.4,args=f)[0]
else:
    f = 0.812

if boundary == 1:
    if ndim == 1:
        edgemask = np.ones(Nz,bool)
        edgemask[10:-10] = False
    else:
        edgemask = np.ones((Nz,)*ndim,bool)
        edgemask[10:-10,10:-10] = False

if skyrmion:
    f = 0
