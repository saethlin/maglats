'''
The first function here is what I used for the ansatz in the 2015 thesis.
The second function is probably a much better guess, but it is poorly tested
I STRONGLY encourage future users to try and improve on this
'''

import numpy as np
from parameters import *


# Make an antiskyrmion by inverting the y or x
def make_skyrmion(size):
    # arr is the distance to the center of a region of radius size
    arr = np.sqrt(np.sum((np.mgrid[:Nz, :Nz] - ((Nz - 1) / 2)) ** 2, 0))
    # mask is the not-skyrmion region
    mask = arr > size
    # Make everything not skyrmion a simple domain
    arr[mask] = 1
    dst = arr.copy()
    # Normalize the array to a value of pi so that we can evalute sin/cos
    # on the array for x and y
    arr /= np.max(arr)
    arr *= np.pi

    # Compute z value- based on -cosine so that it is -1 at the middle
    z = -np.cos(arr)
    z[mask] = 1

    # New position array
    arr = (np.mgrid[:Nz, :Nz] - ((Nz - 1) / 2)).astype(float)
    arr[:, mask] = 0
    # Normalize again
    arr /= np.max(arr)
    arr *= np.pi

    # Compute sine of the array, but this time it has two components, x and y
    # So evaluate sine over it produces sine of x and sine of y

    xy = np.sin(arr)

    y = -xy[0] / dst
    x = xy[1] / dst

    # Stack together the spin components
    spin = np.dstack((x, y, z))

    # Normalize each of the components so that |s|=1
    nfactor = np.sqrt(np.sum(spin ** 2, 2))

    spin[..., 0] /= nfactor
    spin[..., 1] /= nfactor
    spin[..., 2] /= nfactor

    return spin


def put_skyrmion(spin, size, y, x, regular):
    # A new array to hold the skyrmion, only as large as needed
    new = np.zeros((2 * size + 1, 2 * size + 1, 3))

    # x and y positions along the skyrmion array from the center
    ypos, xpos = np.mgrid[-size:size + 1, -size:size + 1].astype(float)

    # The x and y coordinates are the vector components (roughly) for a hedgehog
    new[..., 0] = xpos
    new[..., 1] = ypos

    # Compute radial distances at each point and make a circular mask
    rad = np.hypot(ypos, xpos)
    mask = rad <= size

    # Normalize the x and y components to be constant with distance
    rad[rad == 0] = 1
    new[..., 0] /= rad
    new[..., 1] /= rad

    # Compute the z component: a cosine shape that is -1 in the center
    new[..., 2] = -np.cos(rad / size * np.pi)

    # Force the inverse of the cosine shape from z onto the x and y components
    new[..., 0] *= (1 - abs(new[..., 2])) ** 3
    new[..., 1] *= (1 - abs(new[..., 2])) ** 3

    # Normalize all the spins to length 1
    length = np.sqrt(np.sum(new ** 2, 2))
    new[..., 0] /= length
    new[..., 1] /= length
    new[..., 2] /= length

    # Flip the x components if antiskyrmion is specified
    if not regular:
        new[..., 0] *= -1

    # Insert the skyrmion at the given position
    spin[y - size:y + size + 1, x - size:x + size + 1][mask] = new[mask]
    return spin
