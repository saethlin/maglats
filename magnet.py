from parameters import *
import numba
'''
Finds the magnetization from spin, by applying a mean over all the spatial
axes of the system. The last axis specifies x,y,z.
'''
@numba.jit(nopython=True)
def magnet(spin):
    '''
    if spin.ndim == 2:
        x = np.mean(spin[:, 0])
        y = np.mean(spin[:, 1])
        z = np.mean(spin[:, 2])
        return M0 * np.array((x, y, z))
    elif spin.ndim == 3:
        x = np.mean(spin[:, :, 0])
        y = np.mean(spin[:, :, 1])
        z = np.mean(spin[:, :, 2])
        return M0 * np.array((x, y, z))
    '''
    x = np.mean(spin[:, :, 0])
    y = np.mean(spin[:, :, 1])
    z = np.mean(spin[:, :, 2])
    return M0 * np.array((x, y, z))

