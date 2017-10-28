from parameters import *
'''
Finds the magnetization from spin, by applying a mean over all the spatial
axes of the system. The last axis specifies x,y,z.
'''
def magnet(spin):
    return M0 *  np.apply_over_axes(np.mean, spin,
                                    np.arange(spin.ndim-1)).reshape(3)