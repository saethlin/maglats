import numba

@numba.jit(nopython=True)
def RK4(y, update, h):
    k1 = update(y)
    k2 = update(y + h / 2 * k1)
    k3 = update(y + h / 2 * k2)
    k4 = update(y + h * k3)

    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
