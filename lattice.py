"""
Negative interlayer coupling constant makes the simulation antiferromagnetic
"""

import numba
import numpy as np
from scipy.misc import imsave
from tqdm import tqdm
import cv2

from RK4 import RK4
from neighborsum import neighborsum

spec = [
    ('demag_strength', numba.float64),
    ('anisotropy', numba.float64),
    ('interlayer_coupling', numba.float64),
    ('shape_parameter', numba.float64),
    ('demag_diagonal', numba.float64[:]),
    ('spin', numba.float64[:, :, :]),
]


@numba.jitclass(spec)
class Lattice:
    def __init__(self, ndim, Nz, shape_parameter, interlayer_coupling, anisotropy, demag_strength,
                 noise_amplitude):

        self.demag_strength = demag_strength  # M0
        self.anisotropy = anisotropy  # D
        self.interlayer_coupling = interlayer_coupling  # J
        self.shape_parameter = shape_parameter  # beta

        self.spin = np.zeros((Nz, Nz, 3))
        # Uniform mode amplitude
        self.spin[..., 0] = 0.812

        # Add noise
        self.spin[..., 0] += noise_amplitude * (np.random.rand(Nz, Nz) - 0.5) * 2

        # normalize the spins by the z component
        self.spin[..., 2] = np.sqrt(1 - self.spin[..., 0] ** 2)

        self.demag_diagonal = np.array([shape_parameter,
                                        shape_parameter,
                                        -2 * shape_parameter])  # N

    def step(self, time_step):
        self.spin = RK4(self.spin, self.effective_field, time_step)
        # Just in case???
        # self.spin /= np.sqrt(np.sum(self.spin**2, axis=-1))[:, :, None]

    def effective_field(self, spin):
        H_eff = (2 * self.interlayer_coupling * neighborsum(spin)) + (2 * self.anisotropy * spin)

        if self.shape_parameter != 0:
            H_eff -= self.demag_diagonal * self.magnetization()

        output = np.empty_like(self.spin)
        output[..., 0] = (self.spin[..., 2] * H_eff[..., 3]) - (self.spin[..., 3] * H_eff[..., 2])
        output[..., 1] = (self.spin[..., 3] * H_eff[..., 1]) - (self.spin[..., 1] * H_eff[..., 3])
        output[..., 2] = (self.spin[..., 1] * H_eff[..., 2]) - (self.spin[..., 2] * H_eff[..., 1])

        return output
        # return np.cross(self.spin, H_eff)

    def magnetization(self):
        mean = np.array([0., 0., 0.])
        for r in range(self.spin.shape[0]):
            for c in range(self.spin.shape[1]):
                mean += self.spin[r, c]
        return mean
        # return self.demag_strength * np.mean(self.spin, axis=(0, 1))

    def energy(self):
        # Components of the lattice energy: exchange, anisotropy, demagnetization
        en_exch = -np.sum(self.interlayer_coupling * self.spin * neighborsum(self.spin), axis=-1)
        en_anis = -np.sum(self.anisotropy * self.spin ** 2, axis=-1)
        en_demag = np.sum(self.demag_diagonal * self.spin * self.magnetization(), axis=-1)

        return en_exch + en_anis + en_demag


if __name__ == '__main__':
    lattice = Lattice(
        ndim=2,
        Nz=100,
        shape_parameter=-1 / 3,
        interlayer_coupling=-100.0,
        demag_strength=400.0,
        anisotropy=100.0,
        noise_amplitude=0.01,
    )

    rk_per_period = 256
    periods = 1
    uniform_mode_frequency = 447.2

    time_step = (2 * np.pi / uniform_mode_frequency) / rk_per_period

    frame = 0
    for step in tqdm(range(periods * rk_per_period)):
        lattice.step(time_step)
        if step % 20 == 0:
            imsave(f'step{frame:0>6}.png', lattice.energy())
            frame += 1
