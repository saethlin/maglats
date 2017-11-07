import numpy as np
from scipy.misc import imsave

from tqdm import tqdm
from RK4 import RK4
from neighborsum import neighborsum


class Lattice:
    def __init__(self, *, ndim, Nz, afm, shape_parameter, interlayer_coupling, anisotropy, demag_strength,
                 noise_amplitude):

        self.uniform_mode_period = 256.0

        self.noise_amplitude = noise_amplitude
        self.demag_strength = demag_strength  # M0
        self.anisotropy = anisotropy  # D
        self.interlayer_coupling = interlayer_coupling  # J
        self.shape_parameter = shape_parameter  # beta
        self.ndim = ndim
        self.Nz = Nz

        self.spin = np.random.rand(Nz, Nz, 3)
        self.spin /= np.sqrt(np.sum(self.spin**2, axis=-1))[:, :, None]

        self.demag_diagonal = np.array([shape_parameter,
                                        shape_parameter,
                                        -2 * shape_parameter])  # N

    def step(self, time_step):
        self.spin = RK4(self.spin, self.effective_field, time_step)

    def effective_field(self, spin):
        H_eff = (2 * self.interlayer_coupling * neighborsum(spin)) + (2 * self.anisotropy * spin)

        if self.shape_parameter != 0:
            H_eff -= self.demag_diagonal * self.magnetization()

        """
        # Landau Damping
        if np.any(self.lmbda != 0):
            # I don't know why this is done twice, but that's how it was written in
            # the original FORTRAN code
            for repeat in range(2):
                dSdt = np.cross(self.spin, H_eff)
                H_eff -= self.lmbda * dSdt
        """

        return np.cross(self.spin, H_eff)

    def magnetization(self):
        return self.demag_strength * np.apply_over_axes(np.mean, self.spin,
                                                        np.arange(self.spin.ndim - 1)).reshape(3)


if __name__ == '__main__':
    lattice = Lattice(
        ndim=2,
        Nz=100,
        afm=True,
        shape_parameter=-1/3,
        interlayer_coupling=100.0,
        demag_strength=400.0,
        anisotropy=100.0,
        noise_amplitude=0.01,
    )

    rk_per_period = 256
    periods = 10
    uniform_mode_frequency = 447.2

    time_step = (2*np.pi/uniform_mode_frequency)/rk_per_period
    #for step in tqdm(range(periods * rk_per_period)):
    for step in tqdm(range(1000)):
        lattice.step(time_step)
        if step % 10 == 0:
            imsave(f"step{step}.png", lattice.spin[..., 0])
