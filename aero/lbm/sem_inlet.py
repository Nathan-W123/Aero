"""
Synthetic Eddy Method (SEM) for generating correlated turbulent inlet fluctuations.

Generates spatially correlated velocity fluctuations at an LBM inlet plane using
random eddies with tent-function kernels. A 2D Fourier divergence-free projection
on the inlet (y-z) plane ensures ∂v'/∂y + ∂w'/∂z = 0.

Reference: Jarrin et al. (2006), Intl. J. Heat Fluid Flow 27:585-593.
"""

from typing import Tuple
import numpy as np


class SEMInlet:
    """
    Synthetic Eddy Method inlet condition for 3D LBM.

    Parameters
    ----------
    Ny, Nz   : int   — inlet plane dimensions (vertical, spanwise)
    u0       : float — mean streamwise velocity
    Tu       : float — turbulence intensity (RMS / u0), e.g. 0.05 = 5%
    L_int    : float — integral length scale in lattice units
    N_eddies : int   — number of synthetic eddies (default 200)
    rng_seed : int   — random seed for reproducibility
    """

    def __init__(
        self,
        Ny: int,
        Nz: int,
        u0: float,
        Tu: float,
        L_int: float,
        N_eddies: int = 200,
        rng_seed: int = 0,
    ) -> None:
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.u0 = float(u0)
        self.Tu = float(Tu)
        self.sigma = float(L_int)
        self.N = int(N_eddies)
        self._r = 3.0 * self.sigma

        self._rng = np.random.default_rng(rng_seed)

        # Eddy positions: x in [-r, r], y in [0, Ny], z in [0, Nz]
        self.xk = self._rng.uniform(-self._r, self._r, self.N)
        self.yk = self._rng.uniform(0.0, float(self.Ny), self.N)
        self.zk = self._rng.uniform(0.0, float(self.Nz), self.N)
        # Random sign intensities: (3, N) for u, v, w components
        self.eps = self._rng.choice(
            np.array([-1.0, 1.0]), size=(3, self.N)
        ).astype(np.float64)

    def step(self, dt: float = 1.0) -> None:
        """Advance eddy positions by u0*dt; recycle eddies that exit the box."""
        self.xk += self.u0 * dt
        wrap = self.xk > self._r
        n_wrap = int(np.sum(wrap))
        if n_wrap > 0:
            self.xk[wrap] = -self._r
            self.yk[wrap] = self._rng.uniform(0.0, float(self.Ny), n_wrap)
            self.zk[wrap] = self._rng.uniform(0.0, float(self.Nz), n_wrap)
            self.eps[:, wrap] = self._rng.choice(
                np.array([-1.0, 1.0]), size=(3, n_wrap)
            ).astype(np.float64)

    def fluctuation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute inlet velocity fluctuations u', v', w' at the inlet plane.

        Returns
        -------
        u_prime, v_prime, w_prime : (Nz, Ny) float64 arrays
            Each component has zero mean and std ≈ Tu * u0.
            (v_prime, w_prime) are divergence-free: ∂v'/∂y + ∂w'/∂z ≈ 0.
        """
        zeros = np.zeros((self.Nz, self.Ny), dtype=np.float64)
        if self.Tu <= 0.0 or self.sigma <= 0.0 or self.N == 0:
            return zeros.copy(), zeros.copy(), zeros.copy()

        # Cell-centre coordinates on inlet plane
        y = np.arange(self.Ny, dtype=np.float64) + 0.5   # (Ny,)
        z = np.arange(self.Nz, dtype=np.float64) + 0.5   # (Nz,)
        ZZ, YY = np.meshgrid(z, y, indexing="ij")         # (Nz, Ny)

        # Vectorised tent kernel: (N, Nz, Ny)
        dy = YY[np.newaxis, :, :] - self.yk[:, np.newaxis, np.newaxis]
        dz = ZZ[np.newaxis, :, :] - self.zk[:, np.newaxis, np.newaxis]
        r = np.sqrt(
            self.xk[:, np.newaxis, np.newaxis] ** 2 + dy ** 2 + dz ** 2
        )
        fk = np.maximum(0.0, 1.0 - r / self.sigma)  # (N, Nz, Ny)

        inv_sqrt_N = 1.0 / np.sqrt(float(self.N))
        u_raw = inv_sqrt_N * (self.eps[0, :, np.newaxis, np.newaxis] * fk).sum(axis=0)
        v_raw = inv_sqrt_N * (self.eps[1, :, np.newaxis, np.newaxis] * fk).sum(axis=0)
        w_raw = inv_sqrt_N * (self.eps[2, :, np.newaxis, np.newaxis] * fk).sum(axis=0)

        # Divergence-free projection on (v, w) in Fourier space
        # Removes curl-free part so ∂v'/∂y + ∂w'/∂z = 0.
        # Nyquist rows/cols are zeroed first: these modes break conjugate
        # symmetry under the projection and would produce large imaginary parts.
        v_hat = np.fft.fft2(v_raw)
        w_hat = np.fft.fft2(w_raw)
        if self.Nz % 2 == 0:
            v_hat[self.Nz // 2, :] = 0.0
            w_hat[self.Nz // 2, :] = 0.0
        if self.Ny % 2 == 0:
            v_hat[:, self.Ny // 2] = 0.0
            w_hat[:, self.Ny // 2] = 0.0
        kz = (np.fft.fftfreq(self.Nz) * 2.0 * np.pi)[:, np.newaxis]  # (Nz, 1)
        ky = (np.fft.fftfreq(self.Ny) * 2.0 * np.pi)[np.newaxis, :]  # (1, Ny)
        k2 = kz ** 2 + ky ** 2
        k2[0, 0] = 1.0  # avoid division by zero at DC (mean) component
        div_hat = ky * v_hat + kz * w_hat
        v_hat -= ky / k2 * div_hat
        w_hat -= kz / k2 * div_hat
        v_df = np.real(np.fft.ifft2(v_hat))
        w_df = np.real(np.fft.ifft2(w_hat))

        # Normalise each component to target RMS = Tu * u0.
        # v and w share the same scale factor to preserve the divergence-free condition.
        target = self.Tu * self.u0

        def _scale(arr: np.ndarray, std: float) -> np.ndarray:
            return arr * (target / std) if std > 1e-14 else zeros.copy()

        u_std = float(np.std(u_raw))
        # Combined RMS of (v, w) for shared normalisation
        vw_rms = float(np.sqrt(np.mean(v_df ** 2 + w_df ** 2) / 2.0))

        return _scale(u_raw, u_std), _scale(v_df, vw_rms), _scale(w_df, vw_rms)
