"""
Spectral post-processing for LBM velocity fields.

Functions
---------
energy_spectrum_3d    — spherically-averaged 3D kinetic energy spectrum E(k)
two_point_correlation_z — spanwise two-point velocity correlation R(r)
integral_length_scale — Lz from R(r) via trapezoidal integration
"""

from typing import Tuple
import numpy as np


def energy_spectrum_3d(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spherically-averaged 3D turbulent kinetic energy spectrum.

    Parameters
    ----------
    ux, uy, uz : (Nz, Ny, Nx) float arrays — velocity components

    Returns
    -------
    k_bins : (M,) float — wavenumber bin centres (in units of 2π/L where L=1)
    E_k    : (M,) float — E(k), energy per unit wavenumber bin
    """
    Nz, Ny, Nx = ux.shape

    # 3D FFT of each component; normalise by total cell count
    N_total = float(Nx * Ny * Nz)
    ux_hat = np.fft.fftn(ux) / N_total
    uy_hat = np.fft.fftn(uy) / N_total
    uz_hat = np.fft.fftn(uz) / N_total

    # Energy density in k-space: 0.5 * |û|² (factor of 2 cancelled by real pair)
    energy_k = 0.5 * (
        np.abs(ux_hat) ** 2
        + np.abs(uy_hat) ** 2
        + np.abs(uz_hat) ** 2
    )

    # Wavenumber magnitudes for each mode
    kx = np.fft.fftfreq(Nx, d=1.0 / Nx)
    ky = np.fft.fftfreq(Ny, d=1.0 / Ny)
    kz = np.fft.fftfreq(Nz, d=1.0 / Nz)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    k_mag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

    # Bin by integer wavenumber shell
    k_max = int(np.floor(min(Nx, Ny, Nz) / 2))
    k_bins = np.arange(0, k_max + 1, dtype=np.float64)
    E_k = np.zeros(k_max + 1, dtype=np.float64)

    k_int = np.round(k_mag).astype(np.int32)
    mask = k_int <= k_max
    np.add.at(E_k, k_int[mask], energy_k[mask])

    return k_bins, E_k


def two_point_correlation_z(
    u: np.ndarray,
    iz_ref: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spanwise two-point velocity correlation R(r) averaged over x-y planes.

    Parameters
    ----------
    u      : (Nz, Ny, Nx) velocity component (e.g. ux)
    iz_ref : reference z-index (default 0)

    Returns
    -------
    r   : (Nz,) float — spanwise separations in lattice units
    R_z : (Nz,) float — R(r) normalised by R(0) = 1
    """
    Nz, Ny, Nx = u.shape
    u_prime = u - u.mean()

    ref = u_prime[iz_ref]  # (Ny, Nx)
    R_z = np.zeros(Nz, dtype=np.float64)
    for dz in range(Nz):
        iz = (iz_ref + dz) % Nz
        R_z[dz] = float(np.mean(ref * u_prime[iz]))

    # Normalise so R(0) = 1
    if R_z[0] > 0.0:
        R_z /= R_z[0]

    r = np.arange(Nz, dtype=np.float64)
    return r, R_z


def integral_length_scale(r: np.ndarray, R_z: np.ndarray) -> float:
    """
    Integral length scale Lz = integral_0^r_zero R(r) dr.

    Integration stops at first zero crossing of R(r) to avoid aliasing.
    """
    zero_crossings = np.where(R_z[1:] <= 0.0)[0]
    r_zero = int(zero_crossings[0] + 1) if len(zero_crossings) else len(r)
    return float(np.trapezoid(R_z[:r_zero], r[:r_zero]) if hasattr(np, "trapezoid") else np.trapz(R_z[:r_zero], r[:r_zero]))
