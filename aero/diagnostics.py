"""
Runtime stability checks, convergence detection, and flow diagnostics.
"""

import warnings
from typing import List, Optional
import numpy as np


def check_stability(
    f: np.ndarray,
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    step: int,
) -> None:
    """
    Raise RuntimeError on divergence; warn on concerning but non-fatal conditions.
    Called periodically during the solver loop.
    """
    if np.any(np.isnan(f)):
        raise RuntimeError(
            f"[step {step}] NaN detected in f — simulation has diverged. "
            "Check tau (must be > 0.5), inlet velocity, and grid resolution."
        )

    if np.any(rho <= 0.0):
        raise RuntimeError(
            f"[step {step}] Non-positive density detected — simulation has diverged."
        )

    umag_max = float(np.sqrt(ux**2 + uy**2).max())
    if umag_max > 0.3:
        warnings.warn(
            f"[step {step}] Max velocity = {umag_max:.4f} lattice units "
            f"(Ma ≈ {umag_max * (3**0.5):.3f}). "
            "Compressibility artifacts likely. Reduce u0 or increase grid size.",
            RuntimeWarning,
            stacklevel=2,
        )

    rho_mean = float(rho.mean())
    if abs(rho_mean - 1.0) > 0.05:
        warnings.warn(
            f"[step {step}] Mean density drifted to {rho_mean:.4f} (>5% from 1.0). "
            "Check outlet BC or total mass conservation.",
            RuntimeWarning,
            stacklevel=2,
        )


def check_convergence(
    Cd_history: List[float],
    window: int = 5000,
    tol: float = 0.01,
) -> bool:
    """
    Return True when Cd has converged: std/mean < tol over the last `window` steps.

    Used to allow early termination when steady state is reached.
    """
    if len(Cd_history) < window:
        return False
    recent = np.array(Cd_history[-window:])
    mean = float(np.mean(np.abs(recent)))
    if mean < 1e-12:
        return True
    return float(np.std(recent)) / mean < tol


def compute_strouhal(
    Cl_history: List[float],
    D: float,
    u0: float,
) -> Optional[float]:
    """
    Estimate the vortex-shedding Strouhal number from the Cl time series.

    Uses the FFT peak frequency of the (mean-subtracted) Cl signal:

        St = f_peak * D / u0

    where f_peak is in units of (1/timestep).

    Returns None if the signal is too short (< 512 samples) or if the
    peak amplitude is negligible (no clear shedding detected).

    Parameters
    ----------
    Cl_history : list of instantaneous Cl values recorded every timestep
    D          : characteristic length in lattice cells
    u0         : inlet velocity in lattice units

    Returns
    -------
    St : float or None
    """
    if len(Cl_history) < 512 or u0 <= 0.0 or D <= 0.0:
        return None

    cl = np.array(Cl_history, dtype=np.float64)
    cl -= cl.mean()

    if np.max(np.abs(cl)) < 1e-6:
        return None   # no oscillation

    N = len(cl)
    fft_mag = np.abs(np.fft.rfft(cl))
    freqs   = np.fft.rfftfreq(N, d=1.0)   # units: cycles per timestep

    peak_idx = int(np.argmax(fft_mag[1:])) + 1   # skip DC component
    f_peak   = float(freqs[peak_idx])

    return f_peak * D / u0


def validate_parameters(
    tau: float,
    u0: float,
    D: float,
    Ny: int,
) -> List[str]:
    """
    Pre-flight parameter validation.

    Returns a list of warning/error strings.  An entry starting with
    'ERROR' means the simulation will be rejected.
    """
    msgs = []
    Ma       = u0 * (3.0 ** 0.5)
    blockage = D / Ny

    if tau <= 0.5:
        msgs.append(f"ERROR: tau={tau:.4f} <= 0.5 — BGK is unconditionally unstable.")
    elif tau < 0.55:
        msgs.append(
            f"WARNING: tau={tau:.4f} is very close to 0.5. "
            "Simulation may be marginally stable. Consider reducing Re or increasing u0/D."
        )

    if Ma > 0.15:
        msgs.append(
            f"WARNING: Ma={Ma:.4f} > 0.15. "
            "Compressibility errors will be significant. Reduce u0."
        )

    if blockage > 0.40:
        msgs.append(
            f"WARNING: blockage ratio D/Ny={blockage:.2f} > 40%. "
            "Wall confinement will significantly inflate Cd. Increase Ny."
        )

    return msgs
