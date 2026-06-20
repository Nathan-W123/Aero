"""
Runtime stability checks, convergence detection, and flow diagnostics.
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ConvergenceReport:
    converged: bool
    window: int
    ratio: float
    mean_abs: float
    std: float
    threshold: float


@dataclass
class StrouhalReport:
    strouhal: Optional[float]
    peak_frequency: Optional[float]
    peak_amplitude: float
    sample_count: int
    window: int
    stationary: bool = False
    relative_drift: Optional[float] = None


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
    return analyze_convergence(Cd_history, window=window, tol=tol).converged


def analyze_convergence(
    history: List[float],
    window: int = 5000,
    tol: float = 0.01,
) -> ConvergenceReport:
    """Return rolling-window convergence metrics for a coefficient history."""
    window = max(int(window), 1)
    if len(history) < window:
        return ConvergenceReport(
            converged=False,
            window=window,
            ratio=float("inf"),
            mean_abs=0.0,
            std=0.0,
            threshold=float(tol),
        )

    recent = np.asarray(history[-window:], dtype=np.float64)
    mean_abs = float(np.mean(np.abs(recent)))
    std = float(np.std(recent))
    if mean_abs < 1e-12:
        ratio = 0.0
        converged = True
    else:
        ratio = std / mean_abs
        converged = ratio < tol
    return ConvergenceReport(
        converged=converged,
        window=window,
        ratio=ratio,
        mean_abs=mean_abs,
        std=std,
        threshold=float(tol),
    )


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
    return analyze_strouhal(Cl_history, D=D, u0=u0).strouhal


def analyze_strouhal(
    Cl_history: List[float],
    D: float,
    u0: float,
    *,
    window: Optional[int] = None,
    min_samples: int = 512,
) -> StrouhalReport:
    """
    Estimate Strouhal data from the recent lift history.

    If `window` is given, only the most recent window is analysed.
    """
    if u0 <= 0.0 or D <= 0.0:
        return StrouhalReport(None, None, 0.0, len(Cl_history), int(window or 0))

    samples = np.asarray(Cl_history, dtype=np.float64)
    if window is not None and window > 0:
        samples = samples[-int(window):]
    used_window = int(samples.size)
    if used_window < max(int(min_samples), 2):
        return StrouhalReport(None, None, 0.0, len(Cl_history), used_window)

    signal = samples - float(np.mean(samples))
    peak_amplitude = float(np.max(np.abs(signal)))
    if peak_amplitude < 1e-6:
        return StrouhalReport(None, None, peak_amplitude, len(Cl_history), used_window)

    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0)
    if fft_mag.size <= 1:
        return StrouhalReport(None, None, peak_amplitude, len(Cl_history), used_window)

    peak_idx = int(np.argmax(fft_mag[1:])) + 1
    peak_frequency = float(freqs[peak_idx])
    return StrouhalReport(
        strouhal=peak_frequency * D / u0,
        peak_frequency=peak_frequency,
        peak_amplitude=peak_amplitude,
        sample_count=len(Cl_history),
        window=used_window,
    )


def detect_statistical_stationarity(
    Cd_history: List[float],
    Cl_history: List[float],
    D: float,
    u0: float,
    *,
    convergence_window: int = 5000,
    convergence_tol: float = 0.01,
    strouhal_window: int = 4096,
    strouhal_tol: float = 0.05,
) -> tuple[ConvergenceReport, StrouhalReport, bool]:
    """
    Combine rolling Cd convergence with Strouhal stability on recent lift data.

    Returns `(cd_report, st_report, stop_now)`.
    """
    cd_report = analyze_convergence(Cd_history, window=convergence_window, tol=convergence_tol)
    st_report = analyze_strouhal(Cl_history, D=D, u0=u0, window=strouhal_window)

    if st_report.strouhal is None or st_report.window < max(strouhal_window, 1024):
        return cd_report, st_report, cd_report.converged

    half_window = st_report.window // 2
    first = analyze_strouhal(Cl_history[-st_report.window:-half_window], D=D, u0=u0, min_samples=128)
    second = analyze_strouhal(Cl_history[-half_window:], D=D, u0=u0, min_samples=128)
    if first.strouhal is None or second.strouhal is None:
        return cd_report, st_report, cd_report.converged

    drift = abs(second.strouhal - first.strouhal) / max(abs(first.strouhal), 1e-12)
    st_report.stationary = drift < strouhal_tol
    st_report.relative_drift = drift
    return cd_report, st_report, cd_report.converged and st_report.stationary


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
