"""
D2Q9 Multiple Relaxation Time (MRT) matrices.

Reference: Lallemand & Luo, Phys. Rev. E 61, 6546 (2000).

Moment ordering (rows of M):
  0  rho   — density            (conserved)
  1  e     — energy
  2  eps   — energy-square
  3  jx    — x-momentum         (conserved)
  4  qx    — energy-flux x
  5  jy    — y-momentum         (conserved)
  6  qy    — energy-flux y
  7  pxx   — stress diagonal
  8  pxy   — stress off-diagonal

Velocity ordering (columns) matches d2q9.py E:
  i=0:(0,0)  1:(1,0)  2:(0,1)  3:(-1,0)  4:(0,-1)
  5:(1,1)    6:(-1,1) 7:(-1,-1) 8:(1,-1)

Relaxation rates S = diag(s0..s8):
  s0, s3, s5 = 1.0   (conserved — value doesn't matter)
  s7 = s8    = omega  (stress modes → kinematic viscosity nu = cs²*(1/omega - 0.5))
  s1 (se), s2 (sep), s4=s6 (sq) are free tuning parameters.

The "magic parameter" choice sq = 8*(2-sv)/(8-sv) makes the scheme
equivalent to TRT and gives 4th-order accurate diffusion.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Transformation matrix M  (9 × 9)
# ---------------------------------------------------------------------------

M9 = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1],   # rho
    [-4, -1, -1, -1, -1,  2,  2,  2,  2],   # e
    [ 4, -2, -2, -2, -2,  1,  1,  1,  1],   # eps
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1],   # jx
    [ 0, -2,  0,  2,  0,  1, -1, -1,  1],   # qx
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1],   # jy
    [ 0,  0, -2,  0,  2,  1,  1, -1, -1],   # qy
    [ 0,  1, -1,  1, -1,  0,  0,  0,  0],   # pxx
    [ 0,  0,  0,  0,  0,  1, -1,  1, -1],   # pxy
], dtype=np.float64)

M9_inv = np.linalg.inv(M9)


# Entries of s that depend on the local relaxation rate.  A subgrid model
# makes omega a per-cell field, and only these rows may then vary per cell.
VISCOUS_MODES = np.array([7, 8], dtype=np.int64)     # pxx, pxy
ENERGY_FLUX_MODES = np.array([4, 6], dtype=np.int64)  # qx, qy — sq(omega)


def magic_sq(sv: float) -> float:
    """Energy-flux rate tied to the viscous rate (4th-order accurate diffusion)."""
    return 8.0 * (2.0 - sv) / (8.0 - sv)


def build_s_vec(omega: float, se: float = 1.64, sep: float = 1.54) -> np.ndarray:
    """
    Build the MRT relaxation vector s for D2Q9.

    Parameters
    ----------
    omega : float — BGK-equivalent relaxation rate (1/tau); controls viscosity
    se    : float — energy mode relaxation (default 1.64, Lallemand 2000)
    sep   : float — energy-square mode relaxation (default 1.54)

    Returns
    -------
    s : (9,) float64 — diagonal of S matrix
    """
    sv = omega                            # stress modes → viscosity
    sq = magic_sq(sv)                     # magic parameter (4th-order accurate)
    return np.array([
        1.0,   # s0  rho      (conserved)
        se,    # s1  e
        sep,   # s2  eps
        1.0,   # s3  jx       (conserved)
        sq,    # s4  qx
        1.0,   # s5  jy       (conserved)
        sq,    # s6  qy
        sv,    # s7  pxx
        sv,    # s8  pxy
    ], dtype=np.float64)
