"""
Double-distribution-function (DDF) thermal LBM extension.

Adds a second distribution g for temperature transport alongside the fluid
distribution f. Supports:
  - Passive scalar mode: temperature advected by the flow, no buoyancy
  - Active mode: Boussinesq buoyancy coupling (natural convection, RB instability)

Temperature equilibrium (passive advection):
    g_eq_i = w_i * T * (1 + 3*(e_i · u))

BGK collision with thermal relaxation rate omega_T = 1/(3*alpha + 0.5)
where alpha is the thermal diffusivity.

Boussinesq body force (gravity in -y direction, hot at y=0, cold at y=Ny-1):
    F_i = w_i * 3 * rho * g_grav * beta * (T - T_ref) * e_y_i

References
----------
He, Chen & Doolen (1998) Int. J. Heat Mass Transfer 41:3387-3396.
"""

from typing import Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 2D thermal distribution (D2Q9)
# ---------------------------------------------------------------------------

def init_g_2d(
    T0: float,
    Ny: int,
    Nx: int,
    w: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
) -> np.ndarray:
    """Initialise g to equilibrium at uniform temperature T0, zero velocity."""
    Q = len(w)
    g = np.empty((Q, Ny, Nx), dtype=np.float64)
    for i in range(Q):
        g[i] = w[i] * T0
    return g


def collide_g_2d(
    g: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    T: np.ndarray,
    omega_T: float,
    solid: np.ndarray,
) -> np.ndarray:
    """BGK collision step for the 2D temperature distribution."""
    Q = g.shape[0]
    from .d2q9 import E, W
    g_post = np.empty_like(g)
    for i in range(Q):
        eu = float(E[i, 0]) * ux + float(E[i, 1]) * uy
        g_eq = W[i] * T * (1.0 + 3.0 * eu)
        g_post[i] = g[i] - omega_T * (g[i] - g_eq)
    # Solid cells: relax to zero-velocity equilibrium at local T
    g_post[:, solid] = 0.0
    for i in range(Q):
        g_post[i][solid] = W[i] * T[solid]
    return g_post


def stream_g_2d(g: np.ndarray, ex: np.ndarray, ey: np.ndarray) -> np.ndarray:
    """Streaming step for the 2D temperature distribution."""
    Q = g.shape[0]
    g_new = np.empty_like(g)
    for i in range(Q):
        g_new[i] = np.roll(g[i], int(ey[i]), axis=0)
        g_new[i] = np.roll(g_new[i], int(ex[i]), axis=1)
    return g_new


def extract_T(g: np.ndarray) -> np.ndarray:
    """Extract temperature field: T = sum_i g_i."""
    return g.sum(axis=0)


def apply_temperature_bc_2d(
    g: np.ndarray,
    T_hot: float,
    T_cold: float,
    w: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
) -> None:
    """
    Dirichlet temperature BC: hot wall at y=0, cold wall at y=Ny-1.
    Sets g to the zero-velocity equilibrium at the wall temperature.
    """
    Q = len(w)
    for i in range(Q):
        g[i, 0, :]    = w[i] * T_hot    # bottom — hot
        g[i, -1, :]   = w[i] * T_cold   # top — cold


def guo_buoyancy_force_2d(
    rho: np.ndarray,
    T: np.ndarray,
    T_ref: float,
    g_gravity: float,
    beta: float,
    ey: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Boussinesq buoyancy body force (gravity in +y direction by convention).

    F_i = w_i * 3 * rho * g_gravity * beta * (T - T_ref) * e_y_i

    Add this to f after collision to drive natural convection.
    """
    Q = len(w)
    buoy = g_gravity * beta * (T - T_ref)  # (Ny, Nx)
    F = np.empty((Q,) + T.shape, dtype=np.float64)
    for i in range(Q):
        F[i] = w[i] * 3.0 * rho * buoy * float(ey[i])
    return F


# ---------------------------------------------------------------------------
# 3D thermal distribution (D3Q19)
# ---------------------------------------------------------------------------

def init_g_3d(
    T0: float,
    Nz: int,
    Ny: int,
    Nx: int,
    w: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
) -> np.ndarray:
    """Initialise 3D g to equilibrium at uniform T0, zero velocity."""
    Q = len(w)
    g = np.empty((Q, Nz, Ny, Nx), dtype=np.float64)
    for i in range(Q):
        g[i] = w[i] * T0
    return g


def collide_g_3d(
    g: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    T: np.ndarray,
    omega_T: float,
    solid: np.ndarray,
) -> np.ndarray:
    """BGK collision step for the 3D temperature distribution."""
    from .d3q19 import E3, W3
    Q = g.shape[0]
    g_post = np.empty_like(g)
    for i in range(Q):
        eu = float(E3[i, 0]) * ux + float(E3[i, 1]) * uy + float(E3[i, 2]) * uz
        g_eq = W3[i] * T * (1.0 + 3.0 * eu)
        g_post[i] = g[i] - omega_T * (g[i] - g_eq)
    g_post[:, solid] = 0.0
    for i in range(Q):
        g_post[i][solid] = W3[i] * T[solid]
    return g_post


def stream_g_3d(
    g: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
) -> np.ndarray:
    """Streaming step for the 3D temperature distribution."""
    Q = g.shape[0]
    g_new = np.empty_like(g)
    for i in range(Q):
        tmp = np.roll(g[i], int(ez[i]), axis=0)
        tmp = np.roll(tmp, int(ey[i]), axis=1)
        g_new[i] = np.roll(tmp, int(ex[i]), axis=2)
    return g_new


def apply_temperature_bc_3d(
    g: np.ndarray,
    T_hot: float,
    T_cold: float,
    w: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
) -> None:
    """
    Dirichlet temperature BCs for 3D: hot at y=0, cold at y=Ny-1.
    """
    Q = len(w)
    for i in range(Q):
        g[i, :, 0, :]  = w[i] * T_hot   # bottom wall
        g[i, :, -1, :] = w[i] * T_cold  # top wall


def guo_buoyancy_force_3d(
    rho: np.ndarray,
    T: np.ndarray,
    T_ref: float,
    g_gravity: float,
    beta: float,
    ey: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Boussinesq buoyancy body force for 3D (gravity along y)."""
    from .d3q19 import E3
    Q = len(w)
    buoy = g_gravity * beta * (T - T_ref)
    F = np.empty((Q,) + T.shape, dtype=np.float64)
    for i in range(Q):
        F[i] = w[i] * 3.0 * rho * buoy * float(E3[i, 1])
    return F
