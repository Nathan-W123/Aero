"""
Boundary condition implementations for the D2Q9 LBM solver.

Applied each timestep in this order, after streaming:
  1. Mid-link bounce-back   (obstacle surface)
  2. Inlet BC               (left wall, x=0)   — velocity or pressure Zou-He
  3. Outlet BC              (right wall, x=Nx-1) — convective or pressure Zou-He
  4. Wall BC                (top y=Ny-1, bottom y=0) — slip or no-slip
"""

import numpy as np
from .d2q9 import E, OPP


def apply_inlet_zou_he(
    f: np.ndarray,
    u0: float,
    *,
    uy_amp: float = 0.0,
    step: int = 0,
) -> None:
    """
    Zou-He velocity BC at left wall (x=0): impose ux=u0, uy≈0.

    Optional ``uy_amp`` adds a travelling sinusoidal transverse perturbation
    (fraction of u0) to trigger vortex shedding at supercritical Re.
    """
    col = 0

    rho_in = (
        (f[0, :, col] + f[2, :, col] + f[4, :, col])
        + 2.0 * (f[3, :, col] + f[6, :, col] + f[7, :, col])
    ) / (1.0 - u0)

    f[1, :, col] = f[3, :, col] + (2.0 / 3.0) * rho_in * u0
    f[5, :, col] = f[7, :, col] - 0.5 * (f[2, :, col] - f[4, :, col]) + (1.0 / 6.0) * rho_in * u0
    f[8, :, col] = f[6, :, col] + 0.5 * (f[2, :, col] - f[4, :, col]) + (1.0 / 6.0) * rho_in * u0

    if uy_amp > 0.0:
        ny = f.shape[1]
        uy = uy_amp * u0 * np.sin(2.0 * np.pi * np.arange(ny) / max(ny, 1) + 0.17 * step)
        f[2, :, col] += uy * 0.25
        f[4, :, col] -= uy * 0.25
        f[5, :, col] += uy * 0.15
        f[8, :, col] -= uy * 0.15


def apply_outlet_zero_gradient(f: np.ndarray) -> None:
    """
    Zero-gradient (extrapolation) outlet at right wall (x=Nx-1).

    Copies distributions from second-to-last column.
    Kept for reference; prefer apply_outlet_convective for production runs.
    """
    f[:, :, -1] = f[:, :, -2]


def apply_outlet_convective(
    f: np.ndarray,
    f_outlet_prev: np.ndarray,
    u_conv: float,
) -> None:
    """
    Convective (advective) outflow BC at right wall (x=Nx-1).

    Discretises the advection equation for each distribution function:

        ∂f_i/∂t + u_conv * ∂f_i/∂x = 0

    Using forward-Euler in time and first-order upwind in x (Δt=Δx=1):

        f_i^{n+1}[x_out] = f_i^n[x_out] - u_conv * (f_i^n[x_out] - f_i^n[x_out-1])
                         = (1 - u_conv) * f_i^n[x_out]  +  u_conv * f_i^n[x_out-1]

    where:
        f_i^n[x_out]   = f_outlet_prev  (outlet column from previous timestep)
        f_i^n[x_out-1] = f[:, :, -2]   (penultimate column, current step post-stream)

    Stability:  u_conv ≤ 1  (CFL).  Since u0 ≈ 0.05 << 1 this is always satisfied.

    This BC is transparent to outgoing vortices, avoiding the wake-reflection
    artifact seen with zero-gradient when the vortex street reaches the outlet.

    Parameters
    ----------
    f             : (9, Ny, Nx) — full f array; outlet column is overwritten in-place
    f_outlet_prev : (9, Ny)    — outlet column from the previous timestep; updated in-place
    u_conv        : float      — convection speed (use inlet u0)
    """
    u = float(np.clip(u_conv, 0.0, 1.0))          # CFL guard
    new_outlet = (1.0 - u) * f_outlet_prev + u * f[:, :, -2]
    f[:, :, -1] = new_outlet
    f_outlet_prev[:] = new_outlet                  # advance stored state


def apply_inlet_zou_he_pressure(f: np.ndarray, rho_in: float) -> None:
    """
    Zou-He pressure BC at left wall (x=0): impose rho=rho_in, uy=0.

    rho_in is given; ux is derived from moment constraints, then the three
    unknown rightward populations (f[1], f[5], f[8]) are set analytically.

    Math:
        ux = 1 - (f[0]+f[2]+f[4] + 2*(f[3]+f[6]+f[7])) / rho_in
        f[1] = f[3] + (2/3)*rho_in*ux
        f[5] = f[7] - 0.5*(f[2]-f[4]) + (1/6)*rho_in*ux
        f[8] = f[6] + 0.5*(f[2]-f[4]) + (1/6)*rho_in*ux
    """
    col = 0
    ux_in = 1.0 - (
        f[0, :, col] + f[2, :, col] + f[4, :, col]
        + 2.0 * (f[3, :, col] + f[6, :, col] + f[7, :, col])
    ) / rho_in
    f[1, :, col] = f[3, :, col] + (2.0 / 3.0) * rho_in * ux_in
    f[5, :, col] = f[7, :, col] - 0.5 * (f[2, :, col] - f[4, :, col]) + (1.0 / 6.0) * rho_in * ux_in
    f[8, :, col] = f[6, :, col] + 0.5 * (f[2, :, col] - f[4, :, col]) + (1.0 / 6.0) * rho_in * ux_in


def apply_outlet_zou_he_pressure(f: np.ndarray, rho_out: float) -> None:
    """
    Zou-He pressure BC at right wall (x=Nx-1): impose rho=rho_out, uy=0.

    ux is derived from moment constraints; the three unknown leftward
    populations (f[3], f[6], f[7]) are set analytically.

    Math:
        ux = -1 + (f[0]+f[2]+f[4] + 2*(f[1]+f[5]+f[8])) / rho_out
        f[3] = f[1] - (2/3)*rho_out*ux
        f[7] = f[5] + 0.5*(f[2]-f[4]) - (1/6)*rho_out*ux
        f[6] = f[8] - 0.5*(f[2]-f[4]) - (1/6)*rho_out*ux
    """
    col = -1
    ux_out = -1.0 + (
        f[0, :, col] + f[2, :, col] + f[4, :, col]
        + 2.0 * (f[1, :, col] + f[5, :, col] + f[8, :, col])
    ) / rho_out
    f[3, :, col] = f[1, :, col] - (2.0 / 3.0) * rho_out * ux_out
    f[7, :, col] = f[5, :, col] + 0.5 * (f[2, :, col] - f[4, :, col]) - (1.0 / 6.0) * rho_out * ux_out
    f[6, :, col] = f[8, :, col] - 0.5 * (f[2, :, col] - f[4, :, col]) - (1.0 / 6.0) * rho_out * ux_out


def apply_noslip_walls(f: np.ndarray) -> None:
    """
    Full bounce-back (no-slip) at top and bottom walls.

    At each wall row, incoming populations are reflected to their exact
    opposite directions (reversing both velocity components), enforcing
    zero velocity at the wall mid-link.

    Bottom wall (y=0):   i=4→2, i=7→5, i=8→6
    Top    wall (y=Ny-1): i=2→4, i=5→7, i=6→8

    Cf. apply_slip_walls which reflects only the y-component (slip BC).
    """
    # Bottom wall (y=0): reverse down-going into up-going (OPP)
    f[2, 0, :] = f[4, 0, :]
    f[5, 0, :] = f[7, 0, :]
    f[6, 0, :] = f[8, 0, :]

    # Top wall (y=-1): reverse up-going into down-going (OPP)
    f[4, -1, :] = f[2, -1, :]
    f[7, -1, :] = f[5, -1, :]
    f[8, -1, :] = f[6, -1, :]


def apply_slip_walls(f: np.ndarray) -> None:
    """
    Specular reflection (slip / free-slip) at top and bottom walls.

    Reverses the y-component of velocity: directions that point into
    a wall are replaced by their y-mirrored counterpart.

    Bottom wall y=0: directions 4, 7, 8 point down (ey=-1).
    Top    wall y=-1: directions 2, 5, 6 point up  (ey=+1).

    Mirror pairs (y-reflection only):
      2 <-> 4,  5 <-> 8,  6 <-> 7
    """
    # Bottom wall (y=0): incoming from above → reflect down-going back up
    f[2, 0, :] = f[4, 0, :]
    f[5, 0, :] = f[8, 0, :]
    f[6, 0, :] = f[7, 0, :]

    # Top wall (y=-1): incoming from below → reflect up-going back down
    f[4, -1, :] = f[2, -1, :]
    f[8, -1, :] = f[5, -1, :]
    f[7, -1, :] = f[6, -1, :]


from typing import Optional, Tuple


def build_surface_links(
    solid: np.ndarray,
    phi: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute surface link list for bounce-back.

    A surface link is a tuple (i, y, x) where:
      - cell (y, x) is fluid
      - cell (y + E[i,1], x + E[i,0]) is solid

    Parameters
    ----------
    solid : (Ny, Nx) bool
    phi   : (Ny, Nx) float64 optional signed-distance field.
            If provided, q_vals[k] = clip(phi[y, x], 1e-4, 1.0) giving the
            wall fraction for Bouzidi interpolated bounce-back.
            If None, q_vals = 0.5 (exact mid-link).

    Returns
    -------
    links  : ndarray int32, shape (N_links, 3)  columns: [i, y, x]
    q_vals : ndarray float32, shape (N_links,)  wall fractions
    """
    Ny, Nx = solid.shape
    fluid = ~solid

    links = []
    for i in range(1, 9):  # skip rest direction i=0
        y_idx, x_idx = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing='ij')
        yn_idx = np.clip(y_idx + int(E[i, 1]), 0, Ny - 1)
        xn_idx = np.clip(x_idx + int(E[i, 0]), 0, Nx - 1)
        mask = fluid & solid[yn_idx, xn_idx]
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            links.append((i, int(y), int(x)))

    if not links:
        return np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.float32)

    links_arr = np.array(links, dtype=np.int32)
    if phi is not None:
        y_arr = links_arr[:, 1]
        x_arr = links_arr[:, 2]
        q_vals = np.clip(phi[y_arr, x_arr].astype(np.float32), 1e-4, 1.0)
    else:
        q_vals = np.full(len(links_arr), 0.5, dtype=np.float32)

    return links_arr, q_vals


def apply_bounce_back(f: np.ndarray, f_pre: np.ndarray, links: np.ndarray) -> None:
    """
    Mid-link bounce-back on obstacle surface.

    For each surface link (i, y, x):
      f[opp[i], y, x] = f_pre[i, y, x]

    where f_pre is the post-collision / pre-streaming snapshot.

    Modifies f in-place.
    """
    if links.shape[0] == 0:
        return
    i_arr = links[:, 0]
    y_arr = links[:, 1]
    x_arr = links[:, 2]
    opp_i = OPP[i_arr]
    f[opp_i, y_arr, x_arr] = f_pre[i_arr, y_arr, x_arr]


def apply_bounce_back_bouzidi(
    f: np.ndarray,
    f_pre: np.ndarray,
    links: np.ndarray,
    q_vals: np.ndarray,
) -> None:
    """
    Bouzidi (2001) interpolated bounce-back — 2nd-order accurate at curved walls.

    For each surface link (i, y, x) with wall fraction q = phi[y, x]:

    q < 0.5  (wall closer than midpoint):
        f[opp[i], y, x] = 2q * f_pre[i, y, x] + (1-2q) * f_pre[opp[i], y, x]

    q >= 0.5 (wall farther than midpoint):
        f[opp[i], y, x] = (1/2q) * f_pre[i, y, x]
                         + (1 - 1/2q) * f_pre[i, y_nb, x_nb]
        where (y_nb, x_nb) is the fluid neighbour in the opp[i] direction.

    At q = 0.5 both cases reduce to standard mid-link bounce-back.
    Modifies f in-place.
    """
    if links.shape[0] == 0:
        return

    Ny, Nx = f.shape[1], f.shape[2]
    i_arr = links[:, 0]
    y_arr = links[:, 1]
    x_arr = links[:, 2]
    opp_i = OPP[i_arr]
    q = q_vals.astype(np.float64)

    # Case 1: q < 0.5
    m = q < 0.5
    if m.any():
        f[opp_i[m], y_arr[m], x_arr[m]] = (
            2.0 * q[m] * f_pre[i_arr[m], y_arr[m], x_arr[m]]
            + (1.0 - 2.0 * q[m]) * f_pre[opp_i[m], y_arr[m], x_arr[m]]
        )

    # Case 2: q >= 0.5 — needs neighbour node in opp direction
    m2 = ~m
    if m2.any():
        yn = np.clip(y_arr[m2] + E[opp_i[m2], 1].astype(int), 0, Ny - 1)
        xn = np.clip(x_arr[m2] + E[opp_i[m2], 0].astype(int), 0, Nx - 1)
        inv2q = 1.0 / (2.0 * q[m2])
        f[opp_i[m2], y_arr[m2], x_arr[m2]] = (
            inv2q * f_pre[i_arr[m2], y_arr[m2], x_arr[m2]]
            + (1.0 - inv2q) * f_pre[i_arr[m2], yn, xn]
        )
