"""
Boundary conditions for the D3Q19 LBM solver.

Applied each timestep in this order (after streaming):
  1. Mid-link bounce-back   (obstacle surface)
  2. Inlet BC               (left face, x=0)   — Zou-He velocity
  3. Outlet BC              (right face, x=Nx-1) — convective or zero-gradient
  4. Wall BC                (top y=Ny-1, bottom y=0) — slip or no-slip
  (z direction: periodic by default via streaming)

f array layout: (Q, Nz, Ny, Nx)
"""

from typing import Optional, Tuple

import numpy as np
from .d3q19 import E3, OPP3, Y_MIR3, compute_feq_3d, compute_macroscopic_3d


# ---------------------------------------------------------------------------
# Inlet: Zou-He 3D velocity BC (ux=u0, uy=0, uz=0) at x=0
# ---------------------------------------------------------------------------

def apply_inlet_zou_he_3d(
    f: np.ndarray,
    u0: float,
    *,
    uy_amp: float = 0.0,
    uz_amp: float = 0.0,
    step: int = 0,
) -> None:
    """
    3D Zou-He velocity BC at left face (x=0): impose ux=u0, uy≈0, uz≈0.

    Optional spanwise uz_amp and vertical uy_amp perturbations (fraction of u0)
    trigger 3D shedding at supercritical Re.
    """
    c = 0  # x=0 column index

    F_minus = (f[2,:,:,c] + f[8,:,:,c] + f[10,:,:,c]
               + f[12,:,:,c] + f[14,:,:,c])
    F_neut  = (f[0,:,:,c] + f[3,:,:,c] + f[4,:,:,c]
               + f[5,:,:,c] + f[6,:,:,c]
               + f[15,:,:,c] + f[16,:,:,c] + f[17,:,:,c] + f[18,:,:,c])
    rho_in = (2.0 * F_minus + F_neut) / (1.0 - u0)

    cy = (f[3,:,:,c] - f[4,:,:,c]
          + f[15,:,:,c] - f[16,:,:,c] + f[17,:,:,c] - f[18,:,:,c])
    cz = (f[5,:,:,c] - f[6,:,:,c]
          + f[15,:,:,c] + f[16,:,:,c] - f[17,:,:,c] - f[18,:,:,c])

    if uy_amp > 0.0 or uz_amp > 0.0:
        nz, ny = f.shape[1], f.shape[2]
        yy = np.arange(ny, dtype=np.float64)
        zz = np.arange(nz, dtype=np.float64)
        phase = 0.17 * step
        if uy_amp > 0.0:
            uy = uy_amp * u0 * np.sin(2.0 * np.pi * yy / max(ny, 1) + phase)
            cy = cy + rho_in * uy
        if uz_amp > 0.0:
            uz = uz_amp * u0 * np.sin(2.0 * np.pi * zz / max(nz, 1) + phase)
            cz = cz + rho_in * uz[:, None]

    f[1, :,:,c]  = f[2, :,:,c] + (1.0/3.0) * rho_in * u0
    f[7, :,:,c]  = f[10,:,:,c] + (1.0/6.0) * rho_in * u0 - 0.5 * cy
    f[9, :,:,c]  = f[8, :,:,c] + (1.0/6.0) * rho_in * u0 + 0.5 * cy
    f[11,:,:,c]  = f[14,:,:,c] + (1.0/6.0) * rho_in * u0 - 0.5 * cz
    f[13,:,:,c]  = f[12,:,:,c] + (1.0/6.0) * rho_in * u0 + 0.5 * cz


def apply_inlet_velocity_field_3d(
    f: np.ndarray,
    ux_target: np.ndarray,
    uy_target: np.ndarray,
    uz_target: np.ndarray,
) -> None:
    """3D Zou-He inlet with per-cell target velocity fields at x=0."""
    c = 0
    ux_target = np.asarray(ux_target, dtype=np.float64)
    uy_target = np.asarray(uy_target, dtype=np.float64)
    uz_target = np.asarray(uz_target, dtype=np.float64)
    F_minus = (f[2,:,:,c] + f[8,:,:,c] + f[10,:,:,c] + f[12,:,:,c] + f[14,:,:,c])
    F_neut  = (f[0,:,:,c] + f[3,:,:,c] + f[4,:,:,c] + f[5,:,:,c] + f[6,:,:,c]
               + f[15,:,:,c] + f[16,:,:,c] + f[17,:,:,c] + f[18,:,:,c])
    rho_in = (2.0 * F_minus + F_neut) / np.maximum(1.0 - ux_target, 1e-8)
    cy = rho_in * uy_target
    cz = rho_in * uz_target

    f[1, :,:,c]  = f[2, :,:,c] + (1.0/3.0) * rho_in * ux_target
    f[7, :,:,c]  = f[10,:,:,c] + (1.0/6.0) * rho_in * ux_target - 0.5 * cy
    f[9, :,:,c]  = f[8, :,:,c] + (1.0/6.0) * rho_in * ux_target + 0.5 * cy
    f[11,:,:,c]  = f[14,:,:,c] + (1.0/6.0) * rho_in * ux_target - 0.5 * cz
    f[13,:,:,c]  = f[12,:,:,c] + (1.0/6.0) * rho_in * ux_target + 0.5 * cz


def apply_inlet_sem_3d(
    f: np.ndarray,
    u0: float,
    u_prime: np.ndarray,
    v_prime: np.ndarray,
    w_prime: np.ndarray,
) -> None:
    """
    Zou-He 3D velocity BC at x=0 with SEM fluctuations.

    Imposes (ux, uy, uz) = (u0 + u_prime, v_prime, w_prime) on the inlet face
    using the standard Zou-He non-equilibrium bounce-back formulation.

    Parameters
    ----------
    u_prime, v_prime, w_prime : (Nz, Ny) arrays from SEMInlet.fluctuation()
    """
    c = 0
    ux_field = u0 + u_prime  # (Nz, Ny)

    F_minus = (f[2,:,:,c] + f[8,:,:,c] + f[10,:,:,c]
               + f[12,:,:,c] + f[14,:,:,c])
    F_neut  = (f[0,:,:,c] + f[3,:,:,c] + f[4,:,:,c]
               + f[5,:,:,c] + f[6,:,:,c]
               + f[15,:,:,c] + f[16,:,:,c] + f[17,:,:,c] + f[18,:,:,c])
    rho_in = (2.0 * F_minus + F_neut) / np.clip(1.0 - ux_field, 1e-10, None)

    # cy_base enforces uy=0; subtract rho*v_prime to impose uy=+v_prime
    cy = (f[3,:,:,c] - f[4,:,:,c]
          + f[15,:,:,c] - f[16,:,:,c] + f[17,:,:,c] - f[18,:,:,c])
    cy = cy - rho_in * v_prime

    # cz_base enforces uz=0; subtract rho*w_prime to impose uz=+w_prime
    cz = (f[5,:,:,c] - f[6,:,:,c]
          + f[15,:,:,c] + f[16,:,:,c] - f[17,:,:,c] - f[18,:,:,c])
    cz = cz - rho_in * w_prime

    f[1, :,:,c]  = f[2, :,:,c] + (1.0/3.0) * rho_in * ux_field
    f[7, :,:,c]  = f[10,:,:,c] + (1.0/6.0) * rho_in * ux_field - 0.5 * cy
    f[9, :,:,c]  = f[8, :,:,c] + (1.0/6.0) * rho_in * ux_field + 0.5 * cy
    f[11,:,:,c]  = f[14,:,:,c] + (1.0/6.0) * rho_in * ux_field - 0.5 * cz
    f[13,:,:,c]  = f[12,:,:,c] + (1.0/6.0) * rho_in * ux_field + 0.5 * cz


# ---------------------------------------------------------------------------
# Outlet BCs at x=Nx-1
# ---------------------------------------------------------------------------

def apply_outlet_zero_gradient_3d(f: np.ndarray) -> None:
    """Zero-gradient (copy) outlet BC at right face (x=Nx-1)."""
    f[:,:,:,-1] = f[:,:,:,-2]


def apply_outlet_convective_3d(
    f: np.ndarray,
    f_outlet_prev: np.ndarray,
    u_conv: float,
) -> None:
    """
    Convective outflow BC at right face (x=Nx-1).

        f_i^{n+1}[x_out] = (1-u)*f_i^n[x_out] + u*f_i^n[x_out-1]

    Same advection scheme as the 2D version; f_outlet_prev has shape (Q,Nz,Ny).
    """
    u = float(np.clip(u_conv, 0.0, 1.0))
    new_outlet = (1.0 - u) * f_outlet_prev + u * f[:,:,:,-2]
    f[:,:,:,-1] = new_outlet
    f_outlet_prev[:] = new_outlet


def apply_streamwise_periodic_3d(f: np.ndarray) -> None:
    """Explicit no-op helper for fully periodic streamwise domains."""
    return None


def apply_recycling_rescaling_inlet_3d(
    f: np.ndarray,
    target_u0: float,
    *,
    recycle_index: int = -2,
    inlet_index: int = 0,
) -> None:
    """
    Recycle the downstream plane to the inlet and rescale its mean ux to `target_u0`.

    This preserves cross-stream structure and turbulence content better than a
    uniform inlet while remaining lightweight enough for the current solver.
    """
    rho, ux, uy, uz = compute_macroscopic_3d(f)
    recycle_rho = rho[:, :, recycle_index]
    recycle_ux = ux[:, :, recycle_index]
    recycle_uy = uy[:, :, recycle_index]
    recycle_uz = uz[:, :, recycle_index]
    mean_ux = float(np.mean(recycle_ux))
    scale = 1.0 if abs(mean_ux) < 1e-12 else float(target_u0) / mean_ux
    feq = compute_feq_3d(
        recycle_rho[:, :, None],
        (recycle_ux * scale)[:, :, None],
        (recycle_uy * scale)[:, :, None],
        (recycle_uz * scale)[:, :, None],
    )
    f[:, :, :, inlet_index] = feq[:, :, :, 0]


# ---------------------------------------------------------------------------
# Wall BCs: top (y=Ny-1) and bottom (y=0)
# ---------------------------------------------------------------------------

def apply_slip_walls_3d(f: np.ndarray) -> None:
    """
    Specular reflection at top and bottom walls (preserve ex,ez, flip ey).

    y-mirror pairs (ex,ez fixed, ey flipped):
      (3↔4), (7↔9), (8↔10), (15↔16), (17↔18)
    """
    # Bottom wall (y=0): incoming ey<0 directions → reflect to ey>0
    f[3,:,0,:] = f[4,:,0,:]   # ey+ ← ey-
    f[7,:,0,:] = f[9,:,0,:]   # ex+,ey+ ← ex+,ey-
    f[8,:,0,:] = f[10,:,0,:]  # ex-,ey+ ← ex-,ey-
    f[15,:,0,:] = f[16,:,0,:] # ey+,ez+ ← ey-,ez+
    f[17,:,0,:] = f[18,:,0,:] # ey+,ez- ← ey-,ez-

    # Top wall (y=Ny-1): incoming ey>0 directions → reflect to ey<0
    f[4,:,-1,:] = f[3,:,-1,:]
    f[9,:,-1,:] = f[7,:,-1,:]
    f[10,:,-1,:] = f[8,:,-1,:]
    f[16,:,-1,:] = f[15,:,-1,:]
    f[18,:,-1,:] = f[17,:,-1,:]


def apply_noslip_walls_3d(f: np.ndarray) -> None:
    """
    Full bounce-back (no-slip) at top and bottom walls.

    Reverses all velocity components: f[opp[i]] ← f[i] for incoming directions.

    Bounce-back pairs vs slip pairs differ for diagonal directions:
      slip: ex,ez fixed, ey flipped     (7↔9, 8↔10, 15↔16, 17↔18)
      noslip: full reversal via OPP3    (7↔10, 8↔9, 15↔18, 16↔17)
    """
    # Bottom wall (y=0): ey<0 incoming → reverse
    f[3,:,0,:]  = f[4,:,0,:]
    f[7,:,0,:]  = f[10,:,0,:]  # OPP3[10]=7 ← differs from slip
    f[8,:,0,:]  = f[9,:,0,:]   # OPP3[9]=8  ← differs from slip
    f[15,:,0,:] = f[18,:,0,:]  # OPP3[18]=15
    f[17,:,0,:] = f[16,:,0,:]  # OPP3[16]=17

    # Top wall (y=Ny-1): ey>0 incoming → reverse
    f[4,:,-1,:]  = f[3,:,-1,:]
    f[10,:,-1,:] = f[7,:,-1,:]
    f[9,:,-1,:]  = f[8,:,-1,:]
    f[18,:,-1,:] = f[15,:,-1,:]
    f[16,:,-1,:] = f[17,:,-1,:]


def apply_moving_walls_3d(
    f: np.ndarray,
    *,
    u_top: float = 0.0,
    u_bottom: float = 0.0,
) -> None:
    """Moving-wall bounce-back with tangential x-velocity on top/bottom y-walls."""
    rho_bottom = np.maximum(np.sum(f[:, :, 0, :], axis=0), 1e-12)
    rho_top = np.maximum(np.sum(f[:, :, -1, :], axis=0), 1e-12)

    f[3,:,0,:]  = f[4,:,0,:]
    f[7,:,0,:]  = f[10,:,0,:] + 6.0 * (1.0 / 36.0) * rho_bottom * float(u_bottom)
    f[8,:,0,:]  = f[9,:,0,:]  - 6.0 * (1.0 / 36.0) * rho_bottom * float(u_bottom)
    f[15,:,0,:] = f[18,:,0,:]
    f[17,:,0,:] = f[16,:,0,:]

    f[4,:,-1,:]  = f[3,:,-1,:]
    f[10,:,-1,:] = f[7,:,-1,:]  - 6.0 * (1.0 / 36.0) * rho_top * float(u_top)
    f[9,:,-1,:]  = f[8,:,-1,:]  + 6.0 * (1.0 / 36.0) * rho_top * float(u_top)
    f[18,:,-1,:] = f[15,:,-1,:]
    f[16,:,-1,:] = f[17,:,-1,:]


# ---------------------------------------------------------------------------
# Obstacle: mid-link bounce-back
# ---------------------------------------------------------------------------

def build_surface_links_3d(
    solid: np.ndarray,
    phi: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute surface links for 3D bounce-back.

    A surface link (i, z, y, x) satisfies:
      - cell (z, y, x) is fluid
      - neighbour (z+ez[i], y+ey[i], x+ex[i]) is solid

    Parameters
    ----------
    solid : (Nz, Ny, Nx) bool
    phi   : optional (Nz, Ny, Nx) float64 signed-distance field.
            If provided, q_vals[k] = clip(phi[z, y, x], 1e-4, 1.0).
            If None, q_vals = 0.5 (exact mid-link).

    Returns
    -------
    links  : ndarray int32, shape (N_links, 4) — columns [i, z, y, x]
    q_vals : ndarray float32, shape (N_links,)
    """
    Nz, Ny, Nx = solid.shape
    fluid = ~solid
    links = []
    for i in range(1, 19):
        ex_, ey_, ez_ = int(E3[i, 0]), int(E3[i, 1]), int(E3[i, 2])
        z_g, y_g, x_g = np.meshgrid(
            np.arange(Nz), np.arange(Ny), np.arange(Nx), indexing='ij'
        )
        zn = np.clip(z_g + ez_, 0, Nz - 1)
        yn = np.clip(y_g + ey_, 0, Ny - 1)
        xn = np.clip(x_g + ex_, 0, Nx - 1)
        mask = fluid & solid[zn, yn, xn]
        zs, ys, xs = np.where(mask)
        for z, y, x in zip(zs.tolist(), ys.tolist(), xs.tolist()):
            links.append((i, int(z), int(y), int(x)))

    if not links:
        return np.empty((0, 4), dtype=np.int32), np.empty(0, dtype=np.float32)

    links_arr = np.array(links, dtype=np.int32)
    if phi is not None:
        z_arr = links_arr[:, 1]
        y_arr = links_arr[:, 2]
        x_arr = links_arr[:, 3]
        q_vals = np.clip(phi[z_arr, y_arr, x_arr].astype(np.float32), 1e-4, 1.0)
    else:
        q_vals = np.full(len(links_arr), 0.5, dtype=np.float32)

    return links_arr, q_vals


def apply_bounce_back_3d(
    f: np.ndarray,
    f_pre: np.ndarray,
    links: np.ndarray,
) -> None:
    """
    Mid-link bounce-back on 3D obstacle surface.

    For each link (i, z, y, x):  f[opp[i], z, y, x] = f_pre[i, z, y, x]
    """
    if links.shape[0] == 0:
        return
    i_arr = links[:, 0]
    z_arr = links[:, 1]
    y_arr = links[:, 2]
    x_arr = links[:, 3]
    opp_i = OPP3[i_arr]
    f[opp_i, z_arr, y_arr, x_arr] = f_pre[i_arr, z_arr, y_arr, x_arr]


def apply_bounce_back_bouzidi_3d(
    f: np.ndarray,
    f_pre: np.ndarray,
    links: np.ndarray,
    q_vals: np.ndarray,
) -> None:
    """
    Bouzidi (2001) interpolated bounce-back for 3D — 2nd-order at curved walls.

    q < 0.5:  f[opp[i], z, y, x] = 2q*f_pre[i,z,y,x] + (1-2q)*f_pre[opp[i],z,y,x]
    q >= 0.5: f[opp[i], z, y, x] = (1/2q)*f_pre[i,z,y,x]
                                  + (1-1/2q)*f_pre[i, z_nb, y_nb, x_nb]
    """
    if links.shape[0] == 0:
        return

    Nz, Ny, Nx = f.shape[1], f.shape[2], f.shape[3]
    i_arr = links[:, 0]
    z_arr = links[:, 1]
    y_arr = links[:, 2]
    x_arr = links[:, 3]
    opp_i = OPP3[i_arr]
    q = q_vals.astype(np.float64)

    m = q < 0.5
    if m.any():
        f[opp_i[m], z_arr[m], y_arr[m], x_arr[m]] = (
            2.0 * q[m] * f_pre[i_arr[m], z_arr[m], y_arr[m], x_arr[m]]
            + (1.0 - 2.0 * q[m]) * f_pre[opp_i[m], z_arr[m], y_arr[m], x_arr[m]]
        )

    m2 = ~m
    if m2.any():
        zn = np.clip(z_arr[m2] + E3[opp_i[m2], 2].astype(int), 0, Nz - 1)
        yn = np.clip(y_arr[m2] + E3[opp_i[m2], 1].astype(int), 0, Ny - 1)
        xn = np.clip(x_arr[m2] + E3[opp_i[m2], 0].astype(int), 0, Nx - 1)
        inv2q = 1.0 / (2.0 * q[m2])
        f[opp_i[m2], z_arr[m2], y_arr[m2], x_arr[m2]] = (
            inv2q * f_pre[i_arr[m2], z_arr[m2], y_arr[m2], x_arr[m2]]
            + (1.0 - inv2q) * f_pre[i_arr[m2], zn, yn, xn]
        )
