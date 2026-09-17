"""
Boundary conditions for the 3D LBM solver (D3Q19 or D3Q27).

Applied each timestep in this order (after streaming):
  1. Mid-link bounce-back   (obstacle surface)
  2. Inlet BC               (left face, x=0)   — Zou-He velocity
  3. Outlet BC              (right face, x=Nx-1) — convective or zero-gradient
  4. Wall BC                (top y=Ny-1, bottom y=0) — slip or no-slip
  (z direction: periodic by default via streaming)

f array layout: (Q, Nz, Ny, Nx)

Every routine here is written against a :class:`~aero.lbm.lattice3d.Lattice3D`
descriptor rather than against hard-coded direction indices, and defaults to
D3Q19.  That is not tidiness for its own sake: a 27-entry index table written
out by hand is exactly the kind of thing whose transcription errors survive
review and then show up as a slow momentum leak.
"""

from typing import Optional, Tuple

import numpy as np

from .lattice3d import D3Q19, Lattice3D, compute_feq, compute_macroscopic


# ---------------------------------------------------------------------------
# Inlet: Zou-He 3D velocity BC at x=0
# ---------------------------------------------------------------------------

def _zou_he_x_inlet(
    f: np.ndarray,
    ux_t: np.ndarray,
    uy_t: np.ndarray,
    uz_t: np.ndarray,
    lat: Lattice3D,
    c: int = 0,
) -> np.ndarray:
    """
    Impose ``(ux, uy, uz) = (ux_t, uy_t, uz_t)`` on the x-normal face ``c``.

    The unknowns after streaming are the directions with ``ex > 0``; everything
    else arrived from the interior.  Three steps, each exact:

    1.  Density from the mass and x-momentum closure.  Splitting f by the sign
        of ``ex`` gives ``rho = (2 F_minus + F_zero) / (1 - ux)``, independent
        of the transverse target.

    2.  Non-equilibrium bounce-back for each unknown,
        ``f_i = f_opp(i) + 6 w_i rho (e_i . u)``, the even part of the
        equilibrium cancelling in the difference.  This lands mass and
        x-momentum exactly, because ``sum_{ex>0} w_i = 1/6`` for any lattice
        whose x-components are in {-1, 0, 1}.

    3.  Transverse momentum.  Step 2 delivers only ``rho u_y / 3`` of the
        target and leaves the ex=0 plane's own transverse momentum in place,
        so the shortfall is spread back over the unknowns as
        ``delta_i = lambda w_i e_iy``.  That shape is forced: it is the only
        one that is odd in ``e_y`` (so it cannot disturb mass or x-momentum)
        while still carrying y-momentum.  The normalisation is
        ``S = sum_{ex>0} w_i e_iy^2``, computed from the lattice.

    Doing it this way rather than from a table is what lets D3Q27 work here
    unchanged, where the unknown set is nine directions rather than five and
    three of them carry ``e_y > 0`` instead of one.
    """
    U = lat.x_plus
    opp_u = lat.OPP[U].astype(np.intp)
    ex, ey, ez, w = lat.ex, lat.ey, lat.ez, lat.W

    known_minus = f[lat.x_minus, :, :, c].sum(axis=0)
    known_zero = f[lat.x_zero, :, :, c].sum(axis=0)
    rho = (2.0 * known_minus + known_zero) / np.clip(1.0 - ux_t, 1e-10, None)

    # transverse momentum already carried by the directions that stay put
    zero_idx = lat.x_zero
    cy = np.einsum("i,izy->zy", ey[zero_idx], f[zero_idx, :, :, c])
    cz = np.einsum("i,izy->zy", ez[zero_idx], f[zero_idx, :, :, c])

    for k, i in enumerate(U):
        eu = ex[i] * ux_t + ey[i] * uy_t + ez[i] * uz_t
        f[i, :, :, c] = f[opp_u[k], :, :, c] + 6.0 * w[i] * rho * eu

    s = lat.transverse_norm
    lam_y = ((2.0 / 3.0) * rho * uy_t - cy) / s
    lam_z = ((2.0 / 3.0) * rho * uz_t - cz) / s
    for i in U:
        if ey[i] != 0.0:
            f[i, :, :, c] += lam_y * w[i] * ey[i]
        if ez[i] != 0.0:
            f[i, :, :, c] += lam_z * w[i] * ez[i]
    return rho


def apply_inlet_zou_he_3d(
    f: np.ndarray,
    u0: float,
    *,
    uy_amp: float = 0.0,
    uz_amp: float = 0.0,
    step: int = 0,
    lattice: Lattice3D = D3Q19,
) -> None:
    """
    3D Zou-He velocity BC at left face (x=0): impose ux=u0, uy≈0, uz≈0.

    Optional spanwise uz_amp and vertical uy_amp perturbations (fraction of u0)
    trigger 3D shedding at supercritical Re.
    """
    nz, ny = f.shape[1], f.shape[2]
    shape = (nz, ny)
    ux_t = np.full(shape, float(u0))
    uy_t = np.zeros(shape)
    uz_t = np.zeros(shape)

    if uy_amp > 0.0 or uz_amp > 0.0:
        phase = 0.17 * step
        if uy_amp > 0.0:
            yy = np.arange(ny, dtype=np.float64)
            uy_t += uy_amp * u0 * np.sin(2.0 * np.pi * yy / max(ny, 1) + phase)
        if uz_amp > 0.0:
            zz = np.arange(nz, dtype=np.float64)
            uz_t += (
                uz_amp * u0
                * np.sin(2.0 * np.pi * zz / max(nz, 1) + phase)
            )[:, None]

    _zou_he_x_inlet(f, ux_t, uy_t, uz_t, lattice)


def apply_inlet_velocity_field_3d(
    f: np.ndarray,
    ux_target: np.ndarray,
    uy_target: np.ndarray,
    uz_target: np.ndarray,
    *,
    lattice: Lattice3D = D3Q19,
) -> None:
    """3D Zou-He inlet with per-cell target velocity fields at x=0."""
    _zou_he_x_inlet(
        f,
        np.asarray(ux_target, dtype=np.float64),
        np.asarray(uy_target, dtype=np.float64),
        np.asarray(uz_target, dtype=np.float64),
        lattice,
    )


def apply_inlet_sem_3d(
    f: np.ndarray,
    u0: float,
    u_prime: np.ndarray,
    v_prime: np.ndarray,
    w_prime: np.ndarray,
    *,
    lattice: Lattice3D = D3Q19,
) -> None:
    """
    Zou-He 3D velocity BC at x=0 with SEM fluctuations.

    Imposes (ux, uy, uz) = (u0 + u_prime, v_prime, w_prime) on the inlet face.

    Parameters
    ----------
    u_prime, v_prime, w_prime : (Nz, Ny) arrays from SEMInlet.fluctuation()
    """
    _zou_he_x_inlet(
        f,
        u0 + np.asarray(u_prime, dtype=np.float64),
        np.asarray(v_prime, dtype=np.float64),
        np.asarray(w_prime, dtype=np.float64),
        lattice,
    )


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
    lattice: Lattice3D = D3Q19,
) -> None:
    """
    Recycle the downstream plane to the inlet and rescale its mean ux to `target_u0`.

    This preserves cross-stream structure and turbulence content better than a
    uniform inlet while remaining lightweight enough for the current solver.
    """
    rho, ux, uy, uz = compute_macroscopic(f, lattice)
    recycle_rho = rho[:, :, recycle_index]
    recycle_ux = ux[:, :, recycle_index]
    recycle_uy = uy[:, :, recycle_index]
    recycle_uz = uz[:, :, recycle_index]
    mean_ux = float(np.mean(recycle_ux))
    scale = 1.0 if abs(mean_ux) < 1e-12 else float(target_u0) / mean_ux
    feq = compute_feq(
        recycle_rho[:, :, None],
        (recycle_ux * scale)[:, :, None],
        (recycle_uy * scale)[:, :, None],
        (recycle_uz * scale)[:, :, None],
        lattice,
    )
    f[:, :, :, inlet_index] = feq[:, :, :, 0]


# ---------------------------------------------------------------------------
# Wall BCs: top (y=Ny-1) and bottom (y=0)
# ---------------------------------------------------------------------------

def apply_slip_walls_3d(f: np.ndarray, *, lattice: Lattice3D = D3Q19) -> None:
    """
    Specular reflection at top and bottom walls (preserve ex,ez, flip ey).

    The unknowns at the bottom wall are the directions with ``ey > 0``; each
    takes the value of its y-mirror image.  For D3Q19 that reproduces the pairs
    (3<->4), (7<->9), (8<->10), (15<->16), (17<->18); D3Q27 adds the corners.
    """
    up = lattice.y_plus
    dn = lattice.y_minus
    mir = lattice.Y_MIR.astype(np.intp)

    # Bottom wall (y=0): incoming ey<0 directions → reflect to ey>0
    f[up, :, 0, :] = f[mir[up], :, 0, :]
    # Top wall (y=Ny-1): incoming ey>0 directions → reflect to ey<0
    f[dn, :, -1, :] = f[mir[dn], :, -1, :]


def apply_noslip_walls_3d(f: np.ndarray, *, lattice: Lattice3D = D3Q19) -> None:
    """
    Full bounce-back (no-slip) at top and bottom walls.

    Reverses all velocity components: f[i] ← f[opp[i]] for the unknowns.

    Bounce-back differs from slip exactly in the diagonals, where the full
    reversal also flips ex and ez:
      slip:   ex,ez fixed, ey flipped  (7<->9, 8<->10, 15<->16, 17<->18)
      noslip: full reversal via OPP    (7<->10, 8<->9, 15<->18, 16<->17)
    """
    up = lattice.y_plus
    dn = lattice.y_minus
    opp = lattice.OPP.astype(np.intp)

    f[up, :, 0, :] = f[opp[up], :, 0, :]
    f[dn, :, -1, :] = f[opp[dn], :, -1, :]


def apply_moving_walls_3d(
    f: np.ndarray,
    *,
    u_top: float = 0.0,
    u_bottom: float = 0.0,
    lattice: Lattice3D = D3Q19,
) -> None:
    """
    Moving-wall bounce-back with tangential x-velocity on top/bottom y-walls.

    Bounce-back plus the momentum the wall injects,
    ``f_i = f_opp(i) + 6 w_i rho (e_i . u_wall)``, which for a purely
    streamwise wall velocity is non-zero only on directions with ``ex != 0``.
    """
    up = lattice.y_plus
    dn = lattice.y_minus
    opp = lattice.OPP.astype(np.intp)
    w, ex = lattice.W, lattice.ex

    rho_bottom = np.maximum(np.sum(f[:, :, 0, :], axis=0), 1e-12)
    rho_top = np.maximum(np.sum(f[:, :, -1, :], axis=0), 1e-12)

    f[up, :, 0, :] = (
        f[opp[up], :, 0, :]
        + 6.0 * (w[up] * ex[up])[:, None, None] * rho_bottom * float(u_bottom)
    )
    f[dn, :, -1, :] = (
        f[opp[dn], :, -1, :]
        + 6.0 * (w[dn] * ex[dn])[:, None, None] * rho_top * float(u_top)
    )


# ---------------------------------------------------------------------------
# Obstacle: mid-link bounce-back
# ---------------------------------------------------------------------------

def build_surface_links_3d(
    solid: np.ndarray,
    phi: Optional[np.ndarray] = None,
    *,
    lattice: Lattice3D = D3Q19,
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
    z_line = np.arange(Nz)
    y_line = np.arange(Ny)
    x_line = np.arange(Nx)

    per_dir = []
    for i in range(1, lattice.Q):
        ex_, ey_, ez_ = (int(v) for v in lattice.E[i])
        # Clamped (not wrapped) neighbour indices — see the 2D builder.
        zn = np.clip(z_line + ez_, 0, Nz - 1)
        yn = np.clip(y_line + ey_, 0, Ny - 1)
        xn = np.clip(x_line + ex_, 0, Nx - 1)
        mask = fluid & solid[zn[:, None, None], yn[None, :, None], xn[None, None, :]]
        zs, ys, xs = np.nonzero(mask)
        if zs.size:
            per_dir.append(
                np.stack(
                    (np.full(zs.size, i, dtype=np.int32),
                     zs.astype(np.int32),
                     ys.astype(np.int32),
                     xs.astype(np.int32)),
                    axis=1,
                )
            )

    if not per_dir:
        return np.empty((0, 4), dtype=np.int32), np.empty(0, dtype=np.float32)

    links_arr = np.concatenate(per_dir, axis=0)
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
    *,
    lattice: Lattice3D = D3Q19,
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
    opp_i = lattice.OPP[i_arr]
    f[opp_i, z_arr, y_arr, x_arr] = f_pre[i_arr, z_arr, y_arr, x_arr]


def apply_bounce_back_bouzidi_3d(
    f: np.ndarray,
    f_pre: np.ndarray,
    links: np.ndarray,
    q_vals: np.ndarray,
    *,
    lattice: Lattice3D = D3Q19,
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
    opp_i = lattice.OPP[i_arr]
    q = q_vals.astype(np.float64)

    m = q < 0.5
    if m.any():
        f[opp_i[m], z_arr[m], y_arr[m], x_arr[m]] = (
            2.0 * q[m] * f_pre[i_arr[m], z_arr[m], y_arr[m], x_arr[m]]
            + (1.0 - 2.0 * q[m]) * f_pre[opp_i[m], z_arr[m], y_arr[m], x_arr[m]]
        )

    m2 = ~m
    if m2.any():
        zn = np.clip(z_arr[m2] + lattice.E[opp_i[m2], 2].astype(int), 0, Nz - 1)
        yn = np.clip(y_arr[m2] + lattice.E[opp_i[m2], 1].astype(int), 0, Ny - 1)
        xn = np.clip(x_arr[m2] + lattice.E[opp_i[m2], 0].astype(int), 0, Nx - 1)
        inv2q = 1.0 / (2.0 * q[m2])
        f[opp_i[m2], z_arr[m2], y_arr[m2], x_arr[m2]] = (
            inv2q * f_pre[i_arr[m2], z_arr[m2], y_arr[m2], x_arr[m2]]
            + (1.0 - inv2q) * f_pre[i_arr[m2], zn, yn, xn]
        )
