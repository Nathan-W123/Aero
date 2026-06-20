"""
3D LBM result visualisation.

Baseline outputs are Matplotlib midplane slice PNGs.
When PyVista is installed, optional 3D artifacts can also be produced:
  - VTK image-data volume export
  - off-screen screenshots of a midplane scalar slice and solid geometry
"""

import pathlib
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    pv = None  # type: ignore[assignment]
    HAS_PYVISTA = False


def _midplane(arr3d: np.ndarray) -> np.ndarray:
    """Return the midplane z-slice of a (Nz, Ny, Nx) array."""
    return arr3d[arr3d.shape[0] // 2]


def build_image_data(result: dict, solid3d: np.ndarray):
    """
    Build a PyVista ImageData volume from simulation outputs.

    The underlying arrays are cell-centered with shape (Nz, Ny, Nx), so the VTK
    point dimensions are (Nx+1, Ny+1, Nz+1).
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")

    Nz, Ny, Nx = solid3d.shape
    grid = pv.ImageData(dimensions=(Nx + 1, Ny + 1, Nz + 1))
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (1.0, 1.0, 1.0)

    rho = np.asarray(result["rho"], dtype=np.float64)
    ux = np.asarray(result["ux"], dtype=np.float64)
    uy = np.asarray(result["uy"], dtype=np.float64)
    uz = np.asarray(result["uz"], dtype=np.float64)
    umag = np.sqrt(ux**2 + uy**2 + uz**2)

    grid.cell_data["rho"] = rho.ravel(order="C")
    grid.cell_data["ux"] = ux.ravel(order="C")
    grid.cell_data["uy"] = uy.ravel(order="C")
    grid.cell_data["uz"] = uz.ravel(order="C")
    grid.cell_data["umag"] = umag.ravel(order="C")
    grid.cell_data["solid"] = solid3d.astype(np.uint8).ravel(order="C")
    return grid


def list_vti_files(output_dir: str = "./outputs3d") -> List[pathlib.Path]:
    out = pathlib.Path(output_dir)
    if not out.exists():
        return []
    return sorted(path for path in out.iterdir() if path.suffix.lower() == ".vti")


def load_volume(path: Union[str, pathlib.Path]):
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")
    return pv.read(path)


def extract_solid_surface(grid):
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")
    solid_mesh = grid.threshold(value=0.5, scalars="solid")
    return solid_mesh.extract_surface().triangulate() if solid_mesh.n_cells > 0 else solid_mesh


def extract_solid_mesh(grid, smooth_iterations: int = 40, relaxation_factor: float = 0.12):
    """Smooth shaded solid geometry for interactive viewing."""
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")
    surface = extract_solid_surface(grid)
    if surface.n_cells == 0:
        return surface
    smoothed = surface.smooth(
        n_iter=smooth_iterations,
        relaxation_factor=relaxation_factor,
        feature_smoothing=False,
        boundary_smoothing=True,
    )
    return smoothed.compute_normals(cell_normals=False, split_vertices=False)


def extract_surface_wireframe(surface_mesh):
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")
    if surface_mesh.n_cells == 0:
        return surface_mesh
    return surface_mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=True,
        manifold_edges=False,
    )


def _cell_dimensions(grid) -> Tuple[int, int, int]:
    dims = grid.dimensions
    return int(dims[0] - 1), int(dims[1] - 1), int(dims[2] - 1)


def _flat_velocity_arrays(grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ux = np.asarray(grid.cell_data["ux"], dtype=np.float64)
    uy = np.asarray(grid.cell_data["uy"], dtype=np.float64)
    uz = np.asarray(grid.cell_data["uz"], dtype=np.float64)
    solid = np.asarray(grid.cell_data["solid"]).astype(bool)
    speed = np.sqrt(ux**2 + uy**2 + uz**2)
    return ux, uy, uz, solid, speed


def build_tunnel_wireframe(grid, divisions: int = 10):
    """Subdivided wind-tunnel style wireframe around the simulation domain."""
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")

    x0, x1, y0, y1, z0, z1 = grid.bounds
    points: List[np.ndarray] = []
    lines: List[int] = []

    def add_segment(p0, p1) -> None:
        start = len(points)
        points.append(np.asarray(p0, dtype=np.float64))
        points.append(np.asarray(p1, dtype=np.float64))
        lines.extend((2, start, start + 1))

    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    for i, j in (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ):
        add_segment(corners[i], corners[j])

    n = max(divisions, 2)
    ys = np.linspace(y0, y1, n)
    zs = np.linspace(z0, z1, n)
    xs = np.linspace(x0, x1, n)

    for y in ys:
        for z in (z0, z1):
            add_segment((x0, y, z), (x1, y, z))
    for z in zs:
        for y in (y0, y1):
            add_segment((x0, y, z), (x1, y, z))
    for x in xs:
        for z in (z0, z1):
            add_segment((x, y0, z), (x, y1, z))
    for z in zs:
        for x in (x0, x1):
            add_segment((x, y0, z), (x, y1, z))

    if not points:
        return pv.PolyData()
    mesh = pv.PolyData(np.vstack(points))
    mesh.lines = np.asarray(lines, dtype=np.int64)
    return mesh


def build_velocity_glyphs(grid, stride: int = 5, scale_factor: float = 14.0, max_points: int = 1800):
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")

    Nx, Ny, Nz = _cell_dimensions(grid)
    ux, uy, uz, solid, speed = _flat_velocity_arrays(grid)
    zz, yy, xx = np.indices((Nz, Ny, Nx))
    base_mask = (
        (xx.ravel(order="C") % max(stride, 1) == 0)
        & (yy.ravel(order="C") % max(stride, 1) == 0)
        & (zz.ravel(order="C") % max(stride, 1) == 0)
        & (~solid)
        & (speed > 1e-10)
    )
    if not np.any(base_mask):
        return pv.PolyData()

    indices = np.flatnonzero(base_mask)
    if indices.size > max_points:
        step = max(1, int(np.ceil(indices.size / max_points)))
        indices = indices[::step]

    centers = grid.cell_centers().points[indices]
    poly = pv.PolyData(centers)
    vectors = np.column_stack([ux[indices], uy[indices], uz[indices]])
    poly["vectors"] = vectors
    poly["speed"] = speed[indices]
    return poly.glyph(orient="vectors", scale="speed", factor=scale_factor)


def build_interactive_scene(path: Union[str, pathlib.Path]):
    """
    Load a volume and build static tunnel + solid meshes for the GUI viewport.

    Must run on the Qt main thread — VTK is not thread-safe.
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")

    grid = load_volume(path)
    return {
        "grid": grid,
        "path": pathlib.Path(path),
        "tunnel": build_tunnel_wireframe(grid, divisions=4),
        "solid": extract_solid_surface(grid),
    }


def load_volume_grid(path: Union[str, pathlib.Path]):
    """Read only the volume grid (first load step)."""
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")
    return load_volume(path)


def build_tunnel_and_solid(grid):
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")
    return build_tunnel_wireframe(grid, divisions=4), extract_solid_surface(grid)


def build_velocity_glyphs_for_grid(
    grid,
    *,
    glyph_stride: int = 5,
    glyph_scale: float = 14.0,
    max_glyphs: int = 900,
):
    """Rebuild only the velocity glyph mesh (for density slider updates)."""
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")
    return build_velocity_glyphs(
        grid,
        stride=max(glyph_stride, 2),
        scale_factor=glyph_scale,
        max_points=max_glyphs,
    )


def fluid_speed_range(grid) -> Tuple[float, float]:
    """Return (vmin, vmax) for fluid speed colormapping."""
    _, _, _, solid, speed = _flat_velocity_arrays(grid)
    fluid = speed[~solid.astype(bool)]
    if fluid.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(fluid, 5))
    vmax = float(np.percentile(fluid, 95))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def compute_flow_advect_dt(grid) -> float:
    """Time step so animated markers visibly traverse the tunnel each second."""
    _, _, _, solid, speed = _flat_velocity_arrays(grid)
    fluid = speed[~solid.astype(bool)]
    fluid = fluid[fluid > 1e-9]
    mean_speed = float(np.mean(fluid)) if fluid.size else 0.05
    x0, x1, _, _, _, _ = grid.bounds
    span = max(float(x1 - x0), 1.0)
    target_step = 0.06 * span
    return max(target_step / max(mean_speed, 1e-6), 2.0)


def seed_inlet_streamlets(
    grid,
    count: int = 160,
    inlet_fraction: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Seed inlet streamlets focused on the object cross-section for wake visualization."""
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")

    rng = rng or np.random.default_rng(0)
    x0, x1, y0, y1, z0, z1 = grid.bounds
    _, _, _, solid, _ = _flat_velocity_arrays(grid)
    n_x, n_y, n_z = _cell_dimensions(grid)
    solid3d = solid.reshape(n_z, n_y, n_x)

    coords_y = y0 + (y1 - y0) * (np.arange(n_y) + 0.5) / max(n_y, 1)
    coords_z = z0 + (z1 - z0) * (np.arange(n_z) + 0.5) / max(n_z, 1)
    wy = np.any(solid3d, axis=(0, 2)).astype(np.float64)
    wz = np.any(solid3d, axis=(1, 2)).astype(np.float64)
    cy = float(np.average(coords_y, weights=wy + 1e-9))
    cz = float(np.average(coords_z, weights=wz + 1e-9))

    x_in = x0 + inlet_fraction * max(x1 - x0, 1.0)
    side = int(np.ceil(np.sqrt(count)))
    ys = np.linspace(y0 + 0.1 * (y1 - y0), y1 - 0.1 * (y1 - y0), side)
    zs = np.linspace(z0 + 0.1 * (z1 - z0), z1 - 0.1 * (z1 - z0), side)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    flat_y = yy.ravel()
    flat_z = zz.ravel()
    n_pts = min(count, flat_y.size)

    points = np.empty((n_pts, 3), dtype=np.float64)
    points[:, 0] = x_in + rng.uniform(0.0, 0.35 * max(x1 - x0, 1.0), size=n_pts)
    points[:, 1] = 0.65 * flat_y[:n_pts] + 0.35 * cy + rng.normal(0.0, 0.08 * (y1 - y0), n_pts)
    points[:, 2] = 0.65 * flat_z[:n_pts] + 0.35 * cz + rng.normal(0.0, 0.08 * (z1 - z0), n_pts)
    points[:, 1] = np.clip(points[:, 1], y0 + 0.05 * (y1 - y0), y1 - 0.05 * (y1 - y0))
    points[:, 2] = np.clip(points[:, 2], z0 + 0.05 * (z1 - z0), z1 - 0.05 * (z1 - z0))
    return points


def seed_particles(
    grid,
    count: int = 240,
    inlet_fraction: float = 0.08,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")

    rng = rng or np.random.default_rng(0)
    x0, x1, y0, y1, z0, z1 = grid.bounds
    y_pad = 0.08 * max(y1 - y0, 1.0)
    z_pad = 0.08 * max(z1 - z0, 1.0)
    x_seed = x0 + inlet_fraction * max(x1 - x0, 1.0)

    points = np.empty((count, 3), dtype=np.float64)
    points[:, 0] = x_seed
    points[:, 1] = rng.uniform(y0 + y_pad, y1 - y_pad, size=count)
    points[:, 2] = rng.uniform(z0 + z_pad, z1 - z_pad, size=count)
    return points


def sample_velocity(grid, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")

    Nx, Ny, Nz = _cell_dimensions(grid)
    ux, uy, uz, solid, speed = _flat_velocity_arrays(grid)
    origin = np.asarray(grid.origin, dtype=np.float64)
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    cell_coords = np.floor((points - origin) / spacing).astype(int)
    cell_coords[:, 0] = np.clip(cell_coords[:, 0], 0, Nx - 1)
    cell_coords[:, 1] = np.clip(cell_coords[:, 1], 0, Ny - 1)
    cell_coords[:, 2] = np.clip(cell_coords[:, 2], 0, Nz - 1)
    flat = cell_coords[:, 0] + Nx * (cell_coords[:, 1] + Ny * cell_coords[:, 2])
    vectors = np.column_stack([ux[flat], uy[flat], uz[flat]])
    return vectors, speed[flat], solid[flat]


def advect_particles(
    grid,
    points: np.ndarray,
    dt: float = 0.9,
    inlet_fraction: float = 0.08,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")

    rng = rng or np.random.default_rng(0)
    vectors, _, solid = sample_velocity(grid, points)
    next_points = points + dt * vectors
    x0, x1, y0, y1, z0, z1 = grid.bounds
    out_of_bounds = (
        (next_points[:, 0] > x1)
        | (next_points[:, 1] < y0)
        | (next_points[:, 1] > y1)
        | (next_points[:, 2] < z0)
        | (next_points[:, 2] > z1)
    )
    respawn = out_of_bounds | solid
    if np.any(respawn):
        next_points[respawn] = seed_particles(
            grid,
            count=int(np.count_nonzero(respawn)),
            inlet_fraction=inlet_fraction,
            rng=rng,
        )
    vectors, speed, _ = sample_velocity(grid, next_points)
    return next_points, vectors, speed


def build_particle_glyphs(points: np.ndarray, vectors: np.ndarray, speed: np.ndarray, scale_factor: float = 18.0):
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")

    poly = pv.PolyData(points)
    poly["vectors"] = vectors
    poly["speed"] = speed
    return poly.glyph(orient="vectors", scale="speed", factor=scale_factor)


def build_flow_lines(
    points: np.ndarray,
    vectors: np.ndarray,
    speed: np.ndarray,
    *,
    scale: float = 0.55,
):
    """Lightweight velocity segments for real-time animation in the GUI."""
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed.")

    if len(points) == 0:
        return pv.PolyData()

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    ends = points + scale * vectors / norms
    n = len(points)
    all_points = np.vstack([points, ends])
    lines = np.empty(n * 3, dtype=np.int64)
    for i in range(n):
        base = i * 3
        lines[base] = 2
        lines[base + 1] = i
        lines[base + 2] = n + i
    poly = pv.PolyData(all_points)
    poly.lines = lines
    poly["speed"] = np.concatenate([speed, speed])
    return poly


def save_slices(
    result: dict,
    solid3d: np.ndarray,
    u0: float,
    Re: float,
    shape_name: str,
    steps: int,
    output_dir: str = "./outputs3d",
) -> List[pathlib.Path]:
    """
    Save midplane slice PNGs for a 3D LBM result.

    Returns the paths written.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[pathlib.Path] = []

    ux = _midplane(result["ux"])
    uy = _midplane(result["uy"])
    rho = _midplane(result["rho"])
    solid = _midplane(solid3d)

    Ny, Nx = ux.shape
    umag = np.sqrt(ux**2 + uy**2)
    umag[solid] = np.nan
    ux_plot = ux.copy()
    ux_plot[solid] = np.nan
    uy_plot = uy.copy()
    uy_plot[solid] = np.nan

    prefix = f"{shape_name}_re{Re:.0f}"
    Cd = result.get("Cd_mean", float("nan"))
    Cly = result.get("Cly_mean", float("nan"))

    # velocity
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(umag, origin="lower", cmap="viridis", vmin=0, vmax=1.4 * u0)
    plt.colorbar(im, ax=ax, label="|u| (lattice units)")
    x_g = np.arange(Nx)
    y_g = np.arange(Ny)
    ax.streamplot(x_g, y_g, ux_plot, uy_plot, color="white", linewidth=0.5, density=1.2)
    ax.set_title(f"3D {shape_name} midplane  Re={Re:.0f}  Cd={Cd:.3f}  Cly={Cly:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    path = out / f"{prefix}_slice_velocity.png"
    fig.savefig(path, dpi=120)
    written.append(path)
    plt.close(fig)

    # pressure
    pres = rho / 3.0
    pres[solid] = np.nan
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pres, origin="lower", cmap="coolwarm")
    plt.colorbar(im, ax=ax, label="p = rho/3")
    ax.set_title(f"3D {shape_name} midplane pressure  Re={Re:.0f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    path = out / f"{prefix}_slice_pressure.png"
    fig.savefig(path, dpi=120)
    written.append(path)
    plt.close(fig)

    # vorticity
    dux_dy = np.gradient(ux_plot, axis=0)
    duy_dx = np.gradient(uy_plot, axis=1)
    vort = dux_dy - duy_dx
    vort[solid] = np.nan
    vmax = float(np.nanpercentile(np.abs(vort), 99)) or 0.01
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(vort, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="ωz = dux/dy − duy/dx")
    ax.set_title(f"3D {shape_name} midplane vorticity  Re={Re:.0f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    path = out / f"{prefix}_slice_vorticity.png"
    fig.savefig(path, dpi=120)
    written.append(path)
    plt.close(fig)

    # force history
    Cd_hist = result.get("Cd_history", [])
    if Cd_hist:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(Cd_hist, lw=0.8, color="steelblue", label="Cd")
        Cly_hist = result.get("Cly_history", [])
        if Cly_hist:
            ax.plot(Cly_hist, lw=0.8, color="tomato", label="Cl_y")
        ax.axhline(Cd, color="steelblue", ls="--", lw=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Coefficient")
        ax.set_title(f"3D {shape_name}  Re={Re:.0f}  force history")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out / f"{prefix}_cd_history.png"
        fig.savefig(path, dpi=120)
        written.append(path)
        plt.close(fig)

    return written


def save_pyvista_artifacts(
    result: dict,
    solid3d: np.ndarray,
    u0: float,
    Re: float,
    shape_name: str,
    output_dir: str = "./outputs3d",
    export_vtk: bool = False,
    render_images: bool = True,
) -> List[pathlib.Path]:
    """
    Save optional PyVista-based artifacts.

    Produces an off-screen 3D screenshot set when PyVista is installed.
    If `export_vtk` is True, also writes a `.vti` volume file.
    """
    if not HAS_PYVISTA:
        return []

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[pathlib.Path] = []
    prefix = f"{shape_name}_re{Re:.0f}"

    grid = build_image_data(result, solid3d)

    if export_vtk:
        path = out / f"{prefix}_volume.vti"
        grid.save(path)
        written.append(path)

    if render_images:
        solid_mesh = grid.threshold(value=0.5, scalars="solid")
        z_mid = solid3d.shape[0] / 2.0
        slice_mesh = grid.slice(
            normal=(0.0, 0.0, 1.0),
            origin=(solid3d.shape[2] / 2.0, solid3d.shape[1] / 2.0, z_mid),
        )

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(slice_mesh, scalars="umag", cmap="viridis", clim=(0.0, max(1.4 * u0, 1e-6)))
        if solid_mesh.n_cells > 0:
            plotter.add_mesh(solid_mesh.outline(), color="black", line_width=2)
        plotter.view_isometric()
        plotter.add_axes()
        path = out / f"{prefix}_pyvista_slice.png"
        plotter.screenshot(path)
        written.append(path)
        plotter.close()

        plotter = pv.Plotter(off_screen=True)
        if solid_mesh.n_cells > 0:
            plotter.add_mesh(solid_mesh, color="lightgray", smooth_shading=False, show_edges=False)
        plotter.view_isometric()
        plotter.add_axes()
        path = out / f"{prefix}_pyvista_solid.png"
        plotter.screenshot(path)
        written.append(path)
        plotter.close()

    return written


def save_all_3d(
    result: dict,
    solid3d: np.ndarray,
    u0: float,
    Re: float,
    shape_name: str,
    steps: int,
    output_dir: str = "./outputs3d",
    viz_mode: str = "auto",
    export_vtk: bool = False,
) -> List[pathlib.Path]:
    """
    Save all requested 3D visualisation artifacts.

    Modes
    -----
    auto     : slice PNGs always, PyVista extras when available
    slices   : slice PNGs only
    pyvista  : PyVista extras only (falls back to no output if unavailable)
    all      : slice PNGs + PyVista extras
    """
    if viz_mode not in ("auto", "slices", "pyvista", "all"):
        raise ValueError(f"Unknown viz_mode '{viz_mode}'.")

    written: List[pathlib.Path] = []

    want_slices = viz_mode in ("auto", "slices", "all")
    want_pyvista = viz_mode in ("all", "pyvista") or (viz_mode == "auto" and HAS_PYVISTA)

    if want_slices:
        written.extend(save_slices(result, solid3d, u0, Re, shape_name, steps, output_dir))

    if want_pyvista:
        written.extend(save_pyvista_artifacts(result, solid3d, u0, Re, shape_name, output_dir, export_vtk))

    return written
