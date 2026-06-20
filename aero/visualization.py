"""
Matplotlib-based visualization for the LBM wind tunnel results.

Saves PNG images and a plain-text coefficient summary to an output directory.
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _mask_solid(arr: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Return copy of arr with solid cells set to NaN (for clean plots)."""
    out = arr.astype(np.float64).copy()
    out[solid] = np.nan
    return out


def plot_velocity(
    ux: np.ndarray,
    uy: np.ndarray,
    solid: np.ndarray,
    u0: float,
    Cd: float,
    Cl: float,
    Re: float,
    step: int,
    out_path: pathlib.Path,
) -> None:
    """
    Velocity magnitude colormap with streamline overlay.
    """
    umag = np.sqrt(ux**2 + uy**2)
    umag_plot = _mask_solid(umag, solid)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        umag_plot,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.6 * u0,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="Velocity magnitude (lattice units)")

    # Streamlines on subsampled grid
    Ny, Nx = ux.shape
    skip = max(1, min(Nx, Ny) // 50)
    xs = np.arange(0, Nx, skip)
    ys = np.arange(0, Ny, skip)
    xx, yy = np.meshgrid(xs, ys)
    ux_s = ux[::skip, ::skip]
    uy_s = uy[::skip, ::skip]
    # Mask solid streamline seeds
    solid_s = solid[::skip, ::skip]
    ux_s = np.where(solid_s, np.nan, ux_s)
    uy_s = np.where(solid_s, np.nan, uy_s)

    ax.streamplot(
        xs.astype(float), ys.astype(float),
        ux_s, uy_s,
        color="white", linewidth=0.5, density=1.2, arrowsize=0.8,
    )

    ax.set_title(
        f"Velocity magnitude — Re={Re:.0f}  Cd={Cd:.3f}  Cl={Cl:.3f}  step={step}",
        fontsize=10,
    )
    ax.set_xlabel("x (lattice cells)")
    ax.set_ylabel("y (lattice cells)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pressure(
    rho: np.ndarray,
    solid: np.ndarray,
    Cd: float,
    Cl: float,
    Re: float,
    step: int,
    out_path: pathlib.Path,
) -> None:
    """
    Pressure field (p = rho / 3) with coolwarm colormap.
    """
    p = _mask_solid(rho / 3.0, solid)
    p_mean = float(np.nanmean(p))

    fig, ax = plt.subplots(figsize=(12, 5))
    dev = float(np.nanstd(p)) * 3 or 0.01
    im = ax.imshow(
        p,
        origin="lower",
        cmap="coolwarm",
        vmin=p_mean - dev,
        vmax=p_mean + dev,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="Pressure (lattice units)")
    ax.set_title(
        f"Pressure field — Re={Re:.0f}  Cd={Cd:.3f}  Cl={Cl:.3f}  step={step}",
        fontsize=10,
    )
    ax.set_xlabel("x (lattice cells)")
    ax.set_ylabel("y (lattice cells)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vorticity(
    ux: np.ndarray,
    uy: np.ndarray,
    solid: np.ndarray,
    Cd: float,
    Cl: float,
    Re: float,
    step: int,
    out_path: pathlib.Path,
) -> None:
    """
    z-vorticity (dux/dy − duy/dx) via central differences.
    """
    # Central differences (interior); boundary rows/cols use forward/backward
    dvdx = np.zeros_like(ux)
    dudy = np.zeros_like(ux)

    dvdx[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) / 2.0
    dvdx[:, 0]    = uy[:, 1] - uy[:, 0]
    dvdx[:, -1]   = uy[:, -1] - uy[:, -2]

    dudy[1:-1, :] = (ux[2:, :] - ux[:-2, :]) / 2.0
    dudy[0, :]    = ux[1, :] - ux[0, :]
    dudy[-1, :]   = ux[-1, :] - ux[-2, :]

    vort = _mask_solid(dudy - dvdx, solid)
    v_lim = float(np.nanpercentile(np.abs(vort), 99)) or 0.01

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        vort,
        origin="lower",
        cmap="RdBu_r",
        vmin=-v_lim,
        vmax=v_lim,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="Vorticity (1/timestep)")
    ax.set_title(
        f"Vorticity field — Re={Re:.0f}  Cd={Cd:.3f}  Cl={Cl:.3f}  step={step}",
        fontsize=10,
    )
    ax.set_xlabel("x (lattice cells)")
    ax.set_ylabel("y (lattice cells)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coefficient_history(
    Cd_history: list[float],
    Cl_history: list[float],
    Re: float,
    out_path: pathlib.Path,
) -> None:
    """
    Time-series plot of Cd and Cl over the full simulation.
    """
    steps = np.arange(1, len(Cd_history) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(steps, Cd_history, color="steelblue", linewidth=0.8)
    ax1.set_ylabel("Cd (drag coefficient)")
    ax1.axhline(np.mean(Cd_history[-len(Cd_history)//5:]), color="red",
                linestyle="--", linewidth=1.0, label="trailing avg")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, Cl_history, color="darkorange", linewidth=0.8)
    ax2.set_ylabel("Cl (lift coefficient)")
    ax2.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Timestep")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Coefficient history — Re={Re:.0f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scalar(
    scalar: np.ndarray,
    solid: np.ndarray,
    Re: float,
    step: int,
    out_path: pathlib.Path,
) -> None:
    """Passive scalar / temperature field colormap."""
    scalar_plot = _mask_solid(scalar, solid)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(scalar_plot, origin="lower", cmap="plasma", aspect="equal")
    plt.colorbar(im, ax=ax, label="Scalar")
    ax.set_title(f"Scalar field — Re={Re:.0f}  step={step}", fontsize=10)
    ax.set_xlabel("x (lattice cells)")
    ax.set_ylabel("y (lattice cells)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_coefficients(
    out_path: pathlib.Path,
    shape: str,
    Re: float,
    steps: int,
    avg_window: int,
    Cd: float,
    Cl: float,
    Cd_std: float,
    Cl_std: float,
) -> None:
    """Write plain-text coefficient summary."""
    text = (
        "# Aero CFD Results\n"
        f"shape       = {shape}\n"
        f"Re          = {Re:.4f}\n"
        f"steps       = {steps}\n"
        f"avg_window  = {avg_window}\n"
        f"Cd          = {Cd:.6f}\n"
        f"Cd_std      = {Cd_std:.6f}\n"
        f"Cl          = {Cl:.6f}\n"
        f"Cl_std      = {Cl_std:.6f}\n"
    )
    out_path.write_text(text)
    print(text)


def save_all(
    result: dict,
    solid: np.ndarray,
    u0: float,
    Re: float,
    shape_name: str,
    steps: int,
    output_dir,
) -> None:
    """
    Generate and save all standard output files for a completed simulation.

    Parameters
    ----------
    result     : dict returned by Solver.run()
    solid      : bool ndarray (Ny, Nx)
    u0         : float — inlet velocity
    Re         : float — Reynolds number
    shape_name : str   — label for filenames / text output
    steps      : int   — total steps run
    output_dir : path  — directory to write files into
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rho = result["rho"]
    ux  = result["ux"]
    uy  = result["uy"]
    Cd  = result["Cd_mean"]
    Cl  = result["Cl_mean"]
    avg_window = max(1, steps // 5)

    tag = f"{shape_name}_re{Re:.0f}"

    plot_velocity(ux, uy, solid, u0, Cd, Cl, Re, steps, out / f"{tag}_velocity.png")
    print(f"  Saved: {out / f'{tag}_velocity.png'}")

    plot_pressure(rho, solid, Cd, Cl, Re, steps, out / f"{tag}_pressure.png")
    print(f"  Saved: {out / f'{tag}_pressure.png'}")

    plot_vorticity(ux, uy, solid, Cd, Cl, Re, steps, out / f"{tag}_vorticity.png")
    print(f"  Saved: {out / f'{tag}_vorticity.png'}")

    plot_coefficient_history(
        result["Cd_history"], result["Cl_history"], Re,
        out / f"{tag}_history.png",
    )
    print(f"  Saved: {out / f'{tag}_history.png'}")

    scalar = result.get("scalar")
    if scalar is not None:
        plot_scalar(scalar, solid, Re, steps, out / f"{tag}_scalar.png")
        print(f"  Saved: {out / f'{tag}_scalar.png'}")

    save_coefficients(
        out / f"{tag}_coefficients.txt",
        shape=shape_name,
        Re=Re,
        steps=steps,
        avg_window=avg_window,
        Cd=Cd,
        Cl=Cl,
        Cd_std=result["Cd_std"],
        Cl_std=result["Cl_std"],
    )
    print(f"  Saved: {out / f'{tag}_coefficients.txt'}")
