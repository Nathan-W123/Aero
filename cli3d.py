#!/usr/bin/env python3
"""
Aero CFD — 3D LBM Wind Tunnel Simulator (D3Q19)
CLI entry point.

Usage
-----
  python3 cli3d.py --shape sphere --re 100
  python3 cli3d.py --shape sphere --re 100 --nz 64 --ny 64 --nx 128 --steps 5000
  python3 cli3d.py --shape box --width 10 --height 10 --depth 10 --re 100
  python3 cli3d.py --backend numpy --shape sphere --re 50 --no-plot
  python3 cli3d.py --shape sphere --re 100 --checkpoint-every 5000 --checkpoint-dir ./ckpt3d
"""

import argparse
import math
import pathlib
import sys
import time
from typing import Tuple

from aero.geometry3d.sphere import Sphere
from aero.geometry3d.box import Box
from aero.geometry3d.cylinder3d import Cylinder3D
from aero.benchmarks import build_uncertainty_report, build_validation_report
from aero.autoconfig import autoconfigure_3d
from aero.lbm.solver3d import Solver3D
from aero.case import SimulationCase
from aero.visualization3d import save_all_3d, HAS_PYVISTA


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="3D LBM D3Q19 Wind Tunnel CFD Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shape
    p.add_argument("--shape", choices=["sphere", "box", "cylinder", "mesh"], default="sphere")
    p.add_argument("--radius", type=float, default=10.0,
                   help="Sphere radius (lattice cells)")
    p.add_argument("--width",  type=float, default=10.0,
                   help="Box streamwise width (lattice cells)")
    p.add_argument("--height", type=float, default=10.0,
                   help="Box vertical height (lattice cells)")
    p.add_argument("--depth",  type=float, default=10.0,
                   help="Box spanwise depth (lattice cells)")
    p.add_argument("--length", type=float, default=20.0,
                   help="Cylinder spanwise length (lattice cells)")
    p.add_argument("--stl-path", type=str, default=None,
                   help="Path to STL file for --shape mesh")
    p.add_argument("--stl-fit", type=float, default=0.35,
                   help="Max cross-stream mesh extent as fraction of min(Ny,Nz)")
    p.add_argument("--mesh-orient", choices=["auto", "none"], default="auto",
                   help="Auto-orient STL: stream +x, span +y, thin +z (PCA)")
    p.add_argument("--mesh-rot-x", type=float, default=0.0,
                   help="Extra mesh rotation about stream +x axis (degrees)")
    p.add_argument("--mesh-rot-y", type=float, default=0.0,
                   help="Extra mesh rotation about vertical +y axis (degrees)")
    p.add_argument("--mesh-rot-z", type=float, default=0.0,
                   help="Extra mesh rotation about span +z axis (degrees)")
    p.add_argument("--cx-frac", type=float, default=1.0/3.0,
                   help="Obstacle centre x as fraction of Nx")
    p.add_argument("--cy-frac", type=float, default=0.5)
    p.add_argument("--cz-frac", type=float, default=0.5)

    # Flow
    p.add_argument("--re",  type=float, default=100.0, help="Reynolds number")
    p.add_argument("--u0",  type=float, default=0.05,  help="Inlet velocity (lattice units)")

    # Grid
    p.add_argument("--nx", type=int, default=128, help="Streamwise cells")
    p.add_argument("--ny", type=int, default=64,  help="Vertical cells")
    p.add_argument("--nz", type=int, default=64,  help="Spanwise cells (periodic)")

    # BCs
    p.add_argument("--wall-bc", choices=["slip", "noslip", "moving"], default="slip")
    p.add_argument("--outlet-bc", choices=["convective", "zerogradient"], default="convective")
    p.add_argument("--streamwise-bc", choices=["open", "periodic", "recycling"], default="open")

    # Backend
    p.add_argument("--backend", choices=["auto", "numpy", "numba"], default="auto")
    p.add_argument("--collision", choices=["bgk", "mrt", "trt"], default="bgk")
    p.add_argument("--trt-lambda", type=float, default=0.25)
    p.add_argument("--inlet-perturbation", type=float, default=0.0,
                   help="Inlet uz perturbation amplitude (fraction of u0)")
    p.add_argument("--sponge-cells", type=int, default=0)
    p.add_argument("--sponge-strength", type=float, default=0.1)
    p.add_argument("--les", action="store_true")
    p.add_argument("--les-cs", type=float, default=0.16)
    p.add_argument("--les-model", choices=["smagorinsky", "wale"], default="smagorinsky")
    p.add_argument("--bouzidi", action="store_true", help="Use Bouzidi 2nd-order curved bounce-back")
    p.add_argument("--van-driest", action="store_true", help="Van Driest wall damping for LES (requires --les)")
    p.add_argument("--van-driest-A", type=float, default=25.0, help="Van Driest A+ constant (default 25)")
    p.add_argument("--export-hdf5", type=str, default=None, metavar="PATH",
                   help="Write HDF5+XDMF time series to PATH (requires h5py)")
    p.add_argument("--hdf5-every", type=int, default=None, metavar="N",
                   help="Write HDF5 snapshot every N steps")
    p.add_argument("--mesh-bc", choices=["voxel", "ibm"], default="voxel",
                   help="STL mesh boundary: voxel bounce-back or Guo IBM")
    p.add_argument("--body-force-x", type=float, default=0.0,
                   help="Uniform streamwise body force for periodic/internal-flow cases")
    p.add_argument("--body-force-y", type=float, default=0.0)
    p.add_argument("--body-force-z", type=float, default=0.0)
    p.add_argument("--wall-velocity-top", type=float, default=0.0)
    p.add_argument("--wall-velocity-bottom", type=float, default=0.0)
    p.add_argument("--synthetic-inflow", action="store_true",
                   help="Use filtered synthetic inlet fluctuations on the open inlet")
    p.add_argument("--synthetic-inflow-intensity", type=float, default=0.03)
    p.add_argument("--scalar", "--thermal", dest="thermal", action="store_true",
                   help="Enable passive scalar / thermal transport")
    p.add_argument("--scalar-hot", dest="T_hot", type=float, default=1.0,
                   help="Bottom-wall scalar value")
    p.add_argument("--scalar-cold", dest="T_cold", type=float, default=0.0,
                   help="Top-wall scalar value")
    p.add_argument("--scalar-diffusivity", dest="alpha_T", type=float, default=1e-3,
                   help="Passive scalar diffusivity")
    p.add_argument("--scalar-ref", dest="T_ref", type=float, default=0.5,
                   help="Reference scalar for buoyancy coupling")
    p.add_argument("--buoyancy", action="store_true",
                   help="Enable Boussinesq buoyancy coupling on the scalar field")
    p.add_argument("--gravity", dest="g_gravity", type=float, default=0.0,
                   help="Buoyancy gravity magnitude")
    p.add_argument("--beta", type=float, default=1e-3,
                   help="Buoyancy expansion coefficient")
    # SEM synthetic turbulence inlet
    p.add_argument("--sem-inlet", action="store_true",
                   help="Enable Synthetic Eddy Method turbulent inlet")
    p.add_argument("--sem-tu", type=float, default=0.05,
                   help="SEM turbulence intensity (RMS/u0, default 0.05)")
    p.add_argument("--sem-lint", type=float, default=10.0,
                   help="SEM integral length scale in lattice units (default 10)")
    p.add_argument("--sem-n", type=int, default=200,
                   help="Number of SEM synthetic eddies (default 200)")
    p.add_argument("--allow-high-blockage", action="store_true",
                   help="Allow very confined runs (>30% frontal blockage)")
    p.add_argument("--autoconfigure", choices=["off", "safe"], default="off",
                   help="Automatically adjust recoverable preflight settings")
    # Multi-block static z-refinement
    p.add_argument("--refine-z-lo", type=int, default=None,
                   help="Start of 2× refined z-slab (coarse grid index)")
    p.add_argument("--refine-z-hi", type=int, default=None,
                   help="End of 2× refined z-slab (exclusive, coarse grid index)")

    # Run control
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--check-every", type=int, default=500)
    p.add_argument("--auto-stop", action="store_true",
                   help="Stop when rolling Cd and Strouhal metrics become stationary")

    # Checkpoint
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--checkpoint-dir",   type=str, default=None)
    p.add_argument("--resume-from",      type=str, default=None)

    # Output
    p.add_argument("--output", type=str, default="./outputs3d")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--viz3d", choices=["auto", "slices", "pyvista", "all"], default="auto",
                   help="3D visualisation mode: slices, pyvista, all, or auto")
    p.add_argument("--export-vtk", action="store_true",
                   help="Export a VTK image-data volume (.vti) when PyVista is available")
    p.add_argument("--verbose", action="store_true", default=True)

    p.add_argument("--save-case", action="store_true",
                   help="Save config + results as a named case in ./cases/")
    p.add_argument("--cases-root", type=str, default="./cases",
                   help="Root directory for all saved cases")

    return p


def compute_blockage_metrics(args: argparse.Namespace, D: float) -> Tuple[float, float, float]:
    """
    Return blockage ratios in the two cross-stream directions and their max.

    Flow is along x, so blockage is assessed on the frontal y-z plane.
    """
    if args.shape == "sphere":
        frontal_y = D
        frontal_z = D
    elif args.shape == "box":
        frontal_y = args.height
        frontal_z = args.depth
    elif args.shape == "cylinder":
        frontal_y = D
        frontal_z = args.length
    elif args.shape == "mesh":
        fit = getattr(args, "stl_fit", 0.35)
        frontal_y = args.ny * fit
        frontal_z = args.nz * fit
    else:
        frontal_y = D
        frontal_z = D

    by = frontal_y / args.ny
    bz = frontal_z / args.nz
    return by, bz, max(by, bz)


def assess_blockage(blockage: float, allow_high_blockage: bool) -> tuple[str, bool]:
    """
    Classify blockage and decide whether the run should be blocked.

    Returns
    -------
    label : str
        Human-readable assessment.
    blocked : bool
        True when the run should not proceed without explicit override.
    """
    if blockage <= 0.10:
        return "good", False
    if blockage <= 0.20:
        return "elevated", False
    if blockage <= 0.30:
        return "high", False
    if allow_high_blockage:
        return "very high (override)", False
    return "very high", True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    geom = None
    solid = None
    phi_field = None
    ibm_enabled = False
    mesh_blockage_tuple = None

    autoconfig_report = autoconfigure_3d(args, policy=args.autoconfigure)
    if autoconfig_report is not None and autoconfig_report.applied:
        print(f"[i] Autoconfigure ({autoconfig_report.policy}) applied {len(autoconfig_report.changes)} change(s):")
        for item in autoconfig_report.changes:
            print(f"    - {item.field}: {item.old} -> {item.new} ({item.reason})")

    # Derived LBM parameters (D may come from voxelized mesh)
    if args.shape == "mesh":
        if not args.stl_path:
            print("[X] --stl-path required for --shape mesh")
            return 1
        from aero.geometry3d.mesh_mask import MeshMask
        from aero.geometry3d.signed_distance import compute_phi_field
        from aero.geometry3d.stl_io import load_stl_triangles
        from aero.geometry3d.stl_prep import compute_frontal_blockage

        geom = MeshMask(
            path=args.stl_path,
            cx_frac=args.cx_frac,
            cy_frac=args.cy_frac,
            cz_frac=args.cz_frac,
            fit_frac=args.stl_fit,
            mesh_orient=args.mesh_orient,
            mesh_rot_x=args.mesh_rot_x,
            mesh_rot_y=args.mesh_rot_y,
            mesh_rot_z=args.mesh_rot_z,
        )
        solid = geom.mark_solid(args.nz, args.ny, args.nx)
        ibm_enabled = args.mesh_bc == "ibm"
        if ibm_enabled:
            tris = load_stl_triangles(args.stl_path)
            phi_field = compute_phi_field(
                tris, args.nz, args.ny, args.nx,
                args.cx_frac, args.cy_frac, args.cz_frac, args.stl_fit,
                args.mesh_orient,
                mesh_rot_x=args.mesh_rot_x,
                mesh_rot_y=args.mesh_rot_y,
                mesh_rot_z=args.mesh_rot_z,
            )
            solid = phi_field <= 0.0
        D = geom.reference_length()
        mesh_blockage_tuple = compute_frontal_blockage(solid)
    elif args.shape == "sphere":
        D = 2.0 * args.radius
    elif args.shape == "box":
        D = args.height
    elif args.shape == "cylinder":
        D = 2.0 * args.radius
    else:
        D = args.ny * 0.2

    nu_lbm = args.u0 * D / args.re if args.re > 0 else 0.02
    tau    = 3.0 * nu_lbm + 0.5
    omega  = 1.0 / tau
    Ma     = args.u0 * math.sqrt(3.0)
    if mesh_blockage_tuple is not None:
        blockage_y, blockage_z, blockage = mesh_blockage_tuple
    else:
        blockage_y, blockage_z, blockage = compute_blockage_metrics(args, D)
    blockage_label, blockage_blocked = assess_blockage(blockage, args.allow_high_blockage)

    if tau < 0.55:
        print(f"[X] tau={tau:.4f} < 0.55 — unstable. Reduce Re or increase grid.")
        return 1
    if Ma > 0.15:
        print(f"[!] WARNING: Ma={Ma:.4f} > 0.15 — compressibility error may be significant.")
    if blockage > 0.10:
        print(
            f"[!] WARNING: frontal blockage is {blockage*100:.1f}% "
            f"(y={blockage_y*100:.1f}%, z={blockage_z*100:.1f}%)."
        )
        if blockage <= 0.20:
            print("[!] External-flow forces may be mildly inflated. Target <=10% for cleaner benchmarks.")
        elif blockage <= 0.30:
            print("[!] External-flow forces may be significantly inflated. Consider enlarging Ny/Nz.")
        else:
            print("[!] This is a very confined run. Use --allow-high-blockage only if confinement is intentional.")
    if blockage_blocked:
        print("[X] Refusing run because frontal blockage exceeds 30% without explicit override.")
        return 1

    print()
    print("=" * 60)
    print("  Aero CFD — 3D LBM D3Q19 Wind Tunnel Simulator")
    print("=" * 60)
    print(f"  Shape     : {args.shape}")
    print(f"  Grid      : {args.nx} x {args.ny} x {args.nz}  (Nx x Ny x Nz)")
    print(f"  Re        : {args.re:.2f}")
    print(f"  u0_lbm    : {args.u0:.5f}")
    print(f"  D (ref L) : {D:.1f} cells")
    print(f"  nu_lbm    : {nu_lbm:.6f}")
    print(f"  tau       : {tau:.6f}")
    print(f"  omega     : {omega:.6f}")
    print(f"  Ma        : {Ma:.5f}")
    print(f"  Blockage  : {blockage*100:.1f}%  (y={blockage_y*100:.1f}%, z={blockage_z*100:.1f}%, {blockage_label})")
    print(f"  Steps     : {args.steps}")
    print(f"  wall_bc   : {args.wall_bc}")
    print(f"  x_bc      : {args.streamwise_bc}")
    print(f"  outlet_bc : {args.outlet_bc}")
    print(f"  collision : {args.collision}")
    print(f"  backend   : {args.backend}")
    print(f"  viz3d     : {args.viz3d}")
    print("=" * 60)
    print()

    case = None
    if args.save_case:
        case = SimulationCase.from_args(args, {
            "D": D, "nu_lbm": nu_lbm, "tau": tau, "omega": omega, "Ma": Ma,
        }, cases_root=args.cases_root)
        case.config["mode"] = "3d"
        case.config["nz"] = args.nz
        case.config["viz3d"] = args.viz3d
        case.save_config()
        output_dir = str(case.case_dir)
    else:
        output_dir = args.output

    # Geometry (analytic shapes; mesh was voxelized above)
    if args.shape == "mesh":
        if tau < 0.55:
            print(f"[X] tau={tau:.4f} < 0.55 — unstable after mesh fit. Reduce Re or enlarge grid.")
            return 1
        print(
            f"Obstacle  : {solid.sum()} solid cells (STL mesh, D≈{D:.1f}, "
            f"orient={args.mesh_orient})"
        )
    elif args.shape == "sphere":
        geom = Sphere(radius=args.radius,
                      cx_frac=args.cx_frac, cy_frac=args.cy_frac, cz_frac=args.cz_frac)
    elif args.shape == "box":
        geom = Box(width=args.width, height=args.height, depth=args.depth,
                   cx_frac=args.cx_frac, cy_frac=args.cy_frac, cz_frac=args.cz_frac)
    elif args.shape == "cylinder":
        geom = Cylinder3D(radius=args.radius, length=args.length,
                          cx_frac=args.cx_frac, cy_frac=args.cy_frac, cz_frac=args.cz_frac)
    else:
        print(f"[X] Unknown shape: {args.shape}")
        return 1

    if args.shape != "mesh":
        solid = geom.mark_solid(args.nz, args.ny, args.nx)
        print(f"Obstacle  : {solid.sum()} solid cells")

    ibm_active = args.shape == "mesh" and args.mesh_bc == "ibm"
    if args.bouzidi and not ibm_active and args.shape != "mesh":
        phi_for_solver = geom.sdf_field(args.nz, args.ny, args.nx)
    elif ibm_active:
        phi_for_solver = phi_field
    else:
        phi_for_solver = None

    # Collect common Solver3D kwargs to allow multiblock reuse
    _use_multiblock = (args.refine_z_lo is not None and args.refine_z_hi is not None)
    _solver_kw = dict(
        Nz=args.nz, Ny=args.ny, Nx=args.nx,
        solid=solid,
        omega=omega,
        u0=args.u0,
        D=D,
        rho0=1.0,
        wall_bc=args.wall_bc,
        outlet_bc=args.outlet_bc,
        streamwise_bc=args.streamwise_bc,
        backend=args.backend,
        collision=args.collision,
        inlet_perturbation=args.inlet_perturbation,
        trt_lambda=args.trt_lambda,
        sponge_thickness=args.sponge_cells,
        sponge_strength=args.sponge_strength,
        les=args.les,
        les_cs=args.les_cs,
        les_model=args.les_model,
        bouzidi=args.bouzidi,
        van_driest=args.van_driest,
        van_driest_A=args.van_driest_A,
        ibm_enabled=ibm_active,
        phi=phi_for_solver,
        body_force_x=args.body_force_x,
        body_force_y=args.body_force_y,
        body_force_z=args.body_force_z,
        wall_velocity_top=args.wall_velocity_top,
        wall_velocity_bottom=args.wall_velocity_bottom,
        synthetic_inflow=args.synthetic_inflow,
        synthetic_inflow_intensity=args.synthetic_inflow_intensity,
        sem_inlet=args.sem_inlet,
        sem_Tu=args.sem_tu,
        sem_L_int=args.sem_lint,
        sem_N=args.sem_n,
        thermal=args.thermal,
        T_hot=args.T_hot,
        T_cold=args.T_cold,
        alpha_T=args.alpha_T,
        buoyancy=args.buoyancy,
        g_gravity=args.g_gravity,
        beta=args.beta,
        T_ref=args.T_ref,
    )
    if _use_multiblock:
        from aero.lbm.multiblock import MultiblockSolver3D
        _mb_kw = {k: v for k, v in _solver_kw.items() if k not in ("Nz", "Ny", "Nx", "solid")}
        solver = MultiblockSolver3D(
            Nz=args.nz, Ny=args.ny, Nx=args.nx,
            solid=solid,
            refine_z_lo=args.refine_z_lo,
            refine_z_hi=args.refine_z_hi,
            **_mb_kw,
        )
        print(f"Multi-block: refined z=[{args.refine_z_lo},{args.refine_z_hi})")
    else:
        solver = Solver3D(**_solver_kw)
        print(f"Surf links: {solver.surface_links.shape[0]}")
    print()

    if args.resume_from:
        solver.load_checkpoint(args.resume_from)
        print(f"Resumed from step {solver.step_count}")

    print("Running 3D simulation...")
    print()

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None and args.checkpoint_every is not None:
        ckpt_dir = args.output
    if ckpt_dir:
        pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = solver.run(
        steps=args.steps,
        check_every=args.check_every,
        verbose=args.verbose,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=ckpt_dir,
        auto_stop=args.auto_stop,
        hdf5_path=args.export_hdf5,
        hdf5_every=args.hdf5_every or args.check_every,
    )
    elapsed = time.perf_counter() - t0
    if autoconfig_report is not None:
        result["autoconfig_report"] = autoconfig_report.as_dict()

    print()
    print("  === RESULTS ===")
    print(f"  Cd    (mean) = {result['Cd_mean']:.4f}  ±  {result['Cd_std']:.4f}")
    if "Cd_p_mean" in result:
        print(f"  Cd_press       = {result['Cd_p_mean']:.4f}  (pressure drag)")
        print(f"  Cd_visc        = {result['Cd_v_mean']:.4f}  (viscous drag)")
    print(f"  Cl_y  (mean) = {result['Cly_mean']:.4f}  ±  {result['Cly_std']:.4f}")
    print(f"  Cl_z  (mean) = {result['Clz_mean']:.4f}  ±  {result['Clz_std']:.4f}")
    print(f"  Elapsed      = {elapsed:.1f} s  ({args.steps/elapsed:.0f} steps/s)")
    print(f"  Stop reason  = {result.get('stop_reason', 'max_steps')}")
    if result.get("scalar_stats"):
        scalar_stats = result["scalar_stats"]
        print(
            f"  Scalar       = mean {scalar_stats['mean']:.4f}  "
            f"range [{scalar_stats['min']:.4f}, {scalar_stats['max']:.4f}]"
        )
    if args.shape == "sphere":
        if abs(args.re - 20.0) < 1e-12:
            print(f"  Expected Cd  ≈ 2–5  (sphere Re=20; confinement can inflate drag)")
        elif abs(args.re - 100.0) < 1e-12:
            print(f"  Expected Cd  ≈ 1.0–1.1  (sphere Re=100, literature)")
    print()

    if case is not None:
        case.save_results(result, elapsed)

    if not args.no_plot:
        if args.viz3d in ("auto", "all", "pyvista") and not HAS_PYVISTA:
            print("PyVista not installed; falling back to slice visualisations.")
        print("Saving 3D visualisations...")
        save_all_3d(
            result=result,
            solid3d=solid,
            u0=args.u0,
            Re=args.re,
            shape_name=args.shape,
            steps=args.steps,
            output_dir=output_dir,
            viz_mode=args.viz3d,
            export_vtk=args.export_vtk,
        )

    from aero.run_manifest import write_run_manifest
    validation = build_validation_report(
        mode="3d",
        shape=args.shape,
        params=vars(args),
        cd=result.get("Cd_mean"),
        grid_cd_values=result.get("grid_cd_values"),
    )
    uncertainty = build_uncertainty_report(
        mode="3d",
        shape=args.shape,
        params=vars(args),
        result=result,
    )
    write_run_manifest(
        output_dir,
        {
            "mode": "3d",
            "shape": args.shape,
            "re": float(args.re),
            "steps": int(args.steps),
            "u0": float(args.u0),
            "Cd_mean": float(result["Cd_mean"]),
            "Cd_std": float(result["Cd_std"]),
            "Cly_mean": float(result["Cly_mean"]),
            "Cly_std": float(result["Cly_std"]),
            "Clz_mean": float(result["Clz_mean"]),
            "Clz_std": float(result["Clz_std"]),
            "Cmx_mean": float(result.get("Cmx_mean", 0.0)),
            "Cmy_mean": float(result.get("Cmy_mean", 0.0)),
            "Cmz_mean": float(result.get("Cmz_mean", 0.0)),
            "Cd_history": [float(x) for x in result.get("Cd_history", [])],
            "Cly_history": [float(x) for x in result.get("Cly_history", [])],
            "Clz_history": [float(x) for x in result.get("Clz_history", [])],
            "Cmx_history": [float(x) for x in result.get("Cmx_history", [])],
            "Cmy_history": [float(x) for x in result.get("Cmy_history", [])],
            "Cmz_history": [float(x) for x in result.get("Cmz_history", [])],
            "volume_file": f"{args.shape}_re{args.re:.0f}_volume.vti" if args.export_vtk else None,
            "stop_reason": result.get("stop_reason", "max_steps"),
            "observables": result.get("observables"),
            "scalar_stats": result.get("scalar_stats"),
            "autoconfig_report": None if autoconfig_report is None else autoconfig_report.as_dict(),
            "validation_report": {
                "benchmark_status": validation.benchmark_status,
                "grid_status": validation.grid_status,
                "collision_status": validation.collision_status,
                "overall_ok": validation.overall_ok,
            },
            "uncertainty_report": {
                "overall_status": uncertainty.overall_status,
                "summary": uncertainty.summary,
                "components": uncertainty.components,
            },
        },
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
