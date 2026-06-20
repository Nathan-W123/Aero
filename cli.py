#!/usr/bin/env python3
"""
Aero CFD — 2D LBM Wind Tunnel Simulator
CLI entry point.

Usage
-----
  python3 cli.py --shape cylinder --re 100
  python3 cli.py --shape rectangle --width 40 --height 20 --re 200 --nx 600 --ny 300
  python3 cli.py --shape polygon --polygon-verts "0.3,0.3 0.5,0.3 0.5,0.7 0.3,0.7"
  python3 cli.py --shape image --image-path my_shape.png
  python3 cli.py --wall-bc noslip --inlet-bc pressure --outlet-bc pressure --rho-in 1.001 --re 100
  python3 cli.py --save-case                          # auto-names the case
  python3 cli.py --load-case cases/cylinder_re100_*/  # re-run from saved config
  python3 cli.py --list-cases                         # list all saved cases
  python3 cli.py --resume-from cases/my_case/checkpoint_00010000.npz --steps 5000
  python3 cli.py --checkpoint-every 5000 --checkpoint-dir ./checkpoints
"""

import argparse
import math
import pathlib
import sys
import json
import time

from aero.geometry.cylinder import Cylinder
from aero.geometry.rectangle import Rectangle
from aero.geometry.polygon import Polygon
from aero.geometry.image_mask import ImageMask
from aero.lbm.solver import Solver
from aero.diagnostics import validate_parameters, check_convergence, compute_strouhal
from aero.visualization import save_all
from aero.case import SimulationCase


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2D LBM Wind Tunnel CFD Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shape
    p.add_argument("--shape", choices=["cylinder", "rectangle", "polygon", "image"],
                   default="cylinder")
    p.add_argument("--radius", type=float, default=20.0,
                   help="Cylinder radius (lattice cells)")
    p.add_argument("--width",  type=float, default=40.0,
                   help="Rectangle streamwise width (lattice cells)")
    p.add_argument("--height", type=float, default=20.0,
                   help="Rectangle cross-stream height (lattice cells)")
    p.add_argument("--cx-frac", type=float, default=1/3,
                   help="Obstacle centre x as fraction of Nx")
    p.add_argument("--cy-frac", type=float, default=0.5,
                   help="Obstacle centre y as fraction of Ny")
    p.add_argument("--polygon-verts", type=str, default=None,
                   help="Space-separated 'x_frac,y_frac' pairs for polygon shape")
    p.add_argument("--image-path", type=str, default=None,
                   help="Path to image file for --shape image")
    p.add_argument("--image-threshold", type=float, default=0.5,
                   help="Grayscale threshold: pixels <= threshold become solid")
    p.add_argument("--image-invert", action="store_true",
                   help="Invert image mask (light=solid, dark=fluid)")

    # Flow
    p.add_argument("--re",  type=float, default=100.0, help="Reynolds number")
    p.add_argument("--u0",  type=float, default=0.05,
                   help="Inlet velocity (lattice units); ignored for pressure-driven flow")

    # Grid
    p.add_argument("--nx", type=int, default=400)
    p.add_argument("--ny", type=int, default=200)

    # Boundary conditions (Phase 2)
    p.add_argument("--wall-bc", choices=["slip", "noslip"], default="slip",
                   help="Top/bottom wall BC: specular slip or full bounce-back no-slip")
    p.add_argument("--inlet-bc", choices=["velocity", "pressure"], default="velocity",
                   help="Inlet BC: Zou-He velocity or Zou-He pressure")
    p.add_argument("--outlet-bc", choices=["convective", "pressure", "zerogradient"],
                   default="convective",
                   help="Outlet BC: advective, Zou-He pressure, or zero-gradient copy")
    p.add_argument("--rho-in",  type=float, default=None,
                   help="Inlet density for --inlet-bc pressure (default: 1 + 3*u0^2)")
    p.add_argument("--rho-out", type=float, default=1.0,
                   help="Outlet density for --outlet-bc pressure")
    p.add_argument("--inlet-perturbation", type=float, default=0.0,
                   help="Inlet uy perturbation amplitude (fraction of u0) for shedding")

    # Run control
    p.add_argument("--steps",       type=int, default=80000)
    p.add_argument("--check-every", type=int, default=1000)
    p.add_argument("--early-stop",  action="store_true",
                   help="Stop when Cd converges")

    # Checkpoint (Phase 2)
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="Save solver state every N steps")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Directory for checkpoint .npz files (default: case dir or ./checkpoints)")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Resume from a checkpoint .npz file")

    # Backend (Phase 3)
    p.add_argument("--backend", choices=["auto", "numpy", "numba"], default="auto",
                   help="Compute backend: auto=use Numba if installed, else NumPy")
    p.add_argument("--collision", choices=["bgk", "mrt", "trt"], default="bgk",
                   help="Collision operator: BGK, MRT, or TRT")
    p.add_argument("--trt-lambda", type=float, default=0.25, help="TRT magic parameter")
    p.add_argument("--sponge-cells", type=int, default=0, help="Outlet sponge thickness (0=off)")
    p.add_argument("--sponge-strength", type=float, default=0.1, help="Max sponge relaxation")
    p.add_argument("--les", action="store_true", help="Enable Smagorinsky LES")
    p.add_argument("--les-cs", type=float, default=0.16, help="Smagorinsky constant")

    # Output / config
    p.add_argument("--output",  type=str, default="./outputs",
                   help="Output directory for PNGs (ignored when --save-case)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip visualisations")
    p.add_argument("--config",  type=str, default=None,
                   help="Load run parameters from a JSON file")
    p.add_argument("--verbose", action="store_true", default=True)

    # Case system
    p.add_argument("--save-case",  action="store_true",
                   help="Save config + results as a named case in ./cases/")
    p.add_argument("--load-case",  type=str, default=None,
                   help="Load config from an existing case directory and re-run")
    p.add_argument("--cases-root", type=str, default="./cases",
                   help="Root directory for all saved cases")
    p.add_argument("--list-cases", action="store_true",
                   help="List all saved cases and exit")

    return p


def load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def parse_polygon_verts(raw: str):
    pairs = raw.strip().split()
    verts = []
    for pair in pairs:
        x_s, y_s = pair.split(",")
        verts.append((float(x_s), float(y_s)))
    return verts


# ---------------------------------------------------------------------------
# Parameter derivation
# ---------------------------------------------------------------------------

def derive_params(args: argparse.Namespace) -> dict:
    if args.shape == "cylinder":
        D = 2.0 * args.radius
    elif args.shape in ("rectangle",):
        D = args.height
    else:
        D = getattr(args, "height", args.ny * 0.2)   # fallback for polygon/image

    nu_lbm   = args.u0 * D / args.re if args.re > 0 else 0.02
    tau      = 3.0 * nu_lbm + 0.5
    omega    = 1.0 / tau
    Ma       = args.u0 * math.sqrt(3.0)
    blockage = D / args.ny
    return dict(D=D, nu_lbm=nu_lbm, tau=tau, omega=omega, Ma=Ma, blockage=blockage)


def print_summary(args: argparse.Namespace, p: dict) -> None:
    if args.shape == "cylinder":
        detail = f"radius={args.radius}"
    elif args.shape == "rectangle":
        detail = f"width={args.width}  height={args.height}"
    elif args.shape == "polygon":
        detail = f"verts={args.polygon_verts}"
    else:
        detail = f"image={args.image_path}"
    print()
    print("=" * 55)
    print("  Aero CFD — 2D LBM Wind Tunnel Simulator")
    print("=" * 55)
    print(f"  Shape     : {args.shape}  ({detail})")
    print(f"  Grid      : {args.nx} x {args.ny}")
    print(f"  Re        : {args.re:.2f}")
    print(f"  u0_lbm    : {args.u0:.5f}")
    print(f"  D (ref L) : {p['D']:.1f} cells")
    print(f"  nu_lbm    : {p['nu_lbm']:.6f}")
    print(f"  tau       : {p['tau']:.6f}")
    print(f"  omega     : {p['omega']:.6f}")
    print(f"  Ma        : {p['Ma']:.5f}")
    print(f"  Blockage  : {p['blockage']*100:.1f}%")
    print(f"  Steps     : {args.steps}")
    print(f"  wall_bc   : {args.wall_bc}")
    print(f"  inlet_bc  : {args.inlet_bc}")
    print(f"  outlet_bc : {args.outlet_bc}")
    print(f"  collision : {args.collision}")
    print("=" * 55)
    print()


# ---------------------------------------------------------------------------
# List cases helper
# ---------------------------------------------------------------------------

def list_cases(cases_root: str) -> None:
    root = pathlib.Path(cases_root)
    if not root.is_dir():
        print("No cases directory found.")
        return
    dirs = sorted(root.iterdir())
    if not dirs:
        print("No saved cases found.")
        return
    print(f"\nSaved cases in {root}/\n")
    for d in dirs:
        if not d.is_dir():
            continue
        try:
            case = SimulationCase.load(str(d))
            print(f"  {case.summary()}")
        except Exception:
            print(f"  {d.name}  (unreadable)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    # --list-cases shortcut
    if args.list_cases:
        list_cases(args.cases_root)
        return 0

    # --load-case: read config from an existing case directory
    if args.load_case:
        case   = SimulationCase.load(args.load_case)
        cfg    = case.config
        print(f"Loaded case: {case.name}")
        parser.set_defaults(**{k: v for k, v in cfg.items()
                                if hasattr(args, k) and v is not None})
        args = parser.parse_args()

    # --config: plain JSON parameter file
    elif args.config:
        cfg = load_json(args.config)
        parser.set_defaults(**cfg)
        args = parser.parse_args()

    # Derived LBM parameters
    p = derive_params(args)

    # Validate before doing any work
    msgs     = validate_parameters(p["tau"], args.u0, p["D"], args.ny)
    errors   = [m for m in msgs if m.startswith("ERROR")]
    warnings = [m for m in msgs if m.startswith("WARNING")]

    for w in warnings:
        print(f"[!] {w}")
    if errors:
        for e in errors:
            print(f"[X] {e}")
        return 1

    print_summary(args, p)

    # Case setup (before run so config is saved even if run fails)
    case = None
    if args.save_case:
        case = SimulationCase.from_args(args, p, cases_root=args.cases_root)
        case.save_config()
        output_dir = str(case.case_dir)
    else:
        output_dir = args.output

    # Geometry
    if args.shape == "cylinder":
        geom = Cylinder(radius=args.radius, cx_frac=args.cx_frac, cy_frac=args.cy_frac)
    elif args.shape == "rectangle":
        geom = Rectangle(width=args.width, height=args.height,
                         cx_frac=args.cx_frac, cy_frac=args.cy_frac)
    elif args.shape == "polygon":
        if args.polygon_verts is None:
            print("[X] --polygon-verts required for --shape polygon")
            return 1
        verts = parse_polygon_verts(args.polygon_verts)
        geom = Polygon(verts)
    elif args.shape == "image":
        if args.image_path is None:
            print("[X] --image-path required for --shape image")
            return 1
        geom = ImageMask(args.image_path,
                         threshold=args.image_threshold,
                         invert=args.image_invert)
    else:
        print(f"[X] Unknown shape: {args.shape}")
        return 1

    solid = geom.mark_solid(args.ny, args.nx)
    print(f"Obstacle  : {solid.sum()} solid cells")

    # Determine rho_in for pressure-driven inlet
    rho_in = args.rho_in
    if rho_in is None:
        rho_in = 1.0 + 3.0 * args.u0 ** 2   # slight pressure overshoot ≡ velocity inlet

    # Solver
    solver = Solver(
        Ny=args.ny, Nx=args.nx,
        solid=solid,
        omega=p["omega"],
        u0=args.u0,
        D=p["D"],
        rho0=1.0,
        wall_bc=args.wall_bc,
        inlet_bc=args.inlet_bc,
        outlet_bc=args.outlet_bc,
        rho_in=rho_in,
        rho_out=args.rho_out,
        backend=args.backend,
        collision=args.collision,
        inlet_perturbation=args.inlet_perturbation,
        trt_lambda=args.trt_lambda,
        sponge_thickness=args.sponge_cells,
        sponge_strength=args.sponge_strength,
        les=args.les,
        les_cs=args.les_cs,
    )
    print(f"Surf links: {solver.surface_links.shape[0]}")
    print()

    # Resume from checkpoint
    if args.resume_from:
        solver.load_checkpoint(args.resume_from)
        print(f"Resumed from step {solver.step_count} ({args.resume_from})")

    print("Running simulation...")
    print()

    # Checkpoint directory
    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None and args.checkpoint_every is not None:
        ckpt_dir = output_dir
    if ckpt_dir:
        pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Early-stop callback
    def maybe_stop(step, Cd, Cl, rho, ux, uy):
        if args.early_stop and check_convergence(solver.Cd_history):
            print(f"\n[Convergence at step {step}]")
            raise StopIteration

    t0 = time.perf_counter()
    try:
        result = solver.run(
            steps=args.steps,
            check_every=args.check_every,
            verbose=args.verbose,
            callback=maybe_stop if args.early_stop else None,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=ckpt_dir,
        )
    except StopIteration:
        from aero.lbm.d2q9 import compute_macroscopic
        import numpy as np
        rho, ux, uy = compute_macroscopic(solver.f)
        w = max(1, len(solver.Cd_history) // 5)
        result = dict(
            Cd_mean=float(np.mean(solver.Cd_history[-w:])),
            Cl_mean=float(np.mean(solver.Cl_history[-w:])),
            Cd_std =float(np.std(solver.Cd_history[-w:])),
            Cl_std =float(np.std(solver.Cl_history[-w:])),
            Cd_history=solver.Cd_history,
            Cl_history=solver.Cl_history,
            rho=rho, ux=ux, uy=uy,
        )
    elapsed = time.perf_counter() - t0

    # Strouhal number (Phase 2)
    St = compute_strouhal(result["Cl_history"], p["D"], args.u0)

    print()
    print("  === RESULTS ===")
    print(f"  Cd (mean) = {result['Cd_mean']:.4f}  ±  {result['Cd_std']:.4f}")
    print(f"  Cl (mean) = {result['Cl_mean']:.4f}  ±  {result['Cl_std']:.4f}")
    if "Cd_p_mean" in result:
        print(f"  Cd_press  = {result['Cd_p_mean']:.4f}  (pressure drag)")
        print(f"  Cd_visc   = {result['Cd_v_mean']:.4f}  (viscous drag)")
    if St is not None:
        print(f"  St        = {St:.4f}  (Strouhal — expected ~0.164 for cylinder Re=100)")
    print(f"  Elapsed   = {elapsed:.1f} s")
    print()

    # Save case results
    if case is not None:
        case.save_results(result, elapsed)

    # Visualisation
    if not args.no_plot:
        print("Saving visualisations...")
        save_all(
            result=result,
            solid=solid,
            u0=args.u0,
            Re=args.re,
            shape_name=args.shape,
            steps=len(result["Cd_history"]),
            output_dir=output_dir,
        )

    from aero.run_manifest import write_run_manifest
    write_run_manifest(
        output_dir,
        {
            "mode": "2d",
            "shape": args.shape,
            "re": float(args.re),
            "steps": int(len(result["Cd_history"])),
            "u0": float(args.u0),
            "Cd_mean": float(result["Cd_mean"]),
            "Cd_std": float(result["Cd_std"]),
            "Cl_mean": float(result["Cl_mean"]),
            "Cl_std": float(result["Cl_std"]),
            "Cd_history": [float(x) for x in result.get("Cd_history", [])],
            "Cl_history": [float(x) for x in result.get("Cl_history", [])],
        },
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
