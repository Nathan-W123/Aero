#!/usr/bin/env python3
"""
MPI entry point for the 3D LBM wind tunnel.

Run with::

    mpirun -n 4 python mpi_cli3d.py --shape cylinder --nx 120 --ny 60 --nz 60 \
        --re 200 --steps 500

All CLI arguments match cli3d.py. The MPI decomposition splits the domain
into z-slabs, one per rank.
"""

import sys
import argparse
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPI 3D LBM wind tunnel")
    p.add_argument("--shape", choices=["cylinder", "sphere", "rectangle", "stl", "none"],
                   default="cylinder")
    p.add_argument("--nx", type=int, default=80)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--nz", type=int, default=40)
    p.add_argument("--re", type=float, default=200.0)
    p.add_argument("--u0", type=float, default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--check-every", type=int, default=200)
    p.add_argument("--omega", type=float, default=None)
    p.add_argument("--backend", choices=["auto", "numpy", "numba"], default="auto")
    p.add_argument("--collision", choices=["bgk", "mrt", "trt"], default="bgk")
    p.add_argument("--wall-bc", choices=["slip", "noslip"], default="slip")
    p.add_argument("--outlet-bc", choices=["convective", "zerogradient"], default="convective")
    p.add_argument("--no-verbose", action="store_true")
    p.add_argument("--output-dir", type=str, default="./mpi_outputs3d")
    return p.parse_args()


def main() -> None:
    try:
        from mpi4py import MPI
    except ImportError:
        print("ERROR: mpi4py is required. Install with: pip install mpi4py", file=sys.stderr)
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    args = _parse_args()

    Nx, Ny, Nz = args.nx, args.ny, args.nz
    Re = args.re

    # Default u0 and omega derived from Re
    u0 = args.u0 if args.u0 is not None else 0.05
    D = max(Ny // 5, 4)
    nu = u0 * D / Re
    omega = args.omega if args.omega is not None else float(1.0 / (3.0 * nu + 0.5))

    # Build solid mask (rank 0 builds, then broadcasts)
    if rank == 0:
        from aero.geometry3d.sphere import Sphere
        if args.shape in ("cylinder", "sphere"):
            geom = Sphere(radius=D / 2, cx_frac=0.3, cy_frac=0.5, cz_frac=0.5)
            solid = geom.mark_solid(Nz, Ny, Nx)
        else:
            solid = np.zeros((Nz, Ny, Nx), dtype=bool)
    else:
        solid = np.empty((Nz, Ny, Nx), dtype=bool)

    comm.Bcast(solid, root=0)

    from aero.lbm.mpi_solver3d import MPISolver3D

    solver = MPISolver3D(
        comm=comm,
        Nz_global=Nz,
        Ny=Ny,
        Nx=Nx,
        solid_global=solid,
        omega=omega,
        u0=u0,
        D=float(D),
        backend=args.backend,
        collision=args.collision,
        wall_bc=args.wall_bc,
        outlet_bc=args.outlet_bc,
    )

    if rank == 0:
        print(f"MPI 3D LBM — {comm.Get_size()} ranks  |  grid {Nx}×{Ny}×{Nz}  |  Re={Re:.0f}")
        print(f"  rank {rank}: z=[{solver._z_lo}, {solver._z_hi})  Nz_local={solver._Nz_local}")
        print()

    result = solver.run(
        steps=args.steps,
        check_every=args.check_every,
        verbose=not args.no_verbose,
    )

    if rank == 0:
        print(f"\nDone. Steps completed: {result['steps_completed']}")


if __name__ == "__main__":
    main()
