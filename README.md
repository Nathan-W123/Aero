# Aero

From-scratch 2D and 3D lattice Boltzmann wind-tunnel simulator for external and internal flow research workflows.

## What it does

- 2D D2Q9 and 3D D3Q19 solvers
- Analytic geometry support for cylinders, rectangles, polygons, spheres, boxes, and 3D cylinders
- STL voxelization / IBM-style mesh workflows for 3D cases
- BGK, MRT, and TRT collision models
- Outlet sponge layers, Bouzidi curved-wall bounce-back, moving walls, and streamwise periodic / recycling options
- LES support with `smagorinsky` and `wale`
- Convergence detection from rolling drag statistics and lift-history Strouhal analysis
- Force, lift, drag-split, moment, and profile observables
- Passive scalar / thermal transport with optional Boussinesq buoyancy
- Case saving, checkpoint / resume, run manifests, PNG outputs, optional VTK and HDF5 export
- Desktop GUI support

## Status

This codebase is moving toward research-grade validation tooling. It already includes:

- automated convergence/stationarity checks
- uncertainty and validation reporting in saved cases
- 3D streamwise periodic and recycling inlet options
- richer force and moment observables
- synthetic / SEM-style inflow scaffolding
- passive scalar transport interfaces in both CLIs

Still in progress:

- scalar data in HDF5/XDMF exports
- broader canonical validation campaigns
- more advanced turbulence modeling and multiphysics coverage

## Repository layout

- `aero/geometry` — 2D geometry definitions
- `aero/geometry3d` — 3D geometry and STL prep
- `aero/lbm` — solver kernels, boundary conditions, LES, scalar transport
- `aero/gui` — desktop GUI helpers and viewport code
- `cli.py` — 2D command-line entry point
- `cli3d.py` — 3D command-line entry point
- `tests/` — regression and feature tests
- `cases/` — saved simulation cases

## Installation

Base install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

With development tools:

```bash
pip install -e ".[dev]"
```

With 3D visualization:

```bash
pip install -e ".[viz3d]"
```

With GUI dependencies:

```bash
pip install -e ".[gui]"
```

## Quick start

2D cylinder:

```bash
python3 cli.py --shape cylinder --re 100 --nx 400 --ny 200 --steps 20000
```

2D with WALE LES and passive scalar:

```bash
python3 cli.py \
  --shape cylinder \
  --re 100 \
  --les --les-model wale \
  --scalar --scalar-hot 1.0 --scalar-cold 0.0 --scalar-diffusivity 0.01
```

3D sphere:

```bash
python3 cli3d.py --shape sphere --re 100 --nx 128 --ny 64 --nz 64 --steps 5000
```

3D periodic/internal-flow style run:

```bash
python3 cli3d.py \
  --shape box \
  --streamwise-bc periodic \
  --body-force-x 1e-6 \
  --wall-bc noslip \
  --re 100
```

## Common capabilities

### 2D CLI

Notable options in `cli.py`:

- geometry: `--shape`, `--radius`, `--width`, `--height`, `--polygon-verts`, `--image-path`
- BCs: `--wall-bc`, `--inlet-bc`, `--outlet-bc`
- numerics: `--collision`, `--trt-lambda`, `--les`, `--les-model`, `--bouzidi`
- inflow/walls: `--synthetic-inflow`, `--wall-velocity-top`, `--wall-velocity-bottom`
- scalar transport: `--scalar`, `--scalar-hot`, `--scalar-cold`, `--scalar-diffusivity`, `--buoyancy`
- run control: `--early-stop`, `--checkpoint-every`, `--resume-from`, `--export-hdf5`

### 3D CLI

Notable options in `cli3d.py`:

- geometry: `--shape`, `--radius`, `--width`, `--height`, `--depth`, `--length`, `--stl-path`
- BCs: `--wall-bc`, `--outlet-bc`, `--streamwise-bc`
- numerics: `--collision`, `--les`, `--les-model`, `--bouzidi`
- inflow: `--synthetic-inflow`, `--sem-inlet`, `--sem-tu`, `--sem-lint`, `--sem-n`
- internal-flow forcing: `--body-force-x`, `--body-force-y`, `--body-force-z`
- scalar transport: `--scalar`, `--scalar-hot`, `--scalar-cold`, `--scalar-diffusivity`, `--buoyancy`
- output: `--viz3d`, `--export-vtk`, `--export-hdf5`

## Outputs

A typical run can produce:

- `results.json` with scalar metrics and validation/uncertainty summaries
- `history.json` with coefficient histories and convergence metadata
- `config.json` when saved as a case
- PNG visualizations
- VTK volume output for 3D runs when enabled
- HDF5/XDMF snapshots for flow fields when enabled

Saved cases are managed through `aero/case.py` and written under `cases/`.

## Testing

Run the full suite:

```bash
python3 -m pytest
```

Run a focused solver/physics subset:

```bash
python3 -m pytest -q tests/test_solver.py tests/test_les.py tests/test_thermal.py
```

## GUI

Launch the desktop app with:

```bash
python3 gui.py
```

or:

```bash
aero-gui
```

## Research notes

- Low-Mach operation is still important; keep `u0` conservative.
- Validation coverage is improving, but not every physics path is yet benchmark-complete.
- Some advanced features are present as scaffolding or early implementations and should still be checked case-by-case before publication.

## Near-term roadmap

- add passive scalar fields to HDF5/XDMF export
- add canonical passive-scalar validation cases
- continue turbulence-model upgrades
- expand publishable 3D validation campaigns

