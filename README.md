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

3D force coefficients are normalised by the body's **frontal area** (`1/2 rho u0^2 A`):
`pi r^2` for a sphere, `height x depth` for a box, `2r x length` for a 3D cylinder, and
the projected voxel area for an STL mesh.  Blockage above ~10% inflates drag noticeably
(a sphere at 29% blockage reads ~30% above Schiller-Naumann); widen the tunnel before
comparing with unconfined literature values.

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

## Web UI

A browser front end that needs nothing beyond the package itself:

```bash
python3 webui.py          # then open http://localhost:8017
```

- Set the case in the property grid; the checks strip grades it before you run
  (blockage, domain length, resolution, stability) and the viewer previews the
  body in its tunnel.
- 3D shape **mesh** takes an STL (ASCII or binary): **Choose…**, or drop the
  file anywhere on the page.  Size it in cells, orient and rotate it, and the
  preview shows exactly the cells the solver will see.
- The 3D view shows smoke carried by the computed flow, and replays the wake
  and vorticity of the run as a time-lapse.
- The convergence chart has Cd and Cl at every step: wheel to zoom, drag to
  pan, Shift-drag to zoom to a box, double-click for the whole run.
- **Grid study** runs the case on three grids a ratio of √2 apart (your grid
  and two coarser, or one coarser and one finer: Grid study → Levels) and
  reports the observed order of convergence, Cd extrapolated to zero cell
  size, and the Grid Convergence Index on the finest grid (Celik et al.
  2008); the resolution check then shows the measured number instead of an
  estimate.
- **Blockage**: every result also gives **Cd free air**, the drag with the
  tunnel's walls taken out by a confinement model calibrated with this solver
  on cylinders (2D) and spheres (3D) across blockage and Reynolds number
  (`aero/blockage.py`), with its own uncertainty, larger for shapes it was
  not calibrated on.  **Blockage study** measures it for your body instead:
  the case in the tunnel as set and two wider ones, extrapolated to zero
  blockage.
- **Refine near body** (3D) runs a box around the body -- half a body length
  ahead and to the sides, one and a half into the wake -- at twice the
  resolution, coupled to the tunnel grid; the body is voxelized again there
  and the forces come from it.  It costs what the box's cells do (about 3-5×
  a plain run in a crowded tunnel), against 16× for refining the whole
  tunnel.  The view outlines the block in blue.
- Export the view as PNG, the history as CSV, the results as JSON.

## Research notes

- Low-Mach operation is still important; keep `u0` conservative.
- Validation coverage is improving, but not every physics path is yet benchmark-complete.
- Some advanced features are present as scaffolding or early implementations and should still be checked case-by-case before publication.

## Near-term roadmap

- add passive scalar fields to HDF5/XDMF export
- add canonical passive-scalar validation cases
- continue turbulence-model upgrades
- expand publishable 3D validation campaigns

