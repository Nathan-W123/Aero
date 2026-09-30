# Sample STL meshes

Public-domain test geometry for 3D mesh runs (`--shape mesh`).

| File | Source | Description |
|------|--------|-------------|
| `unit_sphere.stl` | [trimesh](https://github.com/mikedh/trimesh) | Unit sphere (~1 cell diameter), good for Cd benchmarks |
| `unit_cube.stl` | trimesh | 1×1×1 cube |
| `cylinder.stl` | trimesh | Radius 1, height 8 |
| `cube_20mm.stl` | trimesh | 20 mm cube (arbitrary units) |
| `sphere.stl` | [stl-creator](https://github.com/elerac/stl-creator) | ASCII sphere, radius 0.5 |
| `simple_plane.stl` | [simple-plane](https://github.com/RLuckom/simple-plane) | Small toy-style airplane (OpenSCAD) |

Use `--mesh-orient auto` (default) to PCA-align the mesh: stream +x, wingspan +y, thickness +z.

## CLI examples

```bash
# Voxel bounce-back (default)
python3 cli3d.py --shape mesh --stl-path samples/stl/unit_sphere.stl \
  --re 20 --nx 64 --ny 32 --nz 32 --stl-fit 0.25 --steps 500

# Guo IBM boundary
python3 cli3d.py --shape mesh --stl-path samples/stl/unit_sphere.stl \
  --mesh-bc ibm --re 20 --nx 64 --ny 32 --nz 32 --stl-fit 0.25 --steps 500

# Cylinder mesh
python3 cli3d.py --shape mesh --stl-path samples/stl/cylinder.stl \
  --re 100 --nx 80 --ny 40 --nz 40 --stl-fit 0.3 --steps 1000

# Simple airplane
python3 cli3d.py --shape mesh --stl-path samples/stl/simple_plane.stl \
  --re 100 --nx 96 --ny 48 --nz 48 --stl-fit 0.35 --steps 1000
```

## Web UI

1. `python3 webui.py`, open http://localhost:8017
2. Mode **3D**, shape **mesh**, then **Choose…** next to *STL file* -- or drop
   the file anywhere on the page
3. Set **Size** (cells across) and check the preview: parts thinner than a cell
   at that size fall between the cell centres (`simple_plane.stl`'s wings fill
   in only from ~80 cells across)

## GUI

1. Mode **3D**, shape **mesh**
2. Click **Browse** next to **stl_path** and pick a file from this folder
3. Set **mesh_bc** to `voxel` or `ibm`

Use a grid large enough to keep blockage below ~20% (`--stl-fit` controls obstacle size relative to the domain).
