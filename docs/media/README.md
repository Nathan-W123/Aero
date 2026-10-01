# Media

Screenshots, animations and graphs of Aero made for the portfolio site
(nathan-w123/Website, `public/projects/aero/`), at commit `b04d414` of this
branch. Runs were made in the browser UI (`python3 webui.py`) on 4 CPU cores,
captured with headless Chromium on a virtual clock so animations are smooth.

| File | What it is | How it was made |
|---|---|---|
| `aircraft_wake_orbit.{mp4,webm,webp}` | `samples/stl/tunnel_plane.stl` in a 144×72×96 tunnel, Wake view (\|u − U∞\|) replayed as a time-lapse while the camera orbits (ping-pong loop) | 3D, mesh, orientation "as in the file", Size 48, regularized, D3Q19, Re 100, 4,000 steps |
| `webui_aircraft_3d.png` | The full UI on that run | Same run, 2× pixel density |
| `webui_cylinder_smoke.{mp4,webm}` | The full UI on a 2D cylinder, smoke advected through the solved field | 2D cylinder r 20, 400×200, Re 100, BGK, 24,000 steps: Cd 1.6552 ± 0.0103 |
| `webui_cylinder_re100.png` | The same run, still | |
| `cylinder_smoke_orbit.mp4` | The viewport alone on that case, smoke while the camera orbits (ping-pong loop) | Re-run at `017cd7f`, same settings, 1.5× pixel density |
| `sphere_drag_confinement_refinement.png` | Sphere drag above Schiller–Naumann vs blockage (Re 20, Re 100), and local refinement cost vs Cd | Numbers from commits 54b92ba, 89dec89, 7b17774, b04d414 |
| `cpu_throughput.png` | MLUPS before / after the tiled, fused kernels | Numbers from commit fa7c70f |
