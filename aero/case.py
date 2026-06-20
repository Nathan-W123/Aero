"""
Simulation case management.

A "case" is a directory containing:
  config.json   — all input parameters (geometry, solver, run settings)
  results.json  — scalar outputs (Cd, Cl, Re, steps, timing)
  *.png         — visualisation outputs

Usage
-----
  case = SimulationCase.from_args(args, derived_params)
  case.save_config()
  # ... run solver ...
  case.save_results(result, elapsed_seconds)

  # Reload later
  case = SimulationCase.load("cases/cylinder_re100/")
  print(case.config)
"""

import json
import pathlib
import datetime
from typing import Optional

from .benchmarks import build_uncertainty_report, build_validation_report


class SimulationCase:
    """
    Container for one simulation case: its config, paths, and results.

    Parameters
    ----------
    name       : str          — human-readable case identifier
    cases_root : str | Path   — parent directory for all cases (default: ./cases)
    config     : dict         — full parameter dict (geometry + solver + run settings)
    """

    def __init__(self, name: str, cases_root="./cases", config: Optional[dict] = None):
        self.name       = name
        self.case_dir   = pathlib.Path(cases_root) / name
        self.config     = config or {}
        self.results    = {}

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def config_path(self) -> pathlib.Path:
        return self.case_dir / "config.json"

    @property
    def results_path(self) -> pathlib.Path:
        return self.case_dir / "results.json"

    @property
    def outputs_dir(self) -> pathlib.Path:
        return self.case_dir

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_config(self) -> None:
        """Write config.json to the case directory (creates dir if needed)."""
        self.case_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as fh:
            json.dump(self.config, fh, indent=2)
        print(f"  Case config saved → {self.config_path}")

    def save_results(self, result: dict, elapsed_seconds: float) -> None:
        """
        Write results.json. Stores scalar outputs and coefficient history sidecar.
        """
        mode = self.config.get("mode", "2d")
        shape = self.config.get("shape", "unknown")
        validation = build_validation_report(
            mode=mode,
            shape=shape,
            params=self.config,
            cd=result.get("Cd_mean"),
            grid_cd_values=result.get("grid_cd_values"),
        )
        uncertainty = build_uncertainty_report(
            mode=mode,
            shape=shape,
            params=self.config,
            result=result,
        )
        self.results = {
            "mode": mode,
            "shape": shape,
            "Cd_mean": result["Cd_mean"],
            "Cd_std": result["Cd_std"],
            "steps_run": len(result.get("Cd_history", [])),
            "steps_completed": int(result.get("steps_completed", len(result.get("Cd_history", [])))),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "stop_reason": result.get("stop_reason", "max_steps"),
            "validation_report": {
                "benchmark_status": validation.benchmark_status,
                "benchmark_message": validation.benchmark_message,
                "grid_status": validation.grid_status,
                "grid_message": validation.grid_message,
                "bc_warnings": list(validation.bc_warnings),
                "collision_status": validation.collision_status,
                "collision_message": validation.collision_message,
                "overall_ok": validation.overall_ok,
            },
            "uncertainty_report": {
                "overall_status": uncertainty.overall_status,
                "summary": uncertainty.summary,
                "components": uncertainty.components,
            },
        }
        if "Cd_p_mean" in result:
            self.results["Cd_p_mean"] = result["Cd_p_mean"]
            self.results["Cd_v_mean"] = result["Cd_v_mean"]
        if "Cm_mean" in result:
            self.results["Cm_mean"] = result["Cm_mean"]
            self.results["Cm_std"] = result.get("Cm_std")
        if mode == "3d":
            self.results["Cly_mean"] = result.get("Cly_mean")
            self.results["Cly_std"] = result.get("Cly_std")
            self.results["Clz_mean"] = result.get("Clz_mean")
            self.results["Clz_std"] = result.get("Clz_std")
            for key in ("Cmx_mean", "Cmy_mean", "Cmz_mean", "Cmx_std", "Cmy_std", "Cmz_std"):
                if key in result:
                    self.results[key] = result.get(key)
        else:
            self.results["Cl_mean"] = result.get("Cl_mean")
            self.results["Cl_std"] = result.get("Cl_std")
        if "observables" in result:
            self.results["observables"] = result["observables"]
        if result.get("scalar_stats") is not None:
            self.results["scalar_stats"] = result["scalar_stats"]
        if result.get("scalar_validation") is not None:
            self.results["scalar_validation"] = result["scalar_validation"]
        if result.get("autoconfig_report") is not None:
            self.results["autoconfig_report"] = result["autoconfig_report"]

        self.case_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as fh:
            json.dump(self.results, fh, indent=2)

        history = {"Cd_history": [float(x) for x in result.get("Cd_history", [])]}
        if mode == "3d":
            history["Cly_history"] = [float(x) for x in result.get("Cly_history", [])]
            history["Clz_history"] = [float(x) for x in result.get("Clz_history", [])]
            for key in ("Cmx_history", "Cmy_history", "Cmz_history"):
                if key in result:
                    history[key] = [float(x) for x in result.get(key, [])]
        else:
            history["Cl_history"] = [float(x) for x in result.get("Cl_history", [])]
            if "Cm_history" in result:
                history["Cm_history"] = [float(x) for x in result.get("Cm_history", [])]
        if result.get("convergence_report") is not None:
            history["convergence_report"] = {
                "converged": bool(result["convergence_report"].converged),
                "window": int(result["convergence_report"].window),
                "ratio": float(result["convergence_report"].ratio),
                "mean_abs": float(result["convergence_report"].mean_abs),
                "std": float(result["convergence_report"].std),
                "threshold": float(result["convergence_report"].threshold),
            }
        if result.get("strouhal_report") is not None:
            report = result["strouhal_report"]
            history["strouhal_report"] = {
                "strouhal": None if report.strouhal is None else float(report.strouhal),
                "peak_frequency": None if report.peak_frequency is None else float(report.peak_frequency),
                "peak_amplitude": float(report.peak_amplitude),
                "sample_count": int(report.sample_count),
                "window": int(report.window),
                "stationary": bool(report.stationary),
                "relative_drift": None if report.relative_drift is None else float(report.relative_drift),
            }
        history_path = self.case_dir / "history.json"
        with open(history_path, "w") as fh:
            json.dump(history, fh, indent=2)

        print(f"  Case results saved → {self.results_path}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, case_dir: str) -> "SimulationCase":
        """
        Load an existing case from its directory.

        Parameters
        ----------
        case_dir : str — path to the case directory

        Returns
        -------
        SimulationCase with .config and .results populated
        """
        path = pathlib.Path(case_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Case directory not found: {path}")

        name = path.name
        inst = cls(name=name, cases_root=str(path.parent))

        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as fh:
                inst.config = json.load(fh)

        results_path = path / "results.json"
        if results_path.exists():
            with open(results_path) as fh:
                inst.results = json.load(fh)

        return inst

    # ------------------------------------------------------------------
    # Factory: build from CLI args
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, args, derived: dict, cases_root: str = "./cases") -> "SimulationCase":
        """
        Build a SimulationCase from parsed argparse Namespace and derived params.

        The case name is auto-generated from shape + Re, with a timestamp
        suffix to avoid collisions.
        """
        ts   = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        name = f"{args.shape}_re{args.re:.0f}_{ts}"

        config = {
            # geometry
            "shape":    args.shape,
            "radius":   getattr(args, "radius", None),
            "width":    getattr(args, "width", None),
            "height":   getattr(args, "height", None),
            "cx_frac":  getattr(args, "cx_frac", 1/3),
            "cy_frac":  getattr(args, "cy_frac", 0.5),
            "polygon_verts": getattr(args, "polygon_verts", None),
            "image_path": getattr(args, "image_path", None),
            "image_threshold": getattr(args, "image_threshold", None),
            "image_invert": getattr(args, "image_invert", None),
            "stl_path": getattr(args, "stl_path", None),
            "stl_fit": getattr(args, "stl_fit", None),
            # flow
            "re":       args.re,
            "u0":       args.u0,
            # grid
            "nx":       args.nx,
            "ny":       args.ny,
            # run
            "steps":    args.steps,
            "wall_bc":  getattr(args, "wall_bc", None) or "slip",
            "inlet_bc": getattr(args, "inlet_bc", None) or "velocity",
            "outlet_bc": getattr(args, "outlet_bc", None) or "convective",
            "streamwise_bc": getattr(args, "streamwise_bc", None) or "open",
            "rho_in":   getattr(args, "rho_in", None),
            "rho_out":  getattr(args, "rho_out", None),
            "backend":  getattr(args, "backend", None),
            "collision": getattr(args, "collision", None),
            "trt_lambda": getattr(args, "trt_lambda", None),
            "sponge_cells": getattr(args, "sponge_cells", None),
            "sponge_strength": getattr(args, "sponge_strength", None),
            "les": getattr(args, "les", None),
            "les_cs": getattr(args, "les_cs", None),
            "les_model": getattr(args, "les_model", None),
            "inlet_perturbation": getattr(args, "inlet_perturbation", None),
            "mesh_bc": getattr(args, "mesh_bc", None),
            "mesh_orient": getattr(args, "mesh_orient", None),
            "mesh_rot_x": getattr(args, "mesh_rot_x", None),
            "mesh_rot_y": getattr(args, "mesh_rot_y", None),
            "mesh_rot_z": getattr(args, "mesh_rot_z", None),
            "body_force_x": getattr(args, "body_force_x", None),
            "body_force_y": getattr(args, "body_force_y", None),
            "body_force_z": getattr(args, "body_force_z", None),
            "auto_stop": getattr(args, "auto_stop", None),
            "wall_velocity_top": getattr(args, "wall_velocity_top", None),
            "wall_velocity_bottom": getattr(args, "wall_velocity_bottom", None),
            "synthetic_inflow": getattr(args, "synthetic_inflow", None),
            "synthetic_inflow_intensity": getattr(args, "synthetic_inflow_intensity", None),
            "autoconfigure": getattr(args, "autoconfigure", None),
            "thermal": getattr(args, "thermal", None),
            "T_hot": getattr(args, "T_hot", None),
            "T_cold": getattr(args, "T_cold", None),
            "alpha_T": getattr(args, "alpha_T", None),
            "buoyancy": getattr(args, "buoyancy", None),
            "g_gravity": getattr(args, "g_gravity", None),
            "beta": getattr(args, "beta", None),
            "T_ref": getattr(args, "T_ref", None),
            # derived LBM params (informational)
            "D":        derived["D"],
            "nu_lbm":   derived["nu_lbm"],
            "tau":      derived["tau"],
            "omega":    derived["omega"],
            "Ma":       derived["Ma"],
        }

        return cls(name=name, cases_root=cases_root, config=config)

    @classmethod
    def from_gui(cls, config, cases_root: str = "./cases") -> "SimulationCase":
        """Build a SimulationCase from the desktop GUI configuration."""
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        mode = config.mode
        shape = config.shape_2d if mode == "2d" else config.shape_3d
        params = config.params_2d if mode == "2d" else config.params_3d
        name = f"{shape}_re{float(params['re']):.0f}_{ts}"

        case_config: dict = {
            "mode": mode,
            "shape": shape,
            "re": float(params["re"]),
            "u0": float(params["u0"]),
            "nx": int(params["nx"]),
            "ny": int(params["ny"]),
            "steps": int(params["steps"]),
            "wall_bc": params.get("wall_bc"),
            "outlet_bc": params.get("outlet_bc"),
            "backend": params.get("backend"),
            "collision": params.get("collision"),
            "trt_lambda": float(params.get("trt_lambda", "0.25") or "0.25"),
            "sponge_cells": int(params.get("sponge_cells", "0") or "0"),
            "sponge_strength": float(params.get("sponge_strength", "0.1") or "0.1"),
            "les": params.get("les", "0") in ("1", "true", "True"),
            "les_cs": float(params.get("les_cs", "0.16") or "0.16"),
            "les_model": params.get("les_model", "smagorinsky"),
            "wall_velocity_top": float(params.get("wall_velocity_top", "0") or "0"),
            "wall_velocity_bottom": float(params.get("wall_velocity_bottom", "0") or "0"),
            "synthetic_inflow": params.get("synthetic_inflow", "0") in ("1", "true", "True"),
            "synthetic_inflow_intensity": float(params.get("synthetic_inflow_intensity", "0.03") or "0.03"),
            "thermal": params.get("thermal", "0") in ("1", "true", "True"),
            "T_hot": float(params.get("T_hot", "1.0") or "1.0"),
            "T_cold": float(params.get("T_cold", "0.0") or "0.0"),
            "alpha_T": float(params.get("alpha_T", "1e-3") or "1e-3"),
            "buoyancy": params.get("buoyancy", "0") in ("1", "true", "True"),
            "g_gravity": float(params.get("g_gravity", "0.0") or "0.0"),
            "beta": float(params.get("beta", "1e-3") or "1e-3"),
            "T_ref": float(params.get("T_ref", "0.5") or "0.5"),
        }
        if mode == "2d":
            case_config["inlet_bc"] = params.get("inlet_bc")
            case_config["inlet_perturbation"] = float(params.get("inlet_perturbation", "0") or "0")
        else:
            case_config["nz"] = int(params["nz"])
            case_config["viz3d"] = params.get("viz3d")
            case_config["inlet_perturbation"] = float(params.get("inlet_perturbation", "0") or "0")
            if shape == "mesh":
                case_config["stl_path"] = params.get("stl_path")
                case_config["stl_fit"] = float(params.get("stl_fit", "0.35") or "0.35")
                case_config["mesh_bc"] = params.get("mesh_bc", "voxel")
                case_config["mesh_orient"] = params.get("mesh_orient", "auto")
                case_config["mesh_rot_x"] = float(params.get("mesh_rot_x", "0") or "0")
                case_config["mesh_rot_y"] = float(params.get("mesh_rot_y", "0") or "0")
                case_config["mesh_rot_z"] = float(params.get("mesh_rot_z", "0") or "0")

        for key in ("radius", "width", "height", "depth", "length"):
            if key in params and params[key] != "":
                case_config[key] = float(params[key])

        return cls(name=name, cases_root=cases_root, config=case_config)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"Case: {self.name}", f"Dir:  {self.case_dir}"]
        if self.config:
            lines.append(f"Re={self.config.get('re')}  "
                         f"shape={self.config.get('shape')}  "
                         f"steps={self.config.get('steps')}")
        if self.results:
            if self.config.get("mode") == "3d" or "Cly_mean" in self.results:
                lines.append(
                    f"Cd={self.results.get('Cd_mean', '?'):.4f}  "
                    f"Cl_y={self.results.get('Cly_mean', '?'):.4f}"
                )
            else:
                lines.append(
                    f"Cd={self.results.get('Cd_mean', '?'):.4f}  "
                    f"Cl={self.results.get('Cl_mean', '?'):.4f}"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<SimulationCase '{self.name}'>"
