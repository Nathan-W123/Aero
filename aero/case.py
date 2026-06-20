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
        self.results = {
            "mode": mode,
            "Cd_mean": result["Cd_mean"],
            "Cd_std": result["Cd_std"],
            "steps_run": len(result.get("Cd_history", [])),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        if mode == "3d":
            self.results["Cly_mean"] = result.get("Cly_mean")
            self.results["Cly_std"] = result.get("Cly_std")
            self.results["Clz_mean"] = result.get("Clz_mean")
            self.results["Clz_std"] = result.get("Clz_std")
        else:
            self.results["Cl_mean"] = result.get("Cl_mean")
            self.results["Cl_std"] = result.get("Cl_std")

        self.case_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as fh:
            json.dump(self.results, fh, indent=2)

        history = {"Cd_history": [float(x) for x in result.get("Cd_history", [])]}
        if mode == "3d":
            history["Cly_history"] = [float(x) for x in result.get("Cly_history", [])]
            history["Clz_history"] = [float(x) for x in result.get("Clz_history", [])]
        else:
            history["Cl_history"] = [float(x) for x in result.get("Cl_history", [])]
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
            # flow
            "re":       args.re,
            "u0":       args.u0,
            # grid
            "nx":       args.nx,
            "ny":       args.ny,
            # run
            "steps":    args.steps,
            "wall_bc":  getattr(args, "wall_bc", None),
            "inlet_bc": getattr(args, "inlet_bc", None),
            "outlet_bc": getattr(args, "outlet_bc", None),
            "rho_in":   getattr(args, "rho_in", None),
            "rho_out":  getattr(args, "rho_out", None),
            "backend":  getattr(args, "backend", None),
            "collision": getattr(args, "collision", None),
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
        }
        if mode == "2d":
            case_config["inlet_bc"] = params.get("inlet_bc")
            case_config["inlet_perturbation"] = float(params.get("inlet_perturbation", "0") or "0")
        else:
            case_config["nz"] = int(params["nz"])
            case_config["viz3d"] = params.get("viz3d")

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
