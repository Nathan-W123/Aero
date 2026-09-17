"""
Running time-averaged flow statistics.

A single instantaneous snapshot is not a result for an unsteady flow.  Anything
with a wake, a shear layer or a subgrid model needs the time-averaged field —
mean velocity, RMS fluctuations, Reynolds stresses, mean pressure — and those
have to be accumulated as the run goes, because storing every snapshot is not
an option.

Accumulation uses Welford's online algorithm rather than running sums of
``x`` and ``x**2``.  The distinction matters for a *streaming* accumulator: a
running ``sum(x**2)`` grows without bound while the variance it encodes is the
difference of two nearly equal numbers, and LBM makes that ratio bad — mean
velocity ~1e-2 against fluctuations that can be ~1e-5, over runs of 1e5-1e6
steps.  Welford keeps the running mean and the centred second moment
separately, so the subtraction never happens and the result stays accurate
however small the fluctuation and however long the run.  (A one-shot
``np.var`` over a stored array is fine — numpy sums pairwise — but storing
every snapshot is exactly what this class exists to avoid.)

Reynolds stresses use the same co-moment update, which is the two-variable
form of Welford:

    M2_ab <- M2_ab + (a - mean_a_old) * (b - mean_b_new)

Usage::

    stats = FlowStatistics(shape=(Ny, Nx), components=2)
    ...
    stats.update(ux, uy, rho=rho)         # once per sampled step
    result = stats.summary()              # mean, rms, Reynolds stresses, TKE
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

CS2 = 1.0 / 3.0


class FlowStatistics:
    """
    Welford accumulator for velocity, its fluctuations, and pressure.

    Parameters
    ----------
    shape      : grid shape, ``(Ny, Nx)`` or ``(Nz, Ny, Nx)``
    components : 2 or 3 velocity components
    start_step : steps before this are not sampled, so the initial transient
                 does not pollute the mean.  The caller decides what counts as
                 settled; ``run()`` exposes it as ``stats_start``.
    """

    #: names of the independent Reynolds-stress components, in order
    STRESS_2D = ("uu", "uv", "vv")
    STRESS_3D = ("uu", "uv", "uw", "vv", "vw", "ww")

    def __init__(
        self,
        shape: Tuple[int, ...],
        components: int = 2,
        start_step: int = 0,
    ) -> None:
        if components not in (2, 3):
            raise ValueError("components must be 2 or 3")
        self.shape = tuple(int(s) for s in shape)
        self.components = int(components)
        self.start_step = int(start_step)
        self.count = 0

        self._mean = [np.zeros(self.shape) for _ in range(components)]
        # centred co-moments; index pairs follow STRESS_2D / STRESS_3D order
        self._pairs = self._pair_indices()
        self._m2 = {name: np.zeros(self.shape) for name in self.stress_names}
        self._rho_mean = np.zeros(self.shape)
        self._rho_m2 = np.zeros(self.shape)

    # ------------------------------------------------------------------

    @property
    def stress_names(self) -> Sequence[str]:
        return self.STRESS_2D if self.components == 2 else self.STRESS_3D

    def _pair_indices(self):
        names = self.STRESS_2D if self.components == 2 else self.STRESS_3D
        pairs = []
        for a in range(self.components):
            for b in range(a, self.components):
                pairs.append((a, b))
        return dict(zip(names, pairs))

    # ------------------------------------------------------------------

    def should_sample(self, step: int) -> bool:
        return step >= self.start_step

    def update(
        self,
        *velocity: np.ndarray,
        rho: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fold one snapshot into the running statistics.

        ``velocity`` is ``(ux, uy)`` or ``(ux, uy, uz)``.
        """
        if len(velocity) != self.components:
            raise ValueError(
                f"expected {self.components} velocity components, got {len(velocity)}"
            )

        self.count += 1
        inv_n = 1.0 / self.count

        # Welford: keep the pre-update means for the co-moment step
        deltas_old = [u - m for u, m in zip(velocity, self._mean)]
        for m, d in zip(self._mean, deltas_old):
            m += d * inv_n
        deltas_new = [u - m for u, m in zip(velocity, self._mean)]

        for name, (a, b) in self._pairs.items():
            self._m2[name] += deltas_old[a] * deltas_new[b]

        if rho is not None:
            d_old = rho - self._rho_mean
            self._rho_mean += d_old * inv_n
            self._rho_m2 += d_old * (rho - self._rho_mean)

    # ------------------------------------------------------------------

    def _variance(self, m2: np.ndarray) -> np.ndarray:
        """Population variance; zero until there are at least two samples."""
        if self.count < 2:
            return np.zeros(self.shape)
        return m2 / self.count

    @property
    def mean_velocity(self) -> Tuple[np.ndarray, ...]:
        return tuple(m.copy() for m in self._mean)

    def reynolds_stress(self, name: str) -> np.ndarray:
        """``<u'_a u'_b>`` for one component pair, e.g. ``"uv"``."""
        return self._variance(self._m2[name])

    @property
    def turbulent_kinetic_energy(self) -> np.ndarray:
        """``k = 1/2 <u'_i u'_i>``."""
        diagonal = ("uu", "vv") if self.components == 2 else ("uu", "vv", "ww")
        return 0.5 * sum(self.reynolds_stress(n) for n in diagonal)

    @property
    def mean_pressure(self) -> np.ndarray:
        """``p = cs^2 rho`` from the mean density."""
        return CS2 * self._rho_mean

    def rms_velocity(self) -> Tuple[np.ndarray, ...]:
        diagonal = ("uu", "vv") if self.components == 2 else ("uu", "vv", "ww")
        return tuple(np.sqrt(self.reynolds_stress(n)) for n in diagonal)

    # ------------------------------------------------------------------

    def fields(self) -> Dict[str, np.ndarray]:
        """Every accumulated field, keyed for export."""
        labels = ("ux", "uy", "uz")[: self.components]
        out: Dict[str, np.ndarray] = {}
        for label, m in zip(labels, self._mean):
            out[f"mean_{label}"] = m.copy()
        for label, r in zip(labels, self.rms_velocity()):
            out[f"rms_{label}"] = r
        for name in self.stress_names:
            out[f"reynolds_{name}"] = self.reynolds_stress(name)
        out["tke"] = self.turbulent_kinetic_energy
        out["mean_rho"] = self._rho_mean.copy()
        out["mean_pressure"] = self.mean_pressure
        out["rms_rho"] = np.sqrt(self._variance(self._rho_m2))
        return out

    def summary(self) -> Dict[str, object]:
        """
        Compact JSON-safe summary: sample bookkeeping plus scalar reductions.

        The full fields stay out of this so a results file does not carry the
        whole domain; use :meth:`fields` for those.
        """
        if self.count == 0:
            return {"samples": 0, "start_step": self.start_step, "converged": False}

        tke = self.turbulent_kinetic_energy
        mean_mag = np.sqrt(sum(m * m for m in self._mean))
        return {
            "samples": int(self.count),
            "start_step": int(self.start_step),
            "mean_speed_max": float(np.max(mean_mag)),
            "mean_speed_mean": float(np.mean(mean_mag)),
            "tke_max": float(np.max(tke)),
            "tke_mean": float(np.mean(tke)),
            "reynolds_stress_peak": {
                name: float(np.max(np.abs(self.reynolds_stress(name))))
                for name in self.stress_names
            },
            "mean_pressure_mean": float(np.mean(self.mean_pressure)),
            "mean_pressure_range": [
                float(np.min(self.mean_pressure)),
                float(np.max(self.mean_pressure)),
            ],
        }

    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Accumulator state, for checkpointing mid-run."""
        out = {
            "stats_count": np.array(self.count, dtype=np.int64),
            "stats_start_step": np.array(self.start_step, dtype=np.int64),
            "stats_rho_mean": self._rho_mean,
            "stats_rho_m2": self._rho_m2,
        }
        for i, m in enumerate(self._mean):
            out[f"stats_mean_{i}"] = m
        for name, m2 in self._m2.items():
            out[f"stats_m2_{name}"] = m2
        return out

    def load_state_dict(self, data) -> None:
        """Restore from :meth:`state_dict` (an npz mapping is fine)."""
        self.count = int(data["stats_count"])
        self.start_step = int(data["stats_start_step"])
        self._rho_mean = np.array(data["stats_rho_mean"])
        self._rho_m2 = np.array(data["stats_rho_m2"])
        self._mean = [np.array(data[f"stats_mean_{i}"]) for i in range(self.components)]
        self._m2 = {name: np.array(data[f"stats_m2_{name}"]) for name in self.stress_names}
