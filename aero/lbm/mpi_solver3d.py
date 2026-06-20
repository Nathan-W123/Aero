"""
MPI z-slab domain decomposition for Solver3D.

Splits the 3D domain into z-slabs, one per MPI rank. Each rank runs a local
Solver3D on its slab (padded with 1-cell ghost layers top and bottom) and
exchanges halo data after each step via MPI Sendrecv.

Usage (4 ranks)::

    mpirun -n 4 python mpi_cli3d.py [cli3d args]

Design notes
------------
- Only z-direction is decomposed; y and x are replicated on all ranks.
- Ghost-layer indexing: local f has shape (Q, Nz_local+2, Ny, Nx).
  Interior cells: [1 .. Nz_local], ghosts: [0] (bottom) and [Nz_local+1] (top).
- Inlet BC is applied only on rank 0; outlet BC only on rank size-1.
- Forces are computed locally and reduced with MPI.Allreduce(SUM) every
  check_every steps.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np


def _slab_ranges(Nz_global: int, size: int) -> List[Tuple[int, int]]:
    """Return (z_lo, z_hi) for each rank in a balanced slab decomposition."""
    ranges = []
    for r in range(size):
        z_lo = r * Nz_global // size
        z_hi = (r + 1) * Nz_global // size
        ranges.append((z_lo, z_hi))
    return ranges


class MPISolver3D:
    """
    MPI-parallel z-slab wrapper around Solver3D.

    Parameters
    ----------
    comm         : MPI communicator (MPI.COMM_WORLD)
    Nz_global    : total number of z cells across all ranks
    Ny, Nx       : y (vertical) and x (streamwise) dimensions
    solid_global : bool ndarray (Nz_global, Ny, Nx) — full domain obstacle mask
    **solver_kw  : passed to each rank's local Solver3D
    """

    def __init__(
        self,
        comm: Any,
        Nz_global: int,
        Ny: int,
        Nx: int,
        solid_global: np.ndarray,
        **solver_kw: Any,
    ) -> None:
        from .solver3d import Solver3D

        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()
        self.Nz_global = int(Nz_global)
        self.Ny = int(Ny)
        self.Nx = int(Nx)

        # Slab assignment
        ranges = _slab_ranges(Nz_global, self._size)
        self._z_lo, self._z_hi = ranges[self._rank]
        self._Nz_local = self._z_hi - self._z_lo

        # Neighbour ranks (use None for boundary ranks)
        self._rank_below = self._rank - 1 if self._rank > 0 else None
        self._rank_above = self._rank + 1 if self._rank < self._size - 1 else None

        # Local solid: extract slab + 1-cell ghost pads (repeat boundary)
        s_lo = max(0, self._z_lo - 1)
        s_hi = min(Nz_global, self._z_hi + 1)
        solid_padded = solid_global[s_lo:s_hi]
        # If at boundary, pad with extra row
        if self._z_lo == 0:
            solid_padded = np.concatenate(
                [solid_global[0:1], solid_padded], axis=0
            )
        if self._z_hi == Nz_global:
            solid_padded = np.concatenate(
                [solid_padded, solid_global[-1:]], axis=0
            )

        # Nz for inner solver includes 2 ghost layers
        Nz_inner = self._Nz_local + 2

        # Force periodic streamwise BC so inner solver doesn't apply z BCs
        kw = dict(solver_kw)
        kw["streamwise_bc"] = "periodic"

        # Only rank 0 uses inlet; only last rank uses outlet
        if self._rank != 0:
            kw["inlet_bc"] = "pressure"
        if self._rank != self._size - 1:
            kw["outlet_bc"] = "zerogradient"

        self._solver = Solver3D(
            Nz=Nz_inner, Ny=Ny, Nx=Nx,
            solid=solid_padded,
            **kw,
        )

        self.step_count: int = 0
        self.Cd_history: List[float] = []
        self.Cly_history: List[float] = []
        self.Clz_history: List[float] = []

    # ------------------------------------------------------------------

    def _halo_exchange(self) -> None:
        """Exchange one-cell ghost layers with neighbouring ranks."""
        from mpi4py import MPI
        f = self._solver.f
        Q, Nz_inner, Ny, Nx = f.shape
        MPI_NULL = MPI.PROC_NULL

        rank_below = self._rank_below if self._rank_below is not None else MPI_NULL
        rank_above = self._rank_above if self._rank_above is not None else MPI_NULL

        # Send interior bottom row to rank below; receive from rank above into top ghost
        send_bot = np.ascontiguousarray(f[:, 1, :, :])
        recv_top = np.empty((Q, Ny, Nx), dtype=f.dtype)
        self._comm.Sendrecv(
            sendbuf=send_bot, dest=rank_below, sendtag=10,
            recvbuf=recv_top, source=rank_above, recvtag=10,
        )
        if self._rank_above is not None:
            self._solver.f[:, Nz_inner - 1, :, :] = recv_top

        # Send interior top row to rank above; receive from rank below into bottom ghost
        send_top = np.ascontiguousarray(f[:, Nz_inner - 2, :, :])
        recv_bot = np.empty((Q, Ny, Nx), dtype=f.dtype)
        self._comm.Sendrecv(
            sendbuf=send_top, dest=rank_above, sendtag=20,
            recvbuf=recv_bot, source=rank_below, recvtag=20,
        )
        if self._rank_below is not None:
            self._solver.f[:, 0, :, :] = recv_bot

    def step(self) -> Tuple[float, float, float]:
        """Execute one global timestep including halo exchange. Returns (Cd, Cly, Clz)."""
        Cd_local, Cly_local, Clz_local = self._solver._step()
        self._halo_exchange()
        self.step_count += 1
        return Cd_local, Cly_local, Clz_local

    def run(
        self,
        steps: int,
        check_every: int = 500,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run for `steps` timesteps with periodic MPI force reduction.
        Returns a results dict on rank 0; other ranks return partial data.
        """
        from mpi4py import MPI

        for i in range(steps):
            Cd_l, Cly_l, Clz_l = self.step()

            if (i + 1) % check_every == 0:
                # Reduce forces across all ranks
                local_arr = np.array([Cd_l, Cly_l, Clz_l], dtype=np.float64)
                global_arr = np.empty_like(local_arr)
                self._comm.Allreduce(local_arr, global_arr, op=MPI.SUM)
                Cd, Cly, Clz = global_arr[0], global_arr[1], global_arr[2]
                self.Cd_history.append(float(Cd))
                self.Cly_history.append(float(Cly))
                self.Clz_history.append(float(Clz))
                if verbose and self._rank == 0:
                    print(f"  step {self.step_count:6d}  Cd={Cd:.4f}  Cly={Cly:.4f}  Clz={Clz:.4f}")

        return {
            "Cd_history": self.Cd_history,
            "Cly_history": self.Cly_history,
            "Clz_history": self.Clz_history,
            "steps_completed": self.step_count,
            "rank": self._rank,
        }

    def save_checkpoint(self, base_path: str) -> None:
        """Save per-rank checkpoint. Rank 0 also saves a manifest."""
        import json
        path = f"{base_path}_rank{self._rank}.npz"
        np.savez_compressed(path, f=self._solver.f, step_count=self.step_count)
        if self._rank == 0:
            manifest = {
                "Nz_global": self.Nz_global,
                "Ny": self.Ny,
                "Nx": self.Nx,
                "nranks": self._size,
                "step_count": self.step_count,
            }
            with open(f"{base_path}_manifest.json", "w") as fp:
                json.dump(manifest, fp)

    @classmethod
    def from_checkpoint(
        cls,
        comm: Any,
        base_path: str,
        Nz_global: int,
        Ny: int,
        Nx: int,
        solid_global: np.ndarray,
        **solver_kw: Any,
    ) -> "MPISolver3D":
        """Load a previously saved MPI checkpoint."""
        obj = cls(comm, Nz_global, Ny, Nx, solid_global, **solver_kw)
        rank = comm.Get_rank()
        data = np.load(f"{base_path}_rank{rank}.npz")
        obj._solver.f = data["f"]
        obj.step_count = int(data["step_count"])
        obj._solver.step_count = obj.step_count
        return obj
