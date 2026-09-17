"""
MPI z-slab domain decomposition for Solver3D.

Splits the 3D domain into z-slabs, one per MPI rank. Each rank runs a local
Solver3D on its slab (padded with 1-cell ghost layers top and bottom) and
exchanges halo data after each step via MPI Sendrecv.

Usage (4 ranks)::

    mpirun -n 4 python mpi_cli3d.py [cli3d args]

Design notes
------------
- Only z is decomposed; y and x are replicated on every rank.  Because x is
  replicated, every rank applies the *same* streamwise inlet/outlet BCs — the
  x boundaries are not a rank boundary and must not be treated as one.
- Ghost-layer indexing: local f has shape (Q, Nz_local+2, Ny, Nx).
  Interior cells: [1 .. Nz_local], ghosts: [0] (bottom) and [Nz_local+1] (top).
  One layer is enough for D3Q19, whose largest |e_z| is 1.
- Surface links in the ghost layers are dropped, because those cells belong to
  a neighbouring rank.  Keeping them would count their momentum exchange twice
  in the Allreduce — once on the owner and once on the rank holding the copy.
- Ghost rows are overwritten wholesale by the halo exchange, so bounce-back
  applied to them locally would be discarded anyway; the owner has already
  applied it before sending.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _slab_ranges(Nz_global: int, size: int) -> List[Tuple[int, int]]:
    """Return (z_lo, z_hi) for each rank in a balanced slab decomposition."""
    ranges = []
    for r in range(size):
        z_lo = r * Nz_global // size
        z_hi = (r + 1) * Nz_global // size
        ranges.append((z_lo, z_hi))
    return ranges


def restrict_links_to_interior(
    links: np.ndarray,
    q_vals: np.ndarray,
    nz_local: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop surface links that live in a ghost layer.

    Local z runs 0 .. nz_local+1, where 0 and nz_local+1 are ghosts owned by
    the neighbouring ranks.  A link at one of those z values describes a cell
    this rank only holds a copy of, so including it in a summed force reduction
    counts that cell twice.

    Returns the filtered ``(links, q_vals)``.
    """
    if links.shape[0] == 0:
        return links, q_vals
    z = links[:, 1]
    owned = (z >= 1) & (z <= nz_local)
    return np.ascontiguousarray(links[owned]), np.ascontiguousarray(q_vals[owned])


class MPISolver3D:
    """
    MPI-parallel z-slab wrapper around Solver3D.

    Parameters
    ----------
    comm         : MPI communicator (MPI.COMM_WORLD)
    Nz_global    : total number of z cells across all ranks
    Ny, Nx       : y (vertical) and x (streamwise) dimensions
    solid_global : bool ndarray (Nz_global, Ny, Nx) — full domain obstacle mask
    reduce_every : reduce and record forces every N steps (1 = every step,
                   matching the serial solvers; raise it to trade history
                   resolution for fewer collectives)
    **solver_kw  : passed to each rank's local Solver3D, unchanged
    """

    def __init__(
        self,
        comm: Any,
        Nz_global: int,
        Ny: int,
        Nx: int,
        solid_global: np.ndarray,
        reduce_every: int = 1,
        **solver_kw: Any,
    ) -> None:
        from .solver3d import Solver3D

        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()
        self.Nz_global = int(Nz_global)
        self.Ny = int(Ny)
        self.Nx = int(Nx)
        self.reduce_every = max(int(reduce_every), 1)

        # Slab assignment
        ranges = _slab_ranges(Nz_global, self._size)
        self._z_lo, self._z_hi = ranges[self._rank]
        self._Nz_local = self._z_hi - self._z_lo

        # Neighbour ranks (use None for boundary ranks)
        self._rank_below = self._rank - 1 if self._rank > 0 else None
        self._rank_above = self._rank + 1 if self._rank < self._size - 1 else None

        # Local solid: the owned slab plus one ghost layer either side.  z is
        # periodic across the global domain, so the ghosts wrap at the ends.
        z_idx = (np.arange(self._z_lo - 1, self._z_hi + 1) % Nz_global)
        solid_padded = np.ascontiguousarray(solid_global[z_idx])

        # Nz for inner solver includes 2 ghost layers
        Nz_inner = self._Nz_local + 2

        # x and y are replicated, so every rank keeps the caller's streamwise
        # and wall BCs.  Overriding them here would silently turn an open wind
        # tunnel into a streamwise-periodic duct with no inlet and no outlet.
        self._solver = Solver3D(
            Nz=Nz_inner, Ny=Ny, Nx=Nx,
            solid=solid_padded,
            **solver_kw,
        )

        # Forces must come only from cells this rank owns.
        self._solver.surface_links, self._solver.q_vals = restrict_links_to_interior(
            self._solver.surface_links, self._solver.q_vals, self._Nz_local
        )

        self.step_count: int = 0
        self.Cd_history: List[float] = []
        self.Cly_history: List[float] = []
        self.Clz_history: List[float] = []

    # ------------------------------------------------------------------

    @property
    def owned_slice(self) -> slice:
        """Slice selecting the interior (non-ghost) z range of the local arrays."""
        return slice(1, self._Nz_local + 1)

    def _halo_exchange(self) -> None:
        """Exchange one-cell ghost layers with neighbouring ranks."""
        from mpi4py import MPI
        f = self._solver.f
        Q, Nz_inner, Ny, Nx = f.shape
        MPI_NULL = MPI.PROC_NULL

        # z is periodic across the global domain, so the end ranks are each
        # other's neighbours rather than having no partner.
        below = self._rank_below if self._rank_below is not None else self._size - 1
        above = self._rank_above if self._rank_above is not None else 0
        if self._size == 1:
            # A single rank is its own neighbour; the inner solver's periodic
            # z streaming already wraps, so only the ghost copies are needed.
            f[:, Nz_inner - 1, :, :] = f[:, 1, :, :]
            f[:, 0, :, :] = f[:, Nz_inner - 2, :, :]
            return

        # Send interior bottom row to rank below; receive from rank above into top ghost
        send_bot = np.ascontiguousarray(f[:, 1, :, :])
        recv_top = np.empty((Q, Ny, Nx), dtype=f.dtype)
        self._comm.Sendrecv(
            sendbuf=send_bot, dest=below, sendtag=10,
            recvbuf=recv_top, source=above, recvtag=10,
        )
        self._solver.f[:, Nz_inner - 1, :, :] = recv_top

        # Send interior top row to rank above; receive from rank below into bottom ghost
        send_top = np.ascontiguousarray(f[:, Nz_inner - 2, :, :])
        recv_bot = np.empty((Q, Ny, Nx), dtype=f.dtype)
        self._comm.Sendrecv(
            sendbuf=send_top, dest=above, sendtag=20,
            recvbuf=recv_bot, source=below, recvtag=20,
        )
        self._solver.f[:, 0, :, :] = recv_bot

    def step(self) -> Tuple[float, float, float]:
        """Execute one global timestep including halo exchange. Returns local (Cd, Cly, Clz)."""
        Cd_local, Cly_local, Clz_local = self._solver._step()
        self._halo_exchange()
        self.step_count += 1
        return Cd_local, Cly_local, Clz_local

    def reduce_forces(self, local: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Sum per-rank coefficients into the global value."""
        from mpi4py import MPI

        local_arr = np.array(local, dtype=np.float64)
        global_arr = np.empty_like(local_arr)
        self._comm.Allreduce(local_arr, global_arr, op=MPI.SUM)
        return float(global_arr[0]), float(global_arr[1]), float(global_arr[2])

    def run(
        self,
        steps: int,
        check_every: int = 500,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run for `steps` timesteps with periodic MPI force reduction.

        Histories are recorded every ``reduce_every`` steps (one by default, to
        match the serial solvers — a history sampled at some other interval
        would silently rescale any Strouhal number derived from it).
        """
        for i in range(steps):
            local = self.step()

            if (i + 1) % self.reduce_every == 0:
                Cd, Cly, Clz = self.reduce_forces(local)
                self.Cd_history.append(Cd)
                self.Cly_history.append(Cly)
                self.Clz_history.append(Clz)

                if verbose and self._rank == 0 and (i + 1) % check_every == 0:
                    print(
                        f"  step {self.step_count:6d}  "
                        f"Cd={Cd:.4f}  Cly={Cly:.4f}  Clz={Clz:.4f}"
                    )

        return {
            "Cd_history": self.Cd_history,
            "Cly_history": self.Cly_history,
            "Clz_history": self.Clz_history,
            "steps_completed": self.step_count,
            "sample_interval": self.reduce_every,
            "rank": self._rank,
        }

    # ------------------------------------------------------------------

    def gather_field(self, local_field: np.ndarray) -> Optional[np.ndarray]:
        """
        Assemble a global (Nz_global, Ny, Nx) field on rank 0 from local slabs.

        ``local_field`` is indexed with the ghost layers included; only the
        owned interior is contributed.
        """
        interior = np.ascontiguousarray(local_field[self.owned_slice])
        gathered = self._comm.gather(interior, root=0)
        if self._rank != 0 or gathered is None:
            return None
        return np.concatenate(gathered, axis=0)

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
                "reduce_every": self.reduce_every,
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
