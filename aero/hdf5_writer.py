"""
HDF5 + XDMF output for ParaView/VisIt.

Writes each checkpoint step as a group inside one HDF5 file, then maintains
a companion .xdmf file that ParaView can open as a temporal collection.

h5py is optional — when absent every method is a no-op and `HAS_HDF5 = False`.

Usage (from solver or post-processing script)::

    writer = HDF5Writer("run.h5", Nz=Nz, Ny=Ny, Nx=Nx, dims=3)
    writer.write_step(step=100, ux=ux, uy=uy, uz=uz, rho=rho)
    writer.close()   # finalises the XDMF file

For 2D pass dims=2 and omit uz.
"""

import os
import pathlib
from typing import Optional
import numpy as np

try:
    import h5py as _h5py  # type: ignore
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False


_XDMF_HEADER = """\
<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
"""

_XDMF_FOOTER = """\
    </Grid>
  </Domain>
</Xdmf>
"""


def _xdmf_grid_3d(h5_basename: str, step: int, Nz: int, Ny: int, Nx: int) -> str:
    grp = f"step_{step:08d}"
    dims = f"{Nz} {Ny} {Nx}"
    return f"""\
      <Grid Name="{grp}" GridType="Uniform">
        <Time Value="{step}"/>
        <Topology TopologyType="3DCoRectMesh" NumberOfElements="{dims}"/>
        <Geometry GeometryType="ORIGIN_DXDYDZ">
          <DataItem Format="XML" DataType="Float" Dimensions="3">0.5 0.5 0.5</DataItem>
          <DataItem Format="XML" DataType="Float" Dimensions="3">1.0 1.0 1.0</DataItem>
        </Geometry>
        <Attribute Name="ux" AttributeType="Scalar" Center="Node">
          <DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="{dims}">
            {h5_basename}:/{grp}/ux
          </DataItem>
        </Attribute>
        <Attribute Name="uy" AttributeType="Scalar" Center="Node">
          <DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="{dims}">
            {h5_basename}:/{grp}/uy
          </DataItem>
        </Attribute>
        <Attribute Name="uz" AttributeType="Scalar" Center="Node">
          <DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="{dims}">
            {h5_basename}:/{grp}/uz
          </DataItem>
        </Attribute>
        <Attribute Name="rho" AttributeType="Scalar" Center="Node">
          <DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="{dims}">
            {h5_basename}:/{grp}/rho
          </DataItem>
        </Attribute>
      </Grid>
"""


def _xdmf_grid_2d(h5_basename: str, step: int, Ny: int, Nx: int) -> str:
    grp = f"step_{step:08d}"
    dims = f"{Ny} {Nx}"
    return f"""\
      <Grid Name="{grp}" GridType="Uniform">
        <Time Value="{step}"/>
        <Topology TopologyType="2DCoRectMesh" NumberOfElements="{dims}"/>
        <Geometry GeometryType="ORIGIN_DXDY">
          <DataItem Format="XML" DataType="Float" Dimensions="2">0.5 0.5</DataItem>
          <DataItem Format="XML" DataType="Float" Dimensions="2">1.0 1.0</DataItem>
        </Geometry>
        <Attribute Name="ux" AttributeType="Scalar" Center="Node">
          <DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="{dims}">
            {h5_basename}:/{grp}/ux
          </DataItem>
        </Attribute>
        <Attribute Name="uy" AttributeType="Scalar" Center="Node">
          <DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="{dims}">
            {h5_basename}:/{grp}/uy
          </DataItem>
        </Attribute>
        <Attribute Name="rho" AttributeType="Scalar" Center="Node">
          <DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="{dims}">
            {h5_basename}:/{grp}/rho
          </DataItem>
        </Attribute>
      </Grid>
"""


class HDF5Writer:
    """
    Writes LBM field snapshots to HDF5 + XDMF for ParaView/VisIt.

    Parameters
    ----------
    path  : str — path to the .h5 file (e.g. "outputs/run.h5")
    Nz, Ny, Nx : grid dimensions (Nz omitted / set to 0 for 2D)
    dims  : 2 or 3
    """

    def __init__(
        self,
        path: str,
        Ny: int,
        Nx: int,
        Nz: int = 0,
        dims: int = 3,
    ) -> None:
        self._active = HAS_HDF5
        if not self._active:
            return

        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._dims = dims
        self._Nz = Nz
        self._Ny = Ny
        self._Nx = Nx
        self._h5 = _h5py.File(str(self._path), "w")
        self._xdmf_path = self._path.with_suffix(".xdmf")
        self._h5_basename = self._path.name
        self._grids: list = []

    def write_step(
        self,
        step: int,
        ux: np.ndarray,
        uy: np.ndarray,
        rho: np.ndarray,
        uz: Optional[np.ndarray] = None,
    ) -> None:
        """Write one time step. uz is required for 3D."""
        if not self._active:
            return
        grp = self._h5.require_group(f"step_{step:08d}")
        grp.create_dataset("ux",  data=ux.astype(np.float64),  compression="gzip", compression_opts=4)
        grp.create_dataset("uy",  data=uy.astype(np.float64),  compression="gzip", compression_opts=4)
        grp.create_dataset("rho", data=rho.astype(np.float64), compression="gzip", compression_opts=4)
        if self._dims == 3 and uz is not None:
            grp.create_dataset("uz", data=uz.astype(np.float64), compression="gzip", compression_opts=4)
        grp.attrs["step"] = step

        if self._dims == 3:
            self._grids.append(_xdmf_grid_3d(self._h5_basename, step, self._Nz, self._Ny, self._Nx))
        else:
            self._grids.append(_xdmf_grid_2d(self._h5_basename, step, self._Ny, self._Nx))

    def close(self) -> None:
        """Flush HDF5 and write the companion XDMF file."""
        if not self._active:
            return
        self._h5.close()
        with open(str(self._xdmf_path), "w") as fh:
            fh.write(_XDMF_HEADER)
            for grid in self._grids:
                fh.write(grid)
            fh.write(_XDMF_FOOTER)

    def __enter__(self) -> "HDF5Writer":
        return self

    def __exit__(self, *_) -> None:
        self.close()
