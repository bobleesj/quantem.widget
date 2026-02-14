"""
Show3DVolume: Orthogonal slice viewer for 3D volumetric data.
Displays XY, XZ, YZ planes with interactive sliders.
All slicing happens in JavaScript for instant response.
"""
import pathlib
from typing import Optional, Union
import anywidget
import numpy as np
import traitlets
from quantem.widget.array_utils import to_numpy


class Show3DVolume(anywidget.AnyWidget):
    """
    3D volume viewer with three orthogonal slice planes.
    Parameters
    ----------
    data : array_like
        3D array of shape (nz, ny, nx).
    title : str, optional
        Title displayed above the viewer.
    cmap : str, default "inferno"
        Colormap name.
    pixel_size_angstrom : float, optional
        Pixel size in angstroms for scale bar.
    show_stats : bool, default True
        Show per-slice statistics.
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast.
    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show3DVolume
    >>> volume = np.random.rand(64, 64, 64).astype(np.float32)
    >>> Show3DVolume(volume, title="My Volume", cmap="viridis")
    """
    _esm = pathlib.Path(__file__).parent / "static" / "show3dvolume.js"
    _css = pathlib.Path(__file__).parent / "static" / "show3dvolume.css"
    # Volume dimensions
    nx = traitlets.Int(1).tag(sync=True)
    ny = traitlets.Int(1).tag(sync=True)
    nz = traitlets.Int(1).tag(sync=True)
    # Slice positions
    slice_x = traitlets.Int(0).tag(sync=True)
    slice_y = traitlets.Int(0).tag(sync=True)
    slice_z = traitlets.Int(0).tag(sync=True)
    # Raw volume data (sent once)
    volume_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Display
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    # Scale bar
    pixel_size_angstrom = traitlets.Float(0.0).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    # UI
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_crosshair = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    # Axis labels (dim 0, 1, 2 → default "Z", "Y", "X")
    dim_labels = traitlets.List(traitlets.Unicode(), default_value=["Z", "Y", "X"]).tag(sync=True)
    # Stats (3 values: xy, xz, yz)
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)

    def __init__(
        self,
        data: Union[np.ndarray, "torch.Tensor"],
        title: str = "",
        cmap: str = "inferno",
        pixel_size_angstrom: float = 0.0,
        scale_bar_visible: bool = True,
        show_controls: bool = True,
        show_stats: bool = True,
        show_crosshair: bool = True,
        show_fft: bool = False,
        log_scale: bool = False,
        auto_contrast: bool = False,
        dim_labels: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if dim_labels is not None:
            self.dim_labels = dim_labels
        data = to_numpy(data)
        if data.ndim != 3:
            raise ValueError(f"Show3DVolume requires 3D data, got {data.ndim}D")
        self._data = data.astype(np.float32)
        self.nz, self.ny, self.nx = self._data.shape
        # Default to middle slices
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2
        self.title = title
        self.cmap = cmap
        self.pixel_size_angstrom = pixel_size_angstrom
        self.scale_bar_visible = scale_bar_visible
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.show_crosshair = show_crosshair
        self.show_fft = show_fft
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self._compute_stats()
        self.volume_bytes = self._data.tobytes()
        self.observe(self._on_slice_change, names=["slice_x", "slice_y", "slice_z"])

    def _compute_stats(self):
        """Compute statistics for the 3 current slices."""
        slices = [
            self._data[self.slice_z, :, :],
            self._data[:, self.slice_y, :],
            self._data[:, :, self.slice_x],
        ]
        self.stats_mean = [float(np.mean(s)) for s in slices]
        self.stats_min = [float(np.min(s)) for s in slices]
        self.stats_max = [float(np.max(s)) for s in slices]
        self.stats_std = [float(np.std(s)) for s in slices]

    def _on_slice_change(self, change):
        self._compute_stats()
