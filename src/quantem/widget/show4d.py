"""
show4d: General-purpose 4D data explorer widget.

Interactive dual-panel viewer for any 4D dataset (nav_x, nav_y, sig_x, sig_y).
Left panel shows a navigation image (real-space), right panel shows the signal
at the selected position. Supports ROI masking on the navigation panel to
average signals from a region.
"""

import pathlib

import numpy as np
import anywidget
import traitlets

from quantem.widget.array_utils import to_numpy


class Show4D(anywidget.AnyWidget):
    """
    General-purpose 4D data explorer.

    Displays a navigation image (left) and signal at selected position (right).
    Click/drag on the navigation image to explore. Draw ROI masks to average
    signals from a region.

    Parameters
    ----------
    data : array_like
        4D array of shape (nav_x, nav_y, sig_x, sig_y).
        Accepts NumPy, PyTorch, CuPy, or any np.asarray()-compatible object.
    nav_image : array_like, optional
        2D array (nav_x, nav_y) to use as navigation image.
        If not provided, defaults to mean over signal dimensions.
    title : str, optional
        Title displayed in the widget header.
    nav_pixel_size : float, optional
        Pixel size for navigation space scale bar.
    sig_pixel_size : float, optional
        Pixel size for signal space scale bar.
    nav_pixel_unit : str, default "px"
        Unit for navigation pixel size ("Å", "nm", "px").
    sig_pixel_unit : str, default "px"
        Unit for signal pixel size ("Å", "nm", "mrad", "px").

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(32, 32, 64, 64).astype(np.float32)
    >>> Show4D(data, title="My 4D Dataset")

    >>> # With custom navigation image
    >>> nav = data.mean(axis=(2, 3))
    >>> Show4D(data, nav_image=nav, nav_pixel_size=2.5, nav_pixel_unit="Å")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show4d.js"
    _css = pathlib.Path(__file__).parent / "static" / "show4d.css"

    # Navigation position
    pos_x = traitlets.Int(0).tag(sync=True)
    pos_y = traitlets.Int(0).tag(sync=True)

    # Shape
    nav_x = traitlets.Int(1).tag(sync=True)
    nav_y = traitlets.Int(1).tag(sync=True)
    sig_x = traitlets.Int(1).tag(sync=True)
    sig_y = traitlets.Int(1).tag(sync=True)

    # Data transfer (raw float32 bytes)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    nav_image_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Data ranges for JS normalization
    nav_data_min = traitlets.Float(0.0).tag(sync=True)
    nav_data_max = traitlets.Float(1.0).tag(sync=True)
    sig_data_min = traitlets.Float(0.0).tag(sync=True)
    sig_data_max = traitlets.Float(1.0).tag(sync=True)

    # ROI on navigation panel
    roi_mode = traitlets.Unicode("off").tag(sync=True)  # "off", "circle", "square", "rect"
    roi_reduce = traitlets.Unicode("mean").tag(sync=True)  # "mean", "max", "min", "sum"
    roi_center_x = traitlets.Float(0.0).tag(sync=True)
    roi_center_y = traitlets.Float(0.0).tag(sync=True)
    roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    roi_radius = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(10.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)

    # Statistics ([mean, min, max, std])
    nav_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)
    sig_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)

    # Scale bars
    nav_pixel_size = traitlets.Float(0.0).tag(sync=True)
    sig_pixel_size = traitlets.Float(0.0).tag(sync=True)
    nav_pixel_unit = traitlets.Unicode("px").tag(sync=True)
    sig_pixel_unit = traitlets.Unicode("px").tag(sync=True)

    # Title
    title = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        data,
        nav_image=None,
        title="",
        nav_pixel_size=None,
        sig_pixel_size=None,
        nav_pixel_unit="px",
        sig_pixel_unit="px",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Dataset duck typing
        if hasattr(data, "array") and hasattr(data, "sampling"):
            units = getattr(data, "units", ["pixels"] * 4)
            if nav_pixel_size is None and units[0] in ("Å", "angstrom", "A"):
                nav_pixel_size = float(data.sampling[0])
                nav_pixel_unit = "Å"
            elif nav_pixel_size is None and units[0] == "nm":
                nav_pixel_size = float(data.sampling[0]) * 10
                nav_pixel_unit = "Å"
            if sig_pixel_size is None and len(units) >= 4:
                if units[2] in ("Å", "angstrom", "A"):
                    sig_pixel_size = float(data.sampling[2])
                    sig_pixel_unit = "Å"
                elif units[2] == "nm":
                    sig_pixel_size = float(data.sampling[2]) * 10
                    sig_pixel_unit = "Å"
                elif units[2] == "mrad":
                    sig_pixel_size = float(data.sampling[2])
                    sig_pixel_unit = "mrad"
            if not title and hasattr(data, "name") and data.name:
                title = data.name
            data = data.array

        # Convert to NumPy float32
        data_np = to_numpy(data).astype(np.float32)

        if data_np.ndim != 4:
            raise ValueError(
                f"Expected 4D array (nav_x, nav_y, sig_x, sig_y), got {data_np.ndim}D"
            )

        self._data = data_np
        self.nav_x = data_np.shape[0]
        self.nav_y = data_np.shape[1]
        self.sig_x = data_np.shape[2]
        self.sig_y = data_np.shape[3]
        self.title = title

        # Scale bar
        self.nav_pixel_size = nav_pixel_size if nav_pixel_size is not None else 0.0
        self.sig_pixel_size = sig_pixel_size if sig_pixel_size is not None else 0.0
        self.nav_pixel_unit = nav_pixel_unit
        self.sig_pixel_unit = sig_pixel_unit

        # Compute navigation image
        if nav_image is not None:
            nav_img = to_numpy(nav_image).astype(np.float32)
        else:
            nav_img = data_np.mean(axis=(2, 3))
        self._nav_image = nav_img

        # Compute global signal range (sample for speed on large datasets)
        n_total = self.nav_x * self.nav_y
        if n_total > 100:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_total, 100, replace=False)
            sampled = data_np.reshape(n_total, self.sig_x, self.sig_y)[indices]
            self.sig_data_min = float(sampled.min())
            self.sig_data_max = float(sampled.max())
        else:
            self.sig_data_min = float(data_np.min())
            self.sig_data_max = float(data_np.max())

        # Nav image range and bytes (sent once)
        self.nav_data_min = float(nav_img.min())
        self.nav_data_max = float(nav_img.max())
        self.nav_image_bytes = nav_img.tobytes()
        self.nav_stats = [
            float(nav_img.mean()), float(nav_img.min()),
            float(nav_img.max()), float(nav_img.std()),
        ]

        # Initial position at center
        self.pos_x = self.nav_x // 2
        self.pos_y = self.nav_y // 2

        # ROI defaults
        default_roi_size = max(3, min(self.nav_x, self.nav_y) * 0.15)
        self.roi_center_x = float(self.nav_x / 2)
        self.roi_center_y = float(self.nav_y / 2)
        self.roi_center = [float(self.nav_x / 2), float(self.nav_y / 2)]
        self.roi_radius = float(default_roi_size)
        self.roi_width = float(default_roi_size * 2)
        self.roi_height = float(default_roi_size)

        # Observers
        self.observe(self._update_frame, names=["pos_x", "pos_y"])
        self.observe(self._on_roi_change, names=[
            "roi_mode", "roi_reduce", "roi_center_x", "roi_center_y",
            "roi_radius", "roi_width", "roi_height",
        ])
        self.observe(self._on_roi_center_change, names=["roi_center"])

        # Initial frame
        self._update_frame()

    def __repr__(self) -> str:
        return (
            f"Show4D(shape=({self.nav_x}, {self.nav_y}, {self.sig_x}, {self.sig_y}), "
            f"pos=({self.pos_x}, {self.pos_y}))"
        )

    @property
    def position(self) -> tuple[int, int]:
        return (self.pos_x, self.pos_y)

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self.pos_x, self.pos_y = value

    @property
    def nav_shape(self) -> tuple[int, int]:
        return (self.nav_x, self.nav_y)

    @property
    def sig_shape(self) -> tuple[int, int]:
        return (self.sig_x, self.sig_y)

    def _update_frame(self, change=None):
        frame = self._data[self.pos_x, self.pos_y]
        self.sig_stats = [
            float(frame.mean()), float(frame.min()),
            float(frame.max()), float(frame.std()),
        ]
        self.frame_bytes = frame.tobytes()

    def _on_roi_change(self, change=None):
        if self.roi_mode == "off":
            self._update_frame()
            return
        self._compute_roi_signal()

    def _on_roi_center_change(self, change=None):
        if self.roi_mode == "off":
            return
        if change and "new" in change:
            x, y = change["new"]
            self.unobserve(self._on_roi_change, names=["roi_center_x", "roi_center_y"])
            self.roi_center_x = x
            self.roi_center_y = y
            self.observe(self._on_roi_change, names=["roi_center_x", "roi_center_y"])
        self._compute_roi_signal()

    def _compute_roi_signal(self):
        row_coords = np.arange(self.nav_x)[:, None]
        col_coords = np.arange(self.nav_y)[None, :]
        cx, cy = self.roi_center_x, self.roi_center_y

        if self.roi_mode == "circle":
            mask = (row_coords - cx) ** 2 + (col_coords - cy) ** 2 <= self.roi_radius ** 2
        elif self.roi_mode == "square":
            mask = (np.abs(row_coords - cx) <= self.roi_radius) & (np.abs(col_coords - cy) <= self.roi_radius)
        elif self.roi_mode == "rect":
            mask = (np.abs(row_coords - cx) <= self.roi_height / 2) & (np.abs(col_coords - cy) <= self.roi_width / 2)
        else:
            self._update_frame()
            return

        n_positions = int(mask.sum())
        if n_positions == 0:
            self._update_frame()
            return

        reduce_ops = {"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum}
        op = reduce_ops.get(self.roi_reduce, np.mean)
        result = op(self._data[mask], axis=0).astype(np.float32)
        self.sig_stats = [
            float(result.mean()), float(result.min()),
            float(result.max()), float(result.std()),
        ]
        self.frame_bytes = result.tobytes()
