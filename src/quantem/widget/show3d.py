"""
show3d: Interactive 3D stack viewer widget with advanced features.

For viewing a stack of 2D images (e.g., defocus sweep, time series, z-stack, movies).
Includes playback controls, statistics, ROI selection, FFT, and more.
"""

import pathlib
from enum import Enum

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy


class Colormap(str, Enum):
    """Available colormaps for image display."""

    INFERNO = "inferno"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    MAGMA = "magma"
    HOT = "hot"
    GRAY = "gray"

    def __str__(self) -> str:
        return self.value


class Show3D(anywidget.AnyWidget):
    """
    Interactive 3D stack viewer with advanced features for electron microscopy.

    View a stack of 2D images along a specific dimension (e.g., defocus sweep,
    time series, depth stack, in-situ movies). Includes playback controls,
    statistics panel, ROI selection, FFT view, and more.

    Parameters
    ----------
    data : array_like
        3D array of shape (N, height, width) where N is the stack dimension.
    labels : list of str, optional
        Labels for each slice (e.g., ["C10=-500nm", "C10=-400nm", ...]).
        If None, uses slice indices.
    title : str, optional
        Title to display above the image.
    cmap : str or Colormap, default Colormap.MAGMA
        Colormap name. Use Colormap enum (Colormap.MAGMA, Colormap.VIRIDIS, etc.)
        or string ("magma", "viridis", "gray", "inferno", "plasma").
    vmin : float, optional
        Minimum value for colormap. If None, uses data min.
    vmax : float, optional
        Maximum value for colormap. If None, uses data max.
    pixel_size : float, optional
        Pixel size in nm for scale bar display.
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast (ignores vmin/vmax).
    percentile_low : float, default 1.0
        Lower percentile for auto-contrast.
    percentile_high : float, default 99.0
        Upper percentile for auto-contrast.
    fps : float, default 5.0
        Frames per second for playback.
    timestamps : list of float, optional
        Timestamps for each frame (e.g., seconds or dose values).
    timestamp_unit : str, default "s"
        Unit for timestamps (e.g., "s", "ms", "e/A2").

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show3D
    >>>
    >>> # View defocus sweep
    >>> labels = [f"C10={c10:.0f}nm" for c10 in np.linspace(-500, -200, 12)]
    >>> Show3D(stack, labels=labels, title="Defocus Sweep")
    >>>
    >>> # View in-situ movie with timestamps
    >>> times = np.arange(100) * 0.1  # 100 frames at 10 fps
    >>> Show3D(movie, timestamps=times, timestamp_unit="s", fps=30)
    >>>
    >>> # With scale bar
    >>> Show3D(data, pixel_size=0.5, title="HRTEM")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show3d.js"
    _css = pathlib.Path(__file__).parent / "static" / "show3d.css"

    # =========================================================================
    # Core State
    # =========================================================================
    slice_idx = traitlets.Int(0).tag(sync=True)
    n_slices = traitlets.Int(1).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("magma").tag(sync=True)

    # =========================================================================
    # Playback Controls
    # =========================================================================
    playing = traitlets.Bool(False).tag(sync=True)
    reverse = traitlets.Bool(False).tag(sync=True)  # Play in reverse direction
    boomerang = traitlets.Bool(False).tag(sync=True)  # Ping-pong playback
    fps = traitlets.Float(5.0).tag(sync=True)  # Default 5 FPS for easier control
    loop = traitlets.Bool(True).tag(sync=True)
    loop_start = traitlets.Int(0).tag(sync=True)  # Start frame for loop range
    loop_end = traitlets.Int(-1).tag(sync=True)  # End frame for loop (-1 = last)
    bookmarked_frames = traitlets.List(traitlets.Int()).tag(sync=True)

    # =========================================================================
    # Statistics Panel
    # =========================================================================
    show_stats = traitlets.Bool(True).tag(sync=True)
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    percentile_low = traitlets.Float(1.0).tag(sync=True)
    percentile_high = traitlets.Float(99.0).tag(sync=True)
    data_min = traitlets.Float(0.0).tag(sync=True)
    data_max = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)  # nm/pixel, 0 = no scale bar
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Timestamps / Dose
    # =========================================================================
    timestamps = traitlets.List(traitlets.Float()).tag(sync=True)
    timestamp_unit = traitlets.Unicode("s").tag(sync=True)
    current_timestamp = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # ROI Selection
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_shape = traitlets.Unicode("circle").tag(sync=True)  # circle, square, rectangle
    roi_row = traitlets.Int(0).tag(sync=True)
    roi_col = traitlets.Int(0).tag(sync=True)
    roi_radius = traitlets.Int(10).tag(sync=True)  # For circle/square: radius or half-size
    roi_width = traitlets.Int(20).tag(sync=True)  # For rectangle
    roi_height = traitlets.Int(20).tag(sync=True)  # For rectangle
    roi_mean = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Sizing
    # =========================================================================
    image_width_px = traitlets.Int(0).tag(sync=True)  # If 0, use frontend defaults

    # =========================================================================
    # Analysis Panels (FFT + Histogram shown together)
    # =========================================================================
    show_fft = traitlets.Bool(False).tag(sync=True)

    # =========================================================================
    # Export (GIF / ZIP of PNGs)
    # =========================================================================
    _gif_export_requested = traitlets.Bool(False).tag(sync=True)
    _gif_data = traitlets.Bytes(b"").tag(sync=True)
    _zip_export_requested = traitlets.Bool(False).tag(sync=True)
    _zip_data = traitlets.Bytes(b"").tag(sync=True)

    # =========================================================================
    # Playback Buffer (sliding prefetch)
    # =========================================================================
    _buffer_bytes = traitlets.Bytes(b"").tag(sync=True)
    _buffer_start = traitlets.Int(0).tag(sync=True)
    _buffer_count = traitlets.Int(0).tag(sync=True)
    _prefetch_request = traitlets.Int(-1).tag(sync=True)

    def __init__(
        self,
        data,
        labels: list[str] | None = None,
        title: str = "",
        cmap: str | Colormap = Colormap.MAGMA,
        vmin: float | None = None,
        vmax: float | None = None,
        pixel_size: float = 0.0,
        log_scale: bool = False,
        auto_contrast: bool = False,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        fps: float = 5.0,
        timestamps: list[float] | None = None,
        timestamp_unit: str = "s",
        show_fft: bool = False,
        show_stats: bool = True,
        image_width_px: int = 0,
        buffer_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Check if data is a Dataset3d and extract metadata
        _extracted_title = None
        _extracted_pixel_size = None
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            _extracted_title = data.name if data.name else None
            # sampling is (z_sampling, y_sampling, x_sampling) - use y/x for pixel size
            if hasattr(data, "sampling") and len(data.sampling) >= 3:
                sampling_val = float(data.sampling[1])
                # pixel_size is in nm — convert if units are Å
                if hasattr(data, "units"):
                    units = list(data.units)
                    if units[1] in ("Å", "angstrom", "A"):
                        sampling_val = sampling_val / 10  # Å → nm
                _extracted_pixel_size = sampling_val
            data = data.array

        # Convert input to NumPy (handles NumPy, CuPy, PyTorch)
        data = to_numpy(data)

        # Ensure 3D
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")

        # Store data as float32 numpy array
        self._data = data.astype(np.float32)

        # Dimensions
        self.n_slices = int(self._data.shape[0])
        self.height = int(self._data.shape[1])
        self.width = int(self._data.shape[2])

        # Color range (global across all frames)
        self._vmin_user = vmin
        self._vmax_user = vmax
        self._vmin = vmin if vmin is not None else float(self._data.min())
        self._vmax = vmax if vmax is not None else float(self._data.max())
        self.data_min = float(self._data.min())
        self.data_max = float(self._data.max())

        # Labels
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [str(i) for i in range(self.n_slices)]

        # Title and colormap - use extracted title if not explicitly provided
        self.title = title if title else (_extracted_title or "")
        self.cmap = str(cmap)  # Convert Colormap enum to string

        # Use extracted pixel_size if not explicitly provided
        if pixel_size == 0.0 and _extracted_pixel_size is not None:
            pixel_size = _extracted_pixel_size

        # Display options
        self.pixel_size = pixel_size
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.fps = fps

        # Timestamps
        if timestamps is not None:
            self.timestamps = [float(t) for t in timestamps]
        else:
            self.timestamps = []
        self.timestamp_unit = timestamp_unit
        self.show_fft = show_fft
        self.show_stats = show_stats
        self.image_width_px = image_width_px
        frame_bytes = self.height * self.width * 4  # float32
        max_buffer_bytes = 64 * 1024 * 1024  # 64 MB cap per transfer
        min_buffer_frames = 8  # guarantee at least 8 frames for large images
        max_frames = max(min_buffer_frames, max_buffer_bytes // frame_bytes)
        self._buffer_size = min(buffer_size, self.n_slices, max_frames)

        # Initial position at middle
        self.slice_idx = int(self.n_slices // 2)

        # Observers
        self.observe(self._on_slice_change, names=["slice_idx"])
        self.observe(
            self._on_roi_change,
            names=[
                "roi_row",
                "roi_col",
                "roi_radius",
                "roi_active",
                "roi_shape",
                "roi_width",
                "roi_height",
            ],
        )
        self.observe(self._on_gif_export, names=["_gif_export_requested"])
        self.observe(self._on_zip_export, names=["_zip_export_requested"])
        self.observe(self._on_playing_change, names=["playing"])
        self.observe(self._on_prefetch, names=["_prefetch_request"])

        # Initial update
        self._update_all()

    def _get_color_range(self, frame: np.ndarray) -> tuple[float, float]:
        """Get vmin/vmax based on current settings."""
        if self.auto_contrast:
            vmin = float(np.percentile(frame, self.percentile_low))
            vmax = float(np.percentile(frame, self.percentile_high))
        else:
            vmin = self._vmin
            vmax = self._vmax
        return vmin, vmax

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame to uint8 with current display settings."""
        # Apply log scale if enabled
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))

        vmin, vmax = self._get_color_range(frame)

        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def _update_all(self):
        """Update frame, stats, and all derived data. Uses hold_sync for batched transfer."""
        frame = self._data[self.slice_idx]
        with self.hold_sync():
            self.stats_mean = float(frame.mean())
            self.stats_min = float(frame.min())
            self.stats_max = float(frame.max())
            self.stats_std = float(frame.std())
            if self.timestamps and self.slice_idx < len(self.timestamps):
                self.current_timestamp = self.timestamps[self.slice_idx]
            if self.roi_active:
                self._update_roi_mean(frame)
            self.frame_bytes = frame.tobytes()

    def _update_roi_mean(self, frame: np.ndarray):
        """Compute mean value within ROI based on shape."""
        r, c = np.ogrid[0 : self.height, 0 : self.width]

        if self.roi_shape == "circle":
            mask = (c - self.roi_col) ** 2 + (r - self.roi_row) ** 2 <= self.roi_radius**2
        elif self.roi_shape == "square":
            half = self.roi_radius
            mask = (np.abs(c - self.roi_col) <= half) & (np.abs(r - self.roi_row) <= half)
        elif self.roi_shape == "rectangle":
            half_w = self.roi_width // 2
            half_h = self.roi_height // 2
            mask = (np.abs(c - self.roi_col) <= half_w) & (
                np.abs(r - self.roi_row) <= half_h
            )
        else:
            # Default to circle
            mask = (c - self.roi_col) ** 2 + (r - self.roi_row) ** 2 <= self.roi_radius**2

        if mask.sum() > 0:
            self.roi_mean = float(frame[mask].mean())
        else:
            self.roi_mean = 0.0

    def _send_buffer(self, start_idx: int):
        end_idx = start_idx + self._buffer_size
        if end_idx <= self.n_slices:
            chunk = self._data[start_idx:end_idx]
        else:
            chunk = np.concatenate(
                [self._data[start_idx:], self._data[: end_idx - self.n_slices]]
            )
        with self.hold_sync():
            self._buffer_start = int(start_idx)
            self._buffer_count = int(chunk.shape[0])
            self._buffer_bytes = chunk.tobytes()

    def _on_playing_change(self, change=None):
        if self.playing:
            self._send_buffer(self.slice_idx)

    def _on_prefetch(self, change=None):
        if self._prefetch_request >= 0 and self.playing:
            self._send_buffer(self._prefetch_request % self.n_slices)

    def _on_slice_change(self, change=None):
        if self.playing:
            return
        self._update_all()

    def _on_roi_change(self, change=None):
        """Handle ROI change."""
        if self.roi_active:
            self._update_roi_mean(self._data[self.slice_idx])

    # =========================================================================
    # Public Methods
    # =========================================================================

    def play(self):
        """Start playback."""
        self.playing = True

    def pause(self):
        """Pause playback."""
        self.playing = False

    def stop(self):
        """Stop playback and reset to beginning."""
        self.playing = False
        self.slice_idx = 0

    def set_roi(self, row: int, col: int, radius: int = 10):
        """Set ROI position and size."""
        self.roi_row = int(row)
        self.roi_col = int(col)
        self.roi_radius = int(radius)
        self.roi_active = True

    def _on_gif_export(self, change=None):
        if not self._gif_export_requested:
            return
        self._gif_export_requested = False
        self._generate_gif()

    def _generate_gif(self):
        import io

        from matplotlib import colormaps
        from PIL import Image

        start = max(0, self.loop_start)
        end = self.loop_end if self.loop_end >= 0 else self.n_slices - 1
        end = min(end, self.n_slices - 1)

        cmap_fn = colormaps.get_cmap(self.cmap)
        duration_ms = int(1000 / max(0.1, self.fps))

        pil_frames = []
        for i in range(start, end + 1):
            frame = self._data[i]
            normalized = self._normalize_frame(frame)
            rgba = cmap_fn(normalized / 255.0)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(rgb))

        if not pil_frames:
            return

        buf = io.BytesIO()
        pil_frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        self._gif_data = buf.getvalue()

    def _on_zip_export(self, change=None):
        if not self._zip_export_requested:
            return
        self._zip_export_requested = False
        self._generate_zip()

    def _generate_zip(self):
        import io
        import zipfile

        from matplotlib import colormaps
        from PIL import Image

        start = max(0, self.loop_start)
        end = self.loop_end if self.loop_end >= 0 else self.n_slices - 1
        end = min(end, self.n_slices - 1)

        cmap_fn = colormaps.get_cmap(self.cmap)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(start, end + 1):
                frame = self._data[i]
                normalized = self._normalize_frame(frame)
                rgba = cmap_fn(normalized / 255.0)
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                img = Image.fromarray(rgb)
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                label = self.labels[i] if self.labels else str(i).zfill(4)
                zf.writestr(f"frame_{label}.png", img_buf.getvalue())
        self._zip_data = buf.getvalue()
