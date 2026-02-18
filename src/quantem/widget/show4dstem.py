"""
show4dstem: Fast interactive 4D-STEM viewer widget.

Apple MPS GPU limit: PyTorch's MPS backend (Apple Silicon) has a hard limit
of ~2.1 billion elements (INT_MAX = 2^31 - 1) per tensor. Datasets exceeding
this automatically fall back to CPU, which is still fast on Apple Silicon
thanks to unified memory (CPU and GPU share the same RAM).

CUDA GPUs do not have this limit.

Common 4D-STEM sizes (float32):

    Scan     Detector   Elements     Size    MPS?
    128×128  128×128       268M    1.0 GB    yes
    128×128  256×256     1,074M    4.0 GB    yes
    256×256  128×128     1,074M    4.0 GB    yes
    256×256  192×192     2,416M    9.0 GB    no (auto CPU, still fast)
    256×256  256×256     4,295M   16.0 GB    no (auto CPU, still fast)
    512×512  256×256    17,180M   64.0 GB    no (auto CPU)

To reduce data size, bin k-space at the dataset level before viewing:

    dataset = dataset.bin(2, axes=(2, 3))  # 2x2 k-space binning
    widget = Show4DSTEM(dataset)
"""

import json
import pathlib
from typing import Self

import anywidget
import numpy as np
import torch
import traitlets

from quantem.core.config import validate_device
from quantem.widget.array_utils import to_numpy


# ============================================================================
# Constants
# ============================================================================
DEFAULT_BF_RATIO = 0.125  # BF disk radius as fraction of detector size (1/8)
SPARSE_MASK_THRESHOLD = 0.2  # Use sparse indexing below this mask coverage
MIN_LOG_VALUE = 1e-10  # Minimum value for log scale to avoid log(0)
DEFAULT_VI_ROI_RATIO = 0.15  # Default VI ROI size as fraction of scan dimension


class Show4DSTEM(anywidget.AnyWidget):
    """
    Fast interactive 4D-STEM viewer with advanced features.

    Optimized for speed with binary transfer and pre-normalization.
    Works with NumPy and PyTorch arrays.

    Parameters
    ----------
    data : Dataset4dstem or array_like
        Dataset4dstem object (calibration auto-extracted) or 4D array
        of shape (scan_rows, scan_cols, det_rows, det_cols).
    scan_shape : tuple, optional
        If data is flattened (N, det_rows, det_cols), provide scan dimensions.
    pixel_size : float, optional
        Pixel size in Å (real-space). Used for scale bar.
        Auto-extracted from Dataset4dstem if not provided.
    k_pixel_size : float, optional
        Detector pixel size in mrad (k-space). Used for scale bar.
        Auto-extracted from Dataset4dstem if not provided.
    center : tuple[float, float], optional
        (center_row, center_col) of the diffraction pattern in pixels.
        If not provided, defaults to detector center.
    bf_radius : float, optional
        Bright field disk radius in pixels. If not provided, estimated as 1/8 of detector size.
    precompute_virtual_images : bool, default True
        Precompute BF/ABF/LAADF/HAADF virtual images for preset switching.
    log_scale : bool, default False
        Use log scale for better dynamic range visualization.

    Examples
    --------
    >>> # From Dataset4dstem (calibration auto-extracted)
    >>> from quantem.core.io.file_readers import read_emdfile_to_4dstem
    >>> dataset = read_emdfile_to_4dstem("data.h5")
    >>> Show4DSTEM(dataset)

    >>> # From raw array with manual calibration
    >>> import numpy as np
    >>> data = np.random.rand(64, 64, 128, 128)
    >>> Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)

    >>> # With raster animation
    >>> widget = Show4DSTEM(dataset)
    >>> widget.raster(step=2, interval_ms=50)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show4dstem.js"
    _css = pathlib.Path(__file__).parent / "static" / "show4dstem.css"

    # Position in scan space
    pos_row = traitlets.Int(0).tag(sync=True)
    pos_col = traitlets.Int(0).tag(sync=True)

    # Shape of scan space (for slider bounds)
    shape_rows = traitlets.Int(1).tag(sync=True)
    shape_cols = traitlets.Int(1).tag(sync=True)

    # Detector shape for frontend
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)

    # Raw float32 frame as bytes (JS handles scale/colormap for real-time interactivity)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Global min/max for DP normalization (computed once from sampled frames)
    dp_global_min = traitlets.Float(0.0).tag(sync=True)
    dp_global_max = traitlets.Float(1.0).tag(sync=True)

    # =========================================================================
    # Detector Calibration (for presets and scale bar)
    # =========================================================================
    center_col = traitlets.Float(0.0).tag(sync=True)  # Detector center col
    center_row = traitlets.Float(0.0).tag(sync=True)  # Detector center row
    bf_radius = traitlets.Float(0.0).tag(sync=True)  # BF disk radius (pixels)

    # =========================================================================
    # ROI Drawing (for virtual imaging)
    # roi_radius is multi-purpose by mode:
    #   - circle: radius of circle
    #   - square: half-size (distance from center to edge)
    #   - annular: outer radius (roi_radius_inner = inner radius)
    #   - rect: uses roi_width/roi_height instead
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_mode = traitlets.Unicode("point").tag(sync=True)
    roi_center_col = traitlets.Float(0.0).tag(sync=True)
    roi_center_row = traitlets.Float(0.0).tag(sync=True)
    # Compound trait for batched row+col updates (JS sends both at once, 1 observer fires)
    roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    roi_radius = traitlets.Float(10.0).tag(sync=True)
    roi_radius_inner = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(20.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)

    # =========================================================================
    # Virtual Image (ROI-based, updates as you drag ROI on DP)
    # =========================================================================
    virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)  # Raw float32
    vi_data_min = traitlets.Float(0.0).tag(sync=True)  # Min of current VI for normalization
    vi_data_max = traitlets.Float(1.0).tag(sync=True)  # Max of current VI for normalization

    # =========================================================================
    # VI ROI (real-space region selection for summed DP)
    # =========================================================================
    vi_roi_mode = traitlets.Unicode("off").tag(sync=True)  # "off", "circle", "rect"
    vi_roi_center_row = traitlets.Float(0.0).tag(sync=True)
    vi_roi_center_col = traitlets.Float(0.0).tag(sync=True)
    vi_roi_radius = traitlets.Float(5.0).tag(sync=True)
    vi_roi_width = traitlets.Float(10.0).tag(sync=True)
    vi_roi_height = traitlets.Float(10.0).tag(sync=True)
    summed_dp_bytes = traitlets.Bytes(b"").tag(sync=True)  # Summed DP from VI ROI
    summed_dp_count = traitlets.Int(0).tag(sync=True)  # Number of positions summed

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(1.0).tag(sync=True)  # Å per pixel (real-space)
    k_pixel_size = traitlets.Float(1.0).tag(sync=True)  # mrad per pixel (k-space)
    k_calibrated = traitlets.Bool(False).tag(sync=True)  # True if k-space has mrad calibration

    # =========================================================================
    # Path Animation (programmatic crosshair control)
    # =========================================================================
    path_playing = traitlets.Bool(False).tag(sync=True)
    path_index = traitlets.Int(0).tag(sync=True)
    path_length = traitlets.Int(0).tag(sync=True)
    path_interval_ms = traitlets.Int(100).tag(sync=True)  # ms between frames
    path_loop = traitlets.Bool(True).tag(sync=True)  # loop when reaching end

    # =========================================================================
    # Auto-detection trigger (frontend sets to True, backend resets to False)
    # =========================================================================
    auto_detect_trigger = traitlets.Bool(False).tag(sync=True)

    # =========================================================================
    # Statistics for display (mean, min, max, std)
    # =========================================================================
    dp_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)
    vi_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)
    mask_dc = traitlets.Bool(True).tag(sync=True)  # Mask center pixel for DP stats
    log_scale = traitlets.Bool(False).tag(sync=True)  # Log scale for DP display

    # Export (GIF)
    _gif_export_requested = traitlets.Bool(False).tag(sync=True)
    _gif_data = traitlets.Bytes(b"").tag(sync=True)

    # Line Profile (for DP panel)
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)

    def __init__(
        self,
        data: "Dataset4dstem | np.ndarray",
        scan_shape: tuple[int, int] | None = None,
        pixel_size: float | None = None,
        k_pixel_size: float | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        precompute_virtual_images: bool = False,
        log_scale: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.log_scale = log_scale  # Set trait value from constructor

        # Extract calibration from Dataset4dstem if provided
        k_calibrated = False
        if hasattr(data, "sampling") and hasattr(data, "array"):
            # Dataset4dstem: extract calibration and array
            # sampling = [scan_rows, scan_cols, det_rows, det_cols]
            units = getattr(data, "units", ["pixels"] * 4)
            if pixel_size is None and units[0] in ("Å", "angstrom", "A", "nm"):
                pixel_size = float(data.sampling[0])
                if units[0] == "nm":
                    pixel_size *= 10  # Convert nm to Å
            if k_pixel_size is None and units[2] in ("mrad", "1/Å", "1/A"):
                k_pixel_size = float(data.sampling[2])
                k_calibrated = True
            data = data.array

        # Store calibration values (default to 1.0 if not provided)
        self.pixel_size = pixel_size if pixel_size is not None else 1.0
        self.k_pixel_size = k_pixel_size if k_pixel_size is not None else 1.0
        self.k_calibrated = k_calibrated or (k_pixel_size is not None)
        # Path animation (configured via set_path() or raster())
        self._path_points: list[tuple[int, int]] = []
        # Convert to NumPy then PyTorch tensor using quantem device config
        data_np = to_numpy(data)
        device_str, _ = validate_device(None)  # Get device from quantem config
        # MPS backend can't handle tensors >INT_MAX elements; fall back to CPU
        # (still fast on Apple Silicon thanks to unified memory)
        if data_np.size > 2**31 - 1 and device_str == "mps":
            device_str = "cpu"
        self._device = torch.device(device_str)
        self._data = torch.from_numpy(data_np.astype(np.float32)).to(self._device)
        # Remove saturated hot pixels (65535 for uint16, 255 for uint8)
        # Use torch.where instead of boolean indexing to avoid nonzero() INT_MAX limit
        saturated_value = 65535.0 if data_np.dtype == np.uint16 else 255.0 if data_np.dtype == np.uint8 else None
        if saturated_value is not None:
            self._data = torch.where(self._data >= saturated_value, torch.zeros(1, device=self._device), self._data)
        # Handle flattened data — use self._data.shape (accounts for binning)
        ndim = self._data.ndim
        if ndim == 3:
            if scan_shape is not None:
                self._scan_shape = scan_shape
            else:
                n = self._data.shape[0]
                side = int(n ** 0.5)
                if side * side != n:
                    raise ValueError(
                        f"Cannot infer square scan_shape from N={n}. "
                        f"Provide scan_shape explicitly."
                    )
                self._scan_shape = (side, side)
            self._det_shape = (self._data.shape[1], self._data.shape[2])
        elif ndim == 4:
            self._scan_shape = (self._data.shape[0], self._data.shape[1])
            self._det_shape = (self._data.shape[2], self._data.shape[3])
        else:
            raise ValueError(f"Expected 3D or 4D array, got {ndim}D")

        self.shape_rows = self._scan_shape[0]
        self.shape_cols = self._scan_shape[1]
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]
        # Initial position at center
        self.pos_row = self.shape_rows // 2
        self.pos_col = self.shape_cols // 2
        # Precompute global range for consistent scaling (hot pixels already removed)
        self.dp_global_min = max(float(self._data.min()), MIN_LOG_VALUE)
        self.dp_global_max = float(self._data.max())
        # Cache coordinate tensors for mask creation (avoid repeated torch.arange)
        self._det_row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[:, None]
        self._det_col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[None, :]
        self._scan_row_coords = torch.arange(self.shape_rows, device=self._device, dtype=torch.float32)[:, None]
        self._scan_col_coords = torch.arange(self.shape_cols, device=self._device, dtype=torch.float32)[None, :]
        # Setup center and BF radius
        det_size = min(self.det_rows, self.det_cols)
        if center is not None and bf_radius is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
            self.bf_radius = float(bf_radius)
        elif center is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
            self.bf_radius = det_size * DEFAULT_BF_RATIO
        elif bf_radius is not None:
            self.center_col = float(self.det_cols / 2)
            self.center_row = float(self.det_rows / 2)
            self.bf_radius = float(bf_radius)
        else:
            # Neither provided - auto-detect from data
            # Set defaults first (will be overwritten by auto-detect)
            self.center_col = float(self.det_cols / 2)
            self.center_row = float(self.det_rows / 2)
            self.bf_radius = det_size * DEFAULT_BF_RATIO
            # Auto-detect center and bf_radius from the data
            self.auto_detect_center(update_roi=False)

        # Pre-compute and cache common virtual images (BF, ABF, ADF)
        # Each cache stores (bytes, stats) tuple
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        if precompute_virtual_images:
            self._precompute_common_virtual_images()

        # Update frame when position changes (scale/colormap handled in JS)
        self.observe(self._update_frame, names=["pos_row", "pos_col"])
        # Observe individual ROI params (for backward compatibility)
        self.observe(self._on_roi_change, names=[
            "roi_center_col", "roi_center_row", "roi_radius", "roi_radius_inner",
            "roi_active", "roi_mode", "roi_width", "roi_height"
        ])
        # Observe compound roi_center for batched updates from JS
        self.observe(self._on_roi_center_change, names=["roi_center"])

        # Initialize default ROI at BF center
        self.roi_center_col = self.center_col
        self.roi_center_row = self.center_row
        self.roi_center = [self.center_row, self.center_col]
        self.roi_radius = self.bf_radius * 0.5  # Start with half BF radius
        self.roi_active = True
        
        # Compute initial virtual image and frame
        self._compute_virtual_image_from_roi()
        self._update_frame()
        
        # Path animation: observe index changes from frontend
        self.observe(self._on_path_index_change, names=["path_index"])
        self.observe(self._on_gif_export, names=["_gif_export_requested"])

        # Auto-detect trigger: observe changes from frontend
        self.observe(self._on_auto_detect_trigger, names=["auto_detect_trigger"])

        # VI ROI: observe changes for summed DP computation
        # Initialize VI ROI center to scan center with reasonable default sizes
        self.vi_roi_center_row = float(self.shape_rows / 2)
        self.vi_roi_center_col = float(self.shape_cols / 2)
        # Set initial ROI size based on scan dimension
        default_roi_size = max(3, min(self.shape_rows, self.shape_cols) * DEFAULT_VI_ROI_RATIO)
        self.vi_roi_radius = float(default_roi_size)
        self.vi_roi_width = float(default_roi_size * 2)
        self.vi_roi_height = float(default_roi_size)
        self.observe(self._on_vi_roi_change, names=[
            "vi_roi_mode", "vi_roi_center_row", "vi_roi_center_col",
            "vi_roi_radius", "vi_roi_width", "vi_roi_height"
        ])

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = json.loads(pathlib.Path(state).read_text())
            self.load_state_dict(state)

    def set_image(self, data, scan_shape=None):
        """Replace the 4D-STEM data. Preserves all display and ROI settings."""
        if hasattr(data, "sampling") and hasattr(data, "array"):
            data = data.array
        data_np = to_numpy(data)
        self._data = torch.from_numpy(data_np.astype(np.float32)).to(self._device)
        saturated_value = 65535.0 if data_np.dtype == np.uint16 else 255.0 if data_np.dtype == np.uint8 else None
        if saturated_value is not None:
            self._data[self._data >= saturated_value] = 0
        if data_np.ndim == 3:
            if scan_shape is not None:
                self._scan_shape = scan_shape
            else:
                n = data_np.shape[0]
                side = int(n ** 0.5)
                if side * side != n:
                    raise ValueError(f"Cannot infer square scan_shape from N={n}. Provide scan_shape explicitly.")
                self._scan_shape = (side, side)
            self._det_shape = (data_np.shape[1], data_np.shape[2])
        elif data_np.ndim == 4:
            self._scan_shape = (data_np.shape[0], data_np.shape[1])
            self._det_shape = (data_np.shape[2], data_np.shape[3])
        else:
            raise ValueError(f"Expected 3D or 4D array, got {data_np.ndim}D")
        self.shape_rows = self._scan_shape[0]
        self.shape_cols = self._scan_shape[1]
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]
        self.dp_global_min = max(float(self._data.min()), MIN_LOG_VALUE)
        self.dp_global_max = float(self._data.max())
        self._det_row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[:, None]
        self._det_col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[None, :]
        self._scan_row_coords = torch.arange(self.shape_rows, device=self._device, dtype=torch.float32)[:, None]
        self._scan_col_coords = torch.arange(self.shape_cols, device=self._device, dtype=torch.float32)[None, :]
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        self.pos_row = min(self.pos_row, self.shape_rows - 1)
        self.pos_col = min(self.pos_col, self.shape_cols - 1)
        self._compute_virtual_image_from_roi()
        self._update_frame()

    def __repr__(self) -> str:
        k_unit = "mrad" if self.k_calibrated else "px"
        return (
            f"Show4DSTEM(shape=({self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols}), "
            f"sampling=({self.pixel_size} Å, {self.k_pixel_size} {k_unit}), "
            f"pos=({self.pos_row}, {self.pos_col}))"
        )

    def state_dict(self):
        return {
            "pixel_size": self.pixel_size,
            "k_pixel_size": self.k_pixel_size,
            "k_calibrated": self.k_calibrated,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "bf_radius": self.bf_radius,
            "roi_active": self.roi_active,
            "roi_mode": self.roi_mode,
            "roi_center_row": self.roi_center_row,
            "roi_center_col": self.roi_center_col,
            "roi_radius": self.roi_radius,
            "roi_radius_inner": self.roi_radius_inner,
            "roi_width": self.roi_width,
            "roi_height": self.roi_height,
            "vi_roi_mode": self.vi_roi_mode,
            "vi_roi_center_row": self.vi_roi_center_row,
            "vi_roi_center_col": self.vi_roi_center_col,
            "vi_roi_radius": self.vi_roi_radius,
            "vi_roi_width": self.vi_roi_width,
            "vi_roi_height": self.vi_roi_height,
            "mask_dc": self.mask_dc,
            "log_scale": self.log_scale,
            "path_interval_ms": self.path_interval_ms,
            "path_loop": self.path_loop,
            "profile_line": self.profile_line,
            "profile_width": self.profile_width,
        }

    def save(self, path: str):
        pathlib.Path(path).write_text(json.dumps(self.state_dict(), indent=2))

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        lines = ["Show4DSTEM", "═" * 32]
        lines.append(f"Scan:     {self.shape_rows}×{self.shape_cols} ({self.pixel_size:.2f} Å/px)")
        k_unit = "mrad" if self.k_calibrated else "px"
        lines.append(f"Detector: {self.det_rows}×{self.det_cols} ({self.k_pixel_size:.4f} {k_unit}/px)")
        lines.append(f"Position: ({self.pos_row}, {self.pos_col})")
        lines.append(f"Center:   ({self.center_row:.1f}, {self.center_col:.1f})  BF r={self.bf_radius:.1f} px")
        display_parts = []
        if self.log_scale:
            display_parts.append("log scale")
        if self.mask_dc:
            display_parts.append("DC masked")
        lines.append(f"Display:  {', '.join(display_parts) if display_parts else 'default'}")
        if self.roi_active:
            lines.append(f"ROI:      {self.roi_mode} at ({self.roi_center_row:.1f}, {self.roi_center_col:.1f}) r={self.roi_radius:.1f}")
        if self.vi_roi_mode != "off":
            lines.append(f"VI ROI:   {self.vi_roi_mode} at ({self.vi_roi_center_row:.1f}, {self.vi_roi_center_col:.1f}) r={self.vi_roi_radius:.1f}")
        if self.profile_line and len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) -> ({p1['row']:.0f}, {p1['col']:.0f}) width={self.profile_width}")
        print("\n".join(lines))

    # =========================================================================
    # Convenience Properties
    # =========================================================================

    @property
    def position(self) -> tuple[int, int]:
        """Current scan position as (row, col) tuple."""
        return (self.pos_row, self.pos_col)

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        """Set scan position from (row, col) tuple."""
        self.pos_row, self.pos_col = value

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Scan dimensions as (rows, cols) tuple."""
        return (self.shape_rows, self.shape_cols)

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector dimensions as (rows, cols) tuple."""
        return (self.det_rows, self.det_cols)

    # =========================================================================
    # Line Profile
    # =========================================================================

    def set_profile(self, start: tuple, end: tuple) -> Self:
        row0, col0 = start
        row1, col1 = end
        self.profile_line = [
            {"row": float(row0), "col": float(col0)},
            {"row": float(row1), "col": float(col1)},
        ]
        return self

    def clear_profile(self) -> Self:
        self.profile_line = []
        return self

    @property
    def profile(self) -> list[tuple[float, float]]:
        if len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            return [(p0["row"], p0["col"]), (p1["row"], p1["col"])]
        return []

    @property
    def profile_values(self):
        if len(self.profile_line) != 2:
            return None
        p0, p1 = self.profile_line[0], self.profile_line[1]
        frame = self._get_frame(self.pos_row, self.pos_col)
        return self._sample_line(frame, p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self) -> float:
        if len(self.profile_line) != 2:
            return 0.0
        p0, p1 = self.profile_line[0], self.profile_line[1]
        dist_px = np.sqrt((p1["row"] - p0["row"]) ** 2 + (p1["col"] - p0["col"]) ** 2)
        if self.k_calibrated:
            return float(dist_px * self.k_pixel_size)
        return float(dist_px)

    def _sample_line(self, frame, row0, col0, row1, col1):
        h, w = frame.shape[:2]
        dc = col1 - col0
        dr = row1 - row0
        length = np.sqrt(dc * dc + dr * dr)
        n = max(2, int(np.ceil(length)))
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            t = i / (n - 1)
            c = col0 + t * dc
            r = row0 + t * dr
            ci, ri = int(np.floor(c)), int(np.floor(r))
            cf, rf = c - ci, r - ri
            c0c = max(0, min(w - 1, ci))
            c1c = max(0, min(w - 1, ci + 1))
            r0c = max(0, min(h - 1, ri))
            r1c = max(0, min(h - 1, ri + 1))
            out[i] = (
                frame[r0c, c0c] * (1 - cf) * (1 - rf)
                + frame[r0c, c1c] * cf * (1 - rf)
                + frame[r1c, c0c] * (1 - cf) * rf
                + frame[r1c, c1c] * cf * rf
            )
        return out

    # =========================================================================
    # Path Animation Methods
    # =========================================================================
    
    def set_path(
        self,
        points: list[tuple[int, int]],
        interval_ms: int = 100,
        loop: bool = True,
        autoplay: bool = True,
    ) -> Self:
        """
        Set a custom path of scan positions to animate through.

        Parameters
        ----------
        points : list[tuple[int, int]]
            List of (row, col) scan positions to visit.
        interval_ms : int, default 100
            Time between frames in milliseconds.
        loop : bool, default True
            Whether to loop when reaching end.
        autoplay : bool, default True
            Start playing immediately.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget.set_path([(0, 0), (10, 10), (20, 20), (30, 30)])
        >>> widget.set_path([(i, i) for i in range(48)], interval_ms=50)
        """
        self._path_points = list(points)
        self.path_length = len(self._path_points)
        self.path_index = 0
        self.path_interval_ms = interval_ms
        self.path_loop = loop
        if autoplay and self.path_length > 0:
            self.path_playing = True
        return self
    
    def play(self) -> Self:
        """Start playing the path animation."""
        if self.path_length > 0:
            self.path_playing = True
        return self
    
    def pause(self) -> Self:
        """Pause the path animation."""
        self.path_playing = False
        return self
    
    def stop(self) -> Self:
        """Stop and reset path animation to beginning."""
        self.path_playing = False
        self.path_index = 0
        return self
    
    def goto(self, index: int) -> Self:
        """Jump to a specific index in the path."""
        if 0 <= index < self.path_length:
            self.path_index = index
        return self
    
    def _on_path_index_change(self, change):
        """Called when path_index changes (from frontend timer)."""
        idx = change["new"]
        if 0 <= idx < len(self._path_points):
            row, col = self._path_points[idx]
            # Clamp to valid range
            self.pos_row = max(0, min(self.shape_rows - 1, row))
            self.pos_col = max(0, min(self.shape_cols - 1, col))

    def _on_auto_detect_trigger(self, change):
        """Called when auto_detect_trigger is set to True from frontend."""
        if change["new"]:
            self.auto_detect_center()
            # Reset trigger to allow re-triggering
            self.auto_detect_trigger = False

    # =========================================================================
    # Path Animation Patterns
    # =========================================================================

    def raster(
        self,
        step: int = 1,
        bidirectional: bool = False,
        interval_ms: int = 100,
        loop: bool = True,
    ) -> Self:
        """
        Play a raster scan path (row by row, left to right).

        This mimics real STEM scanning: left→right, step down, left→right, etc.

        Parameters
        ----------
        step : int, default 1
            Step size between positions.
        bidirectional : bool, default False
            If True, use snake/boustrophedon pattern (alternating direction).
            If False (default), always scan left→right like real STEM.
        interval_ms : int, default 100
            Time between frames in milliseconds.
        loop : bool, default True
            Whether to loop when reaching the end.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.
        """
        points = []
        for r in range(0, self.shape_rows, step):
            cols = list(range(0, self.shape_cols, step))
            if bidirectional and (r // step % 2 == 1):
                cols = cols[::-1]  # Alternate direction for snake pattern
            for c in cols:
                points.append((r, c))
        return self.set_path(points=points, interval_ms=interval_ms, loop=loop)
    
    # =========================================================================
    # ROI Mode Methods
    # =========================================================================
    
    def roi_circle(self, radius: float | None = None) -> Self:
        """
        Switch to circle ROI mode for virtual imaging.
        
        In circle mode, the virtual image integrates over a circular region
        centered at the current ROI position (like a virtual bright field detector).
        
        Parameters
        ----------
        radius : float, optional
            Radius of the circle in pixels. If not provided, uses current value
            or defaults to half the BF radius.
            
        Returns
        -------
        Show4DSTEM
            Self for method chaining.
            
        Examples
        --------
        >>> widget.roi_circle(20)  # 20px radius circle
        >>> widget.roi_circle()    # Use default radius
        """
        self.roi_mode = "circle"
        if radius is not None:
            self.roi_radius = float(radius)
        return self
    
    def roi_point(self) -> Self:
        """
        Switch to point ROI mode (single-pixel indexing).
        
        In point mode, the virtual image shows intensity at the exact ROI position.
        This is the default mode.
        
        Returns
        -------
        Show4DSTEM
            Self for method chaining.
        """
        self.roi_mode = "point"
        return self

    def roi_square(self, half_size: float | None = None) -> Self:
        """
        Switch to square ROI mode for virtual imaging.

        In square mode, the virtual image integrates over a square region
        centered at the current ROI position.

        Parameters
        ----------
        half_size : float, optional
            Half-size of the square in pixels (distance from center to edge).
            A half_size of 15 creates a 30x30 pixel square.
            If not provided, uses current roi_radius value.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget.roi_square(15)  # 30x30 pixel square (half_size=15)
        >>> widget.roi_square()    # Use default size
        """
        self.roi_mode = "square"
        if half_size is not None:
            self.roi_radius = float(half_size)
        return self

    def roi_annular(
        self, inner_radius: float | None = None, outer_radius: float | None = None
    ) -> Self:
        """
        Set ROI mode to annular (donut-shaped) for ADF/HAADF imaging.
        
        Parameters
        ----------
        inner_radius : float, optional
            Inner radius in pixels. If not provided, uses current roi_radius_inner.
        outer_radius : float, optional
            Outer radius in pixels. If not provided, uses current roi_radius.
            
        Returns
        -------
        Show4DSTEM
            Self for method chaining.
            
        Examples
        --------
        >>> widget.roi_annular(20, 50)  # ADF: inner=20px, outer=50px
        >>> widget.roi_annular(30, 80)  # HAADF: larger angles
        """
        self.roi_mode = "annular"
        if inner_radius is not None:
            self.roi_radius_inner = float(inner_radius)
        if outer_radius is not None:
            self.roi_radius = float(outer_radius)
        return self

    def roi_rect(
        self, width: float | None = None, height: float | None = None
    ) -> Self:
        """
        Set ROI mode to rectangular.
        
        Parameters
        ----------
        width : float, optional
            Width in pixels. If not provided, uses current roi_width.
        height : float, optional
            Height in pixels. If not provided, uses current roi_height.
            
        Returns
        -------
        Show4DSTEM
            Self for method chaining.
            
        Examples
        --------
        >>> widget.roi_rect(30, 20)  # 30px wide, 20px tall
        >>> widget.roi_rect(40, 40)  # 40x40 rectangle
        """
        self.roi_mode = "rect"
        if width is not None:
            self.roi_width = float(width)
        if height is not None:
            self.roi_height = float(height)
        return self

    def auto_detect_center(self, update_roi: bool = True) -> Self:
        """
        Automatically detect BF disk center and radius using centroid.

        This method analyzes the summed diffraction pattern to find the
        bright field disk center and estimate its radius. The detected
        values are applied to the widget's calibration (center_row, center_col,
        bf_radius).

        Parameters
        ----------
        update_roi : bool, default True
            If True, also update ROI center and recompute cached virtual images.
            Set to False during __init__ when ROI is not yet initialized.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget = Show4DSTEM(data)
        >>> widget.auto_detect_center()  # Auto-detect and apply
        """
        # Sum all diffraction patterns to get average (PyTorch)
        if self._data.ndim == 4:
            summed_dp = self._data.sum(dim=(0, 1))
        else:
            summed_dp = self._data.sum(dim=0)

        # Threshold at mean + std to isolate BF disk
        threshold = summed_dp.mean() + summed_dp.std()
        mask = summed_dp > threshold

        # Avoid division by zero
        total = mask.sum()
        if total == 0:
            return self

        # Calculate centroid using cached coordinate grids
        cx = float((self._det_col_coords * mask).sum() / total)
        cy = float((self._det_row_coords * mask).sum() / total)

        # Estimate radius from mask area (A = pi*r^2)
        radius = float(torch.sqrt(total / torch.pi))

        # Apply detected values
        self.center_col = cx
        self.center_row = cy
        self.bf_radius = radius

        if update_roi:
            # Also update ROI to center
            self.roi_center_col = cx
            self.roi_center_row = cy
            # Recompute cached virtual images with new calibration
            self._precompute_common_virtual_images()

        return self

    def _get_frame(self, row: int, col: int) -> np.ndarray:
        """Get single diffraction frame at position (row, col) as numpy array."""
        if self._data.ndim == 3:
            idx = row * self.shape_cols + col
            return self._data[idx].cpu().numpy()
        else:
            return self._data[row, col].cpu().numpy()

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))
        fmin, fmax = float(frame.min()), float(frame.max())
        if fmax > fmin:
            return np.clip((frame - fmin) / (fmax - fmin) * 255, 0, 255).astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def _on_gif_export(self, change=None):
        if not self._gif_export_requested:
            return
        self._gif_export_requested = False
        self._generate_gif()

    def _generate_gif(self):
        import io

        from matplotlib import colormaps
        from PIL import Image

        if not self._path_points:
            return

        cmap_fn = colormaps.get_cmap("inferno")
        duration_ms = max(10, self.path_interval_ms)

        pil_frames = []
        for row, col in self._path_points:
            row = max(0, min(self.shape_rows - 1, row))
            col = max(0, min(self.shape_cols - 1, col))
            frame = self._get_frame(row, col).astype(np.float32)
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

    def _update_frame(self, change=None):
        """Send raw float32 frame to frontend (JS handles scale/colormap)."""
        # Get frame as tensor (stays on device)
        if self._data.ndim == 3:
            idx = self.pos_row * self.shape_cols + self.pos_col
            frame = self._data[idx]
        else:
            frame = self._data[self.pos_row, self.pos_col]

        # Apply log scale if enabled
        if self.log_scale:
            frame = torch.log1p(frame)

        # Compute stats from frame (optionally mask DC component)
        if self.mask_dc and self.det_rows > 3 and self.det_cols > 3:
            # Mask center 3x3 region for stats using detected center (not geometric center)
            cr = int(round(self.center_row))
            cc = int(round(self.center_col))
            cr = max(1, min(self.det_rows - 2, cr))
            cc = max(1, min(self.det_cols - 2, cc))
            mask = torch.ones_like(frame, dtype=torch.bool)
            mask[cr-1:cr+2, cc-1:cc+2] = False
            masked_vals = frame[mask]
            self.dp_stats = [
                float(masked_vals.mean()),
                float(masked_vals.min()),
                float(masked_vals.max()),
                float(masked_vals.std()),
            ]
        else:
            self.dp_stats = [
                float(frame.mean()),
                float(frame.min()),
                float(frame.max()),
                float(frame.std()),
            ]

        # Convert to numpy only for sending bytes to frontend
        self.frame_bytes = frame.cpu().numpy().astype(np.float32).tobytes()

    def _on_roi_change(self, change=None):
        """Recompute virtual image when individual ROI params change.

        This handles legacy setters (setRoiCenterX/Y) from button handlers.
        High-frequency updates use the compound roi_center trait instead.
        """
        if not self.roi_active:
            return
        self._compute_virtual_image_from_roi()

    def _on_roi_center_change(self, change=None):
        """Handle batched roi_center updates from JS (single observer for row+col).

        This is the fast path for drag operations. JS sends [row, col] as a single
        compound trait, so only one observer fires per mouse move.
        """
        if not self.roi_active:
            return
        if change and "new" in change:
            row, col = change["new"]
            # Sync to individual traits (without triggering _on_roi_change observers)
            self.unobserve(self._on_roi_change, names=["roi_center_col", "roi_center_row"])
            self.roi_center_row = row
            self.roi_center_col = col
            self.observe(self._on_roi_change, names=["roi_center_col", "roi_center_row"])
        self._compute_virtual_image_from_roi()

    def _on_vi_roi_change(self, change=None):
        """Compute summed DP when VI ROI changes."""
        if self.vi_roi_mode == "off":
            self.summed_dp_bytes = b""
            self.summed_dp_count = 0
            return
        self._compute_summed_dp_from_vi_roi()

    def _compute_summed_dp_from_vi_roi(self):
        """Sum diffraction patterns from positions inside VI ROI (PyTorch)."""
        # Create mask in scan space using cached coordinates
        if self.vi_roi_mode == "circle":
            mask = (self._scan_row_coords - self.vi_roi_center_row) ** 2 + (self._scan_col_coords - self.vi_roi_center_col) ** 2 <= self.vi_roi_radius ** 2
        elif self.vi_roi_mode == "square":
            half_size = self.vi_roi_radius
            mask = (torch.abs(self._scan_row_coords - self.vi_roi_center_row) <= half_size) & (torch.abs(self._scan_col_coords - self.vi_roi_center_col) <= half_size)
        elif self.vi_roi_mode == "rect":
            half_w = self.vi_roi_width / 2
            half_h = self.vi_roi_height / 2
            mask = (torch.abs(self._scan_row_coords - self.vi_roi_center_row) <= half_h) & (torch.abs(self._scan_col_coords - self.vi_roi_center_col) <= half_w)
        else:
            return

        # Count positions in mask
        n_positions = int(mask.sum())
        if n_positions == 0:
            self.summed_dp_bytes = b""
            self.summed_dp_count = 0
            return

        self.summed_dp_count = n_positions

        # Compute average DP using masked sum (vectorized)
        if self._data.ndim == 4:
            # (scan_rows, scan_cols, det_rows, det_cols) - sum over masked scan positions
            avg_dp = self._data[mask].mean(dim=0)
        else:
            # Flattened: (N, det_rows, det_cols) - need to convert mask indices
            flat_indices = torch.nonzero(mask.flatten(), as_tuple=True)[0]
            avg_dp = self._data[flat_indices].mean(dim=0)

        # Send raw float32 (consistent with other data paths — JS handles normalization)
        self.summed_dp_bytes = avg_dp.cpu().numpy().astype(np.float32).tobytes()

    def _create_circular_mask(self, cx: float, cy: float, radius: float):
        """Create circular mask (boolean tensor on device)."""
        mask = (self._det_col_coords - cx) ** 2 + (self._det_row_coords - cy) ** 2 <= radius ** 2
        return mask

    def _create_square_mask(self, cx: float, cy: float, half_size: float):
        """Create square mask (boolean tensor on device)."""
        mask = (torch.abs(self._det_col_coords - cx) <= half_size) & (torch.abs(self._det_row_coords - cy) <= half_size)
        return mask

    def _create_annular_mask(
        self, cx: float, cy: float, inner: float, outer: float
    ):
        """Create annular (donut) mask (boolean tensor on device)."""
        dist_sq = (self._det_col_coords - cx) ** 2 + (self._det_row_coords - cy) ** 2
        mask = (dist_sq >= inner ** 2) & (dist_sq <= outer ** 2)
        return mask

    def _create_rect_mask(self, cx: float, cy: float, half_width: float, half_height: float):
        """Create rectangular mask (boolean tensor on device)."""
        mask = (torch.abs(self._det_col_coords - cx) <= half_width) & (torch.abs(self._det_row_coords - cy) <= half_height)
        return mask

    def _precompute_common_virtual_images(self):
        """Pre-compute BF/ABF/ADF virtual images for instant preset switching."""
        cx, cy, bf = self.center_col, self.center_row, self.bf_radius
        # Cache (bytes, stats, min, max) for each preset
        bf_arr = self._fast_masked_sum(self._create_circular_mask(cx, cy, bf))
        abf_arr = self._fast_masked_sum(self._create_annular_mask(cx, cy, bf * 0.5, bf))
        adf_arr = self._fast_masked_sum(self._create_annular_mask(cx, cy, bf, bf * 4.0))

        self._cached_bf_virtual = (
            self._to_float32_bytes(bf_arr, update_vi_stats=False),
            [float(bf_arr.mean()), float(bf_arr.min()), float(bf_arr.max()), float(bf_arr.std())],
            float(bf_arr.min()), float(bf_arr.max())
        )
        self._cached_abf_virtual = (
            self._to_float32_bytes(abf_arr, update_vi_stats=False),
            [float(abf_arr.mean()), float(abf_arr.min()), float(abf_arr.max()), float(abf_arr.std())],
            float(abf_arr.min()), float(abf_arr.max())
        )
        self._cached_adf_virtual = (
            self._to_float32_bytes(adf_arr, update_vi_stats=False),
            [float(adf_arr.mean()), float(adf_arr.min()), float(adf_arr.max()), float(adf_arr.std())],
            float(adf_arr.min()), float(adf_arr.max())
        )

    def _get_cached_preset(self) -> tuple[bytes, list[float], float, float] | None:
        """Check if current ROI matches a cached preset and return (bytes, stats, min, max) tuple."""
        # Must be centered on detector center
        if abs(self.roi_center_col - self.center_col) >= 1 or abs(self.roi_center_row - self.center_row) >= 1:
            return None

        bf = self.bf_radius

        # BF: circle at bf_radius
        if (self.roi_mode == "circle" and abs(self.roi_radius - bf) < 1):
            return self._cached_bf_virtual

        # ABF: annular at 0.5*bf to bf
        if (self.roi_mode == "annular" and
            abs(self.roi_radius_inner - bf * 0.5) < 1 and
            abs(self.roi_radius - bf) < 1):
            return self._cached_abf_virtual

        # ADF: annular at bf to 4*bf (combines LAADF + HAADF)
        if (self.roi_mode == "annular" and
            abs(self.roi_radius_inner - bf) < 1 and
            abs(self.roi_radius - bf * 4.0) < 1):
            return self._cached_adf_virtual

        return None

    def _fast_masked_sum(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked sum using PyTorch.

        Uses sparse indexing for small masks (<20% coverage) which is faster
        because it only processes non-zero pixels:
        - r=10 (1%): ~0.8ms (sparse) vs ~13ms (full)
        - r=30 (8%): ~4ms (sparse) vs ~13ms (full)

        For large masks (≥20%), uses full tensordot which has constant ~13ms.
        """
        mask_float = mask.float()
        n_det = self._det_shape[0] * self._det_shape[1]
        n_nonzero = int(mask.sum())
        coverage = n_nonzero / n_det

        if coverage < SPARSE_MASK_THRESHOLD:
            # Sparse: faster for small masks
            indices = torch.nonzero(mask_float.flatten(), as_tuple=True)[0]
            n_scan = self._scan_shape[0] * self._scan_shape[1]
            data_flat = self._data.reshape(n_scan, n_det)
            result = data_flat[:, indices].sum(dim=1).reshape(self._scan_shape)
        else:
            # Tensordot: faster for large masks
            # Reshape to 4D if needed (3D flattened data)
            if self._data.ndim == 3:
                data_4d = self._data.reshape(self._scan_shape[0], self._scan_shape[1], *self._det_shape)
            else:
                data_4d = self._data
            result = torch.tensordot(data_4d, mask_float, dims=([2, 3], [0, 1]))

        return result

    def _to_float32_bytes(self, arr: torch.Tensor, update_vi_stats: bool = True) -> bytes:
        """Convert tensor to float32 bytes."""
        # Compute min/max (fast on GPU)
        vmin = float(arr.min())
        vmax = float(arr.max())

        # Only update traits when requested (avoids side effects during precomputation)
        if update_vi_stats:
            self.vi_data_min = vmin
            self.vi_data_max = vmax
            self.vi_stats = [float(arr.mean()), vmin, vmax, float(arr.std())]

        return arr.cpu().numpy().astype(np.float32).tobytes()

    def _compute_virtual_image_from_roi(self):
        """Compute virtual image based on ROI mode."""
        cached = self._get_cached_preset()
        if cached is not None:
            # Cached preset returns (bytes, stats, min, max) tuple
            vi_bytes, vi_stats, vi_min, vi_max = cached
            self.virtual_image_bytes = vi_bytes
            self.vi_stats = vi_stats
            self.vi_data_min = vi_min
            self.vi_data_max = vi_max
            return

        cx, cy = self.roi_center_col, self.roi_center_row

        if self.roi_mode == "circle" and self.roi_radius > 0:
            mask = self._create_circular_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "square" and self.roi_radius > 0:
            mask = self._create_square_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "annular" and self.roi_radius > 0:
            mask = self._create_annular_mask(cx, cy, self.roi_radius_inner, self.roi_radius)
        elif self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            mask = self._create_rect_mask(cx, cy, self.roi_width / 2, self.roi_height / 2)
        else:
            # Point mode: single-pixel indexing
            row = int(max(0, min(round(cy), self._det_shape[0] - 1)))
            col = int(max(0, min(round(cx), self._det_shape[1] - 1)))
            if self._data.ndim == 4:
                virtual_image = self._data[:, :, row, col]
            else:
                virtual_image = self._data[:, row, col].reshape(self._scan_shape)
            self.virtual_image_bytes = self._to_float32_bytes(virtual_image)
            return

        self.virtual_image_bytes = self._to_float32_bytes(self._fast_masked_sum(mask))
