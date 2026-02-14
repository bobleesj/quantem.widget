"""
show2d: Static 2D image viewer with optional FFT and histogram analysis.

For displaying a single image or a static gallery of multiple images.
Unlike Show3D (interactive), Show2D focuses on static visualization.
"""

import pathlib
from enum import StrEnum
from typing import Optional, Union, List

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy


def _resize_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image using bilinear interpolation (pure numpy, no scipy)."""
    h, w = img.shape

    # If same size, return as-is
    if h == target_h and w == target_w:
        return img

    # Create coordinate grids for target size
    y_new = np.linspace(0, h - 1, target_h)
    x_new = np.linspace(0, w - 1, target_w)
    x_grid, y_grid = np.meshgrid(x_new, y_new)

    # Get integer and fractional parts for bilinear interpolation
    y0 = np.floor(y_grid).astype(int)
    x0 = np.floor(x_grid).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)

    fy = y_grid - y0
    fx = x_grid - x0

    # Vectorized bilinear interpolation
    result = (
        img[y0, x0] * (1 - fy) * (1 - fx) +
        img[y0, x1] * (1 - fy) * fx +
        img[y1, x0] * fy * (1 - fx) +
        img[y1, x1] * fy * fx
    )
    return result.astype(img.dtype)


class Colormap(StrEnum):
    INFERNO = "inferno"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    PLASMA = "plasma"
    GRAY = "gray"


class Show2D(anywidget.AnyWidget):
    """
    Static 2D image viewer with optional FFT and histogram analysis.

    Display a single image or multiple images in a gallery layout.
    For interactive stack viewing with playback, use Show3D instead.

    Parameters
    ----------
    data : array_like
        2D array (height, width) for single image, or
        3D array (N, height, width) for multiple images displayed as gallery.
    labels : list of str, optional
        Labels for each image in gallery mode.
    title : str, optional
        Title to display above the image(s).
    cmap : str, default "inferno"
        Colormap name ("magma", "viridis", "gray", "inferno", "plasma").
    pixel_size_angstrom : float, optional
        Pixel size in angstroms for scale bar display.
    show_fft : bool, default False
        Show FFT and histogram panels.
    show_stats : bool, default True
        Show statistics (mean, min, max, std).
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast.
    ncols : int, default 3
        Number of columns in gallery mode.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show2D
    >>>
    >>> # Single image with FFT
    >>> Show2D(image, title="HRTEM Image", show_fft=True, pixel_size_angstrom=1.0)
    >>>
    >>> # Gallery of multiple images
    >>> labels = ["Raw", "Filtered", "FFT"]
    >>> Show2D([img1, img2, img3], labels=labels, ncols=3)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show2d.js"
    _css = pathlib.Path(__file__).parent / "static" / "show2d.css"

    # =========================================================================
    # Core State
    # =========================================================================
    n_images = traitlets.Int(1).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    ncols = traitlets.Int(3).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    percentile_low = traitlets.Float(1.0).tag(sync=True)
    percentile_high = traitlets.Float(99.0).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size_angstrom = traitlets.Float(0.0).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    scale_bar_length_px = traitlets.Int(50).tag(sync=True)
    scale_bar_thickness_px = traitlets.Int(4).tag(sync=True)
    scale_bar_font_size_px = traitlets.Int(16).tag(sync=True)

    # =========================================================================
    # Sizing & Customization
    # =========================================================================
    panel_size_px = traitlets.Int(150).tag(sync=True)
    image_width_px = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # UI Visibility
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)

    # =========================================================================
    # Analysis Panels (FFT + Histogram shown together)
    # =========================================================================
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_bytes = traitlets.Bytes(b"").tag(sync=True)
    histogram_bins = traitlets.List(traitlets.Float()).tag(sync=True)
    histogram_counts = traitlets.List(traitlets.Int()).tag(sync=True)

    # =========================================================================
    # Selected Image (for single-image analysis display)
    # =========================================================================
    selected_idx = traitlets.Int(0).tag(sync=True)

    def __init__(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        labels: Optional[List[str]] = None,
        title: str = "",
        cmap: Union[str, Colormap] = Colormap.INFERNO,
        pixel_size_angstrom: float = 0.0,
        scale_bar_visible: bool = True,
        show_fft: bool = False,
        show_controls: bool = True,
        show_stats: bool = True,
        log_scale: bool = False,
        auto_contrast: bool = False,
        ncols: int = 3,
        scale_bar_length_px: int = 50,
        scale_bar_thickness_px: int = 4,
        scale_bar_font_size_px: int = 16,
        panel_size_px: int = 150,
        image_width_px: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Convert input to NumPy (handles NumPy, CuPy, PyTorch)
        if isinstance(data, list):
            images = [to_numpy(d) for d in data]

            # Check if all images have the same shape
            shapes = [img.shape for img in images]
            if len(set(shapes)) > 1:
                # Different sizes - resize all to the largest
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                images = [_resize_image(img, max_h, max_w) for img in images]

            data = np.stack(images)
        else:
            data = to_numpy(data)

        # Ensure 3D shape (N, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._data = data.astype(np.float32)
        self.n_images = int(data.shape[0])
        self.height = int(data.shape[1])
        self.width = int(data.shape[2])

        # Labels
        if labels is None:
            self.labels = [f"Image {i+1}" for i in range(self.n_images)]
        else:
            self.labels = list(labels)

        # Options
        self.title = title
        self.cmap = cmap
        self.pixel_size_angstrom = pixel_size_angstrom
        self.scale_bar_visible = scale_bar_visible
        self.scale_bar_length_px = scale_bar_length_px
        self.scale_bar_thickness_px = scale_bar_thickness_px
        self.scale_bar_font_size_px = scale_bar_font_size_px
        self.panel_size_px = panel_size_px
        self.image_width_px = image_width_px
        self.show_fft = show_fft
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.ncols = ncols

        # Compute initial stats
        self._compute_all_stats()

        # Send raw float32 data to JS (normalization happens in JS for speed)
        self._update_all_frames()

        # Initial FFT/histogram for selected image
        if show_fft:
            self._compute_fft()
            self._compute_histogram()

        # Observe changes
        self.observe(self._on_selected_change, names=["selected_idx"])
        self.observe(self._on_fft_change, names=["show_fft"])

    def _compute_all_stats(self):
        """Compute statistics for all images."""
        means, mins, maxs, stds = [], [], [], []
        for i in range(self.n_images):
            img = self._data[i]
            means.append(float(np.mean(img)))
            mins.append(float(np.min(img)))
            maxs.append(float(np.max(img)))
            stds.append(float(np.std(img)))
        self.stats_mean = means
        self.stats_min = mins
        self.stats_max = maxs
        self.stats_std = stds

    def _update_all_frames(self):
        """Send raw float32 data to JS (normalization happens in JS for speed)."""
        self.frame_bytes = self._data.astype(np.float32).tobytes()

    def _compute_fft(self):
        """Compute FFT of selected image."""
        img = self._data[self.selected_idx]
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        log_magnitude = np.log1p(magnitude)

        vmin = np.min(log_magnitude)
        vmax = np.max(log_magnitude)
        if vmax - vmin > 1e-10:
            normalized = (log_magnitude - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(log_magnitude)

        fft_uint8 = (normalized * 255).astype(np.uint8)
        self.fft_bytes = fft_uint8.tobytes()

    def _compute_histogram(self):
        """Compute histogram of selected image."""
        img = self._data[self.selected_idx]
        counts, bins = np.histogram(img.ravel(), bins=50)
        self.histogram_bins = [float(b) for b in bins[:-1]]
        self.histogram_counts = [int(c) for c in counts]

    def _on_selected_change(self, change):
        """Update FFT/histogram when selection changes."""
        if self.show_fft:
            self._compute_fft()
            self._compute_histogram()

    def _on_fft_change(self, change):
        """Compute FFT and histogram when show_fft enabled."""
        if change["new"]:
            self._compute_fft()
            self._compute_histogram()


