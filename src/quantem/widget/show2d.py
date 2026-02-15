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

from quantem.widget.array_utils import to_numpy, _resize_image


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

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size_angstrom = traitlets.Float(0.0).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
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

    # =========================================================================
    # Selected Image (for single-image analysis display)
    # =========================================================================
    selected_idx = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # Line Profile
    # =========================================================================
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)

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
        image_width_px: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Check if data is a Dataset2d and extract metadata
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            if not title and data.name:
                title = data.name
            if pixel_size_angstrom == 0.0 and hasattr(data, "units"):
                units = list(data.units)
                sampling_val = float(data.sampling[-1])
                if units[-1] in ("nm",):
                    pixel_size_angstrom = sampling_val * 10  # nm → Å
                elif units[-1] in ("Å", "angstrom", "A"):
                    pixel_size_angstrom = sampling_val
            data = data.array

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

        self.selected_idx = 0

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

    def _sample_profile(self, row0, col0, row1, col1):
        """Sample intensity values along a line using bilinear interpolation."""
        img = self._data[self.selected_idx]
        h, w = img.shape
        dc, dr = col1 - col0, row1 - row0
        length = (dc**2 + dr**2) ** 0.5
        n = max(2, int(np.ceil(length)))
        t = np.linspace(0, 1, n)
        cs = col0 + t * dc
        rs = row0 + t * dr
        ci = np.floor(cs).astype(int)
        ri = np.floor(rs).astype(int)
        cf = cs - ci
        rf = rs - ri
        c0c = np.clip(ci, 0, w - 1)
        c1c = np.clip(ci + 1, 0, w - 1)
        r0c = np.clip(ri, 0, h - 1)
        r1c = np.clip(ri + 1, 0, h - 1)
        vals = (img[r0c, c0c] * (1 - cf) * (1 - rf) +
                img[r0c, c1c] * cf * (1 - rf) +
                img[r1c, c0c] * (1 - cf) * rf +
                img[r1c, c1c] * cf * rf)
        return vals.astype(np.float32)

    def set_profile(self, row0: float, col0: float, row1: float, col1: float):
        """Set a line profile between two points (image pixel coordinates).

        Parameters
        ----------
        row0, col0 : float
            Start point in pixel coordinates.
        row1, col1 : float
            End point in pixel coordinates.
        """
        self.profile_line = [
            {"row": float(row0), "col": float(col0)},
            {"row": float(row1), "col": float(col1)},
        ]

    def clear_profile(self):
        """Clear the current line profile."""
        self.profile_line = []

    @property
    def profile(self):
        """Get profile line endpoints as [(row0, col0), (row1, col1)] or [].

        Returns
        -------
        list of tuple
            Line endpoints in pixel coordinates, or empty list if no profile.
        """
        return [(p["row"], p["col"]) for p in self.profile_line]

    @property
    def profile_values(self):
        """Get intensity values along the profile line as a numpy array.

        Returns
        -------
        np.ndarray or None
            Float32 array of sampled intensities, or None if no profile.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        return self._sample_profile(p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self):
        """Get total distance of the profile line in calibrated units.

        Returns
        -------
        float or None
            Distance in angstroms (if pixel_size_angstrom > 0) or pixels.
            None if no profile line is set.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        dc = p1["col"] - p0["col"]
        dr = p1["row"] - p0["row"]
        dist_px = (dc**2 + dr**2) ** 0.5
        if self.pixel_size_angstrom > 0:
            return dist_px * self.pixel_size_angstrom
        return dist_px


