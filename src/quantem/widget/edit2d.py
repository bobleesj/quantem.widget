"""
edit2d: Interactive crop/pad/mask tool for 2D images.

Visually define a rectangular output region on a 2D image.
Region inside image bounds crops; region outside pads.
Mask mode allows painting a binary mask on the image.
"""

import json
import pathlib
from typing import Optional, Union, List, Tuple

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy, _resize_image


class Edit2D(anywidget.AnyWidget):
    """
    Interactive visual crop/pad tool for 2D images.

    Display a 2D image with a draggable crop rectangle. The rectangle
    can be positioned anywhere -- inside the image for cropping, extending
    beyond image bounds for padding, or fully enclosing the image for
    pure padding.

    Parameters
    ----------
    data : array_like
        2D array (height, width) for a single image, or
        3D array (N, height, width) or list of 2D arrays for multi-image mode.
        All images are cropped with the same region.
    bounds : tuple of int, optional
        Initial crop bounds as (top, left, bottom, right) in image pixel
        coordinates. Negative values and values exceeding image dimensions
        are allowed for padding. If None, defaults to the full image extent.
    fill_value : float, default 0.0
        Fill value for padded regions outside the original image bounds.
    title : str, default ""
        Title displayed in the widget header.
    cmap : str, default "gray"
        Colormap name.
    pixel_size_angstrom : float, default 0.0
        Pixel size in angstroms for scale bar display.
    show_stats : bool, default True
        Show statistics bar.
    show_controls : bool, default True
        Show control row.
    log_scale : bool, default False
        Log intensity mapping.
    auto_contrast : bool, default True
        Percentile-based contrast.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Edit2D
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> crop = Edit2D(img)
    >>> crop  # display, draw crop region interactively
    >>> crop.result  # returns cropped NumPy array
    >>> crop.crop_bounds  # (top, left, bottom, right) tuple
    """

    _esm = pathlib.Path(__file__).parent / "static" / "edit2d.js"
    _css = pathlib.Path(__file__).parent / "static" / "edit2d.css"

    # =========================================================================
    # Core State
    # =========================================================================
    n_images = traitlets.Int(1).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("gray").tag(sync=True)

    # =========================================================================
    # Crop Region (synced bidirectionally with JS)
    # =========================================================================
    crop_top = traitlets.Int(0).tag(sync=True)
    crop_left = traitlets.Int(0).tag(sync=True)
    crop_bottom = traitlets.Int(0).tag(sync=True)
    crop_right = traitlets.Int(0).tag(sync=True)
    fill_value = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size_angstrom = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # UI Visibility
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Mode: "crop" or "mask"
    # =========================================================================
    mode = traitlets.Unicode("crop").tag(sync=True)

    # =========================================================================
    # Mask State
    # =========================================================================
    mask_bytes = traitlets.Bytes(b"").tag(sync=True)
    mask_tool = traitlets.Unicode("brush").tag(sync=True)
    brush_size = traitlets.Int(10).tag(sync=True)
    mask_action = traitlets.Unicode("add").tag(sync=True)

    # =========================================================================
    # Gallery (multi-image)
    # =========================================================================
    selected_idx = traitlets.Int(0).tag(sync=True)

    def __init__(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        bounds: Optional[Tuple[int, int, int, int]] = None,
        fill_value: float = 0.0,
        mode: str = "crop",
        labels: Optional[List[str]] = None,
        title: str = "",
        cmap: str = "gray",
        pixel_size_angstrom: float = 0.0,
        show_controls: bool = True,
        show_stats: bool = True,
        log_scale: bool = False,
        auto_contrast: bool = True,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode

        # Check if data is a Dataset2d and extract metadata
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            if not title and data.name:
                title = data.name
            if pixel_size_angstrom == 0.0 and hasattr(data, "units"):
                units = list(data.units)
                sampling_val = float(data.sampling[-1])
                if units[-1] in ("nm",):
                    pixel_size_angstrom = sampling_val * 10  # nm -> angstrom
                elif units[-1] in ("\u00c5", "angstrom", "A"):
                    pixel_size_angstrom = sampling_val
            data = data.array

        # Convert input to NumPy (handles NumPy, CuPy, PyTorch)
        if isinstance(data, list):
            images = [to_numpy(d) for d in data]
            shapes = [img.shape for img in images]
            if len(set(shapes)) > 1:
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                images = [_resize_image(img, max_h, max_w) for img in images]
            data = np.stack(images)
        else:
            data = to_numpy(data)

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._data = data.astype(np.float32)
        self.n_images = int(data.shape[0])
        self.height = int(data.shape[1])
        self.width = int(data.shape[2])

        # Labels
        if labels is None:
            if self.n_images == 1:
                self.labels = ["Image"]
            else:
                self.labels = [f"Image {i+1}" for i in range(self.n_images)]
        else:
            self.labels = list(labels)

        # Options
        self.title = title
        self.cmap = cmap
        self.pixel_size_angstrom = pixel_size_angstrom
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.fill_value = fill_value

        # Crop bounds
        if bounds is not None:
            self.crop_top, self.crop_left, self.crop_bottom, self.crop_right = bounds
        else:
            self.crop_top = 0
            self.crop_left = 0
            self.crop_bottom = self.height
            self.crop_right = self.width

        # Compute stats for current image
        self._compute_stats()

        # Send raw float32 data to JS
        self.frame_bytes = self._data.tobytes()

        self.selected_idx = 0

        # State restoration (must be last)
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = json.loads(pathlib.Path(state).read_text())
            self.load_state_dict(state)

    def _compute_stats(self):
        img = self._data[self.selected_idx]
        self.stats_mean = float(np.mean(img))
        self.stats_min = float(np.min(img))
        self.stats_max = float(np.max(img))
        self.stats_std = float(np.std(img))

    def _crop_single(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        out_h = self.crop_bottom - self.crop_top
        out_w = self.crop_right - self.crop_left

        if out_h <= 0 or out_w <= 0:
            return np.empty((0, 0), dtype=img.dtype)

        result = np.full((out_h, out_w), self.fill_value, dtype=img.dtype)

        # Compute overlap between crop region and image
        src_top = max(0, self.crop_top)
        src_left = max(0, self.crop_left)
        src_bottom = min(h, self.crop_bottom)
        src_right = min(w, self.crop_right)

        if src_top >= src_bottom or src_left >= src_right:
            return result

        # Where to place in output
        dst_top = src_top - self.crop_top
        dst_left = src_left - self.crop_left
        dst_bottom = dst_top + (src_bottom - src_top)
        dst_right = dst_left + (src_right - src_left)

        result[dst_top:dst_bottom, dst_left:dst_right] = img[src_top:src_bottom, src_left:src_right]
        return result

    def _apply_mask(self, img: np.ndarray, m: np.ndarray) -> np.ndarray:
        out = img.copy()
        out[m] = self.fill_value
        return out

    @property
    def mask(self) -> np.ndarray:
        """Current mask as a boolean array (H, W). True = masked."""
        if not self.mask_bytes:
            return np.zeros((self.height, self.width), dtype=bool)
        arr = np.frombuffer(self.mask_bytes, dtype=np.uint8).reshape(
            self.height, self.width
        )
        return arr > 0

    @property
    def result(self) -> Union[np.ndarray, List[np.ndarray]]:
        """Return result based on current mode.

        Crop mode: cropped/padded image(s).
        Mask mode: image(s) with masked pixels set to fill_value.
        """
        if self.mode == "mask":
            m = self.mask
            if self.n_images == 1:
                return self._apply_mask(self._data[0], m)
            return [self._apply_mask(self._data[i], m) for i in range(self.n_images)]
        if self.n_images == 1:
            return self._crop_single(self._data[0])
        return [self._crop_single(self._data[i]) for i in range(self.n_images)]

    @property
    def crop_bounds(self) -> Tuple[int, int, int, int]:
        """Current crop bounds as (top, left, bottom, right)."""
        return (self.crop_top, self.crop_left, self.crop_bottom, self.crop_right)

    @crop_bounds.setter
    def crop_bounds(self, bounds: Tuple[int, int, int, int]):
        self.crop_top, self.crop_left, self.crop_bottom, self.crop_right = bounds

    @property
    def crop_size(self) -> Tuple[int, int]:
        """Output size as (height, width)."""
        return (self.crop_bottom - self.crop_top, self.crop_right - self.crop_left)

    def set_image(self, data, **kwargs):
        """Replace the image data."""
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            if "title" not in kwargs and data.name:
                self.title = data.name
            data = data.array

        if isinstance(data, list):
            images = [to_numpy(d) for d in data]
            shapes = [img.shape for img in images]
            if len(set(shapes)) > 1:
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                images = [_resize_image(img, max_h, max_w) for img in images]
            data = np.stack(images)
        else:
            data = to_numpy(data)

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._data = data.astype(np.float32)
        self.n_images = int(data.shape[0])
        self.height = int(data.shape[1])
        self.width = int(data.shape[2])
        self.crop_top = 0
        self.crop_left = 0
        self.crop_bottom = self.height
        self.crop_right = self.width
        self.mask_bytes = b""
        self._compute_stats()
        self.frame_bytes = self._data.tobytes()

    # =========================================================================
    # State Protocol
    # =========================================================================

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "mode": self.mode,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "show_controls": self.show_controls,
            "show_stats": self.show_stats,
            "pixel_size_angstrom": self.pixel_size_angstrom,
            "fill_value": self.fill_value,
            "crop_top": self.crop_top,
            "crop_left": self.crop_left,
            "crop_bottom": self.crop_bottom,
            "crop_right": self.crop_right,
            "brush_size": self.brush_size,
        }

    def save(self, path: str):
        pathlib.Path(path).write_text(json.dumps(self.state_dict(), indent=2))

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        name = self.title if self.title else "Edit2D"
        lines = [name, "═" * 32]
        lines.append(f"Image:    {self.height}×{self.width}")
        if self.n_images > 1:
            lines[-1] += f" ({self.n_images} images)"
        if self.pixel_size_angstrom > 0:
            ps = self.pixel_size_angstrom
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        lines.append(f"Mode:     {self.mode}")
        if self.mode == "crop":
            crop_h, crop_w = self.crop_size
            lines.append(
                f"Crop:     ({self.crop_top}, {self.crop_left}) → "
                f"({self.crop_bottom}, {self.crop_right})  "
                f"= {crop_h}×{crop_w}"
            )
            lines.append(f"Fill:     {self.fill_value}")
        else:
            mask_px = int(np.sum(self.mask)) if self.mask_bytes else 0
            total = self.height * self.width
            pct = 100 * mask_px / total if total > 0 else 0
            lines.append(f"Mask:     {mask_px} px ({pct:.1f}%)")
            lines.append(f"Brush:    {self.brush_size} px")
        scale = "log" if self.log_scale else "linear"
        contrast = "auto" if self.auto_contrast else "manual"
        lines.append(f"Display:  {self.cmap} | {contrast} | {scale}")
        print("\n".join(lines))

    def __repr__(self):
        if self.mode == "mask":
            mask_px = int(np.sum(self.mask)) if self.mask_bytes else 0
            total = self.height * self.width
            pct = 100 * mask_px / total if total > 0 else 0
            return f"Edit2D({self.height}x{self.width}, mask={mask_px}px ({pct:.1f}%))"
        crop_h, crop_w = self.crop_size
        return (
            f"Edit2D({self.height}x{self.width}, "
            f"crop={crop_h}x{crop_w} at ({self.crop_top},{self.crop_left}), "
            f"fill={self.fill_value})"
        )
