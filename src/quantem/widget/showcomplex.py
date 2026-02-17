"""
showcomplex: Interactive complex-valued image viewer.

For displaying complex data from ptychography, holography, and exit wave
reconstruction. Supports amplitude, phase, HSV, real, and imaginary display modes.
"""

import json
import pathlib

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy


class ShowComplex2D(anywidget.AnyWidget):
    """
    Interactive viewer for complex-valued 2D data.

    Display complex images from ptychography, holography, or exit wave
    reconstruction with five visualization modes: amplitude, phase, HSV
    (hue=phase, brightness=amplitude), real part, and imaginary part.

    Parameters
    ----------
    data : array_like (complex) or tuple of (real, imag)
        Complex 2D array of shape (height, width) with dtype complex64 or
        complex128. Also accepts a tuple ``(real, imag)`` of two real arrays.
    display_mode : str, default "amplitude"
        Initial display mode: ``"amplitude"``, ``"phase"``, ``"hsv"``,
        ``"real"``, or ``"imag"``.
    title : str, optional
        Title displayed in the widget header.
    cmap : str, default "inferno"
        Colormap for amplitude/real/imag modes. Phase and HSV modes use
        a fixed cyclic colormap.
    pixel_size_angstrom : float, default 0.0
        Pixel size in angstroms for scale bar display.
    log_scale : bool, default False
        Apply log(1+x) to amplitude before display.
    auto_contrast : bool, default False
        Use percentile-based contrast.
    show_fft : bool, default False
        Show FFT panel.
    show_stats : bool, default True
        Show statistics bar.
    show_controls : bool, default True
        Show control panel.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import ShowComplex2D
    >>>
    >>> # Complex exit wave
    >>> data = np.exp(1j * phase) * amplitude
    >>> ShowComplex2D(data, title="Exit Wave", display_mode="hsv")
    >>>
    >>> # From real and imaginary parts
    >>> ShowComplex2D((real_part, imag_part), display_mode="phase")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showcomplex.js"
    _css = pathlib.Path(__file__).parent / "static" / "showcomplex.css"

    # Core state
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    real_bytes = traitlets.Bytes(b"").tag(sync=True)
    imag_bytes = traitlets.Bytes(b"").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)

    # Display mode
    display_mode = traitlets.Unicode("amplitude").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)

    # Display options
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    percentile_low = traitlets.Float(1.0).tag(sync=True)
    percentile_high = traitlets.Float(99.0).tag(sync=True)

    # Scale bar
    pixel_size_angstrom = traitlets.Float(0.0).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)

    # UI
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    image_width_px = traitlets.Int(0).tag(sync=True)

    # Statistics (recomputed per display_mode)
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)

    def __init__(
        self,
        data,
        display_mode: str = "amplitude",
        title: str = "",
        cmap: str = "inferno",
        pixel_size_angstrom: float = 0.0,
        log_scale: bool = False,
        auto_contrast: bool = False,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        show_fft: bool = False,
        show_stats: bool = True,
        show_controls: bool = True,
        scale_bar_visible: bool = True,
        image_width_px: int = 0,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Dataset duck typing
        _extracted_title = None
        _extracted_pixel_size = None
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            _extracted_title = data.name if data.name else None
            if hasattr(data, "units"):
                units = list(data.units)
                sampling_val = float(data.sampling[-1])
                if units[-1] in ("nm",):
                    _extracted_pixel_size = sampling_val * 10  # nm → Å
                elif units[-1] in ("Å", "angstrom", "A"):
                    _extracted_pixel_size = sampling_val
            data = data.array

        # Handle (real, imag) tuple input
        if isinstance(data, tuple) and len(data) == 2:
            real_arr = to_numpy(data[0]).astype(np.float32)
            imag_arr = to_numpy(data[1]).astype(np.float32)
            if real_arr.shape != imag_arr.shape:
                raise ValueError(
                    f"Real and imaginary parts must have same shape, "
                    f"got {real_arr.shape} and {imag_arr.shape}"
                )
            if real_arr.ndim != 2:
                raise ValueError(f"Expected 2D arrays, got {real_arr.ndim}D")
            self._real = real_arr
            self._imag = imag_arr
        else:
            arr = to_numpy(data)
            if not np.issubdtype(arr.dtype, np.complexfloating):
                raise ValueError(
                    f"Expected complex array (complex64/complex128), got {arr.dtype}. "
                    f"Use ShowComplex2D((real, imag)) for real-valued input."
                )
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got {arr.ndim}D")
            self._real = arr.real.astype(np.float32)
            self._imag = arr.imag.astype(np.float32)

        self.height = int(self._real.shape[0])
        self.width = int(self._real.shape[1])

        # Options
        self.display_mode = display_mode
        self.title = title if title else (_extracted_title or "")
        self.cmap = cmap
        if pixel_size_angstrom == 0.0 and _extracted_pixel_size is not None:
            pixel_size_angstrom = _extracted_pixel_size
        self.pixel_size_angstrom = pixel_size_angstrom
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.show_fft = show_fft
        self.show_stats = show_stats
        self.show_controls = show_controls
        self.scale_bar_visible = scale_bar_visible
        self.image_width_px = image_width_px

        # Compute stats for initial display mode
        self._update_stats()

        # Send data to JS
        self.real_bytes = self._real.tobytes()
        self.imag_bytes = self._imag.tobytes()

        # Observer
        self.observe(self._on_display_mode_change, names=["display_mode"])

        # State restoration (must be last)
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = json.loads(pathlib.Path(state).read_text())
            self.load_state_dict(state)

    def _get_display_data(self) -> np.ndarray:
        if self.display_mode == "amplitude":
            return np.sqrt(self._real ** 2 + self._imag ** 2)
        elif self.display_mode == "phase":
            return np.arctan2(self._imag, self._real)
        elif self.display_mode == "real":
            return self._real
        elif self.display_mode == "imag":
            return self._imag
        else:  # hsv — stats on amplitude
            return np.sqrt(self._real ** 2 + self._imag ** 2)

    def _update_stats(self):
        data = self._get_display_data()
        self.stats_mean = float(data.mean())
        self.stats_min = float(data.min())
        self.stats_max = float(data.max())
        self.stats_std = float(data.std())

    def _on_display_mode_change(self, change=None):
        self._update_stats()

    def set_image(self, data):
        """Replace the complex data. Preserves all display settings."""
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            data = data.array
        if isinstance(data, tuple) and len(data) == 2:
            real_arr = to_numpy(data[0]).astype(np.float32)
            imag_arr = to_numpy(data[1]).astype(np.float32)
            if real_arr.shape != imag_arr.shape:
                raise ValueError(
                    f"Real and imaginary parts must have same shape, "
                    f"got {real_arr.shape} and {imag_arr.shape}"
                )
            if real_arr.ndim != 2:
                raise ValueError(f"Expected 2D arrays, got {real_arr.ndim}D")
            self._real = real_arr
            self._imag = imag_arr
        else:
            arr = to_numpy(data)
            if not np.issubdtype(arr.dtype, np.complexfloating):
                raise ValueError(
                    f"Expected complex array (complex64/complex128), got {arr.dtype}."
                )
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got {arr.ndim}D")
            self._real = arr.real.astype(np.float32)
            self._imag = arr.imag.astype(np.float32)
        self.height = int(self._real.shape[0])
        self.width = int(self._real.shape[1])
        self._update_stats()
        self.real_bytes = self._real.tobytes()
        self.imag_bytes = self._imag.tobytes()

    # =========================================================================
    # State Protocol
    # =========================================================================

    def state_dict(self):
        return {
            "display_mode": self.display_mode,
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "pixel_size_angstrom": self.pixel_size_angstrom,
            "scale_bar_visible": self.scale_bar_visible,
            "show_fft": self.show_fft,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "image_width_px": self.image_width_px,
        }

    def save(self, path: str):
        """Save widget state to a JSON file."""
        pathlib.Path(path).write_text(json.dumps(self.state_dict(), indent=2))

    def load_state_dict(self, state):
        """Restore widget state from a dict."""
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        """Print a human-readable summary of the widget state."""
        name = self.title if self.title else "ShowComplex2D"
        lines = [name, "═" * 32]
        lines.append(f"Image:    {self.height}×{self.width} (complex)")
        if self.pixel_size_angstrom > 0:
            ps = self.pixel_size_angstrom
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        amp = np.sqrt(self._real ** 2 + self._imag ** 2)
        lines.append(
            f"Amp:      min={float(amp.min()):.4g}  max={float(amp.max()):.4g}  "
            f"mean={float(amp.mean()):.4g}"
        )
        phase = np.arctan2(self._imag, self._real)
        lines.append(
            f"Phase:    min={float(phase.min()):.4g}  max={float(phase.max()):.4g}  "
            f"mean={float(phase.mean()):.4g}"
        )
        mode = self.display_mode
        cmap = self.cmap if mode in ("amplitude", "real", "imag") else "hsv (cyclic)"
        scale = "log" if self.log_scale else "linear"
        contrast = "auto" if self.auto_contrast else "manual"
        lines.append(f"Display:  {mode} | {cmap} | {contrast} | {scale}")
        if self.show_fft:
            lines[-1] += " | FFT"
        print("\n".join(lines))

    def __repr__(self) -> str:
        name = self.title if self.title else "ShowComplex2D"
        parts = [f"{name}({self.height}×{self.width}"]
        parts.append(f"mode={self.display_mode}")
        if self.pixel_size_angstrom > 0:
            ps = self.pixel_size_angstrom
            if ps >= 10:
                parts.append(f"px={ps / 10:.2f} nm")
            else:
                parts.append(f"px={ps:.2f} Å")
        if self.log_scale:
            parts.append("log")
        if self.show_fft:
            parts.append("fft")
        return ", ".join(parts) + ")"
