"""
Merge4DSTEM: Stack multiple 4D-STEM datasets along a time axis.

Produces a 5D array (n_sources, scan_r, scan_c, det_r, det_c) backed by
torch.stack on GPU, outputting quantem's native Dataset4dstem.
"""

from __future__ import annotations

import json
import pathlib
import warnings

from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.json_state import save_state_file, resolve_widget_version, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from quantem.core.config import validate_device

    _HAS_VALIDATE_DEVICE = True
except Exception:
    _HAS_VALIDATE_DEVICE = False

_REAL_UNITS = {"Å", "angstrom", "A", "nm"}
_K_UNITS = {"mrad", "1/Å", "1/A"}
_MERGE_ESM = pathlib.Path(__file__).parent / "static" / "merge4dstem.js"
_MERGE_CSS = pathlib.Path(__file__).parent / "static" / "merge4dstem.css"


def _extract_source(source, index: int) -> dict:
    """Normalize a single source into {array, name, sampling, units}."""
    info: dict = {"name": f"source_{index:03d}", "sampling": None, "units": None}

    # File path (Zarr zip)
    if isinstance(source, (str, pathlib.Path)):
        path = pathlib.Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        try:
            from quantem.core.io.serialize import load

            ds = load(str(path))
            info["name"] = path.stem
            if hasattr(ds, "sampling"):
                info["sampling"] = list(getattr(ds, "sampling", []))
            if hasattr(ds, "units"):
                info["units"] = list(getattr(ds, "units", []))
            source = ds.array
        except ImportError:
            raise ImportError(
                "Loading from file requires quantem.core. Install quantem or pass arrays directly."
            )

    # Dataset-like object
    if hasattr(source, "sampling") and hasattr(source, "array"):
        if hasattr(source, "name") and source.name:
            info["name"] = str(source.name)
        info["sampling"] = list(getattr(source, "sampling", []))
        info["units"] = list(getattr(source, "units", []))
        source = source.array

    arr = to_numpy(source, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(
            f"Source {index} must be 4D (scan_r, scan_c, det_r, det_c), got {arr.ndim}D"
        )
    info["array"] = arr
    return info


def _extract_calibration(source_info: dict) -> dict:
    """Extract pixel_size, k_pixel_size, and units from source info."""
    cal: dict = {
        "pixel_size": None,
        "pixel_unit": "px",
        "pixel_calibrated": False,
        "k_pixel_size": None,
        "k_unit": "px",
        "k_calibrated": False,
    }
    sampling = source_info.get("sampling")
    units = source_info.get("units")
    if not sampling or not units:
        return cal

    if len(units) >= 2 and len(sampling) >= 2 and units[0] in _REAL_UNITS:
        sy, sx = float(sampling[0]), float(sampling[1])
        if units[0] == "nm":
            sy *= 10.0
        if units[1] == "nm":
            sx *= 10.0
        cal["pixel_size"] = (sy + sx) / 2.0
        cal["pixel_unit"] = "Å"
        cal["pixel_calibrated"] = True

    if len(units) >= 4 and len(sampling) >= 4 and units[2] in _K_UNITS:
        ky, kx = float(sampling[2]), float(sampling[3])
        cal["k_pixel_size"] = (ky + kx) / 2.0
        cal["k_unit"] = units[2]
        cal["k_calibrated"] = True

    return cal


class Merge4DSTEM(anywidget.AnyWidget):
    """
    Stack multiple 4D-STEM datasets along a time axis.

    Parameters
    ----------
    sources : list
        List of 4D arrays, Dataset4dstem objects, or file paths (Zarr zip).
        All must have identical (scan_r, scan_c, det_r, det_c) shape.
    pixel_size : float, optional
        Override real-space calibration (Å/px).
    k_pixel_size : float, optional
        Override k-space calibration (mrad/px).
    frame_dim_label : str, default "Time"
        Label for the stacked dimension.
    bin_factor : int, default 2
        Detector binning factor. 1 = no binning, 2 = 2x2 average pooling, etc.
    cmap : str, default "inferno"
        Colormap for preview rendering.
    log_scale : bool, default False
        Log scale for preview display.
    """

    _esm = _MERGE_ESM if _MERGE_ESM.exists() else "export function render() {}"
    _css = _MERGE_CSS if _MERGE_CSS.exists() else ""

    # Shape info
    n_sources = traitlets.Int(0).tag(sync=True)
    scan_rows = traitlets.Int(0).tag(sync=True)
    scan_cols = traitlets.Int(0).tag(sync=True)
    det_rows = traitlets.Int(0).tag(sync=True)
    det_cols = traitlets.Int(0).tag(sync=True)

    # Calibration
    pixel_size = traitlets.Float(1.0).tag(sync=True)
    pixel_unit = traitlets.Unicode("px").tag(sync=True)
    pixel_calibrated = traitlets.Bool(False).tag(sync=True)
    k_pixel_size = traitlets.Float(1.0).tag(sync=True)
    k_unit = traitlets.Unicode("px").tag(sync=True)
    k_calibrated = traitlets.Bool(False).tag(sync=True)

    # Source info (JSON string for JS)
    source_info_json = traitlets.Unicode("[]").tag(sync=True)

    # Preview (mean DP of selected source)
    preview_bytes = traitlets.Bytes(b"").tag(sync=True)
    preview_rows = traitlets.Int(0).tag(sync=True)
    preview_cols = traitlets.Int(0).tag(sync=True)
    preview_index = traitlets.Int(0).tag(sync=True)

    # Merge state
    merged = traitlets.Bool(False).tag(sync=True)
    output_shape_json = traitlets.Unicode("[]").tag(sync=True)
    frame_dim_label = traitlets.Unicode("Time").tag(sync=True)

    # Binning
    bin_factor = traitlets.Int(2).tag(sync=True)

    # Status
    status_message = traitlets.Unicode("").tag(sync=True)
    status_level = traitlets.Unicode("ok").tag(sync=True)

    # Merge trigger (JS -> Python)
    _merge_requested = traitlets.Bool(False).tag(sync=True)

    # Display
    title = traitlets.Unicode("Merge4DSTEM").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # Device
    device = traitlets.Unicode("cpu").tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups):
        return normalize_tool_groups("Merge4DSTEM", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_sources: bool = False,
        disable_merge: bool = False,
        disable_preview: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
    ):
        return build_tool_groups(
            "Merge4DSTEM",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "sources": disable_sources,
                "merge": disable_merge,
                "preview": disable_preview,
                "stats": disable_stats,
                "export": disable_export,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_sources: bool = False,
        hide_merge: bool = False,
        hide_preview: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
    ):
        return build_tool_groups(
            "Merge4DSTEM",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "sources": hide_sources,
                "merge": hide_merge,
                "preview": hide_preview,
                "stats": hide_stats,
                "export": hide_export,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    def __init__(
        self,
        sources: list,
        *,
        pixel_size: float | None = None,
        k_pixel_size: float | None = None,
        frame_dim_label: str = "Time",
        bin_factor: int = 2,
        title: str = "Merge4DSTEM",
        cmap: str = "inferno",
        log_scale: bool = False,
        show_controls: bool = True,
        show_stats: bool = True,
        disabled_tools: list[str] | None = None,
        hidden_tools: list[str] | None = None,
        disable_display: bool = False,
        disable_sources: bool = False,
        disable_merge: bool = False,
        disable_preview: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
        hide_display: bool = False,
        hide_sources: bool = False,
        hide_merge: bool = False,
        hide_preview: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
        device: str | None = None,
        state: dict | str | pathlib.Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        if not _HAS_TORCH:
            raise ImportError("Merge4DSTEM requires torch. Install PyTorch to use this widget.")

        if len(sources) < 2:
            raise ValueError("Merge4DSTEM requires at least 2 sources.")

        # Extract and validate sources
        source_infos = []
        for i, src in enumerate(sources):
            source_infos.append(_extract_source(src, i))

        ref_shape = source_infos[0]["array"].shape
        source_table = []
        for i, info in enumerate(source_infos):
            shape = info["array"].shape
            valid = shape == ref_shape
            msg = "OK" if valid else f"Shape mismatch: expected {ref_shape}, got {shape}"
            if not valid:
                raise ValueError(
                    f"Source {i} ({info['name']}): shape {shape} does not match "
                    f"reference shape {ref_shape}. All sources must have identical "
                    f"(scan_r, scan_c, det_r, det_c) dimensions."
                )
            source_table.append({
                "name": info["name"],
                "shape": list(shape),
                "valid": valid,
                "message": msg,
            })

        # Check calibration consistency (warn on mismatch, use first as reference)
        ref_cal = _extract_calibration(source_infos[0])
        cal_warnings = []
        for i, info in enumerate(source_infos[1:], start=1):
            cal = _extract_calibration(info)
            if ref_cal["pixel_calibrated"] and cal["pixel_calibrated"]:
                if abs(ref_cal["pixel_size"] - cal["pixel_size"]) > 1e-6:
                    cal_warnings.append(
                        f"Source {i} ({info['name']}): pixel_size "
                        f"{cal['pixel_size']:.4g} differs from reference {ref_cal['pixel_size']:.4g}"
                    )
            if ref_cal["k_calibrated"] and cal["k_calibrated"]:
                if abs(ref_cal["k_pixel_size"] - cal["k_pixel_size"]) > 1e-6:
                    cal_warnings.append(
                        f"Source {i} ({info['name']}): k_pixel_size "
                        f"{cal['k_pixel_size']:.4g} differs from reference {ref_cal['k_pixel_size']:.4g}"
                    )

        # Set calibration (explicit overrides take priority)
        if pixel_size is not None:
            self.pixel_size = float(pixel_size)
            self.pixel_unit = "Å"
            self.pixel_calibrated = True
        elif ref_cal["pixel_calibrated"]:
            self.pixel_size = ref_cal["pixel_size"]
            self.pixel_unit = ref_cal["pixel_unit"]
            self.pixel_calibrated = True
        else:
            self.pixel_size = 1.0
            self.pixel_unit = "px"
            self.pixel_calibrated = False

        if k_pixel_size is not None:
            self.k_pixel_size = float(k_pixel_size)
            self.k_unit = "mrad"
            self.k_calibrated = True
        elif ref_cal["k_calibrated"]:
            self.k_pixel_size = ref_cal["k_pixel_size"]
            self.k_unit = ref_cal["k_unit"]
            self.k_calibrated = True
        else:
            self.k_pixel_size = 1.0
            self.k_unit = "px"
            self.k_calibrated = False

        # Move all source arrays to torch on GPU
        total_numel = sum(info["array"].size for info in source_infos)
        device_str = self._resolve_torch_device(requested=device, numel=total_numel)
        if device_str is None:
            requested = "auto" if device is None else str(device)
            raise ValueError(f"Unable to initialize torch device '{requested}'")

        self._device = torch.device(device_str)
        self.device = device_str

        self._source_tensors = []
        for info in source_infos:
            arr = np.array(info["array"], dtype=np.float32, copy=True)
            t = torch.from_numpy(arr).to(self._device)
            self._source_tensors.append(t)

        self._source_names = [info["name"] for info in source_infos]
        self._merged_tensor = None

        # Set shape traits
        scan_r, scan_c, det_r, det_c = ref_shape
        self.n_sources = len(source_infos)
        self.scan_rows = scan_r
        self.scan_cols = scan_c
        self.det_rows = det_r
        self.det_cols = det_c

        # Display
        self.title = title
        self.cmap = cmap
        self.log_scale = log_scale
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.frame_dim_label = frame_dim_label
        self.bin_factor = max(1, int(bin_factor))

        # Source table JSON
        self.source_info_json = json.dumps(source_table)

        # Output shape (accounts for binning)
        self._update_output_shape()

        # Compute preview (mean DP of first source)
        self._compute_preview()

        # Status
        if cal_warnings:
            self.status_level = "warn"
            self.status_message = f"Calibration mismatch: {'; '.join(cal_warnings)}"
            for w in cal_warnings:
                warnings.warn(w, stacklevel=2)
        else:
            self.status_level = "ok"
            self.status_message = f"Ready to merge ({self.n_sources} compatible sources)"

        # Observe merge trigger and preview index changes
        self.observe(self._on_merge_requested, names=["_merge_requested"])
        self.observe(self._on_preview_index_changed, names=["preview_index"])
        self.observe(self._on_bin_factor_changed, names=["bin_factor"])

        # Tool visibility
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_sources=disable_sources,
            disable_merge=disable_merge,
            disable_preview=disable_preview,
            disable_stats=disable_stats,
            disable_export=disable_export,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_sources=hide_sources,
            hide_merge=hide_merge,
            hide_preview=hide_preview,
            hide_stats=hide_stats,
            hide_export=hide_export,
            hide_all=hide_all,
        )

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(self) -> Self:
        """Stack all sources along dim=0 to produce a 5D array.

        If bin_factor > 1, detector dimensions are binned via average pooling
        before stacking.
        """
        bf = max(1, self.bin_factor)
        if bf > 1:
            binned = []
            for t in self._source_tensors:
                # t: (scan_r, scan_c, det_r, det_c) -> reshape for avg_pool2d
                sr, sc, dr, dc = t.shape
                # Flatten scan dims, treat det as spatial: (sr*sc, 1, dr, dc)
                flat = t.reshape(sr * sc, 1, dr, dc)
                pooled = F.avg_pool2d(flat, kernel_size=bf, stride=bf)
                binned.append(pooled.reshape(sr, sc, pooled.shape[2], pooled.shape[3]))
            self._merged_tensor = torch.stack(binned, dim=0)
        else:
            self._merged_tensor = torch.stack(self._source_tensors, dim=0)

        self.merged = True
        self.status_level = "ok"
        shape_5d = list(self._merged_tensor.shape)
        self.output_shape_json = json.dumps(shape_5d)
        bin_note = f" (bin {bf}x)" if bf > 1 else ""
        self.status_message = (
            f"Merged {self.n_sources} sources -> {tuple(shape_5d)}{bin_note} on {self.device}"
        )
        return self

    @property
    def result(self):
        """Return merged result as a Dataset4dstem, or None if not merged."""
        if self._merged_tensor is None:
            return None
        try:
            from quantem.core.dataset import Dataset4dstem

            arr = self._merged_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            sampling = []
            units_list = []

            # Build sampling/units for 5D: (n_sources, scan_r, scan_c, det_r, det_c)
            # The frame dimension has no calibration
            if self.pixel_calibrated:
                sampling.extend([self.pixel_size, self.pixel_size])
                units_list.extend([self.pixel_unit, self.pixel_unit])
            else:
                sampling.extend([1.0, 1.0])
                units_list.extend(["pixels", "pixels"])

            if self.k_calibrated:
                sampling.extend([self.k_pixel_size, self.k_pixel_size])
                units_list.extend([self.k_unit, self.k_unit])
            else:
                sampling.extend([1.0, 1.0])
                units_list.extend(["pixels", "pixels"])

            return Dataset4dstem.from_array(
                arr,
                name="merged_4dstem",
                sampling=tuple(sampling),
                units=tuple(units_list),
            )
        except ImportError:
            return None

    @property
    def result_array(self) -> np.ndarray | None:
        """Return merged result as a raw 5D numpy array, or None if not merged."""
        if self._merged_tensor is None:
            return None
        return self._merged_tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    def save_result(self, path: str | pathlib.Path) -> pathlib.Path:
        """Save merged result as a Zarr zip archive."""
        if self._merged_tensor is None:
            raise RuntimeError("No merged result. Call merge() first.")

        output = pathlib.Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        result = self.result
        if result is not None:
            result.save(str(output), mode="o", compression_level=4)
        else:
            # Fallback: save raw numpy
            arr = self.result_array
            np.savez_compressed(str(output), data=arr)

        return output

    def to_show4dstem(self, **kwargs):
        """Open merged result in Show4DSTEM."""
        if self._merged_tensor is None:
            raise RuntimeError("No merged result. Call merge() first.")

        from quantem.widget.show4dstem import Show4DSTEM

        pixel_size = kwargs.pop("pixel_size", self.pixel_size if self.pixel_calibrated else None)
        k_pixel_size = kwargs.pop("k_pixel_size", self.k_pixel_size if self.k_calibrated else None)

        arr = self._merged_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        return Show4DSTEM(arr, pixel_size=pixel_size, k_pixel_size=k_pixel_size, **kwargs)

    def state_dict(self) -> dict:
        return {
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "show_controls": self.show_controls,
            "show_stats": self.show_stats,
            "frame_dim_label": self.frame_dim_label,
            "bin_factor": self.bin_factor,
            "disabled_tools": list(self.disabled_tools),
            "hidden_tools": list(self.hidden_tools),
        }

    def save(self, path: str | pathlib.Path) -> None:
        save_state_file(path, "Merge4DSTEM", self.state_dict())

    def load_state_dict(self, state: dict) -> None:
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self) -> None:
        lines = ["Merge4DSTEM", "=" * 40]
        lines.append(f"Sources:    {self.n_sources}")
        lines.append(
            f"Shape:      scan=({self.scan_rows}, {self.scan_cols}), "
            f"det=({self.det_rows}, {self.det_cols})"
        )
        if self.pixel_calibrated:
            lines.append(f"Real cal:   {self.pixel_size:.4g} {self.pixel_unit}/px")
        if self.k_calibrated:
            lines.append(f"K cal:      {self.k_pixel_size:.4g} {self.k_unit}/px")
        lines.append(f"Bin:        {self.bin_factor}x")
        lines.append(f"Merged:     {self.merged}")
        if self.merged:
            lines.append(f"Output:     {self.output_shape_json}")
        lines.append(f"Device:     {self.device}")
        lines.append(f"Display:    cmap={self.cmap}, log_scale={self.log_scale}")
        if self.disabled_tools:
            lines.append(f"Locked:     {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:     {', '.join(self.hidden_tools)}")
        if self.status_message:
            lines.append(f"Status:     {self.status_level.upper()} - {self.status_message}")
        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_preview(self):
        """Compute mean diffraction pattern from selected source for preview."""
        idx = max(0, min(self.preview_index, len(self._source_tensors) - 1))
        src = self._source_tensors[idx]
        mean_dp = src.mean(dim=(0, 1))
        arr = mean_dp.detach().cpu().contiguous().float().numpy().astype(np.float32, copy=False)
        self.preview_bytes = arr.tobytes()
        self.preview_rows = int(src.shape[2])
        self.preview_cols = int(src.shape[3])

    def _on_merge_requested(self, change=None):
        if self._merge_requested:
            self.merge()
            self._merge_requested = False

    def _on_preview_index_changed(self, change=None):
        self._compute_preview()

    def _on_bin_factor_changed(self, change=None):
        self._update_output_shape()

    def _update_output_shape(self):
        bf = max(1, self.bin_factor)
        out_det_r = self.det_rows // bf if bf > 1 else self.det_rows
        out_det_c = self.det_cols // bf if bf > 1 else self.det_cols
        self.output_shape_json = json.dumps(
            [self.n_sources, self.scan_rows, self.scan_cols, out_det_r, out_det_c]
        )

    def _resolve_torch_device(self, requested: str | None, numel: int) -> str | None:
        if not _HAS_TORCH:
            return None

        if requested is not None:
            device_str = str(requested).strip().lower()
        elif _HAS_VALIDATE_DEVICE:
            device_str, _ = validate_device(None)
        else:
            device_str = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if device_str == "mps" and numel > 2**31 - 1:
            device_str = "cpu"

        try:
            torch.zeros(1, device=torch.device(device_str))
        except Exception:
            return None

        return device_str

    def __repr__(self) -> str:
        return (
            f"Merge4DSTEM(sources={self.n_sources}, "
            f"scan=({self.scan_rows}, {self.scan_cols}), "
            f"det=({self.det_rows}, {self.det_cols}), "
            f"bin={self.bin_factor}x, "
            f"merged={self.merged}, device={self.device})"
        )


bind_tool_runtime_api(Merge4DSTEM, "Merge4DSTEM")
