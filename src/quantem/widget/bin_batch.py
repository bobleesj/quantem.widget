"""
Preset-driven batch binning for 4D-STEM datasets.

This runner is designed for folder-scale preprocessing where files are processed
one-by-one (not all loaded at once), which is useful for large time-series data.

Supported inputs: `.npy`, `.npz`, `.h5`, `.hdf5`, `.emd`
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from quantem.widget.json_state import unwrap_state_payload

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import h5py  # type: ignore

    _HAS_H5PY = True
except Exception:
    h5py = None  # type: ignore[assignment]
    _HAS_H5PY = False

try:
    from quantem.core.config import validate_device

    _HAS_VALIDATE_DEVICE = True
except Exception:
    _HAS_VALIDATE_DEVICE = False

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


@dataclass(frozen=True)
class BinPreset:
    """Serializable binning preset for batch preprocessing."""

    scan_bin_row: int = 1
    scan_bin_col: int = 1
    det_bin_row: int = 1
    det_bin_col: int = 1
    bin_mode: str = "mean"  # "mean" | "sum"
    edge_mode: str = "crop"  # "crop" | "pad" | "error"

    # Optional hints for special layouts
    scan_shape: tuple[int, int] | None = None  # for flattened 3D input (N, det_r, det_c)
    npz_key: str | None = None
    h5_dataset_path: str | None = None  # explicit HDF5 dataset path
    time_axis: int = 0  # for 5D input (time, scan_r, scan_c, det_r, det_c)
    backend: str = "torch"  # torch-only
    device: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BinPreset":
        preset = cls(
            scan_bin_row=int(data.get("scan_bin_row", 1)),
            scan_bin_col=int(data.get("scan_bin_col", 1)),
            det_bin_row=int(data.get("det_bin_row", 1)),
            det_bin_col=int(data.get("det_bin_col", 1)),
            bin_mode=str(data.get("bin_mode", "mean")).lower(),
            edge_mode=str(data.get("edge_mode", "crop")).lower(),
            scan_shape=tuple(data["scan_shape"]) if data.get("scan_shape") is not None else None,
            npz_key=data.get("npz_key"),
            h5_dataset_path=data.get("h5_dataset_path"),
            time_axis=int(data.get("time_axis", 0)),
            backend=str(data.get("backend", "torch")).lower(),
            device=str(data["device"]) if data.get("device") is not None else None,
        )
        preset.validate()
        return preset

    def validate(self) -> None:
        if min(self.scan_bin_row, self.scan_bin_col, self.det_bin_row, self.det_bin_col) < 1:
            raise ValueError("All bin factors must be >= 1")
        if self.bin_mode not in {"mean", "sum"}:
            raise ValueError("bin_mode must be 'mean' or 'sum'")
        if self.edge_mode not in {"crop", "pad", "error"}:
            raise ValueError("edge_mode must be 'crop', 'pad', or 'error'")
        if self.scan_shape is not None and len(self.scan_shape) != 2:
            raise ValueError("scan_shape must be [rows, cols]")
        if self.backend != "torch":
            raise ValueError("backend must be 'torch' (torch-only pipeline)")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scan_bin_row": self.scan_bin_row,
            "scan_bin_col": self.scan_bin_col,
            "det_bin_row": self.det_bin_row,
            "det_bin_col": self.det_bin_col,
            "bin_mode": self.bin_mode,
            "edge_mode": self.edge_mode,
            "time_axis": self.time_axis,
            "backend": self.backend,
        }
        if self.scan_shape is not None:
            out["scan_shape"] = [int(self.scan_shape[0]), int(self.scan_shape[1])]
        if self.npz_key:
            out["npz_key"] = self.npz_key
        if self.h5_dataset_path is not None:
            out["h5_dataset_path"] = self.h5_dataset_path
        if self.device is not None:
            out["device"] = self.device
        return out


def load_preset(path: str | pathlib.Path) -> BinPreset:
    """Load preset JSON from disk."""
    payload = json.loads(pathlib.Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Preset JSON must be an object")
    return BinPreset.from_dict(unwrap_state_payload(payload))


def save_preset(path: str | pathlib.Path, preset: BinPreset) -> None:
    """Save preset JSON to disk."""
    pathlib.Path(path).write_text(json.dumps(preset.to_dict(), indent=2))


def _seconds_to_hms(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0:
        return "unknown"
    total = int(round(float(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _runtime_memory_snapshot(device_str: str) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()  # type: ignore[name-defined]
            out["host_mem_total_gb"] = float(vm.total) / 1_000_000_000.0
            out["host_mem_available_gb"] = float(vm.available) / 1_000_000_000.0
            out["host_mem_used_pct"] = float(vm.percent)
        except Exception:
            pass

    if _HAS_TORCH and str(device_str).startswith("cuda") and torch.cuda.is_available():
        try:
            dev = torch.device(device_str)
            idx = int(dev.index) if dev.index is not None else int(torch.cuda.current_device())
            props = torch.cuda.get_device_properties(idx)
            out["gpu_mem_total_gb"] = float(props.total_memory) / 1_000_000_000.0
            out["gpu_mem_reserved_gb"] = float(torch.cuda.memory_reserved(idx)) / 1_000_000_000.0
            out["gpu_mem_allocated_gb"] = float(torch.cuda.memory_allocated(idx)) / 1_000_000_000.0
            out["gpu_device_name"] = str(props.name)
        except Exception:
            pass

    return out


def _resolve_torch_device(preset: BinPreset, numel: int) -> str:
    if not _HAS_TORCH:
        raise ValueError("Torch is required for bin batch preprocessing.")

    if preset.device is not None:
        device_str = str(preset.device).strip().lower()
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
    except Exception as exc:
        raise ValueError(f"Unable to initialize torch device '{device_str}': {exc}") from exc

    return device_str


def _bin_axis_torch(data, axis: int, factor: int, mode: str, edge: str):
    if factor == 1:
        return data

    n = int(data.shape[axis])

    if edge == "crop":
        n_used = (n // factor) * factor
        if n_used <= 0:
            raise ValueError(
                f"crop mode: axis size {n} is smaller than factor {factor} on axis {axis}"
            )
        trimmed = data.narrow(axis, 0, n_used)
    elif edge == "pad":
        n_used = int(math.ceil(n / factor) * factor)
        pad_amount = n_used - n
        if pad_amount > 0:
            pad_shape = list(data.shape)
            pad_shape[axis] = pad_amount
            pad_block = torch.zeros(pad_shape, dtype=data.dtype, device=data.device)
            trimmed = torch.cat([data, pad_block], dim=axis)
        else:
            trimmed = data
    else:  # "error"
        if n % factor != 0:
            raise ValueError(
                f"error mode: axis size {n} is not divisible by factor {factor} on axis {axis}"
            )
        n_used = n
        trimmed = data

    new_shape = tuple(trimmed.shape[:axis]) + (n_used // factor, factor) + tuple(trimmed.shape[axis + 1 :])
    reshaped = trimmed.reshape(new_shape)
    reduce_axis = axis + 1
    if mode == "sum":
        return reshaped.sum(dim=reduce_axis)
    return reshaped.mean(dim=reduce_axis)


def _binned_axis_shape(n: int, factor: int, edge: str, axis: int) -> tuple[int, int]:
    if factor < 1:
        raise ValueError("bin factors must be >= 1")

    if edge == "crop":
        n_used = (n // factor) * factor
        if n_used <= 0:
            raise ValueError(
                f"crop mode: axis size {n} is smaller than factor {factor} on axis {axis}"
            )
    elif edge == "pad":
        n_used = int(math.ceil(n / factor) * factor)
    else:  # edge == "error"
        if n % factor != 0:
            raise ValueError(
                f"error mode: axis size {n} is not divisible by factor {factor} on axis {axis}"
            )
        n_used = n

    return int(n_used // factor), int(n_used)


def _bin_4d_torch(data4d: np.ndarray, preset: BinPreset, device_str: str) -> np.ndarray:
    writable = np.array(data4d, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(writable).to(torch.device(device_str))
    factors = (
        preset.scan_bin_row,
        preset.scan_bin_col,
        preset.det_bin_row,
        preset.det_bin_col,
    )
    out = tensor
    for axis, factor in enumerate(factors):
        out = _bin_axis_torch(out, axis=axis, factor=int(factor), mode=preset.bin_mode, edge=preset.edge_mode)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def _flattened_to_4d(data3d: np.ndarray, scan_shape: tuple[int, int] | None) -> np.ndarray:
    n, det_r, det_c = int(data3d.shape[0]), int(data3d.shape[1]), int(data3d.shape[2])
    if scan_shape is None:
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(
                f"Flattened 3D input has N={n}; provide scan_shape in preset (rows, cols)."
            )
        scan_shape = (side, side)
    if int(scan_shape[0]) * int(scan_shape[1]) != n:
        raise ValueError(f"scan_shape={scan_shape} does not match flattened N={n}")
    return data3d.reshape(int(scan_shape[0]), int(scan_shape[1]), det_r, det_c)


def apply_preset_to_array(data: np.ndarray, preset: BinPreset) -> np.ndarray:
    """Apply preset to a single array (3D flattened, 4D, or 5D)."""
    preset.validate()
    arr = np.asarray(data)
    device_str = _resolve_torch_device(preset, numel=int(arr.size))

    if arr.ndim == 3:
        prepared = _flattened_to_4d(arr, preset.scan_shape)
        b4 = _bin_4d_torch(prepared, preset, device_str)
        return b4.reshape(b4.shape[0] * b4.shape[1], b4.shape[2], b4.shape[3])

    if arr.ndim == 4:
        return _bin_4d_torch(arr, preset, device_str)

    if arr.ndim == 5:
        if preset.time_axis != 0:
            arr = np.moveaxis(arr, preset.time_axis, 0)
            restore_axis = preset.time_axis
        else:
            restore_axis = 0

        frames = [_bin_4d_torch(arr[i], preset, device_str) for i in range(arr.shape[0])]
        out = np.stack(frames, axis=0)
        if restore_axis != 0:
            out = np.moveaxis(out, 0, restore_axis)
        return out

    raise ValueError(f"Expected 3D, 4D, or 5D array. Got {arr.ndim}D")


def _iter_inputs(input_dir: pathlib.Path, pattern: str, recursive: bool) -> list[pathlib.Path]:
    if recursive:
        return sorted([p for p in input_dir.rglob(pattern) if p.is_file()])
    return sorted([p for p in input_dir.glob(pattern) if p.is_file()])


def _stream_bin_npy_4d(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    preset: BinPreset,
    rows_per_chunk_groups: int = 8,
) -> tuple[int, ...]:
    """Stream 4D `.npy` in scan-row chunks to keep memory bounded."""
    if rows_per_chunk_groups < 1:
        raise ValueError("rows_per_chunk_groups must be >= 1")

    arr = np.load(input_path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError("Streaming 4D path expects a 4D array")

    scan_r, scan_c, det_r, det_c = (int(v) for v in arr.shape)
    out_scan_r, used_scan_r = _binned_axis_shape(
        scan_r, int(preset.scan_bin_row), preset.edge_mode, axis=0
    )
    out_scan_c, _ = _binned_axis_shape(scan_c, int(preset.scan_bin_col), preset.edge_mode, axis=1)
    out_det_r, _ = _binned_axis_shape(det_r, int(preset.det_bin_row), preset.edge_mode, axis=2)
    out_det_c, _ = _binned_axis_shape(det_c, int(preset.det_bin_col), preset.edge_mode, axis=3)

    out_shape = (out_scan_r, out_scan_c, out_det_r, out_det_c)
    out_mm = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=out_shape)

    row_factor = int(preset.scan_bin_row)
    rows_per_chunk = max(row_factor, row_factor * int(rows_per_chunk_groups))
    if rows_per_chunk % row_factor != 0:
        rows_per_chunk = ((rows_per_chunk + row_factor - 1) // row_factor) * row_factor

    # Use an estimated chunk size for robust device selection (important for MPS limits).
    estimated_chunk_rows = min(rows_per_chunk, max(1, scan_r))
    estimated_numel = int(estimated_chunk_rows * scan_c * det_r * det_c)
    device_str = _resolve_torch_device(preset, numel=estimated_numel)

    out_r0 = 0
    for r0 in range(0, used_scan_r, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, used_scan_r)
        chunk = np.asarray(arr[r0:r1], dtype=np.float32)
        if chunk.size == 0:
            continue
        binned = _bin_4d_torch(chunk, preset, device_str)
        out_r1 = out_r0 + int(binned.shape[0])
        out_mm[out_r0:out_r1] = binned
        out_r0 = out_r1

    del out_mm
    return out_shape


def _stream_bin_npy_5d(input_path: pathlib.Path, output_path: pathlib.Path, preset: BinPreset) -> tuple[int, ...]:
    """Stream 5D `.npy` along time axis 0 to keep memory bounded."""
    if preset.time_axis != 0:
        raise ValueError("Streaming 5D `.npy` currently requires time_axis=0")

    arr = np.load(input_path, mmap_mode="r")
    if arr.ndim != 5:
        raise ValueError("Streaming path expects 5D array")

    device_str = _resolve_torch_device(preset, numel=int(arr[0].size))
    first = _bin_4d_torch(np.asarray(arr[0]), preset, device_str)
    out_shape = (int(arr.shape[0]),) + first.shape
    out_mm = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=out_shape)
    out_mm[0] = first

    for t in range(1, int(arr.shape[0])):
        frame = np.asarray(arr[t])
        out_mm[t] = _bin_4d_torch(frame, preset, device_str)

    del out_mm
    return out_shape


def _find_best_h5_dataset(h5f) -> tuple[str, object | None]:
    """Auto-discover the best dataset in an HDF5 file for 4D-STEM binning."""
    candidates: list[tuple[int, str, object]] = []

    def _walk(group, prefix: str = ""):
        for key, item in group.items():
            item_path = f"{prefix}/{key}" if prefix else key
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                try:
                    ndim = int(item.ndim)
                    size = int(item.size)
                    dtype_kind = str(item.dtype.kind)
                except Exception:
                    continue

                if size <= 0 or ndim < 2 or dtype_kind not in {"i", "u", "f", "c"}:
                    continue

                score = size
                # 4D-STEM binning prefers 4D > 5D > 3D
                if ndim == 4:
                    score += 10**15
                elif ndim == 5:
                    score += 9 * 10**14
                elif ndim == 3:
                    score += 8 * 10**14

                lower_path = item_path.lower()
                for token in ["data", "frames", "diffraction", "stem", "signal"]:
                    if token in lower_path:
                        score += 10**12
                for token in ["preview", "thumb", "mask", "meta", "calib"]:
                    if token in lower_path:
                        score -= 10**12

                candidates.append((score, item_path, item))
            elif hasattr(item, "items"):
                _walk(item, item_path)

    _walk(h5f)
    if not candidates:
        return "", None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, ds_path, ds = candidates[0]
    return ds_path, ds


def _stream_bin_h5_4d(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    preset: BinPreset,
    dataset_path: str,
    rows_per_chunk_groups: int = 8,
) -> tuple[int, ...]:
    """Stream 4D HDF5 dataset in scan-row chunks to keep memory bounded."""
    if rows_per_chunk_groups < 1:
        raise ValueError("rows_per_chunk_groups must be >= 1")

    h5f = h5py.File(input_path, "r")
    ds = h5f[dataset_path]
    if ds.ndim != 4:
        h5f.close()
        raise ValueError("Streaming HDF5 4D path expects a 4D dataset")

    scan_r, scan_c, det_r, det_c = (int(v) for v in ds.shape)
    out_scan_r, used_scan_r = _binned_axis_shape(
        scan_r, int(preset.scan_bin_row), preset.edge_mode, axis=0
    )
    out_scan_c, _ = _binned_axis_shape(scan_c, int(preset.scan_bin_col), preset.edge_mode, axis=1)
    out_det_r, _ = _binned_axis_shape(det_r, int(preset.det_bin_row), preset.edge_mode, axis=2)
    out_det_c, _ = _binned_axis_shape(det_c, int(preset.det_bin_col), preset.edge_mode, axis=3)

    out_shape = (out_scan_r, out_scan_c, out_det_r, out_det_c)
    out_mm = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=out_shape)

    row_factor = int(preset.scan_bin_row)
    rows_per_chunk = max(row_factor, row_factor * int(rows_per_chunk_groups))
    if rows_per_chunk % row_factor != 0:
        rows_per_chunk = ((rows_per_chunk + row_factor - 1) // row_factor) * row_factor

    estimated_chunk_rows = min(rows_per_chunk, max(1, scan_r))
    estimated_numel = int(estimated_chunk_rows * scan_c * det_r * det_c)
    device_str = _resolve_torch_device(preset, numel=estimated_numel)

    out_r0 = 0
    for r0 in range(0, used_scan_r, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, used_scan_r)
        chunk = np.asarray(ds[r0:r1], dtype=np.float32)
        if chunk.size == 0:
            continue
        binned = _bin_4d_torch(chunk, preset, device_str)
        out_r1 = out_r0 + int(binned.shape[0])
        out_mm[out_r0:out_r1] = binned
        out_r0 = out_r1

    del out_mm
    h5f.close()
    return out_shape


def process_file(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    preset: BinPreset,
    stream_npy_threshold_gb: float | None = 2.0,
    stream_rows_per_chunk_groups: int = 8,
) -> dict[str, Any]:
    """Process one file with the given preset."""
    preset.validate()
    input_suffix = input_path.suffix.lower()
    resolved_device = "unknown"
    streaming_mode = "none"
    input_size_gb = 0.0

    started = time.perf_counter()

    if input_suffix == ".npy":
        arr = np.load(input_path, mmap_mode="r")
        in_shape = tuple(int(v) for v in arr.shape)
        input_size_gb = float(arr.nbytes) / 1_000_000_000.0
        resolved_device = _resolve_torch_device(preset, numel=int(np.prod(in_shape)))
        if arr.ndim == 5 and preset.time_axis == 0:
            out_shape = _stream_bin_npy_5d(input_path, output_path, preset)
            streaming_mode = "time_axis_5d"
        elif (
            arr.ndim == 4
            and stream_npy_threshold_gb is not None
            and input_size_gb >= float(stream_npy_threshold_gb)
        ):
            out_shape = _stream_bin_npy_4d(
                input_path=input_path,
                output_path=output_path,
                preset=preset,
                rows_per_chunk_groups=int(stream_rows_per_chunk_groups),
            )
            streaming_mode = "scan_rows_4d"
        else:
            out = apply_preset_to_array(np.asarray(arr), preset)
            np.save(output_path, out)
            out_shape = tuple(int(v) for v in out.shape)
            streaming_mode = "none"

    elif input_suffix == ".npz":
        input_size_gb = float(input_path.stat().st_size) / 1_000_000_000.0
        with np.load(input_path, allow_pickle=False) as npz:
            key = preset.npz_key if preset.npz_key in npz.files else (npz.files[0] if npz.files else None)
            if key is None:
                raise ValueError("NPZ archive contains no arrays")
            arr = np.asarray(npz[key])
        in_shape = tuple(int(v) for v in arr.shape)
        resolved_device = _resolve_torch_device(preset, numel=int(np.prod(in_shape)))
        out = apply_preset_to_array(arr, preset)
        np.savez_compressed(output_path, data=out)
        out_shape = tuple(int(v) for v in out.shape)

    elif input_suffix in {".h5", ".hdf5", ".emd"}:
        if not _HAS_H5PY:
            raise RuntimeError("h5py is required for HDF5/EMD files. Install it: pip install h5py")

        with h5py.File(input_path, "r") as h5f:
            if preset.h5_dataset_path is not None:
                ds_path = preset.h5_dataset_path.lstrip("/")
                if ds_path not in h5f:
                    ds_path = "/" + ds_path
                ds = h5f[ds_path]
            else:
                ds_path, ds = _find_best_h5_dataset(h5f)
                if ds is None:
                    raise ValueError(f"No suitable dataset found in {input_path}")

            in_shape = tuple(int(v) for v in ds.shape)
            input_size_gb = float(ds.nbytes) / 1_000_000_000.0

            # Always output as .npy
            if output_path.suffix.lower() != ".npy":
                output_path = output_path.with_suffix(".npy")

            resolved_device = _resolve_torch_device(preset, numel=int(np.prod(in_shape)))

            if (
                ds.ndim == 4
                and stream_npy_threshold_gb is not None
                and input_size_gb >= float(stream_npy_threshold_gb)
            ):
                # Stream large 4D HDF5 in chunks
                pass  # handled below outside context manager
            else:
                arr = np.asarray(ds, dtype=np.float32)
                out = apply_preset_to_array(arr, preset)
                np.save(output_path, out)
                out_shape = tuple(int(v) for v in out.shape)
                streaming_mode = "none"

        # Handle streamed 4D outside the with-block (stream function opens its own handle)
        if (
            len(in_shape) == 4
            and stream_npy_threshold_gb is not None
            and input_size_gb >= float(stream_npy_threshold_gb)
        ):
            out_shape = _stream_bin_h5_4d(
                input_path=input_path,
                output_path=output_path,
                preset=preset,
                dataset_path=ds_path,
                rows_per_chunk_groups=int(stream_rows_per_chunk_groups),
            )
            streaming_mode = "scan_rows_4d_h5"

    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}. Supported: .npy, .npz, .h5, .hdf5, .emd")

    elapsed_sec = float(time.perf_counter() - started)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_shape": list(in_shape),
        "output_shape": list(out_shape),
        "input_size_gb": float(input_size_gb),
        "estimated_output_gb": float(np.prod(out_shape) * 4.0 / 1_000_000_000.0),
        "elapsed_sec": elapsed_sec,
        "backend": preset.backend,
        "device": resolved_device,
        "streaming_mode": streaming_mode,
        "status": "ok",
    }


def run_batch(
    input_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    preset: BinPreset,
    pattern: str = "*.npy",
    recursive: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    name_suffix: str = "_binned",
    manifest_name: str = "bin_batch_manifest.jsonl",
    stream_npy_threshold_gb: float | None = 2.0,
    stream_rows_per_chunk_groups: int = 8,
    max_retries: int = 1,
    fail_fast: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run preset-based binning over a folder one file at a time."""
    preset.validate()
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    in_root = pathlib.Path(input_dir).expanduser().resolve()
    out_root = pathlib.Path(output_dir).expanduser().resolve()

    if not in_root.exists() or not in_root.is_dir():
        raise ValueError(f"Input directory not found: {in_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    files = _iter_inputs(in_root, pattern=pattern, recursive=recursive)
    results: list[dict[str, Any]] = []
    batch_start = time.perf_counter()
    total = len(files)

    manifest_path = out_root / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w"):
        pass

    for idx, src in enumerate(files, start=1):
        rel = src.relative_to(in_root)

        if src.suffix.lower() == ".npz":
            dst = (out_root / rel).with_name(f"{src.stem}{name_suffix}.npz")
        elif src.suffix.lower() in {".h5", ".hdf5", ".emd"}:
            dst = (out_root / rel).with_name(f"{src.stem}{name_suffix}.npy")
        else:
            dst = (out_root / rel).with_name(f"{src.stem}{name_suffix}.npy")

        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            row = {
                "input": str(src),
                "output": str(dst),
                "status": "skipped",
                "reason": "exists",
            }
        elif dry_run:
            row = {
                "input": str(src),
                "output": str(dst),
                "status": "dry_run",
            }
        else:
            row = {}
            last_error = ""
            for attempt in range(1, int(max_retries) + 2):
                try:
                    row = process_file(
                        src,
                        dst,
                        preset,
                        stream_npy_threshold_gb=stream_npy_threshold_gb,
                        stream_rows_per_chunk_groups=stream_rows_per_chunk_groups,
                    )
                    row["attempt"] = int(attempt)
                    row["retries_used"] = int(attempt - 1)
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt >= int(max_retries) + 1:
                        row = {
                            "input": str(src),
                            "output": str(dst),
                            "status": "error",
                            "error": last_error,
                            "attempt": int(attempt),
                            "retries_used": int(attempt - 1),
                        }
                    continue

        elapsed_batch = float(time.perf_counter() - batch_start)
        throughput = float(idx / elapsed_batch) if elapsed_batch > 0 else 0.0
        remaining = max(0, total - idx)
        eta_sec = float(remaining / throughput) if throughput > 0 else None

        row["job_index"] = int(idx)
        row["job_total"] = int(total)
        row["progress_pct"] = float((idx / total) * 100.0) if total > 0 else 100.0
        row["batch_elapsed_sec"] = elapsed_batch
        row["throughput_files_per_sec"] = throughput
        row["eta_sec"] = eta_sec
        row["eta_hms"] = _seconds_to_hms(eta_sec)
        row["max_retries"] = int(max_retries)
        row["fail_fast"] = bool(fail_fast)

        device_str = str(row.get("device", preset.device or "unknown"))
        row["runtime"] = _runtime_memory_snapshot(device_str)

        results.append(row)
        with manifest_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if progress_callback is not None:
            progress_callback(row)

        if bool(fail_fast) and str(row.get("status")) == "error":
            break

    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch bin 4D-STEM files over folders using a JSON preset."
    )
    parser.add_argument("--input-dir", required=True, help="Input folder")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--preset", required=True, help="Preset JSON path")
    parser.add_argument(
        "--pattern",
        default="*.npy",
        help="Glob pattern (e.g. '*.npy' or '*.npz')",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without writing")
    parser.add_argument(
        "--name-suffix",
        default="_binned",
        help="Suffix added to output filenames",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override preset torch device (e.g. cpu, cuda, mps)",
    )
    parser.add_argument(
        "--stream-npy-threshold-gb",
        type=float,
        default=2.0,
        help="Stream 4D .npy files when input size is >= this many GB (set negative to disable)",
    )
    parser.add_argument(
        "--stream-rows-per-chunk-groups",
        type=int,
        default=8,
        help="For streamed 4D .npy, chunk size in multiples of scan_bin_row",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retry failed files this many times before marking as error",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the queue immediately on first error",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable per-file progress output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    preset = load_preset(args.preset)
    if args.device is not None:
        preset = replace(
            preset,
            device=str(args.device),
        )
        preset.validate()

    def _print_progress(row: dict[str, Any]) -> None:
        if bool(args.quiet_progress):
            return
        status = str(row.get("status", "unknown"))
        idx = int(row.get("job_index", 0))
        total = int(row.get("job_total", 0))
        elapsed = float(row.get("elapsed_sec", 0.0))
        eta_hms = str(row.get("eta_hms", "unknown"))
        device = str(row.get("device", "unknown"))
        stream = str(row.get("streaming_mode", "none"))
        retries = int(row.get("retries_used", 0))
        src_name = pathlib.Path(str(row.get("input", ""))).name
        print(
            f"[{idx}/{total}] {status:<7} {src_name} "
            f"t={elapsed:.2f}s eta={eta_hms} device={device} stream={stream} retries={retries}"
        )

    batch_start = time.perf_counter()
    results = run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preset=preset,
        pattern=args.pattern,
        recursive=bool(args.recursive),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        name_suffix=str(args.name_suffix),
        stream_npy_threshold_gb=(
            None if float(args.stream_npy_threshold_gb) < 0 else float(args.stream_npy_threshold_gb)
        ),
        stream_rows_per_chunk_groups=int(args.stream_rows_per_chunk_groups),
        max_retries=int(args.max_retries),
        fail_fast=bool(args.fail_fast),
        progress_callback=_print_progress,
    )

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_err = sum(1 for r in results if r.get("status") == "error")
    n_skip = sum(1 for r in results if r.get("status") in {"skipped", "dry_run"})
    total_elapsed = float(time.perf_counter() - batch_start)

    print(
        "Processed: "
        f"ok={n_ok}, error={n_err}, skipped={n_skip}, total={len(results)}, "
        f"elapsed={total_elapsed:.2f}s"
    )

    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
