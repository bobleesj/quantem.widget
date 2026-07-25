from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
from typing import Any, Literal

import numpy as np


ShowPtychoWebGPUSource = Literal["compressed_hdf5", "bf_columns", "auto"]


def _normalize_webgpu_source(value: str) -> ShowPtychoWebGPUSource:
    """Normalize the ShowPtycho browser source policy."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"hdf5", "compressed", "compressed_hdf5"}:
        return "compressed_hdf5"
    if normalized in {"bf", "bf_column", "bf_columns"}:
        return "bf_columns"
    if normalized == "auto":
        return "auto"
    raise ValueError(
        "webgpu_source must be 'compressed_hdf5', 'bf_columns', or 'auto'; "
        f"got {value!r}"
    )


def _to_numpy(array: object) -> np.ndarray:
    if hasattr(array, "get"):
        array = array.get()
    elif type(array).__module__.split(".", 1)[0] == "torch":
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def _showptycho_fft_mag(array: np.ndarray) -> np.ndarray:
    """Match the ShowPtycho frontend FFT display path."""

    phase = array.astype(np.float32, copy=False)
    centered = (phase - np.float32(phase.mean())).astype(np.float32)
    h, w = centered.shape
    window = (
        np.hanning(h).astype(np.float32)[:, None]
        * np.hanning(w).astype(np.float32)[None, :]
    )
    padded_h = 1 << int(math.ceil(math.log2(h)))
    padded_w = 1 << int(math.ceil(math.log2(w)))
    padded = np.zeros((padded_h, padded_w), dtype=np.complex64)
    padded[:h, :w] = (centered * window).astype(np.complex64)
    fft = np.fft.fft2(padded).astype(np.complex64)
    mag = np.abs(fft[:h, :w]).astype(np.float32)
    return np.log1p(np.fft.fftshift(mag)).astype(np.float32)


def _write_empty_saves_manifest(out_path: pathlib.Path) -> None:
    """Seed optional standalone folder saves so the UI has no startup 404."""

    saves_dir = out_path / "saves"
    saves_dir.mkdir(parents=True, exist_ok=True)
    saves_json = saves_dir / "saves.json"
    if not saves_json.exists():
        saves_json.write_text("[]\n", encoding="utf-8")


def _jsonable_float_list(array: object) -> list[float]:
    return _to_numpy(array).astype(np.float32, copy=False).tolist()


def _jsonable_int_list(array: object) -> list[int]:
    return _to_numpy(array).astype(np.int32, copy=False).tolist()


def _sidecar_calibration(widget: Any, accel: Any) -> dict[str, Any]:
    cache = accel._cache
    g_shape = getattr(accel, "g_shape", None)
    if g_shape is None and hasattr(accel, "G_qk"):
        g_shape = getattr(accel.G_qk, "shape", None)
    if g_shape is None:
        g_shape = (int(cache["num_bf"]), int(cache["ny"]), int(cache["nx"]))
    c10 = float(widget._current_c10())
    c12 = float(widget._current_c12())
    phi12_deg = float(widget._current_phi12_deg())
    phi12 = math.radians(phi12_deg)
    scan_region = getattr(widget, "_scan_region", None)
    if scan_region is None:
        ny = int(cache["ny"])
        nx = int(cache["nx"])
        scan_region_payload = {
            "row_start": 0,
            "row_stop": ny,
            "col_start": 0,
            "col_stop": nx,
            "shape": [ny, nx],
        }
    else:
        row_start, row_stop, col_start, col_stop = scan_region
        scan_region_payload = {
            "row_start": int(row_start),
            "row_stop": int(row_stop),
            "col_start": int(col_start),
            "col_stop": int(col_stop),
            "shape": [int(row_stop - row_start), int(col_stop - col_start)],
        }
    ssb = getattr(widget, "_ssb_ref", None)
    scan_sampling = getattr(ssb, "scan_sampling", None)
    if isinstance(scan_sampling, (tuple, list)):
        scan_sampling_A = float(scan_sampling[0])
    elif scan_sampling is not None:
        scan_sampling_A = float(scan_sampling)
    else:
        scan_sampling_A = float(getattr(widget, "pixel_size", 0.0) or 0.0)
    return {
        "schema_version": 1,
        "kind": "showptycho_webgpu_folder",
        "source_file": "redacted_local_source",
        "source_calibration": "redacted_local_calibration",
        "scan_region": scan_region_payload,
        "backend_reference": "ShowPtycho SSBEngine.reconstruct_with_loss",
        "bf_radius_px": getattr(widget, "_bf_radius_px", None),
        "num_bf": int(cache["num_bf"]),
        "g_shape": [int(g_shape[0]), int(g_shape[1]), int(g_shape[2])],
        "g_dtype": "complex64_interleaved_re_im_native_le",
        "phase_shape": [int(cache["ny"]), int(cache["nx"])],
        "phase_dtype": "float32_native_le",
        "detector_shape": list(getattr(accel, "gpts", (0, 0))),
        "bf_center": [float(accel.bf_center[0]), float(accel.bf_center[1])],
        "bf_rows": _jsonable_int_list(accel.bf_inds_row),
        "bf_cols": _jsonable_int_list(accel.bf_inds_col),
        "kx_bf": _jsonable_float_list(cache["kx_bf"]),
        "ky_bf": _jsonable_float_list(cache["ky_bf"]),
        "qx_1d": _jsonable_float_list(cache["qx_1d"]),
        "qy_1d": _jsonable_float_list(cache["qy_1d"]),
        "aperture_k": _jsonable_float_list(cache["aperture_k_1d"]),
        "alpha_k2": _jsonable_float_list(cache["alpha_k2_1d"]),
        "cos2phi_k": _jsonable_float_list(cache["cos2phi_k_1d"]),
        "sin2phi_k": _jsonable_float_list(cache["sin2phi_k_1d"]),
        "wavelength_A": float(accel.wavelength),
        "semiangle_mrad": float(getattr(ssb, "semiangle_mrad", 0.0) or 0.0),
        "semiangle_rad": float(cache["semiangle_rad"]),
        "scan_sampling_A": scan_sampling_A,
        "voltage_kV": float(getattr(ssb, "voltage_kV", 0.0) or 0.0),
        "det_sampling_mrad_px": list(getattr(ssb, "angular_sampling", (0.0, 0.0))),
        "sampling_A": [float(accel.sampling[0]), float(accel.sampling[1])],
        "angular_sampling_rad": [float(cache["ang_y_rad"]), float(cache["ang_x_rad"])],
        "rotation_angle_deg": float(widget.rotation_deg),
        "rotation_angle_rad": math.radians(float(widget.rotation_deg)),
        "aberrations": {
            "C10": c10,
            "C12": c12,
            "phi12": phi12,
            "phi12_deg": phi12_deg,
        },
        "flip_phase": bool(widget.flip_phase),
        "dc_value": [float(accel._dc_value_host.real), float(accel._dc_value_host.imag)],
    }


def _ensure_supported_webgpu_shape(accel: Any) -> tuple[int, int]:
    """Validate the specialized browser SSB kernels can open this crop."""

    cache = accel._cache
    ny = int(cache.get("ny", 0))
    nx = int(cache.get("nx", 0))
    supported = {128, 256, 512, 1024}
    if ny != nx or ny not in supported:
        raise NotImplementedError(
            "ShowPtycho WebGPU folder export supports square 128, 256, 512, or 1024 "
            f"crops; got {ny}x{nx}."
        )
    return ny, nx


def build_showptycho_webgpu_payload(
    widget: Any,
    *,
    max_bytes: int = 512 * 1024 * 1024,
) -> tuple[dict[str, Any] | None, bytes, str]:
    """Return the in-notebook WebGPU payload for supported ShowPtycho widgets.

    The browser SSB kernel implements specialized 128, 256, 512, and 1024
    C10/C12/phi12 paths. Keep the byte guard here so a full-size notebook does not
    silently sync multi-GB BF-indexed ``G(k)`` through a widget comm.
    """

    accel = widget._accel
    if not hasattr(accel, "G_qk") and hasattr(accel, "_sync_webgpu_export_state"):
        accel._sync_webgpu_export_state()
    if not hasattr(accel, "G_qk") or not hasattr(accel, "_cache"):
        return None, b"", "WebGPU preview requires a CUDA BF-indexed G_qk cache."
    try:
        ny, nx = _ensure_supported_webgpu_shape(widget._accel)
    except NotImplementedError as exc:
        return None, b"", str(exc)

    cache = accel._cache

    g_qk = _to_numpy(accel.G_qk).astype(np.complex64, copy=False)
    if g_qk.ndim != 3 or tuple(g_qk.shape[1:]) != (ny, nx):
        return None, b"", f"WebGPU preview expected G_qk[:,{ny},{nx}], got {g_qk.shape}."
    nbytes = int(g_qk.nbytes)
    if nbytes > int(max_bytes):
        return (
            None,
            b"",
            "WebGPU preview payload is too large for notebook sync: "
            f"{nbytes / 1e6:.1f} MB > {int(max_bytes) / 1e6:.1f} MB.",
        )

    cal = _sidecar_calibration(widget, accel)
    cal["notebook_payload_bytes"] = nbytes
    cal["notebook_preview"] = True
    return cal, g_qk.tobytes(), (
        f"WebGPU preview ready: {g_qk.shape[0]} BF pixels, "
        f"{nbytes / 1e6:.1f} MB payload."
    )


def _write_embedded_widget_html(
    widget: Any,
    html_path: pathlib.Path,
    *,
    title: str,
    calibration: dict[str, Any],
    g_bf_url: str = "",
    h5_source: dict[str, Any] | None = None,
) -> None:
    """Write the same ShowPtycho widget UI with folder-local WebGPU data."""

    from ipywidgets.embed import dependency_state, embed_minimal_html

    from .export import ensure_mobile_viewport

    state_keys = [
        "webgpu_preview_enabled",
        "webgpu_standalone",
        "webgpu_cal_json",
        "webgpu_g_bf_bytes",
        "webgpu_g_bf_url",
        "webgpu_h5_source_json",
        "webgpu_preview_status",
        "phase_bytes",
        "phase_width",
        "phase_height",
    ]
    old_state = {key: getattr(widget, key) for key in state_keys}
    try:
        widget.webgpu_preview_enabled = True
        widget.webgpu_standalone = True
        widget.webgpu_cal_json = json.dumps(calibration)
        widget.webgpu_g_bf_bytes = b""
        widget.webgpu_g_bf_url = g_bf_url
        widget.webgpu_h5_source_json = json.dumps(h5_source or {})
        if h5_source:
            widget.phase_bytes = b""
            widget.phase_width = 0
            widget.phase_height = 0
        if h5_source and h5_source.get("kind") == "bf_columns":
            widget.webgpu_preview_status = (
                "WebGPU folder ready: browser reads detector BF-column counts "
                "and builds BF reducers transiently."
            )
        elif h5_source:
            widget.webgpu_preview_status = (
                "WebGPU folder ready: browser reads compressed HDF5 source "
                "and builds BF reducers transiently."
            )
        else:
            widget.webgpu_preview_status = (
                "WebGPU folder ready: browser fetches explicit BF-G cache "
                "next to this HTML."
            )
        state = dependency_state([widget], drop_defaults=False)
        embed_minimal_html(
            str(html_path),
            views=[widget],
            title=title,
            drop_defaults=False,
            state=state,
        )
    finally:
        for key, value in old_state.items():
            setattr(widget, key, value)
    ensure_mobile_viewport(html_path)


def _link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> str:
    """Make ``dst`` refer to ``src`` without duplicating bytes when possible."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            dst.symlink_to(src)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def _find_hdf5_stack(handle: Any) -> Any:
    """Return the 3D detector stack dataset inside an HDF5 data file."""

    candidates = ("entry/data/data", "entry/data", "data")
    for path in candidates:
        try:
            dataset = handle[path]
        except KeyError:
            continue
        if getattr(dataset, "shape", None) and len(dataset.shape) == 3:
            return dataset
    for _, item in handle.items():
        if getattr(item, "shape", None) and len(item.shape) == 3:
            return item
        if hasattr(item, "visititems"):
            found: list[Any] = []

            def visitor(_name: str, obj: Any) -> None:
                if not found and getattr(obj, "shape", None) and len(obj.shape) == 3:
                    found.append(obj)

            item.visititems(visitor)
            if found:
                return found[0]
    raise ValueError("no 3D detector-stack dataset found in HDF5 data file")


def _hdf5_chunk_index_records(
    src: pathlib.Path,
) -> tuple[dict[str, Any], np.ndarray] | None:
    """Return compact raw-chunk offset metadata for browser range reads."""

    try:
        import h5py
    except ImportError:
        return None

    try:
        with h5py.File(src, "r") as handle:
            dataset = _find_hdf5_stack(handle)
            shape = tuple(int(v) for v in dataset.shape)
            if len(shape) != 3 or not getattr(dataset, "chunks", None):
                return None
            dtype_name = str(dataset.dtype)
            chunk_shape = [int(v) for v in dataset.chunks]
            n_frames = shape[0]
            records = np.zeros((n_frames, 2), dtype="<u8")

            def collect(info: Any) -> None:
                frame = int(info.chunk_offset[0])
                if 0 <= frame < n_frames:
                    records[frame, 0] = int(info.byte_offset)
                    records[frame, 1] = int(info.size)

            dataset.id.chunk_iter(collect)
    except Exception:
        return None
    if not np.all(records[:, 1] > 0):
        return None

    return {
        "frames": int(shape[0]),
        "detector_shape": [int(shape[1]), int(shape[2])],
        "dtype": dtype_name,
        "chunk_shape": chunk_shape,
        "record": "u64le_offset,u64le_size",
    }, records


def _write_hdf5_chunk_index(src: pathlib.Path, dst: pathlib.Path) -> dict[str, Any] | None:
    """Write a compact single-file raw-chunk index for browser range reads."""

    built = _hdf5_chunk_index_records(src)
    if built is None:
        return None
    index, records = built
    dst.parent.mkdir(parents=True, exist_ok=True)
    records.tofile(dst)
    index["path"] = dst.name
    index["bytes"] = int(dst.stat().st_size)
    return index


def _external_hdf5_source_files(master: pathlib.Path) -> list[pathlib.Path]:
    """Return HDF5 data files referenced by external links in a wrapper master."""

    try:
        import h5py
    except ImportError:
        return []

    files: list[pathlib.Path] = []
    try:
        with h5py.File(master, "r") as handle:
            group = handle.get("entry/data")
            if group is None:
                return []
            for name in group:
                link = group.get(name, getlink=True)
                if not isinstance(link, h5py.ExternalLink) or not link.filename:
                    continue
                source = pathlib.Path(link.filename)
                if not source.is_absolute():
                    source = master.parent / source
                files.append(source.expanduser().resolve())
    except OSError:
        return []

    out: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for file in files:
        if file in seen:
            continue
        if not file.exists():
            raise FileNotFoundError(
                f"HDF5 wrapper {master} points at missing data file {file}"
            )
        out.append(file)
        seen.add(file)
    return out


def _collect_hdf5_source_files(master: pathlib.Path) -> list[pathlib.Path]:
    """Return the master and compressed data files used by a native HDF5 scan."""

    master = master.expanduser().resolve()
    external_files = _external_hdf5_source_files(master)
    if external_files:
        return [master, *external_files]
    if not master.name.endswith("_master.h5"):
        raise ValueError(
            "ShowPtycho source must be a *_master.h5 file or an HDF5 wrapper "
            f"with external data links; got {master.name!r}"
        )
    base = master.name[: -len("_master.h5")]
    data_files = sorted(master.parent.glob(f"{base}_data_*.h5"))
    if not data_files:
        raise FileNotFoundError(
            f"no HDF5 data files found next to {master}: expected {base}_data_*.h5"
        )
    return [master, *data_files]


def _prepare_hdf5_source_folder(master: pathlib.Path, out_path: pathlib.Path) -> dict[str, Any]:
    """Expose compressed HDF5 source files inside the review folder."""

    source_dir = out_path / "source"
    files = _collect_hdf5_source_files(master)
    links = []
    chunk_indexes = []
    chunk_index_payloads: list[tuple[dict[str, Any], dict[str, Any], np.ndarray]] = []
    for src in files:
        rel = pathlib.Path("source") / src.name
        mode = _link_or_copy(src, out_path / rel)
        entry = {
            "path": rel.as_posix(),
            "name": src.name,
            "bytes": int(src.stat().st_size),
            "link": mode,
        }
        if src != files[0]:
            built = _hdf5_chunk_index_records(src)
            if built is not None:
                index, records = built
                chunk_index_payloads.append((entry, index, records))
        links.append(entry)
    if chunk_index_payloads and len(chunk_index_payloads) == max(0, len(files) - 1):
        index_rel = pathlib.Path("source") / "chunks.u64"
        index_path = out_path / index_rel
        index_path.parent.mkdir(parents=True, exist_ok=True)
        offset = 0
        with index_path.open("wb") as handle:
            for entry, index, records in chunk_index_payloads:
                payload = records.astype("<u8", copy=False).tobytes(order="C")
                handle.write(payload)
                index = dict(index)
                index["path"] = index_rel.as_posix()
                index["byte_offset"] = offset
                index["bytes"] = len(payload)
                entry["chunk_index"] = index["path"]
                entry["chunk_index_byte_offset"] = offset
                entry["chunk_index_bytes"] = len(payload)
                chunk_indexes.append(index)
                offset += len(payload)
    master_rel = pathlib.Path("source") / files[0].name
    return {
        "kind": "hdf5",
        "master": master_rel.as_posix(),
        "data_files": [entry["path"] for entry in links[1:]],
        "chunk_indexes": chunk_indexes,
        "link_mode": sorted({entry["link"] for entry in links}),
        "files": links,
        "note": (
            "Compressed HDF5 source files are served directly; no persistent "
            "float32 or complex64 BF reducer is stored in this folder."
        ),
    }


def _bf_column_batch_frames() -> int:
    """Return the scan-frame batch used to transpose HDF5 into BF columns."""

    raw = os.environ.get("QUANTEM_SHOWPTYCHO_BF_COLUMN_BATCH", "")
    try:
        frames = int(raw) if raw else 2048
    except ValueError:
        frames = 2048
    # Keep this even so uint4 packing never has to split a nibble pair.
    return max(2, frames + (frames % 2))


def _iter_hdf5_detector_stacks(files: list[pathlib.Path]):
    """Yield detector-stack datasets from data files, skipping the master file."""

    try:
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass
        import h5py
    except ImportError:
        return

    for file in files[1:]:
        handle = h5py.File(file, "r")
        try:
            yield file, handle, _find_hdf5_stack(handle)
        finally:
            handle.close()


def _hdf5_scan_total(files: list[pathlib.Path]) -> tuple[int, tuple[int, int] | None, np.dtype | None]:
    """Return total scan frames, detector shape, and dtype for HDF5 data files."""

    total = 0
    detector_shape: tuple[int, int] | None = None
    dtype: np.dtype | None = None
    for _file, _handle, dataset in _iter_hdf5_detector_stacks(files):
        shape = tuple(int(v) for v in dataset.shape)
        if len(shape) != 3:
            raise ValueError(f"expected a 3D detector stack, got shape {shape}")
        current_detector = (shape[1], shape[2])
        if detector_shape is None:
            detector_shape = current_detector
            dtype = np.dtype(dataset.dtype)
        elif detector_shape != current_detector:
            raise ValueError(
                "all HDF5 data files must share detector shape; "
                f"got {current_detector}, expected {detector_shape}"
            )
        total += shape[0]
    return total, detector_shape, dtype


def _scan_hdf5_bf_max(
    files: list[pathlib.Path],
    *,
    bf_rows: np.ndarray,
    bf_cols: np.ndarray,
) -> int:
    """Return the maximum selected BF count without materializing all columns."""

    max_value = 0
    batch_frames = _bf_column_batch_frames()
    for _file, _handle, dataset in _iter_hdf5_detector_stacks(files):
        n_frames = int(dataset.shape[0])
        for start in range(0, n_frames, batch_frames):
            stop = min(n_frames, start + batch_frames)
            block = np.asarray(dataset[start:stop])
            cols = block[:, bf_rows, bf_cols]
            if cols.size:
                max_value = max(max_value, int(np.max(cols)))
    return max_value


def _write_uint4_columns(
    out: np.memmap,
    values_bf_scan: np.ndarray,
    *,
    scan_start: int,
) -> None:
    """Pack two 4-bit detector counts per byte into detector-major columns."""

    if scan_start % 2:
        raise ValueError("uint4 BF-column writes require even scan_start")
    if values_bf_scan.shape[1] % 2:
        raise ValueError("uint4 BF-column writes require even scan batch length")
    vals = np.asarray(values_bf_scan, dtype=np.uint8)
    if vals.size and int(vals.max()) > 15:
        raise ValueError("uint4 BF-column companion can only store counts <= 15")
    packed = vals[:, 0::2] | (vals[:, 1::2] << np.uint8(4))
    byte_start = scan_start // 2
    out[:, byte_start:byte_start + packed.shape[1]] = packed


def _write_bf_column_companion(
    *,
    files: list[pathlib.Path],
    out_path: pathlib.Path,
    cal: dict[str, Any],
) -> dict[str, Any] | None:
    """Write exact raw BF evidence in detector-major scan columns.

    The companion is not a persistent Fourier/complex reducer.  It stores the
    original selected BF detector counts as ``[bf, scan]`` so the browser can
    fetch only the BF disk columns it needs, then run the same transient WebGPU
    FFT/reduction path as the scan-major HDF5 loader.
    """

    if os.environ.get("QUANTEM_SHOWPTYCHO_DISABLE_BF_COLUMNS"):
        return None
    try:
        bf_rows = np.asarray(cal["bf_rows"], dtype=np.intp)
        bf_cols = np.asarray(cal["bf_cols"], dtype=np.intp)
        num_bf = int(cal["num_bf"])
        plane_shape = tuple(int(v) for v in cal["phase_shape"])
        plane = int(plane_shape[0] * plane_shape[1])
        total_frames, detector_shape, dtype = _hdf5_scan_total(files)
        if total_frames != plane:
            return None
        if detector_shape is None or dtype is None:
            return None
        if bf_rows.shape[0] != num_bf or bf_cols.shape[0] != num_bf:
            return None
        if np.any(bf_rows < 0) or np.any(bf_cols < 0):
            return None
        if np.any(bf_rows >= detector_shape[0]) or np.any(bf_cols >= detector_shape[1]):
            return None
        if not np.issubdtype(dtype, np.integer):
            return None
        max_value = _scan_hdf5_bf_max(files, bf_rows=bf_rows, bf_cols=bf_cols)
        data_frame_counts = [
            int(dataset.shape[0])
            for _file, _handle, dataset in _iter_hdf5_detector_stacks(files)
        ]
        if max_value <= 15:
            if plane % 2 == 0 and all(count % 2 == 0 for count in data_frame_counts):
                encoding = "uint4"
                suffix = "u4"
                itemsize = 0.5
                bytes_per_bf = (plane + 1) // 2
                mmap_dtype = np.uint8
                shape = (num_bf, bytes_per_bf)
            else:
                # Odd split files would make packed-nibble writes cross file
                # boundaries.  Keep exact detector counts and use uint8 instead
                # of writing a hard-to-audit partial nibble stream.
                encoding = "uint8"
                suffix = "u8"
                itemsize = 1
                bytes_per_bf = plane
                mmap_dtype = np.uint8
                shape = (num_bf, plane)
        elif max_value <= 255:
            encoding = "uint8"
            suffix = "u8"
            itemsize = 1
            bytes_per_bf = plane
            mmap_dtype = np.uint8
            shape = (num_bf, plane)
        elif max_value <= 65535:
            encoding = "uint16"
            suffix = "u16"
            itemsize = 2
            bytes_per_bf = plane * 2
            mmap_dtype = np.dtype("<u2")
            shape = (num_bf, plane)
        else:
            return None

        rel = pathlib.Path("source") / f"bf_columns.{suffix}"
        dst = out_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        out = np.memmap(dst, dtype=mmap_dtype, mode="w+", shape=shape)
        scan_offset = 0
        batch_frames = _bf_column_batch_frames()
        for _file, _handle, dataset in _iter_hdf5_detector_stacks(files):
            n_frames = int(dataset.shape[0])
            for start in range(0, n_frames, batch_frames):
                stop = min(n_frames, start + batch_frames)
                block = np.asarray(dataset[start:stop])
                cols = block[:, bf_rows, bf_cols].T
                global_start = scan_offset + start
                if encoding == "uint4":
                    _write_uint4_columns(out, cols, scan_start=global_start)
                else:
                    out[:, global_start:global_start + cols.shape[1]] = cols.astype(mmap_dtype, copy=False)
            scan_offset += n_frames
        out.flush()
        del out
        return {
            "kind": "bf_columns",
            "path": rel.as_posix(),
            "encoding": encoding,
            "dtype": encoding,
            "order": "bf,scan",
            "shape": [num_bf, plane],
            "scan_shape": [int(plane_shape[0]), int(plane_shape[1])],
            "detector_shape": [int(detector_shape[0]), int(detector_shape[1])],
            "bits_per_value": 4 if encoding == "uint4" else int(itemsize * 8),
            "bytes_per_bf": int(bytes_per_bf),
            "max_value": int(max_value),
            "bytes": int(dst.stat().st_size),
            "note": (
                "Detector-major raw BF evidence companion. This is exact count data, "
                "not a persistent float32/complex64 BF-G cache."
            ),
        }
    except Exception:
        return None


def export_showptycho_webgpu_folder(
    widget: Any,
    out_dir: str | pathlib.Path,
    *,
    title: str | None = None,
    overwrite: bool = True,
    source_master: str | pathlib.Path | None = None,
    decode_dtype: str = "uint16",
    webgpu_source: ShowPtychoWebGPUSource | str = "compressed_hdf5",
) -> pathlib.Path:
    """Export a ShowPtycho WebGPU folder backed by compressed HDF5 source files."""

    if decode_dtype not in {"uint8", "uint16", "float32"}:
        raise ValueError(
            "decode_dtype must be 'uint8', 'uint16', or 'float32'; "
            f"got {decode_dtype!r}"
        )
    source_policy = _normalize_webgpu_source(webgpu_source)
    accel = widget._accel
    if not hasattr(accel, "_cache"):
        raise NotImplementedError("ShowPtycho WebGPU folder export requires SSB calibration state.")
    _ensure_supported_webgpu_shape(accel)

    out_path = pathlib.Path(out_dir).expanduser()
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists")
    out_path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale in (
            "g_bf.c64",
            "ref_phase.f32",
            "ref_fft.f32",
            "ref_phase_variance.f32",
            "ref_amplitude.f32",
            "ref_phase.npy",
            "ref_fft.npy",
            "ref_products.npz",
            "PARITY.md",
        ):
            (out_path / stale).unlink(missing_ok=True)
        source_dir = out_path / "source"
        if source_dir.exists() or source_dir.is_symlink():
            if source_dir.is_dir() and not source_dir.is_symlink():
                shutil.rmtree(source_dir)
            else:
                source_dir.unlink()

    raw_source = source_master if source_master is not None else getattr(widget, "_source_file", None)
    if not raw_source:
        raise ValueError(
            "ShowPtycho compressed-source export needs the original *_master.h5 path. "
            "Construct the widget with source_file=... or pass source_master=..."
        )
    master = pathlib.Path(raw_source).expanduser()
    master = master.resolve()
    accel.cache_rotation(math.radians(float(widget.rotation_deg)))
    cal = _sidecar_calibration(widget, accel)
    source = _prepare_hdf5_source_folder(master, out_path)
    source["decode_dtype"] = decode_dtype
    files = _collect_hdf5_source_files(master)
    bf_columns = (
        _write_bf_column_companion(files=files, out_path=out_path, cal=cal)
        if source_policy in {"bf_columns", "auto"}
        else None
    )
    if source_policy == "bf_columns" and bf_columns is None:
        raise ValueError(
            "webgpu_source='bf_columns' was requested, but the exact BF-column "
            "companion could not be written from this HDF5 source. Use "
            "webgpu_source='compressed_hdf5' for the default WebGPU decompression path."
        )
    if bf_columns is not None:
        source["bf_columns"] = bf_columns
        source["preferred_browser_source"] = "bf_columns"
    cal["source_file"] = pathlib.Path(source["master"]).name
    cal["source_transport"] = "bf_columns" if bf_columns is not None else "compressed_hdf5"
    cal["webgpu_source_policy"] = source_policy
    cal["source_files"] = source["data_files"]
    cal["source_decode_dtype"] = decode_dtype
    cal["bf_column_companion"] = bf_columns is not None
    cal["persistent_bf_cache"] = False

    (out_path / "cal.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "format": "quantem.showptycho.webgpu.folder.v2",
        "title": title or "ShowPtycho",
        "index": "index.html",
        "calibration": "cal.json",
        "source": source,
        "arrays": {},
        "persistent_arrays": [],
        "non_goals": [
            "no persistent BF-G cache",
            "no reference float32 image payloads",
            "no detector binning",
        ],
    }
    (out_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    _write_empty_saves_manifest(out_path)
    browser_source = {
        "kind": "hdf5",
        "masterUrl": source["master"],
        "dataUrls": source["data_files"],
        "chunkIndexes": source.get("chunk_indexes", []),
        "decodeDtype": decode_dtype,
    }
    if bf_columns is not None:
        browser_source = {
            "kind": "bf_columns",
            "url": bf_columns["path"],
            "dtype": bf_columns["dtype"],
            "encoding": bf_columns["encoding"],
            "numBf": bf_columns["shape"][0],
            "plane": bf_columns["shape"][1],
            "scanShape": bf_columns["scan_shape"],
            "bytesPerBf": bf_columns["bytes_per_bf"],
            "bitsPerValue": bf_columns["bits_per_value"],
        }
    _write_embedded_widget_html(
        widget,
        out_path / "index.html",
        title=str(manifest["title"]),
        calibration=cal,
        h5_source=browser_source,
    )
    if bf_columns is not None:
        source_note = (
            "The browser reads the detector-major BF-column count companion under "
            "`source/` and builds BF reducers transiently in GPU memory. The original "
            "compressed HDF5 files are also preserved as paper truth and fallback input."
        )
    else:
        source_note = (
            "The browser reads the original compressed HDF5 master/data files under "
            "`source/`, decompresses the selected BF evidence with WebGPU, and builds "
            "BF reducers transiently in GPU memory. No BF-column companion is written "
            "by default; use `webgpu_source='bf_columns'` or "
            "`quantem ptycho --webgpu-source bf-columns` only for an explicit fallback "
            "or comparison export."
        )
    (out_path / "README.md").write_text(
        "# ShowPtycho WebGPU Folder\n\n"
        "Two ways to open this review - no install needed for the first:\n\n"
        "1. **Double-click** `index.html` in Chrome or Edge, click **Open data folder**, "
        "and select this folder. No server, no Python, no terminal. "
        "(If it opens in Safari, drag `index.html` onto Chrome instead.)\n"
        "2. **CLI**: `quantem ptycho <this folder>` serves it and opens the browser "
        "automatically (needs the `quantem-widget` package; `quantem showptycho` "
        "also works as a compatibility alias). Any other Range-capable "
        "static server works too.\n\n"
        f"{source_note} The folder "
        "intentionally does not store `g_bf.c64`, reference `.f32` images, or detector-binned data.\n",
        encoding="utf-8",
    )
    # Double-click launcher (see quantem.widget.command_launcher).
    from quantem.widget.command_launcher import write_command_launcher

    write_command_launcher(out_path, "ShowPtycho")
    return out_path


def export_showptycho_webgpu_sidecar(
    widget: Any,
    out_dir: str | pathlib.Path,
    *,
    title: str | None = None,
    overwrite: bool = True,
) -> pathlib.Path:
    """Export a ShowPtycho WebGPU folder from a live widget instance."""

    accel = widget._accel
    if not hasattr(accel, "G_qk") and hasattr(accel, "_sync_webgpu_export_state"):
        accel._sync_webgpu_export_state()
    if not hasattr(accel, "G_qk") or not hasattr(accel, "_cache"):
        raise NotImplementedError(
            "ShowPtycho WebGPU folder export currently requires the CUDA "
            "SSB accelerator with BF-indexed G_qk."
        )
    ny, nx = _ensure_supported_webgpu_shape(accel)

    out_path = pathlib.Path(out_dir)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists")
    out_path.mkdir(parents=True, exist_ok=True)

    accel.cache_rotation(math.radians(float(widget.rotation_deg)))
    if not hasattr(accel, "G_qk") and hasattr(accel, "_sync_webgpu_export_state"):
        accel._sync_webgpu_export_state()
    c10 = float(widget._current_c10())
    c12 = float(widget._current_c12())
    phi12_deg = float(widget._current_phi12_deg())
    phi12 = math.radians(phi12_deg)
    if hasattr(widget, "_higher_order_arrays"):
        mags_m, angles_rad, any_ho = widget._higher_order_arrays(c10, c12, phi12_deg)
    else:
        mags_m = angles_rad = None
        any_ho = False
    if any_ho and hasattr(accel, "reconstruct_full_with_loss"):
        phase_gpu, loss = accel.reconstruct_full_with_loss(mags_m, angles_rad)
    else:
        phase_gpu, loss = accel.reconstruct_with_loss(c10, c12, phi12)
    phase = _to_numpy(phase_gpu).astype(np.float32, copy=False)
    if bool(widget.flip_phase):
        phase = -phase
    fft_mag = _showptycho_fft_mag(phase)
    mean_phase = getattr(accel, "_mean_phase_buffer", None)
    sumsq = getattr(accel, "_sumsq_buffer", None)
    if mean_phase is None or sumsq is None:
        variance = np.zeros_like(phase, dtype=np.float32)
    else:
        variance = _to_numpy(
            sumsq / float(accel._cache["num_bf"]) - mean_phase ** 2
        ).astype(np.float32, copy=False)
    try:
        amplitude = _to_numpy(accel.reconstruct_object(c10, c12, phi12))
        amplitude = np.abs(amplitude).astype(np.float32, copy=False)
    except Exception:
        amplitude = np.zeros_like(phase, dtype=np.float32)

    g_qk = _to_numpy(accel.G_qk).astype(np.complex64, copy=False)
    if g_qk.ndim != 3 or tuple(g_qk.shape[1:]) != (ny, nx):
        raise ValueError(f"Expected G_qk[:,{ny},{nx}], got {g_qk.shape}.")
    g_qk.tofile(out_path / "g_bf.c64")
    phase.tofile(out_path / "ref_phase.f32")
    fft_mag.tofile(out_path / "ref_fft.f32")
    variance.tofile(out_path / "ref_phase_variance.f32")
    amplitude.tofile(out_path / "ref_amplitude.f32")
    np.save(out_path / "ref_phase.npy", phase)
    np.save(out_path / "ref_fft.npy", fft_mag)
    np.savez_compressed(
        out_path / "ref_products.npz",
        phase=phase,
        fft=fft_mag,
        phase_variance=variance,
        amplitude=amplitude,
    )

    cal = _sidecar_calibration(widget, accel)
    cal["loss"] = float(loss)
    (out_path / "cal.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "format": "quantem.showptycho.webgpu.folder.v1",
        "title": title or "ShowPtycho",
        "index": "index.html",
        "calibration": "cal.json",
        "arrays": {
            "g_bf": {
                "path": "g_bf.c64",
                "shape": list(g_qk.shape),
                "dtype": "complex64",
                "layout": "bf,row,col interleaved float32 re,im",
            },
            "ref_phase": {"path": "ref_phase.f32", "shape": list(phase.shape), "dtype": "float32"},
            "ref_fft": {"path": "ref_fft.f32", "shape": list(fft_mag.shape), "dtype": "float32"},
            "ref_phase_variance": {
                "path": "ref_phase_variance.f32",
                "shape": list(variance.shape),
                "dtype": "float32",
            },
            "ref_amplitude": {
                "path": "ref_amplitude.f32",
                "shape": list(amplitude.shape),
                "dtype": "float32",
            },
        },
        "non_goals": ["no raw diffraction patterns", "no full detector G", "no silent binning"],
    }
    (out_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    _write_empty_saves_manifest(out_path)
    _write_embedded_widget_html(
        widget,
        out_path / "index.html",
        title=str(manifest["title"]),
        calibration=cal,
        g_bf_url="g_bf.c64",
    )
    (out_path / "PARITY.md").write_text(
        "# ShowPtycho WebGPU Folder\n\n"
        "Open `index.html` over HTTP. It renders the same ShowPtycho widget UI "
        "and fetches the folder-local BF-G payload from `g_bf.c64`.\n",
        encoding="utf-8",
    )
    # Double-click launcher: users can open the viewer without a terminal or a
    # File System Access grant (see quantem.widget.command_launcher).
    from quantem.widget.command_launcher import write_command_launcher

    write_command_launcher(out_path, "ShowPtycho")
    return out_path
