from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
from typing import Any

import numpy as np


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


def _write_empty_snapshots_manifest(out_path: pathlib.Path) -> None:
    """Seed optional standalone folder snapshots so the UI has no startup 404."""

    snapshots_dir = out_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshots_json = snapshots_dir / "snapshots.json"
    if not snapshots_json.exists():
        snapshots_json.write_text("[]\n", encoding="utf-8")


def _jsonable_float_list(array: object) -> list[float]:
    return _to_numpy(array).astype(np.float32, copy=False).tolist()


def _jsonable_int_list(array: object) -> list[int]:
    return _to_numpy(array).astype(np.int32, copy=False).tolist()


def _folder_calibration(widget: Any, accel: Any) -> dict[str, Any]:
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

    cal = _folder_calibration(widget, accel)
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
        widget.webgpu_g_bf_url = ""
        widget.webgpu_h5_source_json = json.dumps(h5_source or {})
        if h5_source:
            widget.phase_bytes = b""
            widget.phase_width = 0
            widget.phase_height = 0
        if h5_source and h5_source.get("kind") == "bf_columns":
            widget.webgpu_preview_status = (
                "WebGPU folder ready: browser range-reads exact BF columns "
                "and builds reducers transiently."
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


def _prepare_hdf5_source_folder(
    master: pathlib.Path,
    out_path: pathlib.Path,
    *,
    files: list[pathlib.Path] | None = None,
) -> dict[str, Any]:
    """Expose compressed HDF5 source files inside the review folder."""

    source_dir = out_path / "source"
    files = files or _collect_hdf5_source_files(master)
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


def _source_stack_files(files: list[pathlib.Path]) -> list[pathlib.Path]:
    """Return HDF5 files that hold scan-frame detector stacks."""

    if len(files) > 1:
        return files[1:]
    return files


def _detector_stack_shape(src: pathlib.Path) -> tuple[int, int, int, np.dtype]:
    """Return ``(frames, detector_rows, detector_cols, dtype)`` for ``src``."""

    import h5py

    with h5py.File(src, "r") as handle:
        dataset = _find_hdf5_stack(handle)
        frames, det_rows, det_cols = (int(v) for v in dataset.shape)
        return frames, det_rows, det_cols, np.dtype(dataset.dtype)


def _write_bf_column_source(
    files: list[pathlib.Path],
    out_path: pathlib.Path,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Write exact detector BF columns for browser-first ShowPtycho loading."""

    import h5py

    stack_files = _source_stack_files(files)
    if not stack_files:
        raise ValueError("ShowPtycho BF-column export found no HDF5 detector stack files.")

    bf_rows = np.asarray(calibration["bf_rows"], dtype=np.int64)
    bf_cols = np.asarray(calibration["bf_cols"], dtype=np.int64)
    if bf_rows.shape != bf_cols.shape or bf_rows.ndim != 1:
        raise ValueError("ShowPtycho BF-column export needs 1D bf_rows and bf_cols.")
    num_bf = int(bf_rows.size)
    if num_bf <= 0:
        raise ValueError("ShowPtycho BF-column export found no BF detector pixels.")

    shapes = [_detector_stack_shape(src) for src in stack_files]
    detector_shapes = {(det_rows, det_cols) for _, det_rows, det_cols, _ in shapes}
    if len(detector_shapes) != 1:
        raise ValueError(f"ShowPtycho source files have inconsistent detector shapes: {sorted(detector_shapes)!r}")
    det_rows, det_cols = next(iter(detector_shapes))
    if int(bf_rows.max()) >= det_rows or int(bf_cols.max()) >= det_cols:
        raise ValueError(
            "ShowPtycho BF mask is outside the source detector shape: "
            f"max row/col {(int(bf_rows.max()), int(bf_cols.max()))}, detector {(det_rows, det_cols)}."
        )
    plane = int(sum(frames for frames, _, _, _ in shapes))
    scan_shape = list(calibration.get("scan_region", {}).get("shape") or calibration.get("phase_shape") or [])

    source_dir = out_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = source_dir / "bf_columns.tmp.u16"
    if tmp_path.exists():
        tmp_path.unlink()
    columns_u16 = np.memmap(tmp_path, dtype="<u2", mode="w+", shape=(num_bf, plane))
    offset = 0
    max_value = 0
    for src, (frames, _det_rows, _det_cols, _dtype) in zip(stack_files, shapes, strict=True):
        with h5py.File(src, "r") as handle:
            dataset = _find_hdf5_stack(handle)
            for frame0 in range(0, frames, 1024):
                frame1 = min(frames, frame0 + 1024)
                frames_block = np.asarray(dataset[frame0:frame1])
                block = frames_block[:, bf_rows, bf_cols]
                if block.ndim != 2 or block.shape[1] != num_bf:
                    raise ValueError(
                        f"ShowPtycho BF-column slice from {src} returned {block.shape}, "
                        f"expected ({frame1 - frame0}, {num_bf})."
                    )
                if block.size:
                    max_value = max(max_value, int(np.max(block)))
                columns_u16[:, offset + frame0 : offset + frame1] = block.T.astype("<u2", copy=False)
        offset += frames
    columns_u16.flush()

    if max_value <= 255:
        rel = pathlib.Path("source") / "bf_columns.u8"
        final_path = out_path / rel
        if final_path.exists():
            final_path.unlink()
        columns_u8 = np.memmap(final_path, dtype="u1", mode="w+", shape=(num_bf, plane))
        for bf0 in range(0, num_bf, 256):
            bf1 = min(num_bf, bf0 + 256)
            columns_u8[bf0:bf1, :] = columns_u16[bf0:bf1, :].astype("u1", copy=False)
        columns_u8.flush()
        del columns_u8
        del columns_u16
        tmp_path.unlink(missing_ok=True)
        dtype = "uint8"
        bytes_per_value = 1
    else:
        rel = pathlib.Path("source") / "bf_columns.u16"
        final_path = out_path / rel
        if final_path.exists():
            final_path.unlink()
        del columns_u16
        tmp_path.replace(final_path)
        dtype = "uint16"
        bytes_per_value = 2

    return {
        "kind": "bf_columns",
        "path": rel.as_posix(),
        "url": rel.as_posix(),
        "dtype": dtype,
        "encoding": dtype,
        "num_bf": num_bf,
        "scan_shape": [int(v) for v in scan_shape],
        "plane": plane,
        "bytes_per_bf": int(plane * bytes_per_value),
        "bits_per_value": int(bytes_per_value * 8),
        "bytes": int(final_path.stat().st_size),
        "max_value": int(max_value),
        "note": "Exact detector BF columns; browser range-reads only the BF evidence needed on open.",
    }


def export_showptycho_webgpu_folder(
    widget: Any,
    out_dir: str | pathlib.Path,
    *,
    title: str | None = None,
    overwrite: bool = True,
    source_master: str | pathlib.Path | None = None,
    decode_dtype: str = "uint16",
    webgpu_source: str = "bf_columns",
) -> pathlib.Path:
    """Export a ShowPtycho WebGPU folder backed by browser-ready source files."""

    if decode_dtype not in {"uint8", "uint16", "float32"}:
        raise ValueError(
            "decode_dtype must be 'uint8', 'uint16', or 'float32'; "
            f"got {decode_dtype!r}"
        )
    if webgpu_source not in {"bf_columns", "hdf5"}:
        raise ValueError(
            "webgpu_source must be 'bf_columns' or 'hdf5'; "
            f"got {webgpu_source!r}"
        )
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
            "cal.json",
            "manifest.json",
            "README.md",
            "serve_range.py",
        ):
            (out_path / stale).unlink(missing_ok=True)
        source_dir = out_path / "source"
        if source_dir.exists() or source_dir.is_symlink():
            if source_dir.is_dir() and not source_dir.is_symlink():
                shutil.rmtree(source_dir)
            else:
                source_dir.unlink()
        saves_dir = out_path / "saves"
        if saves_dir.exists() or saves_dir.is_symlink():
            if saves_dir.is_dir() and not saves_dir.is_symlink():
                shutil.rmtree(saves_dir)
            else:
                saves_dir.unlink()
        snapshots_dir = out_path / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            for stale in ("cal.json", "manifest.json", "README.md"):
                (snapshots_dir / stale).unlink(missing_ok=True)

    raw_source = source_master if source_master is not None else getattr(widget, "_source_file", None)
    if not raw_source:
        raise ValueError(
            "ShowPtycho compressed-source export needs the original *_master.h5 path. "
            "Construct the widget with source_file=... or pass source_master=..."
        )
    master = pathlib.Path(raw_source).expanduser()
    master = master.resolve()
    accel.cache_rotation(math.radians(float(widget.rotation_deg)))
    cal = _folder_calibration(widget, accel)
    source_files = _collect_hdf5_source_files(master)
    source = _prepare_hdf5_source_folder(master, out_path, files=source_files)
    source["decode_dtype"] = decode_dtype
    cal["source_file"] = pathlib.Path(source["master"]).name
    cal["source_transport"] = "compressed_hdf5"
    cal["source_files"] = source["data_files"]
    cal["source_decode_dtype"] = decode_dtype
    cal["persistent_bf_cache"] = False
    if webgpu_source == "bf_columns":
        bf_columns = _write_bf_column_source(source_files, out_path, cal)
        source["bf_columns"] = bf_columns
        source["preferred_browser_source"] = "bf_columns"
        cal["bf_column_companion"] = True
        cal["bf_column_companion_path"] = bf_columns["path"]
        cal["bf_column_encoding"] = bf_columns["encoding"]
        cal["webgpu_source_policy"] = "bf_columns_preferred_exact"
    else:
        cal["bf_column_companion"] = False
        cal["webgpu_source_policy"] = "compressed_hdf5_fallback"

    snapshots_dir = out_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    (snapshots_dir / "cal.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "format": "quantem.showptycho.webgpu.folder.v2",
        "title": title or "ShowPtycho",
        "index": "index.html",
        "calibration": "snapshots/cal.json",
        "source": source,
        "arrays": {},
        "persistent_arrays": [],
        "non_goals": [
            "no persistent BF-G cache",
            "no reference float32 image payloads",
            "no detector binning",
        ],
    }
    (snapshots_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    _write_empty_snapshots_manifest(out_path)
    if webgpu_source == "bf_columns":
        bf_columns = source["bf_columns"]
        browser_source = {
            "kind": "bf_columns",
            "url": bf_columns["url"],
            "dtype": bf_columns["dtype"],
            "encoding": bf_columns["encoding"],
            "numBf": bf_columns["num_bf"],
            "plane": bf_columns["plane"],
            "scanShape": bf_columns["scan_shape"],
            "bytesPerBf": bf_columns["bytes_per_bf"],
            "bitsPerValue": bf_columns["bits_per_value"],
        }
    else:
        browser_source = {
            "kind": "hdf5",
            "masterUrl": source["master"],
            "dataUrls": source["data_files"],
            "chunkIndexes": source.get("chunk_indexes", []),
            "decodeDtype": decode_dtype,
        }
    _write_embedded_widget_html(
        widget,
        out_path / "index.html",
        title=str(manifest["title"]),
        calibration=cal,
        h5_source=browser_source,
    )
    if webgpu_source == "bf_columns":
        source_note = (
            "The browser range-reads exact bright-field detector columns from "
            "`source/bf_columns.*` by default, so opening the viewer does not "
            "decode the compressed HDF5 stack unless a fallback path is needed."
        )
    else:
        source_note = (
            "The browser reads the original compressed HDF5 master/data files under "
            "`source/`, decompresses the selected BF evidence with WebGPU, and builds "
            "BF reducers transiently in GPU memory."
        )
    (snapshots_dir / "README.md").write_text(
        "# ShowPtycho WebGPU Folder\n\n"
        "Two ways to open this review - no install needed for the first:\n\n"
        "1. **Double-click** `ShowPtycho.command` (macOS) - it serves this folder "
        "and opens the viewer in Chrome. Or double-click `index.html`, click "
        "**Open data folder**, and select this folder.\n"
        "2. **CLI**: `quantem show <this folder>` serves it and opens the browser "
        "automatically (needs the `quantem-widget` package). Any other Range-capable "
        "static server works too.\n\n"
        f"{source_note} The folder "
        "intentionally does not store `g_bf.c64`, reference `.f32` images, or detector-binned data.\n",
        encoding="utf-8",
    )
    # Double-click launcher (see quantem.widget.command_launcher).
    from quantem.widget.command_launcher import write_command_launcher

    write_command_launcher(out_path, "ShowPtycho")
    return out_path
