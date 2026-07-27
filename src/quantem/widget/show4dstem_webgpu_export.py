"""Local-server Show4DSTEM WebGPU bundle export.

Mirrors the ShowPtycho handoff protocol: the recipient double-clicks one
``Show4DSTEM.command``, a local range-capable HTTP server starts over the data
folder, and Chrome opens a fully vendored viewer page. The current CLI path
uses lazy sidecars for first BF/VI paint and byte ranges into the original HDF5
files for on-demand diffraction frames. No Python package install, no network,
and no folder-grant click are required at view time. Everything the page needs
(require.js, the Jupyter widget manager, anywidget, the server script) ships
from this package's ``static/vendor``.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re
from typing import Any, Sequence

import numpy as np

from quantem.widget.command_launcher import write_command_launcher

_VENDOR = pathlib.Path(__file__).parent / "vendor"
# CDN references embed_minimal_html emits; each is replaced by a vendored copy so
# the bundle works with no network at all (conference wifi is not a dependency).
# Patterns, not exact URLs: the emitted version specifiers drift across
# ipywidgets/anywidget releases (e.g. anywidget@0.11.0 vs anywidget@~0.11.*).
_CDN_REWRITES = (
    (re.compile(r"https://cdnjs\.cloudflare\.com/[^\"']*/require(\.min)?\.js"), "./require.min.js"),
    (re.compile(r"https://cdn\.jsdelivr\.net/[^\"']*html-manager[^\"']*/embed-amd\.js"), "./embed-amd.js"),
    (re.compile(r"\"https://cdn\.jsdelivr\.net/npm/anywidget@[^\"]*\""), '"./anywidget.min"'),
)


def _write_vendor_asset(name: str, viewer: pathlib.Path) -> None:
    """Expand a compressed browser-manager asset into the export viewer."""
    source = _VENDOR / f"{name}.gz"
    if not source.is_file():
        raise FileNotFoundError(
            f"missing vendored Show4DSTEM browser asset: {source}; "
            "rebuild the package with src/quantem/widget/vendor included"
        )
    with gzip.open(source, "rb") as src, (viewer / name).open("wb") as dst:
        dst.write(src.read())


# Promoted WebGPU decode configuration. Native uint16 is the conservative
# default; audited uint8 browse sources can use the low8-only kernel to skip the
# high bitplanes that only hold masked detector sentinels.
def _tuning(*, h5_uint8_lossless: bool) -> str:
    dtype = "uint8" if h5_uint8_lossless else "u2"
    low8 = "true" if h5_uint8_lossless else "false"
    return (
        "<script>\n"
        f'globalThis.__QT_H5_DECODE_DTYPE ??= "{dtype}";\n'
        f"globalThis.__QT_H5_FORCE_LOW8 ??= {low8};\n"
        f"globalThis.__BSLZ4_LOW8_ONLY ??= {low8};\n"
        "globalThis.__BSLZ4_FRAME_WG ??= 64;\n"
        "globalThis.__BSLZ4_PIPELINE_STAGING ??= false;\n"
        "globalThis.__QT_H5_FETCH_WINDOW ??= 8;\n"
        "globalThis.__QT_H5_DECODE_QUEUE ??= 8;\n"
        "globalThis.__QT_H5_PRELOAD_WINDOW ??= 1;\n"
        "globalThis.__QT_H5_LOCAL_GROUP ??= 8;\n"
        "globalThis.__QT_H5_LOCAL_WORKERS ??= 8;\n"
        "</script>\n"
    )


def export_show4dstem_webgpu_bundle(
    widget: Any,
    out_dir: str | pathlib.Path,
    *,
    port: int = 8794,
    title: str | None = None,
) -> pathlib.Path:
    """Write a double-clickable Show4DSTEM WebGPU bundle into ``out_dir``.

    ``out_dir`` must be the folder holding the linked ``*_master.h5`` family and
    any ``*_lazy/`` sidecars the widget references. Produces
    ``Show4DSTEM.command`` at the root and a hidden ``.viewer/`` with the
    vendored page and the range-capable server. Returns the path to the
    launcher. Without this bundle the recipient needs Python, the CDNs, and a
    folder-grant click; with it the demo is one double-click.
    """
    root = pathlib.Path(out_dir)
    if not root.is_dir():
        raise ValueError(f"bundle out_dir must be an existing data folder: {root}")
    masters = sorted(root.glob("*_master.h5"))
    if not masters:
        raise ValueError(f"no *_master.h5 files in {root}; the bundle serves the data folder itself")
    viewer = root / ".viewer"
    viewer.mkdir(exist_ok=True)
    html = viewer / "Show4DSTEM.html"
    widget._write_html_export(html, dtype="uint16", det_bin=1, scan_bin=1, title=title)
    text = html.read_text(encoding="utf-8")
    text = text.replace(
        "<head>",
        "<head>\n"
        + _tuning(h5_uint8_lossless=bool(getattr(widget, "_h5_uint8_lossless", False))),
        1,
    )
    for pattern, local in _CDN_REWRITES:
        text = pattern.sub(local, text)
    html.write_text(text, encoding="utf-8")
    for name in ("require.min.js", "embed-amd.js", "anywidget.min.js"):
        _write_vendor_asset(name, viewer)
    return write_command_launcher(
        root,
        "Show4DSTEM",
        viewer_html=".viewer/Show4DSTEM.html",
        port=int(port),
    )


def bundle_master_urls(folder: str | pathlib.Path, names: Sequence[str] | None = None) -> list[str]:
    """Viewer-relative URLs (``../<basename>``) for masters in a bundle folder.

    The viewer page lives one level down in ``.viewer/``, so data references
    must climb back to the served root; a bare basename would resolve inside
    ``.viewer/`` and 404. ``names`` filters by substring, preserving its order.
    """
    folder = pathlib.Path(folder)
    masters = sorted(p.name for p in folder.glob("*_master.h5"))
    if names:
        picked = []
        for token in names:
            hits = [m for m in masters if token in m]
            if not hits:
                raise ValueError(f"no master matches {token!r} in {folder}")
            picked.append(hits[0])
        masters = picked
    return [f"../{name}" for name in masters]


def build_lazy_show4dstem_sidecar(
    folder: str | pathlib.Path,
    *,
    label: str,
    scan_shape: tuple[int, int],
    detector_shape: tuple[int, int],
) -> str:
    """Build a lazy WebGPU sidecar for an anonymous HDF5 family.

    The sidecar is intentionally small compared with the raw HDF5 family:
    ``profile.bin`` stores radial detector-bin sums per scan position,
    ``index.bin`` stores one range-fetch pointer per scan frame, and ``com.bin``
    stores a full-detector center-of-mass field. Raw detector frames stay in the
    linked ``*_data_*.h5`` files and are fetched on demand by byte range.
    """

    import h5py

    root = pathlib.Path(folder)
    data_files = sorted(root.glob(f"{label}_data_*.h5"))
    if not data_files:
        raise ValueError(f"no linked HDF5 data files found for {label!r} in {root}")
    scan_rows, scan_cols = (int(scan_shape[0]), int(scan_shape[1]))
    det_rows, det_cols = (int(detector_shape[0]), int(detector_shape[1]))
    if det_rows != det_cols:
        raise ValueError(
            "Show4DSTEM lazy WebGPU sidecars currently require a square detector; "
            f"got detector_shape={detector_shape!r}."
        )
    scan_count = scan_rows * scan_cols
    detector_size = det_rows * det_cols
    nbins = max(1, det_rows // 2)
    lazy_dir = root / f"{label}_lazy"
    lazy_dir.mkdir(parents=True, exist_ok=True)

    rows = np.arange(det_rows, dtype=np.float32)[:, None]
    cols = np.arange(det_cols, dtype=np.float32)[None, :]
    radial_bins = np.floor(
        np.hypot(rows - det_rows / 2, cols - det_cols / 2)
    ).astype(np.int32)
    radial_bins = np.clip(radial_bins.reshape(-1), 0, nbins - 1)
    radial_one_hot = np.zeros((detector_size, nbins), dtype=np.float32)
    radial_one_hot[np.arange(detector_size), radial_bins] = 1.0
    row_coords = np.broadcast_to(
        np.arange(det_rows, dtype=np.float32)[:, None], (det_rows, det_cols)
    ).reshape(-1)
    col_coords = np.broadcast_to(
        np.arange(det_cols, dtype=np.float32)[None, :], (det_rows, det_cols)
    ).reshape(-1)

    profile_path = lazy_dir / "profile.bin"
    index_path = lazy_dir / "index.bin"
    com_path = lazy_dir / "com.bin"
    profile = np.memmap(profile_path, mode="w+", dtype=np.float32, shape=(scan_count, nbins))
    frame_index = np.memmap(index_path, mode="w+", dtype=np.uint32, shape=(scan_count, 3))
    com = np.memmap(com_path, mode="w+", dtype=np.float32, shape=(2, scan_count))

    frame_cursor = 0
    for file_index, data_file in enumerate(data_files):
        with h5py.File(data_file, "r") as handle:
            dataset = handle.get("entry/data/data")
            if dataset is None:
                raise ValueError(f"{data_file.name} has no entry/data/data dataset")
            if tuple(int(value) for value in dataset.shape[-2:]) != (det_rows, det_cols):
                raise ValueError(
                    f"{data_file.name} detector shape {dataset.shape[-2:]} does not "
                    f"match {detector_shape!r}."
                )
            n_frames = int(dataset.shape[0])
            for frame in range(n_frames):
                if frame_cursor >= scan_count:
                    raise ValueError(
                        f"{label!r} has more frames than scan_shape={scan_shape!r}."
                    )
                info = dataset.id.get_chunk_info_by_coord((frame, 0, 0))
                frame_index[frame_cursor] = (
                    file_index,
                    int(info.byte_offset),
                    int(info.size),
                )
                frame_cursor += 1
            batch = 512
            start_scan = frame_cursor - n_frames
            for start in range(0, n_frames, batch):
                stop = min(n_frames, start + batch)
                frames = np.asarray(dataset[start:stop], dtype=np.float32).reshape(
                    stop - start, detector_size
                )
                out_slice = slice(start_scan + start, start_scan + stop)
                profile[out_slice, :] = frames @ radial_one_hot
                totals = frames.sum(axis=1)
                safe_totals = np.where(totals > 0, totals, 1.0)
                com[0, out_slice] = (frames @ row_coords) / safe_totals
                com[1, out_slice] = (frames @ col_coords) / safe_totals

    if frame_cursor != scan_count:
        raise ValueError(
            f"{label!r} has {frame_cursor} frames, expected {scan_count} from "
            f"scan_shape={scan_shape!r}."
        )
    profile.flush()
    frame_index.flush()
    com.flush()
    meta = {
        "SR": scan_rows,
        "SC": scan_cols,
        "D": det_rows,
        "NB": nbins,
        "nFrames": scan_count,
        "files": [f"../{path.name}" for path in data_files],
    }
    (lazy_dir / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    return f"{label}_lazy/"
