"""Browser folder for a ShowPtycho: the same widget UI over exact bright-field counts, no kernel.

The folder holds ``index.html`` (the widget with its state embedded and WebGPU switched on),
``source/bf_columns.*`` (the exact detector counts of the bright-field disk, ANS-coded ``.qem`` from a
CUDA session or the ``.u8`` / ``.u16`` companion an MPS session opened), ``snapshots/`` (the calibration
the browser kernels start from, a manifest and the saved aberration states) and a double-click
``ShowPtycho.command`` that serves the folder. No float32 image, complex reducer or copy of the raw
HDF5 acquisition is written: the browser rebuilds the SSB reducers from the counts on its own GPU.
"""

import json
import os
import pathlib
import shutil

import numpy as np

from quantem.widget.command_launcher import write_command_launcher
from quantem.widget.export import write_widget_html

# the browser kernels are specialised per native scan size
BROWSER_SCAN_SIZES = (128, 256, 512, 1024)
# resident scan spectra the browser may keep, in GiB; raising it never drops detector pixels
BROWSER_GPU_MEMORY_GB = 4.5
# every stored record names its layout (js/showptycho/engineAdapter.ts reads the same strings)
BROWSER_CALIBRATION_SCHEMA = "quantem.ssb-browser-calibration/2"
BROWSER_FOLDER_SCHEMA = "quantem.ssb-browser-folder/3"
SNAPSHOTS_SCHEMA = "quantem.ssb-snapshots/1"
# live-session traits the embedded page overrides; they are restored after the page is written
LIVE_TRAITS = (
    "webgpu_preview_enabled", "webgpu_standalone", "webgpu_cal_json", "webgpu_h5_source_json", "webgpu_preview_status",
    "phase_bytes", "phase_width", "phase_height", "stars_path", "calibration_path", "crop_refit_available",
)


def write_showptycho_folder(widget, path: str | pathlib.Path | None, title: str | None) -> pathlib.Path:
    """Write the review folder for ``widget`` and return its path (see the module docstring for the layout)."""
    ssb = widget._ssb
    ssb.set_scan_detector_rotation(float(widget.scan_detector_rotation_deg))
    session = ssb.state()
    state = ssb.browser_state()
    rows, cols = state["scan_shape"]
    if rows != cols or rows not in BROWSER_SCAN_SIZES:
        raise NotImplementedError(f"ShowPtycho WebGPU folder export supports square 128, 256, 512, or 1024 crops; got {rows}x{cols}.")
    if path is not None:
        folder = pathlib.Path(path).expanduser()
    elif widget._source_path:
        master = pathlib.Path(widget._source_path).expanduser().resolve()
        stem = master.name[: -len("_master.h5")] if master.name.endswith("_master.h5") else master.stem
        folder = master.parent / f"{stem}_showptycho"
    else:
        folder = pathlib.Path.cwd() / "showptycho_export"
    folder.mkdir(parents=True, exist_ok=True)
    source_dir = folder / "source"
    # a re-export reuses the counts it wrote last time instead of re-encoding them
    counts_path = state["bright_field_source_path"]
    own_source = counts_path is not None and counts_path.is_file() and counts_path.resolve().parent == source_dir.resolve()
    if not own_source and (source_dir.exists() or source_dir.is_symlink()):
        if source_dir.is_dir() and not source_dir.is_symlink():
            shutil.rmtree(source_dir)
        else:
            source_dir.unlink()
    if session["backend"] == "cuda" and not (counts_path is not None and counts_path.suffix == ".qem" and counts_path.exists()):
        ssb.export_bright_field(source_dir / "bf_columns.qem")
        state = ssb.browser_state()
        counts_path = state["bright_field_source_path"]
    if counts_path is None or state["bright_field_source_dtype"] is None:
        raise RuntimeError("ShowPtycho export needs exact bright-field counts: open the acquisition with SSB.open(...) or use a CUDA session.")

    num_bright_field_pixels = state["num_bright_field_pixels"]
    # geometry arrays are stored in the browser kernels' int32 / float32, the values they compute with
    calibration = {
        "schema": BROWSER_CALIBRATION_SCHEMA,
        "source_path": "redacted_local_source",
        "scan_region": list(widget._scan_region),
        "scan_shape": [rows, cols],
        "source_scan_shape": list(widget._source_scan_shape),
        # the storage the browser SSB kernels compute in; they refuse a calibration that names another
        "precision": {"real_dtype": "float32", "complex_dtype": "complex64"},
        "num_bright_field_pixels": num_bright_field_pixels,
        "g_shape": [num_bright_field_pixels, rows, cols],
        "g_dtype": "complex64_interleaved_re_im_native_le",
        "phase_shape": [rows, cols],
        "phase_dtype": "float32_native_le",
        "detector_shape": state["detector_shape"],
        "bright_field_radius_px": state["bright_field_radius_px"],
        "bright_field_center_px": state["bright_field_center_px"],
        **{name: np.asarray(state[name]).astype(np.int32).tolist() for name in ("bright_field_rows", "bright_field_cols")},
        **{name: np.asarray(state[name]).astype(np.float32).tolist()
           for name in ("bright_field_k_row", "bright_field_k_col", "scan_q_row", "scan_q_col",
                        "aperture_k", "alpha_k2", "cos2phi_k", "sin2phi_k")},
        # the wavelength the session computed from voltage_kV, kept so the browser evaluates the same phase
        "wavelength_A": state["wavelength_A"],
        "voltage_kV": session["voltage_kV"],
        "semiangle_mrad": session["semiangle_mrad"],
        "scan_sampling_A": session["scan_sampling_A"],
        "detector_sampling_mrad": state["detector_sampling_mrad"],
        "scan_detector_rotation_deg": float(widget.scan_detector_rotation_deg) % 360.0,
        "detector_transposed": False,
        "aberrations": widget._current(),
        "flip_phase": bool(widget.flip_phase),
        "dc_value": [float(state["dc_value"].real), float(state["dc_value"].imag)],
        "gpu_memory_gb": BROWSER_GPU_MEMORY_GB,
    }

    # the counts file: link the session's companion into source/ (hardlink, else symlink, else copy)
    dtype = state["bright_field_source_dtype"]
    if counts_path.suffix == ".qem":
        encoding, suffix = "qem", "qem"
    elif dtype == np.dtype(np.uint8):
        encoding, suffix = "uint8", "u8"
    else:
        encoding, suffix = "uint16", "u16"
    relative_path = pathlib.Path("source") / f"bf_columns.{suffix}"
    target = (folder / relative_path).resolve()
    if counts_path.resolve() == target:
        link_mode = "encoded_here"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(counts_path, target)
            link_mode = "hardlink"
        except OSError:
            try:
                target.symlink_to(counts_path)
                link_mode = "symlink"
            except OSError:
                shutil.copy2(counts_path, target)
                link_mode = "copy"
    bf_columns = {
        "kind": "bf_columns", "path": relative_path.as_posix(), "url": relative_path.as_posix(), "dtype": dtype.name, "encoding": encoding,
        "num_bright_field_pixels": num_bright_field_pixels, "scan_shape": [rows, cols], "plane": rows * cols,
        "bytes_per_bright_field_pixel": rows * cols * dtype.itemsize,
        "bits_per_value": dtype.itemsize * 8, "bytes": target.stat().st_size, "max_value": state["bright_field_source_max_value"], "link": link_mode,
        "note": ("Lossless ANS counts decoded on the GPU; changing output sampling reuses the resident native scan spectra."
                 if encoding == "qem" else "Exact detector BF columns, read once and retained on the GPU."),
    }
    calibration.update({
        "source_transport": "bf_columns", "source_files": [bf_columns["path"]], "source_decode_dtype": bf_columns["dtype"],
        "persistent_bf_cache": False, "bf_column_companion": True, "bf_column_companion_path": bf_columns["path"],
        "bf_column_encoding": encoding, "webgpu_source_policy": "bf_columns_preferred_exact",
    })

    snapshots = folder / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    (snapshots / "cal.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    manifest = {
        "schema": BROWSER_FOLDER_SCHEMA,
        "title": title or "ShowPtycho",
        "index": "index.html",
        "calibration": "snapshots/cal.json",
        "source": {"kind": "bf_columns", "bf_columns": bf_columns, "preferred_browser_source": "bf_columns",
                   "note": "Only exact bright-field detector columns are bundled; the private raw HDF5 acquisition is not included."},
        "arrays": {},
        "persistent_arrays": [],
        "non_goals": ["no persistent BF-G cache", "no reference float32 image payloads", "no detector binning"],
    }
    (snapshots / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not (snapshots / "snapshots.json").exists():
        (snapshots / "snapshots.json").write_text(json.dumps({"schema": SNAPSHOTS_SCHEMA, "snapshots": []}) + "\n", encoding="utf-8")
    (snapshots / "README.md").write_text(
        "# ShowPtycho WebGPU Folder\n\n"
        "Two ways to open this review - no install needed for the first:\n\n"
        "1. **Double-click** `ShowPtycho.command` (macOS) - it serves this folder "
        "and opens the viewer in Chrome. Or double-click `index.html`, click "
        "**Open data folder**, and select this folder.\n"
        "2. **Serve it**: run `python -m http.server` in this folder (or any other "
        "Range-capable static server) and open `index.html`.\n\n"
        "The browser loads exact bright-field detector counts from "
        "`source/bf_columns.*` by default, so opening the viewer does not "
        "decode the compressed HDF5 stack unless a fallback path is needed. The folder stores detector evidence rather than derived "
        "Fourier caches, reference images, or detector-binned data.\n",
        encoding="utf-8",
    )

    # embed the widget with the folder source switched on, then restore the live-session traits
    # one spelling per key: the page converts these to the engine's own names (js/showptycho/engineAdapter.ts)
    browser_source = {name: bf_columns[name] for name in ("kind", "url", "dtype", "encoding", "num_bright_field_pixels", "plane",
                                                          "scan_shape", "bytes_per_bright_field_pixel", "bits_per_value")}
    live_values = {name: getattr(widget, name) for name in LIVE_TRAITS}
    try:
        widget.webgpu_preview_enabled = True
        widget.webgpu_standalone = True
        widget.crop_refit_available = False  # refitting needs the live Python session
        widget.webgpu_cal_json = json.dumps(calibration)
        widget.webgpu_h5_source_json = json.dumps(browser_source)
        widget.stars_path = "snapshots/snapshots.json"
        widget.calibration_path = "snapshots/calibration.json"
        widget.phase_bytes, widget.phase_width, widget.phase_height = b"", 0, 0
        widget.webgpu_preview_status = "WebGPU folder ready: browser loads exact BF counts and builds reducers transiently."
        write_widget_html(folder / "index.html", widget, str(manifest["title"]))
    finally:
        for name, value in live_values.items():
            setattr(widget, name, value)
    write_command_launcher(folder, "ShowPtycho")
    return folder
