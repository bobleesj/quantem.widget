import stat

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem_webgpu_export import (
    bundle_master_urls,
    export_show4dstem_webgpu_bundle,
)


def _write_arina_family(folder, stem, n_frames=16, det=32):
    frames = np.zeros((n_frames, det, det), np.uint16)
    frames[:, det // 2, det // 2] = 500  # a >255 count so uint16 matters
    with h5py.File(folder / f"{stem}_data_000001.h5", "w") as f:
        f.create_group("entry/data").create_dataset(
            "data", data=frames, chunks=(1, det, det),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
    with h5py.File(folder / f"{stem}_master.h5", "w") as f:
        g = f.create_group("entry/instrument/detector/detectorSpecific")
        g.create_dataset("ntrigger", data=n_frames)
        g.create_dataset("nimages", data=1)
        g.create_dataset("pixel_mask", data=np.zeros((det, det), np.uint32))


def test_bundle_export_writes_launcher_viewer_and_vendored_page(tmp_path):
    _write_arina_family(tmp_path, "tilt_a")
    _write_arina_family(tmp_path, "tilt_b")
    urls = bundle_master_urls(tmp_path)
    assert urls == ["../tilt_a_master.h5", "../tilt_b_master.h5"]
    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), np.uint8), h5_urls=urls,
        scan_shape=(4, 4), detector_shape=(32, 32), backend="webgpu",
        view_mode="multiple", compare_max_panels=3, compare_group_mode="all",
        precompute_virtual_images=False, verbose=False,
    )
    try:
        launcher = export_show4dstem_webgpu_bundle(widget, tmp_path, port=8899)
    finally:
        widget.close()
    assert launcher.name == "Show4DSTEM.command"
    assert launcher.stat().st_mode & stat.S_IXUSR
    command = launcher.read_text()
    assert "8899" in command and "serve_range.py" in command
    viewer = tmp_path / ".viewer"
    for name in ("Show4DSTEM.html", "require.min.js", "embed-amd.js", "anywidget.min.js", "serve_range.py"):
        assert (viewer / name).exists(), name
    page = (viewer / "Show4DSTEM.html").read_text(encoding="utf-8")
    assert "cdnjs.cloudflare.com" not in page and "cdn.jsdelivr.net" not in page
    assert "__QT_H5_DECODE_DTYPE" in page and "__BSLZ4_FRAME_WG" in page
    assert 'globalThis.__QT_H5_DECODE_DTYPE ??= "u2"' in page
    assert "globalThis.__QT_H5_FORCE_LOW8 ??= false" in page
    assert "../tilt_a_master.h5" in page


def test_bundle_export_uses_low8_for_audited_uint8_h5(tmp_path):
    _write_arina_family(tmp_path, "tilt_a")
    urls = bundle_master_urls(tmp_path)
    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), np.uint8),
        h5_urls=urls,
        h5_uint8_lossless=True,
        scan_shape=(4, 4),
        detector_shape=(32, 32),
        backend="webgpu",
        precompute_virtual_images=False,
        verbose=False,
    )
    try:
        export_show4dstem_webgpu_bundle(widget, tmp_path, port=8899)
    finally:
        widget.close()
    page = (tmp_path / ".viewer" / "Show4DSTEM.html").read_text(encoding="utf-8")
    assert 'globalThis.__QT_H5_DECODE_DTYPE ??= "uint8"' in page
    assert "globalThis.__QT_H5_FORCE_LOW8 ??= true" in page
    assert "globalThis.__BSLZ4_LOW8_ONLY ??= true" in page


def test_bundle_export_requires_masters(tmp_path):
    widget = Show4DSTEM(np.zeros((2, 2, 4, 4), np.uint8), verbose=False)
    try:
        with pytest.raises(ValueError, match="master"):
            export_show4dstem_webgpu_bundle(widget, tmp_path)
    finally:
        widget.close()
