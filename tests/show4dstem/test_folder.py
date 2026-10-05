"""Show4DSTEM.from_folder over small real Arina-style masters on the GPU."""

import warnings
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pytest
from quantem.gpu import detector
from quantem.gpu.device import detect

from quantem.widget import Show4DSTEM


@pytest.fixture(autouse=True)
def _accelerator():
    """io.load keeps acquisitions encoded on CUDA or MPS; there is no CPU route."""
    try:
        detect()
    except RuntimeError as exc:
        pytest.skip(f"encoded acquisitions need CUDA or MPS: {exc}")


def _write_master(folder: Path, stem: str, values: np.ndarray, *, linked: bool = True) -> Path:
    """Write one Arina master that links its frames from an external data file.

    ``values`` has shape (scan_row, scan_col, detector_row, detector_col).
    ``linked=False`` leaves the data file missing, as during an acquisition.
    """
    rows, cols, det_rows, det_cols = values.shape
    chunk = folder / f"{stem}_data_000001.h5"
    master = folder / f"{stem}_master.h5"
    if linked:
        with h5py.File(chunk, "w") as handle:
            handle.create_dataset(
                "entry/data/data",
                data=values.reshape(rows * cols, det_rows, det_cols),
                chunks=(1, det_rows, det_cols),
                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
            )
    with h5py.File(master, "w") as handle:
        handle.require_group("entry/data")["data_000001"] = h5py.ExternalLink(
            chunk.name, "entry/data/data"
        )
        specific = handle.require_group("entry/instrument/detector/detectorSpecific")
        specific.create_dataset("ntrigger", data=rows * cols)
        specific.create_dataset("nimages", data=1)
    return master


def _counts(seed: int, shape=(4, 4, 8, 8)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 200, shape, dtype=np.uint16)


def test_from_folder_fills_every_master_with_exact_panels(tmp_path):
    """C1: three complete masters, expect labels in file order and panels equal to the encoded sums."""
    counts = [_counts(seed) for seed in range(3)]
    for index, values in enumerate(counts):
        _write_master(tmp_path, f"scan_{index:02d}", values)
    widget = Show4DSTEM.from_folder(tmp_path, watch=False)
    try:
        widget.wait_for_folder()
        assert widget.n_frames == 3
        assert list(widget.frame_labels) == ["scan_00", "scan_01", "scan_02"]
        assert widget.compare_panel_indices == [0, 1, 2]
        mask = widget._current_detector_mask().cpu().numpy().astype(bool)
        panels = np.frombuffer(widget.compare_virtual_image_bytes, np.float32).reshape(3, 4, 4)
        for panel, values in zip(panels, counts):
            exact = values[..., mask].sum(axis=-1, dtype=np.uint64)
            np.testing.assert_allclose(panel, exact / mask.sum(), rtol=1e-6)
    finally:
        widget.close()


def test_from_folder_skips_incomplete_masters_quietly_unless_verbose(tmp_path):
    """C2: one master still missing its data file, expect it skipped, and a warning only with verbose."""
    _write_master(tmp_path, "scan_00", _counts(0))
    _write_master(tmp_path, "scan_01", _counts(1), linked=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        widget = Show4DSTEM.from_folder(tmp_path, watch=False)
    try:
        widget.wait_for_folder()
        assert list(widget.frame_labels) == ["scan_00"]
    finally:
        widget.close()
    with pytest.warns(RuntimeWarning, match="scan_01"):
        widget = Show4DSTEM.from_folder(tmp_path, watch=False, verbose=True)
    widget.close()


def test_from_folder_keeps_the_largest_geometry_group(tmp_path):
    """C3: two 4x4 scans and one 2x2 scan, expect only the 4x4 pair in the comparison."""
    _write_master(tmp_path, "scan_00", _counts(0))
    _write_master(tmp_path, "scan_01", _counts(1, (2, 2, 8, 8)))
    _write_master(tmp_path, "scan_02", _counts(2))
    widget = Show4DSTEM.from_folder(tmp_path, watch=False)
    try:
        widget.wait_for_folder()
        assert list(widget.frame_labels) == ["scan_00", "scan_02"]
        assert (widget.shape_rows, widget.shape_cols) == (4, 4)
    finally:
        widget.close()


def test_from_folder_rejects_too_few_or_no_ready_masters(tmp_path):
    """C4: an empty folder, then fewer masters than min_masters, expect corrective errors."""
    with pytest.raises(ValueError, match="No files matching|No ready"):
        Show4DSTEM.from_folder(tmp_path, watch=False)
    _write_master(tmp_path, "scan_00", _counts(0))
    with pytest.raises(ValueError, match="at least 2"):
        Show4DSTEM.from_folder(tmp_path, watch=False, min_masters=2)


def test_from_folder_free_releases_the_acquisitions_it_loaded(tmp_path):
    """C5: free() on a folder viewer, expect every acquisition it loaded closed."""
    for index in range(2):
        _write_master(tmp_path, f"scan_{index:02d}", _counts(index))
    widget = Show4DSTEM.from_folder(tmp_path, watch=False)
    try:
        widget.wait_for_folder()
        acquisitions = list(widget._folder_acquisitions)
        assert len(acquisitions) == 2
        mask = np.ones((8, 8), bool)
        detector.prepare(acquisitions[0]).masked_sum(mask)
        widget.free()
        assert widget._folder_acquisitions == []
        with pytest.raises((RuntimeError, ValueError)):
            detector.prepare(acquisitions[0]).masked_sum(mask)
    finally:
        widget.close()
