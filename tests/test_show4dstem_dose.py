"""Show4DSTEM's electron dose (11-18): a user's calcDose, dose = sum(PACBED) Nx^2 / (ADU FOV^2), is the mean counts
per scan position over (counts per electron x scan step area)."""

import numpy as np
import pytest
import torch

from quantem.widget import Show4DSTEM


def reference_dose(dp, fov, adu):
    """Her code: pacbed = mean over scan positions; dose = sum(pacbed) Nx Nx / (ADU fov fov)."""
    nx = dp.shape[0]
    pacbed = np.mean(dp, axis=(0, 1))
    return np.sum(pacbed) * nx * nx / (adu * fov * fov)


def test_the_dose_matches_the_reference():
    """Measured 11-18 on scan 1256 (512 x 512, 1.0526 A step): 72,038.87 e/A^2 both ways, an independent full sum of
    every count by her formula and this, from the running count total, in 0.12 ms."""
    rng = np.random.default_rng(3)
    dp = rng.poisson(3.0, (8, 8, 4, 4)).astype(np.float32)
    step = 0.5
    viewer = Show4DSTEM(torch.from_numpy(dp), scan_sampling_A=(step, step))
    try:
        assert viewer.dose() == pytest.approx(reference_dose(dp, 8 * step, 1.0), rel=1e-6)
        assert viewer.dose_e_per_A2 == pytest.approx(viewer.dose())
    finally:
        viewer.close()


def test_no_dose_without_a_scan_step():
    viewer = Show4DSTEM(torch.ones((4, 4, 2, 2)))
    try:
        assert viewer.dose_e_per_A2 is None
    finally:
        viewer.close()


def test_source_lines_and_frame_metadata():
    """The folder and file lines above the title (empty hides them), and the browsed dataset's metadata."""
    import numpy as np
    from quantem.widget import Show4DSTEM

    viewer = Show4DSTEM(np.zeros((4, 4, 8, 8), np.float32))
    assert (viewer.source_folder, viewer.source_file) == ("", "")
    viewer.source_folder, viewer.source_file = "/data/live/2026-11-18", "not saved (live)"
    assert isinstance(viewer.frame_metadata(), dict)


def test_frame_metadata_reads_a_live_scans_acquisition(monkeypatch):
    """A live scan is an AcquisitionView without metadata of its own: its acquisition (``.source``) holds it
    (measured on Buffle 11-18: the folder and file lines stayed empty)."""
    from types import SimpleNamespace

    import numpy as np
    from quantem.widget import Show4DSTEM

    viewer = Show4DSTEM(np.zeros((4, 4, 8, 8), np.float32))
    visits = np.ones((4, 4), np.int32)
    visits[2, 1] = 0
    view = SimpleNamespace(source=SimpleNamespace(metadata={"series": "series_1440", "visits_per_position": visits}))
    try:
        monkeypatch.setattr(viewer, "_frame_source", lambda index: view)
        assert viewer.frame_metadata()["series"] == "series_1440"
        viewer._update_unvisited()
        np.testing.assert_array_equal(np.frombuffer(viewer.unvisited_bytes, np.uint8).reshape(4, 4), visits == 0)
    finally:
        viewer.close()
