"""Reuse fitted detector geometry in the interactive viewer."""

import numpy as np

from quantem.gpu import detector
from quantem.widget import Show4DSTEM


def test_fitted_geometry_drives_viewer_bright_field():
    rows, columns = np.indices((32, 40))
    disk = (rows - 11) ** 2 + (columns - 23) ** 2 <= 25
    pattern = np.where(disk, 100, 1).astype(np.uint16)
    data = np.broadcast_to(pattern, (3, 4, 32, 40)).copy()
    mean_dp = detector.mean(data)
    center, radius = detector.fit_probe(mean_dp)
    widget = Show4DSTEM(data, center=center, bf_radius=radius, verbose=False)
    try:
        widget.apply_preset("BF")
        actual = np.frombuffer(widget.virtual_image_bytes, dtype=np.float32).reshape(3, 4)
        expected = detector.bf(data, center=center, radius=radius)
        np.testing.assert_array_equal(actual, expected)
        assert (widget.center_row, widget.center_col) == center
        assert widget.bf_radius == radius
    finally:
        widget.close()

