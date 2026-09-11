"""Packed precision remains resident during live scientific inspection."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Requires an owned CUDA test window and precision-capable quantem.gpu.",
)


@pytest.mark.parametrize("dtype", ["float16", "scaled_uint16"])
def test_saved_precision_viewer_keeps_units_and_report(tmp_path, dtype):
    cp = pytest.importorskip("cupy")
    from quantem.gpu import io
    from quantem.widget import Show4DSTEM

    values = cp.linspace(0, 10, 8 * 8 * 16 * 16, dtype=cp.float32).reshape(8, 8, 16, 16)
    path = tmp_path / "display_master.h5"
    io.save(path, values, dtype=dtype)
    data = io.load(path, verbose=False)
    viewer = Show4DSTEM(data)
    assert viewer.precision_report["storage"] == dtype
    assert viewer.precision_report["report_origin"] == "saved"
    assert not viewer.offline
    viewer.vi_roi_center_row = 6
    viewer.vi_roi_center_col = 5
    assert len(viewer.frame_bytes) == 16 * 16 * 4
    assert len(viewer.virtual_image_bytes) == 8 * 8 * 4
    # The loaded owner stays live after a detector session ends.
    viewer._close_compute()
    assert not data.data.is_released
    data.close()
