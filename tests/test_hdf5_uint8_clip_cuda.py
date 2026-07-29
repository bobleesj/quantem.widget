"""CUDA regression tests for the direct uint8 HDF5 browse path."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quantem.gpu.io.hdf5 import _clip_to_uint8, _clip_to_uint8_count  # noqa: E402


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_clip_to_uint8_count_saturates_and_counts(dtype):
    """The direct dtype='u8' path clips counts above 255 instead of wrapping."""
    src = cp.asarray([0, 1, 254, 255, 256, 1000, 65535], dtype=dtype)
    dst = cp.empty(src.shape, dtype=cp.uint8)

    clipped = _clip_to_uint8_count(src, dst)
    cp.cuda.Device().synchronize()

    np.testing.assert_array_equal(
        cp.asnumpy(dst),
        np.asarray([0, 1, 254, 255, 255, 255, 255], dtype=np.uint8),
    )
    assert int(clipped) == 3


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_clip_to_uint8_saturates_without_counting(dtype):
    """The no-bin browse hot path clips to uint8 without the slow count pass."""
    src = cp.asarray([0, 255, 256, 4096], dtype=dtype)
    dst = cp.empty(src.shape, dtype=cp.uint8)

    assert _clip_to_uint8(src, dst) is True
    cp.cuda.Device().synchronize()

    np.testing.assert_array_equal(
        cp.asnumpy(dst),
        np.asarray([0, 255, 255, 255], dtype=np.uint8),
    )
