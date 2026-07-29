from __future__ import annotations


def test_widget_io_public_api_uses_quantem_gpu_loader() -> None:
    import quantem.gpu.io.hdf5 as gpu_hdf5
    import quantem.widget.io as widget_io

    assert widget_io.load is gpu_hdf5.load
    assert widget_io.bin is gpu_hdf5.bin
    assert widget_io.LoadResult is gpu_hdf5.LoadResult
