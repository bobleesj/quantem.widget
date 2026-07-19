from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


def test_show4dstem_cuda_keeps_cupy_compute_source_for_rawkernel() -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is not available.")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device is not available: {exc}")

    from quantem.widget import Show4DSTEM

    data = cp.ones((4, 4, 12, 12), dtype=cp.uint16)
    widget = Show4DSTEM(
        data,
        precompute_virtual_images=False,
        center=(5.5, 5.5),
        bf_radius=2.0,
    )
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True

    assert widget._compute.__class__.__name__ == "CudaKernelCompute"
    np.testing.assert_array_equal(
        widget._fast_masked_sum(mask),
        np.full((4, 4), int(mask.sum()), dtype=np.float32),
    )
    assert widget._compute._total_cache_uint64 is None


def test_show4dstem_cuda_compare_grid_uses_rawkernel_frames() -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is not available.")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device is not available: {exc}")

    from quantem.widget import Show4DSTEM

    data = cp.ones((2, 4, 4, 12, 12), dtype=cp.uint16)
    widget = Show4DSTEM(
        data,
        precompute_virtual_images=False,
        view_mode="compare",
        compare_max_panels=2,
        center=(5.5, 5.5),
        bf_radius=2.0,
    )
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True

    panels = widget._compare_virtual_images_for_indices([0, 1], mask)

    assert len(panels) == 2
    for panel in panels:
        np.testing.assert_array_equal(panel, np.ones((4, 4), dtype=np.float32))
    assert list(widget._cuda_compare_compute_backends) == [0, 1]

    _ = widget._compare_virtual_images_for_indices([0, 1], mask)
    assert list(widget._cuda_compare_compute_backends) == [0, 1]


def test_show4dstem_mps_contract_uses_quantem_gpu_metal_backend() -> None:
    from quantem.gpu.compute.backends import MetalRawBackend

    source = inspect.getsource(MetalRawBackend.masked_sum)

    assert "fast_vi" in source
    assert "_masked_sum_with_total_cache" in source
    assert "_total_cache" in source
    assert "TorchBackend" not in source


def test_show4dstem_webgpu_engine_has_selected_index_vi_kernel() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "js" / "engine" / "compute.ts").read_text(encoding="utf-8")
    frontend = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert "const maskedSumSrc = (sg: boolean)" in source
    assert "export function buildDetectorMask" in source
    assert "export function buildFullDetectorMask" in source
    assert "export function buildScanMask" in source
    assert "arrayLength(&idx)" in source
    assert "sampleF(base + idx[j]" in source
    assert "maskedSumBuffer(mask: Uint32Array)" in source
    assert "maskedDpcBuffer(mask: Uint32Array" in source
    assert "getDevice(): GPUDevice" in source
    assert "readFloatBuffer(buf: GPUBuffer" in source
    assert "const DPC_MEAN_WGSL" in source
    assert "const DPC_COMPONENT_WGSL" in source
    assert "enable subgroups;" in source
    assert "adoptBuffer(idx: number, buffer: GPUBuffer" in (
        repo / "js" / "colormaps.ts"
    ).read_text(encoding="utf-8")
    assert "renderSlotDirectWithGpuRangeToCanvas" in (
        repo / "js" / "colormaps.ts"
    ).read_text(encoding="utf-8")
    assert "function buildDetectorMask" not in frontend
    assert "function buildScanMask" not in frontend
    assert "buildFullDetectorMask" in frontend
    assert "maskedDpc" in frontend
    assert "roiBufferOnly" in frontend
    assert "dpcBufferOnly" in frontend
    assert "warmStandardViCache" in frontend
    assert "warmCache: () => warmCacheSummary()" in frontend
    assert "suppressViTraitRecompute" in frontend
    assert '"launch_warm_cache"' in frontend
    assert "virtualGpuCanvasRef" in frontend
    assert "renderPanelSlotsDirectToCanvas" in frontend
    assert "renderSlotDirectWithGpuRangeToCanvas" in frontend
