from __future__ import annotations

from pathlib import Path


def test_widget_backend_shims_delegate_to_quantem_gpu() -> None:
    import importlib

    import quantem.gpu.compute.backend as gpu_compute_backend
    import quantem.gpu.compute.backends as gpu_compute_backends
    import quantem.gpu.compute.mps as gpu_compute_mps
    import quantem.gpu.io.backends as gpu_io_backends
    import quantem.gpu.io.bitshuffle as gpu_bitshuffle
    import quantem.gpu.io.constants as gpu_constants
    import quantem.gpu.io.save as gpu_save
    import quantem.widget.io.backends as widget_io_backends
    import quantem.widget.io.bitshuffle as widget_bitshuffle
    import quantem.widget.io.constants as widget_constants
    import quantem.widget.io.save as widget_save
    import quantem.widget.kernels.compute.backend as widget_compute_backend
    import quantem.widget.kernels.compute.backends as widget_compute_backends
    import quantem.widget.kernels.compute.mps as widget_compute_mps

    gpu_detector = importlib.import_module("quantem.gpu.detector")
    gpu_dpc = importlib.import_module("quantem.gpu.dpc")
    widget_backend = importlib.import_module("quantem.widget.backend")
    widget_detector = importlib.import_module("quantem.widget.detector")
    widget_dpc = importlib.import_module("quantem.widget.dpc")

    assert widget_backend.detect_backend is gpu_io_backends.detect_backend
    assert widget_io_backends.resolve_backend is gpu_io_backends.resolve_backend
    assert widget_detector.bf is gpu_detector.bf
    assert widget_detector.virtual_image is gpu_detector.virtual_image
    assert widget_dpc.dpc is gpu_dpc.dpc
    assert widget_dpc.center_of_mass is gpu_dpc.center_of_mass
    assert widget_compute_backend.ComputeBackend is gpu_compute_backend.ComputeBackend
    assert widget_compute_backends.compute_backend is gpu_compute_backends.compute_backend
    assert widget_compute_mps.ChunkedFrames is gpu_compute_mps.ChunkedFrames
    assert widget_constants.BLOCK_SIZE == gpu_constants.BLOCK_SIZE
    assert widget_save.save is gpu_save.save
    assert widget_bitshuffle.__getattr__ is gpu_bitshuffle.__getattr__


def test_widget_webgpu_sources_are_generated_from_quantem_gpu() -> None:
    repo = Path(__file__).resolve().parents[1]

    tracked_engine_sources = sorted(
        path.name for path in (repo / "js" / "engine").glob("*.ts")
    )
    assert tracked_engine_sources == []

    sync_script = (repo / "scripts" / "sync-gpu-webgpu.mjs").read_text(
        encoding="utf-8"
    )
    build_script = (repo / "scripts" / "build.mjs").read_text(encoding="utf-8")
    show4dstem = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )
    showptycho = (repo / "js" / "showptycho" / "index.tsx").read_text(
        encoding="utf-8"
    )
    web_store = (repo / "web" / "src" / "local" / "store.ts").read_text(
        encoding="utf-8"
    )
    web_app = (repo / "web" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert 'targetDir = "js/.generated/engine"' in sync_script
    assert "syncGpuWebgpuSources()" in build_script
    assert "../.generated/engine/bslz4" in show4dstem
    assert "../.generated/engine/local-h5" in show4dstem
    assert "../.generated/engine/showptycho-ssb" in showptycho
    assert "../../../js/.generated/engine/h5reader" in web_store
    assert "../../js/.generated/engine/compute" in web_app
