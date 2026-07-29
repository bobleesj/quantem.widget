from __future__ import annotations

import ast
from pathlib import Path


def test_widget_has_no_duplicate_gpu_backend_modules() -> None:
    import quantem.gpu.io.backends as gpu_io_backends
    import quantem.widget as widget
    import quantem.widget.io as widget_io

    repo = Path(__file__).resolve().parents[1]
    widget_package = repo / "src" / "quantem" / "widget"
    stale_modules = [
        "backend.py",
        "detector.py",
        "dpc.py",
        "io/backends",
        "io/bitshuffle.py",
        "io/constants.py",
        "io/hdf5.py",
        "io/save.py",
        "kernels/compute",
        "kernels/io",
    ]

    stale_files = []
    for relative in stale_modules:
        path = widget_package / relative
        if path.is_file() or (path.is_dir() and any(path.rglob("*.py"))):
            stale_files.append(relative)
    assert stale_files == []
    assert not hasattr(widget, "detect_backend")
    assert not hasattr(widget, "resolve_backend")
    assert widget_io.detect_backend is gpu_io_backends.detect_backend
    assert widget_io.resolve_backend is gpu_io_backends.resolve_backend


def test_widget_source_imports_compute_from_quantem_gpu() -> None:
    repo = Path(__file__).resolve().parents[1]
    widget_package = repo / "src" / "quantem" / "widget"
    stale_imports = (
        "quantem.widget.backend",
        "quantem.widget.detector",
        "quantem.widget.dpc",
        "quantem.widget.io.backends",
        "quantem.widget.io.bitshuffle",
        "quantem.widget.io.constants",
        "quantem.widget.io.hdf5",
        "quantem.widget.io.save",
        "quantem.widget.kernels.compute",
        "quantem.widget.kernels.io",
    )

    offenders = []
    for path in widget_package.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported_modules = {
            name.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for name in node.names
        }
        imported_modules.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        if any(
            module == stale_import or module.startswith(f"{stale_import}.")
            for module in imported_modules
            for stale_import in stale_imports
        ):
            offenders.append(path.relative_to(widget_package).as_posix())
    assert offenders == []


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
    assert "../.generated/engine/webgpu/bslz4" in show4dstem
    assert "../.generated/engine/webgpu/local-h5" in show4dstem
    assert "../.generated/engine/ssb/compute/webgpu/backend" in showptycho
    assert "../../../js/.generated/engine/webgpu/h5reader" in web_store
    assert "../../js/.generated/engine/webgpu/compute" in web_app
