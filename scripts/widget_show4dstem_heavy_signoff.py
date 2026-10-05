#!/usr/bin/env python3
"""Run the local-only real-data Show4DSTEM heavy performance signoff.

This gate is intentionally not normal CI. It uses local 4D-STEM master files.
With ``--backend cuda`` (default) or ``mps`` it loads each master through
``quantem.gpu.io.load`` into encoded GPU storage at full detector resolution
and records load time, resident versus logical bytes, single-viewer and
comparison-viewer build time, and memory before and after release. Encoded
viewers need a live kernel, so their browser interaction belongs to the
live-Jupyter drive. With ``--backend webgpu`` it exports a standalone viewer
that decodes the masters in the browser and drives it in Chromium for FPS.
Generated reports, screenshots, and private lab paths stay under ``/tmp``
unless a maintainer explicitly asks for them.
"""

import argparse
import html
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from widget_browser_smoke import (
    _StaticServer,
    _chrome_executable,
    _free_port,
    _measure_fps,
    _visible_canvas_boxes,
)


DEFAULT_ROOTS = [
    Path("/data"),
    Path("/Volumes"),
]
REPORT_NAME = "show4dstem-heavy-signoff-report.json"
LIVE_KERNEL_NOTE = (
    "Encoded acquisitions need a live kernel and have no offline export; drive "
    "the live viewer in Jupyter for browser interaction, or rerun with "
    "--backend webgpu for the exported browser-decoded viewer."
)


def _timestamp_dir() -> Path:
    return Path("/tmp/quantem-widget-show4dstem-heavy-signoff") / time.strftime("%Y%m%d-%H%M%S")


def _env_roots() -> list[Path]:
    raw = os.environ.get("QUANTEM_WIDGET_4DSTEM_ROOTS") or os.environ.get("QUANTEM_WIDGET_REAL_DATA_ROOTS") or ""
    return [Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip()]


def _memory_snapshot(label: str) -> dict[str, Any]:
    """Host RSS and accelerator allocator state, so each phase records its memory cost."""
    snap: dict[str, Any] = {"label": label, "time": time.time()}
    try:
        import psutil

        snap["rss_mb"] = round(psutil.Process().memory_info().rss / 1024**2, 1)
    except ImportError:
        # ru_maxrss is the process peak, in kilobytes on Linux and bytes on macOS.
        scale = 1024**2 if platform.system() == "Darwin" else 1024
        snap["rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 1)
        snap["rss_source"] = "resource_ru_maxrss"
    if torch.cuda.is_available():
        snap["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        snap["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
        # Encoded acquisitions allocate outside the torch and CuPy pools; only
        # the device total sees them (it also counts other processes).
        free, total = torch.cuda.mem_get_info()
        snap["cuda_device_used_mb"] = round((total - free) / 1024**2, 1)
    if torch.backends.mps.is_available():
        snap["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1024**2, 1)
        snap["mps_driver_allocated_mb"] = round(torch.mps.driver_allocated_memory() / 1024**2, 1)
    return snap


def _roots_from_args(paths: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for root in [*paths, *_env_roots(), *DEFAULT_ROOTS]:
        root = root.expanduser()
        if root.exists() and root not in roots:
            roots.append(root)
    return roots


def _seven_tilt_dir_arg(value: str | None) -> Path | None:
    raw = (
        value
        or os.environ.get("QUANTEM_WIDGET_SHOW4DSTEM_FIXTURE_DIR", "")
        or os.environ.get("QUANTEM_WIDGET_SHOW4DSTEM_7TILT_DIR", "")
    )
    return Path(raw).expanduser() if raw.strip() else None


def _anonymous_master_labels(masters: list[Path]) -> dict[str, str]:
    return {str(path): f"tilt_{idx:02d}" for idx, path in enumerate(masters)}


def _master_label_for_report(master: Path, labels: dict[str, str]) -> str:
    return labels.get(str(master), master.name)


def _private_h5_links(
    artifact_dir: Path,
    masters: list[Path],
    labels: dict[str, str],
) -> list[str]:
    """Create anonymous H5-family symlinks for browser tests without copying data."""
    h5_dir = artifact_dir / "h5"
    h5_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    for master in masters:
        label = _master_label_for_report(master, labels)
        link = h5_dir / f"{label}_master.h5"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(master)
        source_prefix = master.name[: -len("_master.h5")]
        for data_file in sorted(master.parent.glob(f"{source_prefix}_data_*.h5")):
            data_link = h5_dir / data_file.name.replace(source_prefix, label, 1)
            if data_link.exists() or data_link.is_symlink():
                data_link.unlink()
            data_link.symlink_to(data_file.resolve())
        urls.append(f"h5/{link.name}")
    return urls


def _scrub_private_report(value: Any, labels: dict[str, str]) -> Any:
    """Replace private master paths and file names with anonymous tilt labels."""
    if not labels:
        return value
    replacements: dict[str, str] = {}
    for raw, label in labels.items():
        replacements[raw] = label
        replacements[Path(raw).name] = label
        stem = Path(raw).name
        if stem.endswith("_master.h5"):
            replacements[stem[: -len("_master.h5")]] = label
    if isinstance(value, str):
        out = value
        for raw, label in replacements.items():
            out = out.replace(raw, label)
        return out
    if isinstance(value, list):
        return [_scrub_private_report(item, labels) for item in value]
    if isinstance(value, tuple):
        return [_scrub_private_report(item, labels) for item in value]
    if isinstance(value, dict):
        return {
            _scrub_private_report(key, labels): _scrub_private_report(item, labels)
            for key, item in value.items()
        }
    return value


def _discover_real_masters(
    roots: list[Path],
    *,
    pattern: str,
    scan_size: int | None,
    limit: int,
    ready_only: bool,
) -> tuple[list[Path], list[str]]:
    from quantem.gpu.io import discover, inspect

    notes: list[str] = []
    masters: list[Path] = []
    seen: set[str] = set()
    scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
    for root in roots:
        if len(masters) >= limit:
            break
        try:
            found = discover(
                root,
                pattern=pattern,
                recursive=True,
                scan_shape=scan_shape,
                verbose=False,
            )
        except (OSError, ValueError) as exc:
            notes.append(f"{root}: discovery skipped ({str(exc)[:120]})")
            continue
        notes.append(f"{root}: discovered {len(found)} master candidate(s)")
        for item in found:
            path = Path(item).expanduser().resolve()
            key = str(path)
            if key in seen:
                continue
            if ready_only:
                try:
                    if not inspect(path).ready:
                        notes.append(f"{path.name}: not ready yet")
                        continue
                except (OSError, ValueError, KeyError) as exc:
                    notes.append(f"{path.name}: readiness check failed ({str(exc)[:120]})")
                    continue
            masters.append(path)
            seen.add(key)
            if len(masters) >= limit:
                break
    return masters, notes


def _synchronize(backend: str) -> None:
    """Wait for queued device work so a timer covers the whole load, not its launch."""
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()


def _describe_acquisition(acquisition: Any) -> dict[str, Any]:
    resident = acquisition.resident_bytes
    return {
        "shape": list(acquisition.shape),
        "dtype": str(acquisition.dtype),
        "device": str(acquisition.device),
        "representation": acquisition.representation.value,
        "logical_mib": round(acquisition.logical_bytes / 2**20, 1),
        "resident_mib": None if resident is None else round(resident / 2**20, 1),
    }


def _timed(label: str, records: list[dict[str, Any]], func):
    t0 = time.perf_counter()
    before = _memory_snapshot(f"{label}:before")
    result = func()
    after = _memory_snapshot(f"{label}:after")
    records.append(
        {
            "label": label,
            "seconds": round(time.perf_counter() - t0, 3),
            "memory_before": before,
            "memory_after": after,
        }
    )
    return result


def _timed_maybe(label: str, records: list[dict[str, Any]], func):
    """Time ``func`` and record a load failure instead of raising it."""
    t0 = time.perf_counter()
    before = _memory_snapshot(f"{label}:before")
    try:
        result = func()
    except (OSError, ValueError, RuntimeError, MemoryError) as exc:
        records.append(
            {
                "label": label,
                "seconds": round(time.perf_counter() - t0, 3),
                "memory_before": before,
                "memory_after": _memory_snapshot(f"{label}:after_error"),
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        )
        return None, exc
    after = _memory_snapshot(f"{label}:after")
    records.append(
        {
            "label": label,
            "seconds": round(time.perf_counter() - t0, 3),
            "memory_before": before,
            "memory_after": after,
        }
    )
    return result, None


def _cleanup_backend_memory(label: str, records: list[dict[str, Any]]) -> None:
    from quantem.widget import free_gpu

    before = _memory_snapshot(f"{label}:before")
    t0 = time.perf_counter()
    released_gb = float(free_gpu(verbose=True))
    records.append(
        {
            "label": label,
            "seconds": round(time.perf_counter() - t0, 3),
            "released_gb": round(released_gb, 3),
            "memory_before": before,
            "memory_after": _memory_snapshot(f"{label}:after"),
        }
    )


def _export_widget(widget: Any, artifact_dir: Path, *, encoding: str) -> dict[str, Any]:
    path = artifact_dir / f"show4dstem-real-{encoding}.html"
    t0 = time.perf_counter()
    widget.export_html(path, encoding=encoding, title="Show4DSTEM heavy signoff")
    seconds = time.perf_counter() - t0
    return {
        "widget": "show4dstem",
        "variant": f"show4dstem-real-{encoding}",
        "encoding": encoding,
        "path": str(path),
        "seconds": round(seconds, 3),
        "size_mb": round(path.stat().st_size / 1024**2, 2),
        "n_frames": int(getattr(widget, "n_frames", 1) or 1),
        "frame_dim_label": str(getattr(widget, "frame_dim_label", "Frame") or "Frame"),
    }


def _browser_gpu_info(page) -> dict[str, Any]:
    return page.evaluate(
        """async () => {
          const info = { userAgent: navigator.userAgent, webgpu: Boolean(navigator.gpu) };
          if (!navigator.gpu) return info;
          try {
            const adapter = await navigator.gpu.requestAdapter();
            info.adapter = adapter ? (adapter.info || {}) : null;
            info.adapterName = adapter?.info?.description || adapter?.info?.vendor || null;
          } catch (err) {
            info.error = String(err && err.message ? err.message : err);
          }
          return info;
        }"""
    )


def _drag_box(page, box: dict[str, float], *, steps: int = 16) -> float:
    x0 = box["x"] + box["width"] * 0.35
    y0 = box["y"] + box["height"] * 0.35
    x1 = box["x"] + box["width"] * 0.68
    y1 = box["y"] + box["height"] * 0.64
    t0 = time.perf_counter()
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=steps)
    page.mouse.up()
    page.wait_for_timeout(120)
    return round((time.perf_counter() - t0) * 1000, 1)


def _dataset_slider_box(page) -> dict[str, float] | None:
    return page.evaluate(
        """() => {
          const visible = (el) => {
            const rect = el.getBoundingClientRect();
            const style = getComputedStyle(el);
            return rect.width > 50 && rect.height > 6 && style.visibility !== 'hidden' && style.display !== 'none';
          };
          const candidates = [];
          for (const root of [...document.querySelectorAll('.MuiSlider-root')]) {
            if (!visible(root)) continue;
            const thumbs = [...root.querySelectorAll('.MuiSlider-thumb')];
            if (thumbs.length !== 1) continue;
            const inputs = [...root.querySelectorAll('input')];
            const maxVals = inputs
              .map(input => Number(input.getAttribute('aria-valuemax') || input.max || '0'))
              .filter(Number.isFinite);
            const max = maxVals.length ? Math.max(...maxVals) : 0;
            if (max < 1) continue;
            const host = root.closest('.MuiBox-root') || root.parentElement;
            const text = (host?.innerText || '').toLowerCase();
            const score =
              (text.includes('dataset') ? 5 : 0) +
              (text.includes('frame') ? 3 : 0) +
              (text.includes('tilt') ? 3 : 0) +
              (text.includes('time') ? 3 : 0) +
              (text.includes('/') ? 1 : 0);
            const rect = root.getBoundingClientRect();
            candidates.push({x: rect.x, y: rect.y, width: rect.width, height: rect.height, max, score});
          }
          if (!candidates.length) return null;
          candidates.sort((a, b) => (b.score - a.score) || (b.y - a.y) || (b.max - a.max));
          return candidates[0];
        }"""
    )


def _drag_dataset_slider(page) -> dict[str, Any]:
    box = _dataset_slider_box(page)
    if not box:
        return {"found": False, "drag_ms": 0.0}
    y = box["y"] + box["height"] / 2
    t0 = time.perf_counter()
    page.mouse.move(box["x"] + box["width"] * 0.1, y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.9, y, steps=14)
    page.mouse.up()
    page.wait_for_timeout(180)
    return {
        "found": True,
        "drag_ms": round((time.perf_counter() - t0) * 1000, 1),
        "slider_max": int(box.get("max") or 0),
    }


def _drive_browser_export(
    artifact_dir: Path,
    export: dict[str, Any],
    *,
    min_fps: float,
    timeout_ms: int,
    headed: bool,
) -> dict[str, Any]:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is required for Show4DSTEM browser signoff") from exc

    port = _free_port()
    chrome = _chrome_executable()
    launch_kwargs: dict[str, Any] = {
        "headless": not headed,
        "args": [
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-search-engine-choice-screen",
        ],
    }
    if chrome is not None:
        launch_kwargs["executable_path"] = chrome

    results: dict[str, Any] = {
        "file": Path(str(export["path"])).name,
        "errors": [],
        "passed": False,
        "min_fps": float(min_fps),
        "n_frames": int(export.get("n_frames", 1) or 1),
    }
    screenshot = artifact_dir / "show4dstem-browser-signoff.png"
    with _StaticServer(artifact_dir, port) as base_url:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1050})
                console_errors: list[str] = []
                console_warnings: list[str] = []
                page_errors: list[str] = []
                bad_responses: list[dict[str, Any]] = []
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                page.on(
                    "response",
                    lambda response: bad_responses.append(
                        {"status": int(response.status), "url": response.url}
                    )
                    if response.status >= 400 and not response.url.endswith("/favicon.ico")
                    else None,
                )

                def _handle_console(msg) -> None:
                    if msg.type != "error" or "Failed to load resource:" in msg.text:
                        return
                    if "Unable to preventDefault inside passive event listener invocation." in msg.text:
                        console_warnings.append(msg.text)
                        return
                    console_errors.append(msg.text)

                page.on("console", _handle_console)
                page.goto(f"{base_url}/{Path(str(export['path'])).name}", wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_function("document.querySelectorAll('canvas').length >= 2", timeout=timeout_ms)
                page.wait_for_timeout(1000)
                results["browser_gpu"] = _browser_gpu_info(page)
                boxes = _visible_canvas_boxes(page)
                results["canvas_count"] = len(boxes)
                if len(boxes) < 2:
                    results["errors"].append("expected at least two canvases for diffraction + virtual image")
                else:
                    ranked = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)
                    virtual_box = ranked[0]
                    detector_box = ranked[1]
                    results["initial_fps"] = round(float(_measure_fps(page, 1200)), 1)
                    results["scan_position_drag_ms"] = _drag_box(page, virtual_box)
                    results["scan_position_fps"] = round(float(_measure_fps(page, 1200)), 1)
                    t0 = time.perf_counter()
                    results["detector_drag_ms"] = _drag_box(page, detector_box)
                    results["virtual_detector_recompute_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    results["detector_drag_fps"] = round(float(_measure_fps(page, 1200)), 1)
                    page.mouse.wheel(0, -400)
                    page.wait_for_timeout(140)
                    results["wheel_zoom_fps"] = round(float(_measure_fps(page, 1200)), 1)
                    if int(export.get("n_frames", 1) or 1) > 1:
                        flip = _drag_dataset_slider(page)
                        results["dataset_flip"] = flip
                        if flip.get("found"):
                            results["dataset_flip_fps"] = round(float(_measure_fps(page, 1200)), 1)
                        else:
                            results["errors"].append("dataset/frame slider not found for multi-frame export")
                    else:
                        results["dataset_flip"] = {"found": False, "skipped": "single frame export"}
                page.screenshot(path=str(screenshot), full_page=True, timeout=timeout_ms)
                results["screenshot"] = screenshot.name
                results["console_errors"] = console_errors
                results["console_warnings"] = console_warnings
                results["page_errors"] = page_errors
                results["bad_responses"] = bad_responses[:100]
                results["errors"].extend(page_errors)
                results["errors"].extend(console_errors)
                results["errors"].extend(
                    f"HTTP {item['status']} {item['url']}" for item in bad_responses[:20]
                )
                body_text = page.locator("body").inner_text(timeout=timeout_ms).lower()
                if "show4dstem load failed" in body_text or "load failed" in body_text:
                    results["errors"].append("Show4DSTEM load failed text is visible in browser")
            except PlaywrightError as exc:
                results["errors"].append(f"browser drive failed: {exc}")
            finally:
                browser.close()

    for key in ["initial_fps", "scan_position_fps", "detector_drag_fps", "wheel_zoom_fps", "dataset_flip_fps"]:
        if key not in results:
            continue
        value = float(results.get(key, 0) or 0)
        if value < min_fps:
            results["errors"].append(f"{key} {value:.1f} below {min_fps:.1f}")
    results["passed"] = not results["errors"]
    return results


def _run_native(
    *,
    backend: str,
    masters: list[Path],
    master_labels: dict[str, str],
    timing: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    """Load each master into encoded GPU storage, then open single and comparison viewers."""
    from quantem.gpu.io import load
    from quantem.widget import Show4DSTEM

    acquisitions = []
    loads: list[dict[str, Any]] = []
    for master in masters:
        label = _master_label_for_report(master, master_labels)

        def load_master():
            acquisition = load(str(master), backend=backend, verbose=True)
            _synchronize(backend)
            return acquisition

        acquisition, error = _timed_maybe(f"load:{label}", timing, load_master)
        if error is not None:
            errors.append(f"load {label} failed: {error}")
            break
        acquisitions.append(acquisition)
        loads.append({"master": label, "seconds": timing[-1]["seconds"], **_describe_acquisition(acquisition)})

    viewers: list[dict[str, Any]] = []
    if acquisitions:
        single = _timed(
            "build_single_viewer",
            timing,
            lambda: Show4DSTEM(
                acquisitions[0],
                title="Show4DSTEM heavy signoff",
                save_state=False,
                show_controls=True,
            ),
        )
        viewers.append({"kind": "single", "seconds": timing[-1]["seconds"], "n_frames": int(single.n_frames)})
        single.close()
    if len(acquisitions) > 1:
        comparison = _timed(
            "build_comparison_viewer",
            timing,
            lambda: Show4DSTEM(
                acquisitions,
                title="Show4DSTEM heavy signoff",
                save_state=False,
                show_controls=True,
            ),
        )
        viewers.append(
            {
                "kind": "comparison",
                "seconds": timing[-1]["seconds"],
                "n_frames": int(comparison.n_frames),
                "frame_labels": list(comparison.frame_labels),
                "compare_panel_indices": list(comparison.compare_panel_indices),
            }
        )
        comparison.close()

    resident = [item["resident_mib"] for item in loads if item["resident_mib"] is not None]
    logical = [item["logical_mib"] for item in loads]
    before_release = _memory_snapshot("acquisitions:before_close")
    for acquisition in acquisitions:
        acquisition.close()
    return {
        "loads": loads,
        "viewers": viewers,
        "residency": {
            "acquisitions": len(acquisitions),
            "resident_mib": round(sum(resident), 1),
            "logical_mib": round(sum(logical), 1),
            "logical_per_resident": round(sum(logical) / sum(resident), 1) if resident else None,
            "memory_before_close": before_release,
            "memory_after_close": _memory_snapshot("acquisitions:after_close"),
        },
    }


def _run_webgpu(
    args: argparse.Namespace,
    *,
    artifact_dir: Path,
    masters: list[Path],
    master_labels: dict[str, str],
    timing: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    """Export a viewer that decodes the masters in the browser, then drive it."""
    import numpy as np

    from quantem.widget import Show4DSTEM
    from quantem.widget.show4dstem_factory import _master_file_contract

    contract = _master_file_contract(masters[0])
    h5_urls = _private_h5_links(artifact_dir, masters, master_labels)
    labels = [_master_label_for_report(master, master_labels) for master in masters]
    widget = _timed(
        "build_show4dstem_webgpu_h5_viewer",
        timing,
        lambda: Show4DSTEM(
            np.zeros((1, 1, 1, 1), dtype=np.uint8),
            h5_urls=h5_urls,
            backend="webgpu",
            scan_shape=tuple(int(value) for value in contract["scan_shape"]),
            detector_shape=tuple(int(value) for value in contract["detector_shape"]),
            frame_dim_label="Dataset",
            frame_labels=labels,
            title="Show4DSTEM heavy signoff",
            save_state=False,
            verbose=False,
            show_controls=True,
            debug=True,
        ),
    )
    export = _timed(
        f"export_html_{args.encoding}",
        timing,
        lambda: _export_widget(widget, artifact_dir, encoding=args.encoding),
    )
    widget.close()
    browser = None
    skipped: list[str] = []
    if args.skip_browser:
        skipped.append("browser checks skipped by request; export only, not UI signoff")
    else:
        try:
            browser = _drive_browser_export(
                artifact_dir,
                export,
                min_fps=args.min_fps,
                timeout_ms=args.timeout_ms,
                headed=args.headed,
            )
        except RuntimeError as exc:
            browser = {"passed": False, "errors": [str(exc)]}
        errors.extend(browser.get("errors", []))
    return {
        "residency": {
            "type": "WebGPUH5Source",
            "shape": [len(masters), *contract["scan_shape"], *contract["detector_shape"]],
            "h5_urls": h5_urls,
        },
        "exports": [export],
        "browser": browser,
        "skipped": skipped,
    }


def _write_index(artifact_dir: Path, report: dict[str, Any]) -> None:
    report_json = html.escape(json.dumps(report, indent=2))
    exports = "\n".join(
        f"<li><a href='{html.escape(Path(item['path']).name)}'>{html.escape(item['variant'])}</a> "
        f"({item['size_mb']:.2f} MB, {item['seconds']:.2f}s)</li>"
        for item in report.get("exports", [])
    )
    browser = report.get("browser") or {}
    screenshot = browser.get("screenshot")
    shot_html = f"<p><a href='{html.escape(screenshot)}'>Browser screenshot</a></p>" if screenshot else ""
    targets = report.get("targets", {})
    residency = report.get("residency", {})
    target_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in [
            ("backend", targets.get("backend", "")),
            ("requested_master_count", targets.get("requested_master_count", "")),
            ("max_successful_masters", targets.get("max_successful_masters", "")),
            ("resident_mib", residency.get("resident_mib", "")),
            ("logical_mib", residency.get("logical_mib", "")),
            ("encoding", targets.get("encoding", "")),
            ("min_fps", targets.get("min_fps", "")),
        ]
    )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Show4DSTEM heavy performance signoff</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #cbd5df; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f5f7; }}
    pre {{ background: #f5f7f9; padding: 12px; overflow: auto; max-width: 1180px; }}
    .warn {{ border-left: 4px solid #b54708; padding: 8px 12px; background: #fff7ed; }}
  </style>
</head>
<body>
  <h1>Show4DSTEM heavy performance signoff</h1>
  <p class="warn">Local-only real-data report. Do not commit private data paths,
  generated HTML, screenshots, or timing JSON unless explicitly approved.</p>
  <p>Result: <strong>{'PASS' if report['passed'] else 'FAIL'}</strong></p>
  <h2>Targets</h2>
  <table><tbody>{target_rows}</tbody></table>
  <h2>Exports</h2>
  <ul>{exports}</ul>
  {shot_html}
  <h2>Machine-readable report</h2>
  <p><a href="{REPORT_NAME}">{REPORT_NAME}</a></p>
  <pre>{report_json}</pre>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(page, encoding="utf-8")


def _write_reports(artifact_dir: Path, report: dict[str, Any], master_labels: dict[str, str]) -> dict[str, Any]:
    report = _scrub_private_report(report, master_labels)
    (artifact_dir / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_index(artifact_dir, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--seven-tilt", action="store_true", help="Use the private seven-tilt local-data gate and anonymize data labels.")
    parser.add_argument("--seven-tilt-dir", default="", help="Private seven-tilt folder; alternatively set QUANTEM_WIDGET_SHOW4DSTEM_7TILT_DIR.")
    parser.add_argument("--pattern", default="*_master.h5")
    parser.add_argument("--scan-size", type=int, default=None, help="Keep only masters with this square scan size.")
    parser.add_argument("--max-masters", type=int, default=None)
    parser.add_argument("--backend", choices=["cuda", "mps", "webgpu", "auto"], default="cuda")
    parser.add_argument("--encoding", choices=["uint8", "uint16"], default="uint8", help="Count encoding of the --backend webgpu export.")
    parser.add_argument("--min-fps", type=float, default=30.0)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--skip-browser", action="store_true", help="With --backend webgpu, export only; do not claim UI performance signoff.")
    parser.add_argument("--allow-unready", action="store_true", help="Include discovered masters even if readiness checks fail.")
    parser.add_argument("--quick", action="store_true", help="Use one master for script iteration.")
    parser.add_argument("--no-free-gpu-before", action="store_true", help="Do not clear Torch/CuPy/MPS allocator caches before loading.")
    parser.add_argument("--no-free-gpu-after", action="store_true", help="Do not clear Torch/CuPy/MPS allocator caches before exiting.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    artifact_dir = (args.artifact_dir or _timestamp_dir()).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    seven_tilt_dir = _seven_tilt_dir_arg(args.seven_tilt_dir)
    private_seven_tilt = bool(args.seven_tilt or seven_tilt_dir is not None)
    max_masters = (
        1
        if args.quick
        else max(1, int(args.max_masters))
        if args.max_masters is not None
        else 7
        if private_seven_tilt
        else 2
    )
    roots = [seven_tilt_dir] if seven_tilt_dir is not None else _roots_from_args(args.search_root)
    masters, discovery_notes = _discover_real_masters(
        roots,
        pattern=args.pattern,
        scan_size=args.scan_size,
        limit=max_masters,
        ready_only=not args.allow_unready,
    )
    report: dict[str, Any] = {
        "passed": False,
        "local_only": True,
        "normal_ci": False,
        "artifact_dir": str(artifact_dir),
    }
    if not masters:
        report.update(
            reason="no real 4D-STEM master files found",
            search_roots=["private-seven-tilt-dir"] if private_seven_tilt else [str(path) for path in roots],
            discovery_notes=(
                ["private seven-tilt discovery found no usable masters"]
                if private_seven_tilt
                else discovery_notes
            ),
        )
        _write_reports(artifact_dir, report, {})
        print(f"No real Show4DSTEM masters found. Report: {artifact_dir / 'index.html'}")
        return 2

    from quantem.gpu.device import resolve

    timing: list[dict[str, Any]] = []
    cleanup_records: list[dict[str, Any]] = []
    errors: list[str] = []
    backend = resolve(args.backend)
    master_labels = _anonymous_master_labels(masters) if private_seven_tilt else {}
    if not args.no_free_gpu_before:
        _cleanup_backend_memory("free_gpu_before", cleanup_records)
    if backend == "webgpu":
        outcome = _run_webgpu(
            args,
            artifact_dir=artifact_dir,
            masters=masters,
            master_labels=master_labels,
            timing=timing,
            errors=errors,
        )
        max_successful = len(masters) if not errors else 0
    else:
        outcome = _run_native(
            backend=backend,
            masters=masters,
            master_labels=master_labels,
            timing=timing,
            errors=errors,
        )
        outcome.update(exports=[], browser=None, skipped=[LIVE_KERNEL_NOTE])
        max_successful = len(outcome["loads"])
    if not args.no_free_gpu_after:
        _cleanup_backend_memory("free_gpu_after", cleanup_records)

    report.update(
        passed=not errors,
        repo=str(root),
        commit=subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        host={
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        policy={
            "real_data_not_committed": True,
            "normal_ci_excluded": True,
            "browser_and_backend_timings_are_separate": True,
            "browser_required": backend == "webgpu" and not args.skip_browser,
            "private_data_labels_anonymized": private_seven_tilt,
            "full_detector_no_downsample": True,
        },
        targets={
            "masters": [_master_label_for_report(master, master_labels) for master in masters],
            "requested_master_count": len(masters),
            "max_successful_masters": max_successful,
            "backend": backend,
            "encoding": args.encoding if backend == "webgpu" else None,
            "min_fps": args.min_fps,
        },
        discovery_notes=(
            [f"private seven-tilt folder: discovered {len(masters)} master(s)"]
            if private_seven_tilt
            else discovery_notes
        ),
        timing=timing,
        cleanup=cleanup_records,
        **outcome,
        memory_final=_memory_snapshot("final"),
        errors=errors,
    )
    report = _write_reports(artifact_dir, report, master_labels)
    print(f"Show4DSTEM heavy signoff report: {artifact_dir / 'index.html'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
