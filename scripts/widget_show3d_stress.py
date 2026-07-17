#!/usr/bin/env python3
"""Stress-test Show3D exported HTML and folder exports in Chromium.

This is a local-only maintainer gate for real Show3D reports. It can open
existing exact single-file HTML exports, serve Show3D folder exports with HTTP
Range support, and generate a temporary sidecar export from an existing
standalone Show3D HTML so the same dataset is exercised through both browser
paths.
"""

from __future__ import annotations

import argparse
import base64
import html
import http.server
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from playwright.sync_api import sync_playwright

from serve_sidecar_range import RangeRequestHandler
from widget_browser_smoke import (
    _canvas_layout_summary,
    _chrome_executable,
    _exercise_column_select,
    _exercise_fft_toggle,
    _free_port,
    _image_nonblank,
    _measure_fps,
    _visible_canvas_boxes,
)


STATE_SCRIPT_RE = re.compile(
    r'<script[^>]+type=["\']application/vnd\.jupyter\.widget-state\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL,
)

VIEWPORTS = {
    "desktop": {"width": 1500, "height": 1000},
    "wide": {"width": 2200, "height": 1200},
    "narrow": {"width": 900, "height": 900},
}


@dataclass
class TargetSpec:
    name: str
    mode: str
    source: str
    url: str | None = None
    root: Path | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RangeServer:
    root: Path
    port: int
    httpd: http.server.ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None

    def __enter__(self) -> str:
        handler = type(
            "ConfiguredRangeRequestHandler",
            (RangeRequestHandler,),
            {
                "root": self.root,
                "log_message": lambda self, format, *args: None,  # noqa: A002
            },
        )
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_artifact_dir() -> Path:
    return Path("/tmp") / "quantem-widget-show3d-stress" / _timestamp()


def _safe_name(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value)
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean or "show3d"


def _escape(value: object) -> str:
    return html.escape(str(value))


def _extract_widget_state(html_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    text = html_path.read_text(encoding="utf-8")
    match = STATE_SCRIPT_RE.search(text)
    if match is None:
        raise ValueError(f"{html_path} does not contain embedded ipywidgets state")
    state = json.loads(match.group(1))
    for model in state.get("state", {}).values():
        traits = model.get("state", {})
        if traits.get("_view_name") == "AnyView" and "n_slices" in traits and "n_panels" in traits:
            return traits, model
        if "_esm" in traits and "n_slices" in traits and "n_panels" in traits:
            return traits, model
    raise ValueError(f"{html_path} does not look like a Show3D export")


def _buffer_by_trait(model: dict[str, Any], trait: str) -> bytes:
    for item in model.get("buffers", []) or []:
        if item.get("path") == [trait]:
            data = item.get("data") or ""
            return base64.b64decode(data) if data else b""
    value = model.get("state", {}).get(trait)
    if isinstance(value, str) and value:
        try:
            return base64.b64decode(value)
        except Exception:
            return b""
    return b""


def _show3d_metadata_from_html(html_path: Path) -> dict[str, Any]:
    traits, model = _extract_widget_state(html_path)
    buffer_sizes = {
        "_offline_float_stack": len(_buffer_by_trait(model, "_offline_float_stack")),
        "_offline_stack": len(_buffer_by_trait(model, "_offline_stack")),
    }
    return {
        "path": str(html_path),
        "bytes": html_path.stat().st_size,
        "title": traits.get("title", ""),
        "width": int(traits.get("width", 0) or 0),
        "height": int(traits.get("height", 0) or 0),
        "n_slices": int(traits.get("n_slices", 0) or 0),
        "n_panels": int(traits.get("n_panels", 0) or 0),
        "panel_width_px": int(traits.get("panel_width_px", 0) or 0),
        "offline_stack_url": traits.get("_offline_stack_url", ""),
        "buffer_sizes": buffer_sizes,
    }


def _decode_show3d_stack(html_path: Path, *, max_decode_mb: float) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    traits, model = _extract_widget_state(html_path)
    n_slices = int(traits.get("n_slices", 0) or 0)
    height = int(traits.get("height", 0) or 0)
    width = int(traits.get("width", 0) or 0)
    n_panels = int(traits.get("n_panels", 1) or 1)
    if min(n_slices, height, width, n_panels) <= 0:
        raise ValueError(
            f"Show3D export has invalid shape n_slices={n_slices}, height={height}, "
            f"width={width}, n_panels={n_panels}"
        )

    raw = _buffer_by_trait(model, "_offline_float_stack")
    encoding = "float32"
    channels = 1
    if not raw:
        raw = _buffer_by_trait(model, "_offline_stack")
        encoding = "uint8"
    if not raw:
        raise ValueError(f"{html_path} has no embedded offline stack to convert into a sidecar")
    raw_mb = len(raw) / 1024 / 1024
    if raw_mb > float(max_decode_mb):
        raise ValueError(
            f"Refusing to decode {raw_mb:.1f} MB from {html_path}; "
            f"raise --max-decode-mb if this local stress run has enough RAM."
        )

    expected = n_slices * height * width
    if encoding == "float32":
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.size == expected * 3:
            channels = 3
            stack = arr.reshape(n_slices, height, width, 3)
        elif arr.size == expected:
            stack = arr.reshape(n_slices, height, width)
        else:
            raise ValueError(f"float32 offline stack has {arr.size} values, expected {expected} or {expected * 3}")
    else:
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size == expected * 3:
            channels = 3
            stack = arr.reshape(n_slices, height, width, 3).astype(np.float32) / 255.0
        elif arr.size == expected:
            stack = arr.reshape(n_slices, height, width).astype(np.float32)
        else:
            raise ValueError(f"uint8 offline stack has {arr.size} values, expected {expected} or {expected * 3}")

    metadata = {
        "path": str(html_path),
        "source_encoding": encoding,
        "decoded_mb": round(raw_mb, 3),
        "n_slices": n_slices,
        "height": height,
        "width": width,
        "n_panels": n_panels,
        "panel_width_px": int(traits.get("panel_width_px", 0) or 0),
        "channels": channels,
        "title": traits.get("title", ""),
    }
    return stack, traits, metadata


def _generate_sidecar_from_html(html_path: Path, out_dir: Path, *, max_decode_mb: float) -> TargetSpec:
    from quantem.widget import Show3D

    stack, traits, metadata = _decode_show3d_stack(html_path, max_decode_mb=max_decode_mb)
    panel_count = int(metadata["n_panels"])
    panel_width = int(metadata["panel_width_px"]) or int(metadata["width"]) // panel_count
    if panel_width <= 0 or panel_width * panel_count > int(metadata["width"]):
        raise ValueError(f"cannot split panels from width={metadata['width']} and n_panels={panel_count}")

    panels = []
    for panel in range(panel_count):
        start = panel * panel_width
        stop = start + panel_width
        panels.append(np.ascontiguousarray(stack[:, :, start:stop, ...] if stack.ndim == 4 else stack[:, :, start:stop]))

    titles = traits.get("panel_titles") or [f"Panel {idx + 1}" for idx in range(panel_count)]
    out_dir.mkdir(parents=True, exist_ok=True)
    widget = Show3D(
        *panels,
        title=str(traits.get("title") or html_path.stem),
        panel_titles=[str(title) for title in titles[:panel_count]],
        cmap=str(traits.get("cmap") or "gray"),
        max_cols=int(traits.get("max_cols", min(4, panel_count)) or min(4, panel_count)),
        fps=float(traits.get("fps", 30.0) or 30.0),
        avg_window=int(traits.get("avg_window", 1) or 1),
        sampling=float(traits.get("pixel_size", 0.0) or 0.0),
        units=str(traits.get("pixel_unit") or "A"),
        show_fft=bool(traits.get("show_fft", False)),
        fft_layout=str(traits.get("fft_layout") or "bottom"),
        fft_overlay_position=str(traits.get("fft_overlay_position") or "top-left"),
        fft_overlay_size=float(traits.get("fft_overlay_size", 0.35) or 0.35),
        show_controls=True,
        show_zoom_indicator=bool(traits.get("show_zoom_indicator", False)),
        show_scale_bar=bool(traits.get("scale_bar_visible", True)),
        debug=True,
        inter_panel_gap_px=int(traits.get("inter_panel_gap_px", traits.get("panel_gap", 0)) or 0),
        inter_panel_gap_color=str(traits.get("inter_panel_gap_color") or ""),
        gallery_outer_border_px=int(traits.get("gallery_outer_border_px", 0) or 0),
        gallery_outer_border_color=str(traits.get("gallery_outer_border_color") or ""),
        panel_inner_border_px=float(traits.get("panel_inner_border_px", 0.0) or 0.0),
        panel_inner_border_color=str(traits.get("panel_inner_border_color") or "#000000"),
        verbose=False,
    )
    try:
        html_path_out = widget.export_sidecar(out_dir, title=str(traits.get("title") or html_path.stem))
    finally:
        widget.free()

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    metadata.update({
        "sidecar_dir": str(out_dir),
        "sidecar_html": str(html_path_out),
        "sidecar_stack_bytes": int((out_dir / "offline_stack.u8").stat().st_size),
        "manifest": manifest,
    })
    return TargetSpec(
        name=f"{_safe_name(html_path.stem)}-generated-sidecar",
        mode="sidecar",
        source=str(html_path),
        root=out_dir.resolve(),
        metadata=metadata,
    )


def _read_debug(page) -> dict[str, Any]:
    return page.evaluate(
        """() => {
          const raw = window.__quantemShow3DPerf ||
            document.documentElement.__quantemShow3DPerf || {};
          const out = {};
          for (const key of Object.keys(raw)) {
            const value = raw[key];
            if (value == null || ["string", "number", "boolean"].includes(typeof value)) {
              out[key] = value;
            } else if (Array.isArray(value)) {
              out[key] = value.slice(-12);
            } else if (typeof value === "object") {
              try { out[key] = JSON.parse(JSON.stringify(value)); } catch (_) {}
            }
          }
          return out;
        }"""
    )


def _click_button(page, labels: list[str]) -> bool:
    return bool(
        page.evaluate(
            """(labels) => {
              const wanted = labels.map((label) => String(label).toLowerCase());
              const visible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 &&
                  style.display !== "none" && style.visibility !== "hidden" &&
                  Number(style.opacity || "1") > 0.05;
              };
              for (const node of [...document.querySelectorAll("button,[role='button']")]) {
                if (!visible(node)) continue;
                const text = (node.textContent || "").trim().toLowerCase();
                const aria = (node.getAttribute("aria-label") || "").trim().toLowerCase();
                const title = (node.getAttribute("title") || "").trim().toLowerCase();
                if (wanted.some((label) => text === label || aria === label || title === label ||
                    aria.includes(label) || title.includes(label))) {
                  node.click();
                  return true;
                }
              }
              return false;
            }""",
            labels,
        )
    )


def _set_labeled_switch(page, label: str, checked: bool) -> dict[str, Any]:
    """Set a compact MUI switch by its nearby text label."""
    return page.evaluate(
        """({label, checked}) => {
          const wanted = String(label).trim().toLowerCase();
          const visible = (node) => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return rect.width > 0 && rect.height > 0 &&
              style.display !== "none" && style.visibility !== "hidden";
          };
          const labels = [...document.querySelectorAll("span,div,label,p")]
            .filter((node) => (node.textContent || "").trim().toLowerCase() === wanted && visible(node));
          for (const labelNode of labels) {
            const lr = labelNode.getBoundingClientRect();
            const lc = {x: lr.x + lr.width / 2, y: lr.y + lr.height / 2};
            const candidates = [...document.querySelectorAll('input[type="checkbox"]')]
              .map((input) => {
                const host = input.closest(".MuiSwitch-root") || input.closest("label") || input.parentElement || input;
                const r = host.getBoundingClientRect();
                const cx = r.x + r.width / 2;
                const cy = r.y + r.height / 2;
                const dx = Math.abs(cx - lc.x);
                const dy = Math.abs(cy - lc.y);
                return {
                  input,
                  host,
                  score: dy * 8 + dx,
                  dx,
                  dy,
                };
              })
              .filter((item) => visible(item.host) && item.dy <= 32 && item.dx <= 220)
              .sort((a, b) => a.score - b.score);
            if (!candidates.length) continue;
            const input = candidates[0].input;
            const before = Boolean(input.checked);
            if (before !== checked) input.click();
            return {found: true, before, after: Boolean(input.checked)};
          }
          return {found: false, before: null, after: null};
        }""",
        {"label": label, "checked": checked},
    )


def _set_labeled_switch_with_retry(
    page,
    label: str,
    checked: bool,
    *,
    attempts: int = 20,
    interval_ms: int = 200,
) -> dict[str, Any]:
    """Set a compact MUI switch after the exported widget has mounted."""
    last: dict[str, Any] = {"found": False, "before": None, "after": None}
    for attempt in range(1, attempts + 1):
        last = _set_labeled_switch(page, label, checked)
        last["attempts"] = attempt
        if last.get("found") and last.get("after") == checked:
            return last
        page.wait_for_timeout(interval_ms)
    return last


def _first_paint(page, *, timeout_ms: int) -> dict[str, Any]:
    start = time.perf_counter()
    last: dict[str, Any] = {}
    while (time.perf_counter() - start) * 1000 < timeout_ms:
        boxes = _visible_canvas_boxes(page)
        if boxes:
            primary = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)[0]
            png = page.locator("canvas").nth(int(primary["index"])).screenshot()
            nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
            last = {"box": primary, "nonblank": bool(nonblank), "stats": stats}
            if nonblank:
                last["first_paint_ms"] = round((time.perf_counter() - start) * 1000, 1)
                return last
        page.wait_for_timeout(120)
    last["first_paint_ms"] = round((time.perf_counter() - start) * 1000, 1)
    last["timeout"] = True
    return last


def _save_screenshot(page, path: Path) -> dict[str, Any]:
    png = page.screenshot(path=str(path), full_page=False)
    nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
    return {
        "path": str(path),
        "rel": f"screenshots/{path.name}",
        "nonblank": bool(nonblank),
        "stats": stats,
    }


def _primary_canvas_nonblank(page) -> dict[str, Any]:
    boxes = _visible_canvas_boxes(page)
    if not boxes:
        return {"nonblank": False, "error": "no visible canvas"}
    primary = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)[0]
    png = page.locator("canvas").nth(int(primary["index"])).screenshot()
    nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
    return {"box": primary, "nonblank": bool(nonblank), "stats": stats}


def _drive_playback(page, *, wait_ms: int) -> dict[str, Any]:
    before = _read_debug(page)
    before_layout = _canvas_layout_summary(page)
    before_text = page.evaluate("document.body.innerText")
    clicked = _click_button(page, ["play", "start playback", "play animation"])
    page.wait_for_timeout(wait_ms)
    after = _read_debug(page)
    after_layout = _canvas_layout_summary(page)
    after_text = page.evaluate("document.body.innerText")
    paused = _click_button(page, ["pause", "stop playback", "pause animation", "stop"])
    page.wait_for_timeout(250)
    return {
        "clicked": clicked,
        "paused": paused,
        "debug_before": before,
        "debug_after": after,
        "text_changed": before_text != after_text,
        "layout_changed": before_layout.get("all_signature") != after_layout.get("all_signature"),
    }


def _drive_visible_canvas_region(page, box: dict[str, float]) -> None:
    """Zoom/pan inside the visible viewport even when a grid canvas is tall."""
    viewport = page.viewport_size or {"width": 1200, "height": 900}
    max_x = max(24, float(viewport["width"]) - 24)
    max_y = max(64, float(viewport["height"]) - 36)
    x = min(max(float(box["x"]) + float(box["width"]) * 0.52, 24), max_x)
    visible_top = max(float(box["y"]), 72.0)
    preferred_y = float(box["y"]) + min(float(box["height"]) * 0.42, float(viewport["height"]) * 0.55)
    y = min(max(preferred_y, visible_top), max_y)
    page.mouse.move(x, y)
    page.mouse.wheel(0, -450)
    page.wait_for_timeout(140)
    page.mouse.down()
    page.mouse.move(
        min(x + min(44, float(box["width"]) * 0.18), max_x),
        min(y + min(34, float(box["height"]) * 0.08), max_y),
        steps=10,
    )
    page.mouse.up()
    page.wait_for_timeout(180)


def _drive_zoom_pan_stress(page, *, seconds: float) -> dict[str, Any]:
    end_time = time.perf_counter() + max(0.5, seconds)
    cycles = 0
    render_paths: list[str] = []
    samples: list[dict[str, Any]] = []
    last_sample = 0.0
    while time.perf_counter() < end_time:
        boxes = _visible_canvas_boxes(page)
        if not boxes:
            break
        primary = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)[0]
        _drive_visible_canvas_region(page, primary)
        cycles += 1
        dbg = _read_debug(page)
        path = dbg.get("lastInteractionRenderPath")
        if path:
            render_paths.append(str(path))
        now = time.perf_counter()
        if now - last_sample >= 0.9:
            samples.append({
                "fps": round(float(_measure_fps(page, 250)), 1),
                "debug": dbg,
                "layout": _canvas_layout_summary(page),
            })
            last_sample = now
    return {
        "seconds": seconds,
        "cycles": cycles,
        "render_paths": sorted(set(render_paths)),
        "samples": samples,
        "final_canvas": _primary_canvas_nonblank(page),
        "debug": _read_debug(page),
    }


def _run_case(
    page,
    *,
    target: TargetSpec,
    url: str,
    viewport_name: str,
    viewport: dict[str, int],
    artifact_dir: Path,
    seconds: float,
    timeout_ms: int,
    min_fps: float,
    independent_contrast: bool,
) -> dict[str, Any]:
    case_name = _safe_name(f"{target.name}-{viewport_name}")
    screenshots_dir = artifact_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    console_messages: list[dict[str, str]] = []
    page_errors: list[str] = []
    responses: list[dict[str, Any]] = []
    page.on("console", lambda msg: console_messages.append({"type": msg.type, "text": msg.text}))
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
    page.on(
        "response",
        lambda response: responses.append({
            "url": response.url,
            "status": response.status,
            "content_range": response.headers.get("content-range"),
        })
        if "offline_stack" in response.url or response.status >= 400
        else None,
    )

    started = time.perf_counter()
    status = None
    page.set_viewport_size(viewport)
    response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    if response is not None:
        status = response.status
    contrast_step: dict[str, Any] | None = None
    if independent_contrast:
        contrast_step = _set_labeled_switch_with_retry(page, "Contrast", False)
        page.wait_for_timeout(650)
    first = _first_paint(page, timeout_ms=timeout_ms)
    initial_layout = _canvas_layout_summary(page)
    initial_debug = _read_debug(page)
    initial_fps = round(float(_measure_fps(page, 700)), 1)
    initial_shot = _save_screenshot(page, screenshots_dir / f"{case_name}-initial.png")

    col_steps = []
    for cols in (1, 2, 3, 4, 6, 8, 12):
        current = str(initial_debug.get("layoutRequestedMaxCols") or "")
        if current and current == str(cols):
            continue
        step = _exercise_column_select(page, "Show3D maximum columns", cols)
        if step.get("found") or step.get("to_target", {}).get("found"):
            col_steps.append(step)
            break

    fft_step = _exercise_fft_toggle(page)
    playback = _drive_playback(page, wait_ms=max(900, min(2200, int(seconds * 350))))
    stress = _drive_zoom_pan_stress(page, seconds=seconds)
    final_fps = round(float(_measure_fps(page, 900)), 1)
    final_layout = _canvas_layout_summary(page)
    final_debug = _read_debug(page)
    final_shot = _save_screenshot(page, screenshots_dir / f"{case_name}-final.png")
    wall_s = round(time.perf_counter() - started, 3)

    errors: list[str] = []
    warnings: list[str] = []
    if not first.get("nonblank"):
        errors.append("first visible canvas did not become nonblank")
    if not stress.get("final_canvas", {}).get("nonblank"):
        errors.append("canvas became blank after zoom/pan stress")
    if final_fps < min_fps:
        errors.append(f"final browser FPS {final_fps} is below --min-fps={min_fps}")
    if page_errors:
        errors.extend(f"page error: {message}" for message in page_errors)
    for message in console_messages:
        if message.get("type") == "error":
            text = message.get("text", "")
            if "favicon" not in text.lower():
                if text.startswith("Failed to load resource") and "404" in text:
                    warnings.append(f"ignored likely favicon 404 console noise: {text[:180]}")
                    continue
                errors.append(f"console error: {text[:300]}")
    for item in responses:
        if int(item.get("status", 0)) >= 400:
            errors.append(f"HTTP {item['status']} while loading {item['url']}")
    if target.mode == "sidecar" and not any("offline_stack" in item["url"] for item in responses):
        errors.append("sidecar target did not request offline_stack.u8")

    panels = int((target.metadata or {}).get("n_panels", final_debug.get("layoutRequestedMaxCols", 1)) or 1)
    aspect = final_layout.get("primary_aspect")
    if panels >= 4 and aspect is not None and float(aspect) > 4.0:
        errors.append(f"multi-panel primary canvas is still strip-like, aspect={aspect:.2f}")
    if not stress.get("render_paths"):
        warnings.append("zoom/pan stress did not report an interaction render path")
    if target.mode == "sidecar":
        range_hits = [item for item in responses if item.get("status") == 206]
        if not range_hits:
            warnings.append("sidecar loaded without HTTP 206 byte ranges; browser may have fetched the whole stack")
    if independent_contrast and not (contrast_step or {}).get("found"):
        warnings.append("--independent-contrast requested but the Contrast switch was not found")

    return {
        "name": case_name,
        "target": target.name,
        "mode": target.mode,
        "source": target.source,
        "url": url,
        "viewport": {"name": viewport_name, **viewport},
        "http_status": status,
        "wall_s": wall_s,
        "first_paint": first,
        "initial_fps": initial_fps,
        "final_fps": final_fps,
        "initial_layout": initial_layout,
        "final_layout": final_layout,
        "initial_debug": initial_debug,
        "final_debug": final_debug,
        "column_reflow": col_steps,
        "contrast": contrast_step,
        "fft_toggle": fft_step,
        "playback": playback,
        "zoom_pan_stress": stress,
        "screenshots": [initial_shot, final_shot],
        "responses": responses,
        "console_messages": console_messages[-80:],
        "page_errors": page_errors,
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }


def _target_url(target: TargetSpec) -> tuple[str, RangeServer | None]:
    if target.mode == "sidecar":
        if target.root is None:
            raise ValueError(f"sidecar target {target.name} has no root")
        server = RangeServer(target.root.resolve(), _free_port())
        return "", server
    if target.url:
        return target.url, None
    path = Path(target.source).expanduser().resolve()
    return path.as_uri(), None


def _write_report(artifact_dir: Path, report: dict[str, Any]) -> None:
    (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cases = report.get("cases", [])
    errors = [error for case in cases for error in case.get("errors", [])]
    warnings = [warning for case in cases for warning in case.get("warnings", [])]
    status = "PASS" if not errors else "FAIL"
    rows = "".join(
        "<tr>"
        f"<td>{_escape(case.get('name'))}</td>"
        f"<td>{_escape(case.get('mode'))}</td>"
        f"<td>{_escape(case.get('viewport', {}).get('name'))}</td>"
        f"<td>{_escape(case.get('first_paint', {}).get('first_paint_ms'))}</td>"
        f"<td>{_escape(case.get('final_fps'))}</td>"
        f"<td>{_escape(case.get('zoom_pan_stress', {}).get('cycles'))}</td>"
        f"<td>{'PASS' if case.get('passed') else 'FAIL'}</td>"
        "</tr>"
        for case in cases
    )
    cards = []
    for case in cases:
        images = "".join(
            f'<figure><img src="{_escape(shot.get("rel"))}"><figcaption>{_escape(Path(shot.get("path", "")).name)}</figcaption></figure>'
            for shot in case.get("screenshots", [])
        )
        payload = {
            key: case.get(key)
            for key in (
                "source",
                "url",
                "viewport",
                "first_paint",
                "initial_fps",
                "final_fps",
                "contrast",
                "final_layout",
                "final_debug",
                "responses",
                "warnings",
                "errors",
            )
        }
        cards.append(
            f'<section class="card"><h2>{_escape(case.get("name"))}</h2>'
            f'<div class="shots">{images}</div>'
            f"<pre>{_escape(json.dumps(payload, indent=2))}</pre></section>"
        )
    errors_html = "".join(f"<li>{_escape(error)}</li>" for error in errors) or "<li>None</li>"
    warnings_html = "".join(f"<li>{_escape(warning)}</li>" for warning in warnings) or "<li>None</li>"
    sources_html = "".join(
        f"<li><code>{_escape(target.get('mode'))}</code> {_escape(target.get('source'))}</li>"
        for target in report.get("targets", [])
    )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Show3D stress report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; margin: 14px 0 22px; }}
    th, td {{ text-align: left; padding: 7px 12px; border-bottom: 1px solid #d8dee9; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    code {{ background: #f0f2f5; padding: 1px 4px; border-radius: 4px; }}
    pre {{ white-space: pre-wrap; overflow: auto; max-height: 420px; background: #f6f8fa; padding: 12px; border-radius: 6px; }}
    .status {{ font-weight: 800; color: {"#087f23" if status == "PASS" else "#b00020"}; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; margin: 16px 0; }}
    .shots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; border: 1px solid #d8dee9; background: #f7f8fa; }}
    figcaption {{ font-size: 12px; color: #5b6472; }}
  </style>
</head>
<body>
  <h1>Show3D stress report</h1>
  <p class="status">{status}</p>
  <p>Local-only Chromium stress run for exact single-file HTML and folder sidecar paths.</p>
  <h2>Sources</h2>
  <ul>{sources_html}</ul>
  <h2>Summary</h2>
  <table>
    <thead><tr><th>Case</th><th>Mode</th><th>Viewport</th><th>First paint ms</th><th>Final FPS</th><th>Zoom cycles</th><th>Status</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Errors</h2>
  <ul>{errors_html}</ul>
  <h2>Warnings</h2>
  <ul>{warnings_html}</ul>
  {''.join(cards)}
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(doc, encoding="utf-8")


def _build_targets(args: argparse.Namespace, artifact_dir: Path) -> list[TargetSpec]:
    targets: list[TargetSpec] = []
    for html_path in args.html:
        path = Path(html_path).expanduser().resolve()
        metadata = _show3d_metadata_from_html(path)
        targets.append(TargetSpec(name=_safe_name(path.stem), mode="single", source=str(path), metadata=metadata))
    for url in args.url:
        targets.append(TargetSpec(name=_safe_name(url.rsplit("/", 1)[-1] or "url"), mode="url", source=url, url=url))
    for sidecar_dir in args.sidecar_dir:
        root = Path(sidecar_dir).expanduser().resolve()
        manifest_path = root / "manifest.json"
        metadata = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        targets.append(TargetSpec(name=_safe_name(root.name), mode="sidecar", source=str(root), root=root, metadata=metadata))
    for html_path in args.make_sidecar_from_html:
        source = Path(html_path).expanduser().resolve()
        out_dir = artifact_dir / "generated-sidecars" / _safe_name(source.stem)
        targets.append(_generate_sidecar_from_html(source, out_dir, max_decode_mb=args.max_decode_mb))
    if not targets:
        raise SystemExit("Provide at least one --html, --url, --sidecar-dir, or --make-sidecar-from-html target.")
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", action="append", default=[], help="Existing standalone Show3D HTML file.")
    parser.add_argument("--url", action="append", default=[], help="Existing served Show3D HTML URL.")
    parser.add_argument("--sidecar-dir", action="append", default=[], help="Folder export containing index.html and offline_stack.u8.")
    parser.add_argument("--make-sidecar-from-html", action="append", default=[], help="Generate and stress a temporary sidecar from a standalone Show3D HTML file.")
    parser.add_argument("--artifact-dir", default=str(_default_artifact_dir()), help="Output directory for report and screenshots.")
    parser.add_argument("--viewports", default="desktop", help="Comma-separated viewport names: desktop,wide,narrow.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Zoom/pan stress seconds per case.")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Page load and first-paint timeout.")
    parser.add_argument("--min-fps", type=float, default=30.0, help="Minimum final requestAnimationFrame FPS.")
    parser.add_argument("--max-decode-mb", type=float, default=4096.0, help="Safety guard for sidecar generation from embedded HTML.")
    parser.add_argument("--independent-contrast", action="store_true", help="Turn off linked panel contrast before screenshots/stress.")
    parser.add_argument("--headed", action="store_true", help="Open a visible browser window.")
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    targets = _build_targets(args, artifact_dir)
    viewport_names = [item.strip() for item in str(args.viewports).split(",") if item.strip()]
    unknown = [item for item in viewport_names if item not in VIEWPORTS]
    if unknown:
        raise SystemExit(f"unknown viewport(s): {', '.join(unknown)}")

    cases: list[dict[str, Any]] = []
    chrome = _chrome_executable()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headed,
            executable_path=chrome,
            args=["--disable-web-security", "--allow-file-access-from-files"],
        )
        try:
            for target in targets:
                target_url, server = _target_url(target)
                if server is None:
                    for viewport_name in viewport_names:
                        page = browser.new_page()
                        try:
                            cases.append(
                                _run_case(
                                    page,
                                    target=target,
                                    url=target_url,
                                    viewport_name=viewport_name,
                                    viewport=VIEWPORTS[viewport_name],
                                    artifact_dir=artifact_dir,
                                    seconds=args.seconds,
                                    timeout_ms=args.timeout_ms,
                                    min_fps=args.min_fps,
                                    independent_contrast=args.independent_contrast,
                                )
                            )
                        finally:
                            page.close()
                else:
                    with server as base_url:
                        served_url = f"{base_url}/index.html"
                        for viewport_name in viewport_names:
                            page = browser.new_page()
                            try:
                                cases.append(
                                    _run_case(
                                        page,
                                        target=target,
                                        url=served_url,
                                        viewport_name=viewport_name,
                                        viewport=VIEWPORTS[viewport_name],
                                        artifact_dir=artifact_dir,
                                        seconds=args.seconds,
                                        timeout_ms=args.timeout_ms,
                                        min_fps=args.min_fps,
                                        independent_contrast=args.independent_contrast,
                                    )
                                )
                            finally:
                                page.close()
        finally:
            browser.close()

    report = {
        "created_utc": _timestamp(),
        "artifact_dir": str(artifact_dir),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "targets": [
            {
                "name": target.name,
                "mode": target.mode,
                "source": target.source,
                "root": str(target.root) if target.root else None,
                "metadata": target.metadata,
            }
            for target in targets
        ],
        "viewports": viewport_names,
        "cases": cases,
        "passed": all(case.get("passed") for case in cases),
    }
    _write_report(artifact_dir, report)
    print(f"Show3D stress report: {artifact_dir / 'index.html'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
