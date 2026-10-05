#!/usr/bin/env python3
"""Exercise ShowFolder live-folder refresh and selected-viewer handoff.

The 4D-STEM steps write tiny bitshuffle-LZ4 Arina masters and load them for
real through ``quantem.gpu.io.load`` into encoded CUDA or MPS storage; on a
CPU-only host those two steps are reported as skipped.
"""

import argparse
import html
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np
from quantem.gpu.device import detect

from quantem.widget import Show2D, Show3D, Show4DSTEM, ShowFolder
from quantem.widget.render import save_thumbnail


NO_GPU_REASON = "No native CUDA or MPS backend is available."


def _metadata() -> np.ndarray:
    payload = json.dumps({
        "Scan": {"ScanRotation": "0"},
        "BinaryResult": {"PixelSize": {"height": 1e-9, "width": 1e-9}},
    }).encode()
    arr = np.zeros((len(payload) + 1, 1), dtype=np.uint8)
    arr[: len(payload), 0] = np.frombuffer(payload, dtype=np.uint8)
    return arr


def _image_emd(path: Path, *, offset: int) -> None:
    data = np.arange(20 * 24, dtype=np.float32).reshape(20, 24) + float(offset)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("Data/Image/uid")
        group.create_dataset("Data", data=data)
        group.create_dataset("Metadata", data=_metadata())


def _image_npy(path: Path, *, value: float) -> None:
    """Write one tiny exact image used by direct from_folder lifecycle checks."""
    np.save(path, np.full((12, 16), float(value), dtype=np.float32))


def _write_external_master(folder: Path, *, index: int) -> Path:
    """Write one 4x4 scan of 4x4 frames holding the constant count ``index + 1``.

    The master links its frames from an external data file, as an Arina
    acquisition does, so the production readiness check and GPU loader apply.
    Constant counts make every virtual image equal to ``index + 1`` exactly,
    whatever detector mask the viewer chooses.
    """
    data_path = folder / f"scan_{index:03d}_data_000001.h5"
    master_path = folder / f"scan_{index:03d}_master.h5"
    with h5py.File(data_path, "w") as h5:
        h5.create_dataset(
            "entry/data/data",
            data=np.full((16, 4, 4), index + 1, dtype=np.uint16),
            chunks=(1, 4, 4),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
    with h5py.File(master_path, "w") as h5:
        h5.require_group("entry/data")["data_000001"] = h5py.ExternalLink(
            data_path.name,
            "entry/data/data",
        )
        detector = h5.require_group(
            "entry/instrument/detector/detectorSpecific"
        )
        detector.create_dataset("ntrigger", data=16)
    return master_path


def _native_gpu_backend() -> str | None:
    """Return the CUDA or MPS backend io.load encodes on, or None on a CPU-only host."""
    try:
        return detect()
    except RuntimeError:
        return None


def _skipped_step(name: str, kind: str) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "passed": True,
        "skipped": True,
        "skip_reason": NO_GPU_REASON,
        "exports": {},
        "export_rows": [],
    }


def _compare_panel_means(widget: Any) -> list[float]:
    """Mean of each comparison panel's virtual image, in panel order."""
    panels = np.frombuffer(widget.compare_virtual_image_bytes, dtype=np.float32)
    panels = panels.reshape(
        len(widget.compare_panel_indices),
        widget.shape_rows,
        widget.shape_cols,
    )
    return [float(panel.mean()) for panel in panels]


def _export(widget: Any, path: Path, *, title: str) -> Path | None:
    export = getattr(widget, "export_html", None)
    if export is None:
        return None
    return Path(export(path, title=title))


def _browser_export_row(
    path: Path,
    *,
    widget: str,
    variant: str,
    seconds: float = 0.0,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the row shape consumed by widget_browser_smoke.py."""
    return {
        "widget": str(widget),
        "variant": str(variant),
        "path": str(path.resolve()),
        "seconds": round(float(seconds), 3),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "options": dict(options or {}),
    }


def _embedded_watch_contract(path: Path) -> dict[str, Any]:
    """Inspect exported widget model state without trusting visible label text."""
    text = path.read_text(encoding="utf-8")
    states = sorted(
        set(
            re.findall(
                r'"folder_watch_state"\s*:\s*"([^"]+)"',
                text,
            )
        )
    )
    return {
        "embedded_states": states,
        "watching_embedded": "watching" in states,
        "hidden_snapshot": states == ["hidden"],
    }


def _watch_snapshot(
    widget: Any,
    *,
    event: str,
    started: float,
    count_attr: str,
) -> dict[str, Any]:
    """Capture one JSON-safe direct-viewer lifecycle observation.

    Show2D and Show3D have no comparison grid, so their panel fields stay empty.
    """
    return {
        "event": str(event),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "python_id": int(id(widget)),
        "model_id": str(getattr(widget, "model_id", "")),
        "state": str(getattr(widget, "folder_watch_state", "")),
        "detail": str(getattr(widget, "folder_watch_detail", "")),
        "count": int(getattr(widget, count_attr)),
        "compare_panel_indices": [
            int(value)
            for value in getattr(widget, "compare_panel_indices", [])
        ],
        "virtual_image_bytes": len(
            bytes(getattr(widget, "compare_virtual_image_bytes", b""))
        ),
    }


def _write_thumbnail_previews(artifact_dir: Path, browser: Any, *, prefix: str) -> list[dict[str, Any]]:
    preview_dir = artifact_dir / "previews"
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for gallery, image_items in browser.image_galleries:
        data = np.asarray(getattr(gallery, "_data", np.empty((0, 0, 0))))
        if data.ndim < 3:
            continue
        for idx, item in enumerate(image_items):
            if idx >= data.shape[0] or item.file_id in seen:
                continue
            seen.add(item.file_id)
            out = preview_dir / f"{prefix}-{item.file_id}.webp"
            save_thumbnail(data[idx], out, size=96, cmap="inferno")
            rows.append({
                "id": item.file_id,
                "file": item.path.name,
                "webp": out.relative_to(artifact_dir).as_posix(),
                "bytes": out.stat().st_size,
            })
    return rows


def _run_image_live_smoke(artifact_dir: Path) -> dict[str, Any]:
    folder = artifact_dir / "live-images"
    folder.mkdir(parents=True, exist_ok=True)
    _image_emd(folder / "0010 - HAADF live.emd", offset=0)
    _image_emd(folder / "0011 - HAADF live.emd", offset=100)

    widget = ShowFolder(
        folder,
        thumb=10,
        group_by="none",
        cache_dir=artifact_dir / "cache-images",
    )
    assert widget.browser is not None
    assert widget.browser.gallery is not None
    opened = widget.open_both(all_images=True)
    all_2d, all_3d = opened.children
    assert widget.browser._active_selected_modes == {"show2d_all", "show3d_all"}
    assert all_2d._data.shape[0] == 2
    assert all_3d.n_slices == 2
    before_widget_id = id(widget.widget)
    before_items = [item.file_id for item in widget.items]

    widget.watch(start=False)
    _image_emd(folder / "0012 - HAADF live.emd", offset=200)
    changed = widget.watch_once()
    after_items = [item.file_id for item in widget.items]

    assert changed is True
    assert id(widget.widget) == before_widget_id
    assert after_items == ["0010", "0011", "0012"]
    assert widget.browser._selected_show2d_widget is all_2d
    assert widget.browser._selected_show3d_widget is all_3d
    assert widget.browser._active_selected_modes == {"show2d_all", "show3d_all"}
    assert all_2d._data.shape[0] == 3
    assert all_3d.n_slices == 3
    thumbnail_previews = _write_thumbnail_previews(artifact_dir, widget.browser, prefix="live-image")

    exports = {
        "showfolder": _export(
            widget,
            artifact_dir / "showfolder-live-images.html",
            title="ShowFolder live images",
        ),
        "show2d": _export(
            all_2d,
            artifact_dir / "showfolder-live-show2d.html",
            title="ShowFolder live all-image Show2D",
        ),
        "show3d": _export(
            all_3d,
            artifact_dir / "showfolder-live-show3d.html",
            title="ShowFolder live all-image Show3D",
        ),
    }
    export_rows = [
        _browser_export_row(
            path,
            widget=widget_name,
            variant=f"{widget_name}-showfolder-live",
        )
        for widget_name, path in exports.items()
        if path is not None
    ]
    return {
        "name": "ShowFolder live images -> all-image Show2D/Show3D",
        "kind": "showfolder_orchestration",
        "passed": True,
        "before_items": before_items,
        "after_items": after_items,
        "watch_changed": changed,
        "widget_id_preserved": id(widget.widget) == before_widget_id,
        "all_path_names": [item.path.name for item in widget.browser.image_items],
        "show2d_panels": int(all_2d._data.shape[0]),
        "show3d_slices": int(all_3d.n_slices),
        "thumbnail_previews": thumbnail_previews,
        "exports": {key: None if value is None else value.name for key, value in exports.items()},
        "export_rows": export_rows,
    }


def _run_master_live_smoke(artifact_dir: Path) -> dict[str, Any]:
    """Hand a growing master folder from ShowFolder to Show4DSTEM through the real loader."""
    folder = artifact_dir / "live-4dstem"
    folder.mkdir(parents=True, exist_ok=True)
    _write_external_master(folder, index=0)

    widget = ShowFolder(
        folder,
        thumb=10,
        group_by="none",
        cache_dir=artifact_dir / "cache-4dstem",
    )
    first = widget.open_show4dstem(scan_size=4)
    assert first is not None
    first.wait_for_folder()
    assert first.n_frames == 1
    assert widget.master_qc_rows[0]["status"] == "ready"

    widget.watch(start=False)
    _write_external_master(folder, index=1)
    changed = widget.watch_once()
    second = widget.browser._selected_show4dstem_widget
    try:
        assert changed is True
        assert second is not None
        assert second is not first
        # The browser rebuilds the viewer on refresh; the one it replaced must
        # not keep its acquisitions on the GPU.
        replaced_released = first._folder_acquisitions == []
        assert replaced_released
        second.wait_for_folder()
        assert second.n_frames == 2
        assert list(second.frame_labels) == ["scan_000", "scan_001"]
        panel_means = _compare_panel_means(second)
        assert panel_means == [1.0, 2.0]
        return {
            "name": "ShowFolder live 4D-STEM masters -> Show4DSTEM",
            "kind": "showfolder_orchestration",
            "loader_note": (
                "Tiny bitshuffle-LZ4 external-link masters loaded by "
                "quantem.gpu.io.load into encoded GPU storage; no fake loader."
            ),
            "passed": True,
            "watch_changed": changed,
            "first_frames": 1,
            "after_frames": int(second.n_frames),
            "replaced_viewer_released": replaced_released,
            "frame_labels": list(second.frame_labels),
            "panel_means": panel_means,
            "master_qc": widget.master_qc_rows,
            "exports": {},
            "export_rows": [],
        }
    finally:
        if second is not None:
            second.close()


def _run_direct_image_live_smoke(
    artifact_dir: Path,
    *,
    viewer_class: Any,
    widget_name: str,
    count_attr: str,
) -> dict[str, Any]:
    """Exercise one public Show2D/Show3D from_folder lifecycle in place."""
    folder = artifact_dir / f"direct-{widget_name}"
    folder.mkdir(parents=True, exist_ok=True)
    _image_npy(folder / "frame_000.npy", value=1)

    started = time.perf_counter()
    widget = viewer_class.from_folder(
        folder,
        pattern="*.npy",
        watch=True,
        watch_interval=60,
    )
    timeline = [
        _watch_snapshot(
            widget,
            event="mounted_with_initial_candidate_on_probation",
            started=started,
            count_attr=count_attr,
        )
    ]

    def record_state(change: dict[str, Any]) -> None:
        timeline.append(
            _watch_snapshot(
                widget,
                event=f"state:{change['new']}",
                started=started,
                count_attr=count_attr,
            )
        )

    widget.observe(record_state, names="folder_watch_state")
    initial_python_id = id(widget)
    initial_model_id = str(widget.model_id)
    try:
        assert int(getattr(widget, count_attr)) == 0
        assert widget.folder_watch_state == "waiting"

        initial_added = widget.poll_folder()
        assert initial_added == [0]
        timeline.append(
            _watch_snapshot(
                widget,
                event="initial_candidate_confirmed",
                started=started,
                count_attr=count_attr,
            )
        )
        assert widget.folder_watch_state == "watching"

        _image_npy(folder / "frame_001.npy", value=2)
        probation_added = widget.poll_folder()
        assert probation_added == []
        assert widget.folder_watch_state == "waiting"
        timeline.append(
            _watch_snapshot(
                widget,
                event="arrival_probation",
                started=started,
                count_attr=count_attr,
            )
        )

        stable_added = widget.poll_folder()
        assert stable_added == [1]
        timeline.append(
            _watch_snapshot(
                widget,
                event="stable_arrival_applied",
                started=started,
                count_attr=count_attr,
            )
        )
        assert int(getattr(widget, count_attr)) == 2
        assert id(widget) == initial_python_id
        assert str(widget.model_id) == initial_model_id
        np.testing.assert_array_equal(
            np.asarray(widget._data)[:, 0, 0],
            np.asarray([1, 2], dtype=np.float32),
        )

        widget.stop_folder_watch()
        timeline.append(
            _watch_snapshot(
                widget,
                event="stopped_before_static_export",
                started=started,
                count_attr=count_attr,
            )
        )
        assert widget.folder_watch_state == "stopped"
        saved_watch_state = widget.state_dict().get(
            "folder_watch_state",
            "absent",
        )
        assert saved_watch_state != "watching"

        export_path = artifact_dir / f"{widget_name}-from-folder-stopped.html"
        export_started = time.perf_counter()
        exported = _export(
            widget,
            export_path,
            title=f"{widget_name} from_folder stopped snapshot",
        )
        export_seconds = time.perf_counter() - export_started
        assert exported is not None
        static_contract = _embedded_watch_contract(exported)
        assert static_contract["watching_embedded"] is False
        assert static_contract["hidden_snapshot"] is True

        states = [str(item["state"]) for item in timeline]
        for required in ("waiting", "updating", "watching", "stopped"):
            assert required in states
        same_model = (
            {int(item["python_id"]) for item in timeline} == {initial_python_id}
            and {str(item["model_id"]) for item in timeline} == {initial_model_id}
        )
        assert same_model
        export_row = _browser_export_row(
            exported,
            widget=widget_name,
            variant=f"{widget_name}-folder-watch-static",
            seconds=export_seconds,
        )
        return {
            "name": f"Direct {widget_name}.from_folder live lifecycle",
            "kind": "direct_public_from_folder",
            "passed": True,
            "same_mounted_model": same_model,
            "initial_probation_added": initial_added,
            "arrival_probation_added": probation_added,
            "stable_arrival_added": stable_added,
            "final_count": int(getattr(widget, count_attr)),
            "labels": list(widget.labels),
            "saved_state_watch": saved_watch_state,
            "static_watch_contract": static_contract,
            "timeline": timeline,
            "exports": {widget_name: exported.name},
            "export_rows": [export_row],
        }
    finally:
        widget.close()


def _run_direct_show4dstem_live_smoke(artifact_dir: Path) -> dict[str, Any]:
    """Exercise the public Show4DSTEM folder watcher over encoded acquisitions."""
    folder = artifact_dir / "direct-show4dstem"
    folder.mkdir(parents=True, exist_ok=True)
    _write_external_master(folder, index=0)

    started = time.perf_counter()
    widget = Show4DSTEM.from_folder(
        folder,
        scan_size=4,
        watch=True,
        watch_interval=60,
        view_mode="multiple",
        columns=2,
        page_size=4,
        verbose=False,
    )
    timeline = [
        _watch_snapshot(
            widget,
            event="mounted_with_initial_master",
            started=started,
            count_attr="n_frames",
        )
    ]

    def record_state(change: dict[str, Any]) -> None:
        timeline.append(
            _watch_snapshot(
                widget,
                event=f"state:{change['new']}",
                started=started,
                count_attr="n_frames",
            )
        )

    widget.observe(record_state, names="folder_watch_state")
    initial_python_id = id(widget)
    initial_model_id = str(widget.model_id)
    try:
        widget.wait_for_folder()
        assert widget.n_frames == 1
        assert widget.folder_watch_state == "watching"
        # Encoded acquisitions have no offline export; the saved notebook
        # state is their only static form and must never claim Watching.
        saved_while_watching = widget.get_state()["folder_watch_state"]
        assert saved_while_watching == "stopped"

        arrival_start = len(timeline)
        _write_external_master(folder, index=1)
        probation_added = widget.poll_folder()
        assert probation_added == []
        assert widget.folder_watch_state == "waiting"
        timeline.append(
            _watch_snapshot(
                widget,
                event="arrival_headers_on_probation",
                started=started,
                count_attr="n_frames",
            )
        )

        stable_added = widget.poll_folder()
        assert stable_added == [1]
        timeline.append(
            _watch_snapshot(
                widget,
                event="arrival_appended",
                started=started,
                count_attr="n_frames",
            )
        )
        assert widget.n_frames == 2
        assert id(widget) == initial_python_id
        assert str(widget.model_id) == initial_model_id
        assert list(widget.compare_panel_indices) == [0, 1]
        panel_means = _compare_panel_means(widget)
        assert panel_means == [1.0, 2.0]
        assert widget.folder_watch_state == "watching"

        arrival_timeline = timeline[arrival_start:]
        arrival_states = [str(item["state"]) for item in arrival_timeline]
        assert "waiting" in arrival_states
        assert "updating" in arrival_states
        green_points = [
            item for item in arrival_timeline if item["state"] == "watching"
        ]
        assert green_points
        # The badge turns green only once the appended dataset is in the panels.
        green = green_points[-1]
        assert green["count"] == 2
        assert green["compare_panel_indices"] == [0, 1]
        assert green["virtual_image_bytes"] > 0

        widget.stop_folder_watch()
        timeline.append(
            _watch_snapshot(
                widget,
                event="stopped",
                started=started,
                count_attr="n_frames",
            )
        )
        assert widget.folder_watch_state == "stopped"
        saved_after_stop = widget.get_state()["folder_watch_state"]
        assert saved_after_stop != "watching"

        same_model = (
            {int(item["python_id"]) for item in timeline} == {initial_python_id}
            and {str(item["model_id"]) for item in timeline} == {initial_model_id}
        )
        assert same_model
        return {
            "name": "Direct Show4DSTEM.from_folder live lifecycle",
            "kind": "direct_public_from_folder",
            "loader_note": (
                "Tiny bitshuffle-LZ4 external-link masters loaded by "
                "quantem.gpu.io.load into encoded GPU storage; no fake loader."
            ),
            "passed": True,
            "same_mounted_model": same_model,
            "arrival_probation_added": probation_added,
            "stable_arrival_added": stable_added,
            "final_count": int(widget.n_frames),
            "frame_labels": list(widget.frame_labels),
            "compare_panel_indices": list(widget.compare_panel_indices),
            "panel_means": panel_means,
            "saved_state_watch": {
                "while_watching": saved_while_watching,
                "after_stop": saved_after_stop,
            },
            "timeline": timeline,
            "exports": {},
            "export_rows": [],
        }
    finally:
        widget.close()


def _write_browser_plan(artifact_dir: Path, report: dict[str, Any]) -> None:
    """Write standalone-export review instructions for existing browser smoke."""
    pages = []
    for item in report["exports"]:
        direct_static = str(item["variant"]).endswith("folder-watch-static")
        interactions = [
            "open the standalone export and confirm its scientific canvas is nonblank",
            "confirm pan/zoom or the primary image interaction still responds",
        ]
        if direct_static:
            interactions.append(
                "confirm no green Watching badge is present in this stopped static snapshot"
            )
        pages.append(
            {
                "widget": item["widget"],
                "variant": item["variant"],
                "file": Path(str(item["path"])).name,
                "url_path": Path(str(item["path"])).name,
                "expected_folder_watch_state": (
                    "hidden" if direct_static else "not asserted"
                ),
                "required_interactions": interactions,
            }
        )
    plan = {
        "version": 1,
        "description": (
            "Browser-drive the tiny static exports after the Python report has "
            "already proven direct live watcher transitions. Static pages must "
            "never claim that their stopped Python watcher is still Watching."
        ),
        "artifact_dir": str(artifact_dir),
        "pages": pages,
    }
    (artifact_dir / "browser-plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )


def _write_report(artifact_dir: Path, report: dict[str, Any]) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(step['name'])}</td>"
        f"<td>{'skipped' if step.get('skipped') else 'pass' if step['passed'] else 'fail'}</td>"
        f"<td>{html.escape(json.dumps(step, sort_keys=True))}</td>"
        "</tr>"
        for step in report["steps"]
    )
    links = []
    for step in report["steps"]:
        for name in (step.get("exports") or {}).values():
            if name:
                links.append(f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a></li>")
    preview_cards = []
    for step in report["steps"]:
        for preview in step.get("thumbnail_previews") or []:
            src = html.escape(str(preview["webp"]))
            label = html.escape(f"{preview['id']} · {preview['file']}")
            preview_cards.append(
                "<figure>"
                f"<img src='{src}' alt='{label}'>"
                f"<figcaption>{label}</figcaption>"
                "</figure>"
            )
    qc_rows = []
    for step in report["steps"]:
        for row in step.get("master_qc") or []:
            scan = row.get("scan_shape")
            det = row.get("detector_shape")
            qc_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('file', '')))}</td>"
                f"<td>{html.escape(str(row.get('status', '')))}</td>"
                f"<td>{'' if scan is None else html.escape('x'.join(str(v) for v in scan))}</td>"
                f"<td>{'' if det is None else html.escape('x'.join(str(v) for v in det))}</td>"
                f"<td>{html.escape(str(row.get('n_frames') or ''))}</td>"
                f"<td>{html.escape(str(row.get('dtype') or ''))}</td>"
                f"<td>{html.escape(str(row.get('reason', '')))}</td>"
                f"<td>{html.escape(str(row.get('action', '')))}</td>"
                "</tr>"
            )
    timeline_rows = []
    for step in report["steps"]:
        for point in step.get("timeline") or []:
            timeline_rows.append(
                "<tr>"
                f"<td>{html.escape(str(step['name']))}</td>"
                f"<td>{html.escape(str(point.get('event', '')))}</td>"
                f"<td>{html.escape(str(point.get('elapsed_ms', '')))}</td>"
                f"<td>{html.escape(str(point.get('state', '')))}</td>"
                f"<td>{html.escape(str(point.get('count', '')))}</td>"
                f"<td>{html.escape(str(point.get('compare_panel_indices', '')))}</td>"
                f"<td>{html.escape(str(point.get('detail', '')))}</td>"
                "</tr>"
            )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Folder-watching direct-viewer smoke</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 12px; width: min(1180px, 100%); }}
    th, td {{ border: 1px solid #ccd3db; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f5f7; }}
    code, pre {{ background: #f5f7f9; border-radius: 4px; }}
    code {{ padding: 2px 4px; }}
    pre {{ overflow: auto; padding: 12px; max-width: 1180px; }}
    .previews {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }}
    figure {{ margin: 0; border: 1px solid #ccd3db; border-radius: 6px; padding: 8px; background: #f8fafc; }}
    figure img {{ display: block; width: 96px; height: 96px; object-fit: contain; background: #111827; }}
    figcaption {{ max-width: 160px; margin-top: 6px; font-size: 12px; color: #3b4654; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>ShowFolder live-folder smoke: {'PASS' if report['passed'] else 'FAIL'}</h1>
  <p>This lightweight report covers both ShowFolder orchestration and the direct
  public <code>Show2D.from_folder</code>, <code>Show3D.from_folder</code>, and
  <code>Show4DSTEM.from_folder</code> lifecycles. Direct viewers retain one Python
  object and widget model through probation, stable arrival, update, and stop.
  The Show4DSTEM steps load tiny Arina masters through
  <code>quantem.gpu.io.load</code> into encoded GPU storage
  (backend: {html.escape(report['gpu_backend'])}); a poll that appends a master
  has already refreshed the comparison panels when the badge returns to green
  Watching. Heavy real-data and GPU performance remain separate local-only
  signoffs.</p>
  <h2>Review Exports</h2>
  <ul>{''.join(links)}</ul>
  <h2>Thumbnail Previews</h2>
  <div class="previews">{''.join(preview_cards) if preview_cards else '<p>No thumbnail previews.</p>'}</div>
  <h2>4D-STEM Master QC</h2>
  <table>
    <thead><tr><th>Master</th><th>Status</th><th>Scan</th><th>Detector</th><th>Frames</th><th>Dtype</th><th>Reason</th><th>Next step</th></tr></thead>
    <tbody>{''.join(qc_rows) if qc_rows else '<tr><td colspan="8">No master QC rows.</td></tr>'}</tbody>
  </table>
  <h2>Checks</h2>
  <table>
    <thead><tr><th>Scenario</th><th>Status</th><th>Evidence</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Direct Viewer Lifecycle Timeline</h2>
  <p>The state text, not color alone, is authoritative. Open
  <a href="browser-plan.json">browser-plan.json</a> to drive the stopped static
  exports; they must not retain a green Watching claim.</p>
  <table>
    <thead><tr><th>Viewer</th><th>Event</th><th>ms</th><th>State</th><th>Count</th><th>Panel indices</th><th>Detail</th></tr></thead>
    <tbody>{''.join(timeline_rows) if timeline_rows else '<tr><td colspan="7">No direct-viewer timeline.</td></tr>'}</tbody>
  </table>
  <h2>Machine-readable report</h2>
  <p><a href="report.json">report.json</a> · <a href="browser-plan.json">browser-plan.json</a></p>
  <pre>{html.escape(json.dumps(report, indent=2))}</pre>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(page, encoding="utf-8")
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or Path(tempfile.mkdtemp(prefix="quantem-widget-showfolder-live-"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = artifact_dir.resolve()

    started = time.perf_counter()
    backend = _native_gpu_backend()
    steps = [
        _run_image_live_smoke(artifact_dir),
        _run_master_live_smoke(artifact_dir)
        if backend
        else _skipped_step(
            "ShowFolder live 4D-STEM masters -> Show4DSTEM",
            "showfolder_orchestration",
        ),
        _run_direct_image_live_smoke(
            artifact_dir,
            viewer_class=Show2D,
            widget_name="show2d",
            count_attr="n_images",
        ),
        _run_direct_image_live_smoke(
            artifact_dir,
            viewer_class=Show3D,
            widget_name="show3d",
            count_attr="n_slices",
        ),
        _run_direct_show4dstem_live_smoke(artifact_dir)
        if backend
        else _skipped_step(
            "Direct Show4DSTEM.from_folder live lifecycle",
            "direct_public_from_folder",
        ),
    ]
    export_rows = [
        row
        for step in steps
        for row in step.get("export_rows", [])
    ]
    report = {
        "artifact_dir": str(artifact_dir),
        "created_at_unix": int(time.time()),
        "seconds": round(time.perf_counter() - started, 3),
        "gpu_backend": backend or "none",
        "passed": all(step["passed"] for step in steps),
        "steps": steps,
        "exports": export_rows,
        "total_size_mb": round(
            sum(float(row["size_mb"]) for row in export_rows),
            3,
        ),
    }
    _write_browser_plan(artifact_dir, report)
    _write_report(artifact_dir, report)
    print(json.dumps(report, indent=2))
    print(f"ShowFolder live-folder smoke report: {artifact_dir / 'index.html'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
