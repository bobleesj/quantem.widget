#!/usr/bin/env python3
"""Run resumable real-data Show4DSTEM folder endurance on one GPU.

This is an opt-in, local-only signoff runner. The controller waits for the
selected physical NVIDIA device to become idle, then starts every case in a
fresh child process so ``CUDA_VISIBLE_DEVICES`` is fixed before Torch imports.
Each child opens the source folder with ``Show4DSTEM.from_folder``, which keeps
every ready master resident in encoded GPU storage at full detector
resolution, and repeats the canonical page, curation, and diffraction-mode
cycle. Fresh-process open cases record the time to the first viewer and to the
complete folder; the endurance case repeats the cycle until both its cycle
count and its clock budget are met and fails if allocator memory grows. It
writes an atomic live report throughout the run and never mutates source data
or terminates processes it did not create.

Actual Jupyter/browser evidence is a separate required gate. This runner owns
the backend endurance cases and records that browser gate as pending until the
companion live-Jupyter drive attaches its artifacts.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Sequence


SCHEMA_VERSION = 2
DEFAULT_BLOCK_PATTERNS = (
    "overnight_ml_calibration_campaign.py",
    "overnight_zoo_campaign.py",
    "run_noiseless_block.py",
    "run_framewise_block.py",
    "live ptycho",
    "quantem.live.cli.ptycho",
)
FATAL_TOKENS = (
    "cudaerrorillegaladdress",
    "illegal address",
    "out of memory",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # NumPy and Torch scalars convert through item(); arrays through tolist().
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_ready(value), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_text(command: Sequence[str]) -> str:
    return subprocess.check_output(
        list(command),
        text=True,
        stderr=subprocess.STDOUT,
        timeout=30,
    ).strip()


def _git_snapshot(repo: Path) -> dict[str, Any]:
    try:
        commit = _run_text(["git", "-C", str(repo), "rev-parse", "HEAD"])
        status = _run_text(["git", "-C", str(repo), "status", "--short"])
        diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--binary"],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _filesystem_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    result: dict[str, Any] = {"path": str(resolved)}
    try:
        stat = resolved.stat()
        usage = shutil.disk_usage(resolved)
        result.update(
            {
                "device": int(stat.st_dev),
                "free_bytes": int(usage.free),
                "total_bytes": int(usage.total),
            }
        )
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["findmnt"] = _run_text(
            ["findmnt", "-T", str(resolved), "-no", "SOURCE,FSTYPE,TARGET"]
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["findmnt_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _gpu_snapshot() -> dict[str, Any]:
    fields = (
        "index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,"
        "memory.free,utilization.gpu"
    )
    result: dict[str, Any] = {"captured_at": _utc_now(), "gpus": [], "apps": []}
    try:
        raw = _run_text(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 9:
            continue
        result["gpus"].append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "pci_bus_id": parts[2],
                "name": parts[3],
                "driver_version": parts[4],
                "total_mib": int(parts[5]),
                "used_mib": int(parts[6]),
                "free_mib": int(parts[7]),
                "utilization_pct": int(parts[8]),
            }
        )

    try:
        raw = _run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.SubprocessError):
        raw = ""
    apps: list[dict[str, Any]] = []
    pids: list[int] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        pids.append(pid)
        apps.append(
            {
                "gpu_uuid": parts[0],
                "pid": pid,
                "process_name": parts[2],
                "used_mib": None if parts[3] == "[N/A]" else int(parts[3]),
            }
        )
    commands: dict[int, str] = {}
    if pids:
        try:
            raw_ps = _run_text(
                ["ps", "-ww", "-o", "pid=,command=", "-p", ",".join(map(str, pids))]
            )
        except (OSError, subprocess.SubprocessError):
            # ps exits nonzero when every listed process has already ended.
            raw_ps = ""
        for line in raw_ps.splitlines():
            pieces = line.strip().split(maxsplit=1)
            if not pieces:
                continue
            commands[int(pieces[0])] = pieces[1] if len(pieces) > 1 else ""
    for app in apps:
        app["command"] = commands.get(int(app["pid"]), "")
    result["apps"] = apps
    return result


def _idle_decision(
    snapshot: dict[str, Any],
    devices: Sequence[int],
    *,
    max_utilization: int,
    min_free_mib: int,
    block_patterns: Sequence[str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    rows = {int(row["index"]): row for row in snapshot.get("gpus", [])}
    selected_uuids: set[str] = set()
    for device in devices:
        row = rows.get(int(device))
        if row is None:
            reasons.append(f"physical GPU {device} is not visible to nvidia-smi")
            continue
        selected_uuids.add(str(row["uuid"]))
        if int(row["utilization_pct"]) > int(max_utilization):
            reasons.append(
                f"GPU {device} utilization {row['utilization_pct']}% > {max_utilization}%"
            )
        if int(row["free_mib"]) < int(min_free_mib):
            reasons.append(
                f"GPU {device} free {row['free_mib']} MiB < {min_free_mib} MiB"
            )
    lowered_patterns = [token.lower() for token in block_patterns if token]
    for app in snapshot.get("apps", []):
        if str(app.get("gpu_uuid")) not in selected_uuids:
            continue
        command = str(app.get("command", ""))
        lowered = command.lower()
        matched = next((token for token in lowered_patterns if token in lowered), None)
        if matched:
            reasons.append(
                f"GPU campaign PID {app.get('pid')} matches {matched!r}: {command[:260]}"
            )
    return not reasons, reasons


class LiveReport:
    def __init__(self, artifact_dir: Path, initial: dict[str, Any]) -> None:
        self.artifact_dir = artifact_dir.resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.artifact_dir / "report.json"
        self.status_path = self.artifact_dir / "status.json"
        self.events_path = self.artifact_dir / "events.jsonl"
        self.gpu_path = self.artifact_dir / "gpu-telemetry.jsonl"
        self._lock = threading.RLock()
        previous: dict[str, Any] = {}
        if self.path.is_file():
            try:
                previous = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        self.data = {**initial}
        if previous.get("schema_version") == SCHEMA_VERSION:
            self.data["cases"] = list(previous.get("cases", []))
            self.data["errors"] = list(previous.get("errors", []))
            self.data["restart_count"] = int(previous.get("restart_count", 0)) + 1
        self.flush()

    def event(self, event: str, **fields: Any) -> None:
        record = {"time": _utc_now(), "event": event, **fields}
        _append_jsonl(self.events_path, record)

    def gpu(self, snapshot: dict[str, Any], *, phase: str) -> None:
        _append_jsonl(self.gpu_path, {"phase": phase, **snapshot})

    def update(self, **fields: Any) -> None:
        with self._lock:
            self.data.update(fields)
            self.data["heartbeat_at"] = _utc_now()
            self.flush()

    def append_case(self, case: dict[str, Any]) -> None:
        with self._lock:
            cases = [
                item for item in self.data.setdefault("cases", [])
                if item.get("id") != case.get("id")
            ]
            cases.append(case)
            self.data["cases"] = cases
            self.data["completed_cases"] = sum(
                item.get("status") == "pass" for item in cases
            )
            self.data["heartbeat_at"] = _utc_now()
            self.flush()

    def add_error(self, error: dict[str, Any]) -> None:
        with self._lock:
            self.data.setdefault("errors", []).append(error)
            self.flush()

    def completed(self, case_id: str) -> dict[str, Any] | None:
        for case in self.data.get("cases", []):
            if case.get("id") == case_id and case.get("status") == "pass":
                return case
        return None

    def flush(self) -> None:
        with self._lock:
            _atomic_json(self.path, self.data)
            _atomic_json(
                self.artifact_dir / "environment.json",
                self.data.get("environment", {}),
            )
            _atomic_json(
                self.artifact_dir / "gates.json",
                self.data.get("gates", []),
            )
            _atomic_json(
                self.artifact_dir / "manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.data.get("run_id"),
                    "status": self.data.get("status"),
                    "current_phase": self.data.get("current_phase"),
                    "heartbeat_at": self.data.get("heartbeat_at"),
                    "completed_cases": self.data.get("completed_cases", 0),
                    "active_pid": self.data.get("active_pid"),
                    "active_command": self.data.get("active_command"),
                    "report": str(self.path),
                },
            )
            _atomic_json(
                self.status_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": self.data.get("status"),
                    "current_phase": self.data.get("current_phase"),
                    "heartbeat_at": self.data.get("heartbeat_at"),
                    "completed_cases": self.data.get("completed_cases", 0),
                    "last_error": (
                        self.data.get("errors", [])[-1]
                        if self.data.get("errors")
                        else None
                    ),
                },
            )
            self._write_html()

    def _write_html(self) -> None:
        status = html.escape(str(self.data.get("status", "unknown")))
        phase = html.escape(str(self.data.get("current_phase", "")))
        heartbeat = html.escape(str(self.data.get("heartbeat_at", "")))
        rows = []
        for case in self.data.get("cases", []):
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(case.get('id', '')))}</td>"
                f"<td>{html.escape(str(case.get('status', '')))}</td>"
                f"<td>{html.escape(str(case.get('elapsed_seconds', '')))}</td>"
                f"<td><code>{html.escape(str(case.get('error', '')))}</code></td>"
                "</tr>"
            )
        raw = html.escape(json.dumps(_json_ready(self.data), indent=2))
        page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="refresh" content="30"><title>Show4DSTEM overnight signoff</title>
<style>body{{font:14px system-ui;margin:24px;color:#17212b}}.status{{font-weight:700}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3da;padding:6px;text-align:left}}
pre{{white-space:pre-wrap;background:#f5f7f9;padding:12px;border-radius:6px}}code{{font-size:12px}}</style></head>
<body><h1>Show4DSTEM folder endurance overnight</h1>
<p class="status">{status} · {phase}</p><p>Heartbeat: {heartbeat}</p>
<p>This live backend report refreshes every 30 seconds. Browser/Jupyter evidence is a separate gate.</p>
<table><thead><tr><th>Case</th><th>Status</th><th>Seconds</th><th>Error</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table><h2>Machine-readable report</h2>
<p><a href="report.json">report.json</a> · <a href="status.json">status.json</a> ·
<a href="events.jsonl">events.jsonl</a> · <a href="gpu-telemetry.jsonl">gpu-telemetry.jsonl</a></p>
<details><summary>Current report</summary><pre>{raw}</pre></details></body></html>"""
        temporary = self.artifact_dir / f".index.{os.getpid()}.tmp"
        temporary.write_text(page, encoding="utf-8")
        os.replace(temporary, self.artifact_dir / "index.html")


def _child_state(widget: Any) -> dict[str, Any]:
    acquisitions = list(widget._folder_acquisitions)
    return {
        "n_frames": int(widget.n_frames),
        "page_idx": int(widget.compare_page_idx),
        "page_count": int(widget.compare_page_count),
        "panel_indices": list(widget.compare_panel_indices),
        "watch_state": str(widget.folder_watch_state),
        "watch_detail": str(widget.folder_watch_detail),
        "acquisitions": len(acquisitions),
        "resident_bytes": sum(int(item.resident_bytes) for item in acquisitions),
        "logical_bytes": sum(int(item.logical_bytes) for item in acquisitions),
    }


def _memory_sample() -> dict[str, Any]:
    """Allocator and device memory after one cycle.

    Viewer work (virtual images, diffraction reads) allocates through Torch, so
    its counter is the leak signal. Encoded acquisitions allocate outside the
    Torch pool; only the device total sees them, and it also counts other
    processes.
    """
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "time": _utc_now(),
        "torch_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "torch_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
        "device_used_mib": round((total - free) / 2**20, 1),
    }


def _touch_page(widget: Any, page: int) -> dict[str, Any]:
    """Show one comparison page and check that every visible panel is a finite image."""
    import numpy as np

    started = time.perf_counter()
    widget.set_compare_page(int(page))
    indices = [int(value) for value in widget.compare_panel_indices]
    panels = np.frombuffer(widget.compare_virtual_image_bytes, dtype=np.float32)
    elapsed = time.perf_counter() - started
    expected_values = len(indices) * int(widget.shape_rows) * int(widget.shape_cols)
    if not indices or panels.size != expected_values or not np.isfinite(panels).all():
        raise RuntimeError(
            f"Page {page} published {len(indices)} panel(s) with {panels.size} "
            f"virtual-image values; expected {expected_values} finite values."
        )
    return {
        "requested_page": int(page),
        "page_idx": int(widget.compare_page_idx),
        "panel_indices": indices,
        "elapsed_seconds": round(elapsed, 6),
    }


def _run_cycle(widget: Any, result: dict[str, Any], canonical: list[int], page_count: int) -> None:
    """One canonical cycle: pages, rapid navigation, curation, diffraction modes."""
    for page in canonical:
        result["page_actions"].append(_touch_page(widget, page))

    if page_count >= 3:
        rapid_started = time.perf_counter()
        widget.set_compare_page(0)
        widget.set_compare_page(1)
        landed = _touch_page(widget, 2)
        if landed["page_idx"] != 2:
            raise RuntimeError("Rapid page 1 -> 2 -> 3 navigation finished on the wrong page.")
        result["rapid_navigation"] = {
            **landed,
            "elapsed_seconds": round(time.perf_counter() - rapid_started, 6),
        }

    if int(widget.n_frames) >= 3:
        widget.star_compare_panel(2)
        widget.hide_compare_panel(1)
        hidden = list(widget.compare_hidden_panels)
        starred = list(widget.compare_starred_panels)
        widget.show_compare_panel(1)
        widget.unstar_compare_panel(2)
        result["curation"] = {
            "hidden_during": hidden,
            "starred": starred,
            "hidden_after_restore": list(widget.compare_hidden_panels),
            "passed": 1 in hidden and 2 in starred,
        }
        if not result["curation"]["passed"]:
            raise RuntimeError("Hide/star state did not persist through the cycle.")

    hashes: dict[str, str] = {}
    for mode in ("selected", "average", "selected"):
        widget.compare_dp_mode = mode
        hashes[mode] = hashlib.sha256(bytes(widget.frame_bytes)).hexdigest()
    result["diffraction_modes"] = hashes


def _run_child(args: argparse.Namespace) -> int:
    result_path = args.result_path.resolve()
    result: dict[str, Any] = {
        "id": args.case_id,
        "status": "running",
        "case_kind": args.case_kind,
        "started_at": _utc_now(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "page_actions": [],
        "cycle_memory": [],
        "cycles_completed": 0,
        "errors": [],
    }
    _atomic_json(result_path, result)
    widget = None
    started = time.perf_counter()
    try:
        import torch

        from quantem.widget import Show4DSTEM

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU is visible in the child; check CUDA_VISIBLE_DEVICES."
            )
        result["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": {
                "name": torch.cuda.get_device_name(0),
                "total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            },
        }
        result["memory_before_open"] = _memory_sample()
        open_started = time.perf_counter()
        widget = Show4DSTEM.from_folder(
            args.source,
            pattern=args.pattern,
            recursive=True,
            ready_only=True,
            backend="cuda",
            device=0,
            view_mode="multiple",
            columns=args.columns,
            page_size=args.page_size,
            watch=True,
            watch_interval=args.watch_interval,
            precompute_virtual_images=False,
            compare_dp_mode="selected",
            title=f"Show4DSTEM overnight · {args.case_id}",
        )
        result["first_viewer_seconds"] = round(time.perf_counter() - open_started, 6)
        widget.wait_for_folder(timeout=args.fill_timeout)
        if widget._folder_fill_thread.is_alive():
            raise RuntimeError(
                f"The folder did not finish loading within {args.fill_timeout} s."
            )
        result["folder_fill_seconds"] = round(time.perf_counter() - open_started, 6)
        if int(widget.n_frames) < int(args.min_ready):
            raise RuntimeError(
                f"Only {widget.n_frames} ready masters; require {args.min_ready}."
            )
        result["initial_state"] = _child_state(widget)
        result["memory_after_open"] = _memory_sample()
        page_count = max(1, int(widget.compare_page_count))
        canonical = [0, min(1, page_count - 1), page_count - 1, 0]

        while (
            result["cycles_completed"] < args.cycles
            or time.time() < args.deadline_epoch
        ):
            _run_cycle(widget, result, canonical, page_count)
            result["cycles_completed"] += 1
            result["cycle_memory"].append(_memory_sample())
            result["last_state"] = _child_state(widget)
            _atomic_json(result_path, result)

        result["final_state"] = _child_state(widget)
        first, last = result["cycle_memory"][0], result["cycle_memory"][-1]
        growth = last["torch_allocated_mib"] - first["torch_allocated_mib"]
        result["correctness"] = {
            "all_pages_finite": True,
            "torch_allocated_growth_mib": round(growth, 1),
            "bounded_memory": growth <= args.max_memory_growth_mib,
            "watch_worker_alive": bool(
                widget._folder_watch_thread is not None
                and widget._folder_watch_thread.is_alive()
            ),
        }
        if not result["correctness"]["bounded_memory"]:
            raise RuntimeError(
                f"Torch allocated memory grew {growth:.1f} MiB over "
                f"{result['cycles_completed']} cycles; limit {args.max_memory_growth_mib} MiB."
            )
        if not result["correctness"]["watch_worker_alive"]:
            raise RuntimeError("The folder watcher stopped during the run.")
        fatal_text = json.dumps(_json_ready(result)).lower()
        if any(token in fatal_text for token in FATAL_TOKENS):
            raise RuntimeError("A fatal CUDA allocation/runtime token was recorded.")
        result["status"] = "pass"
    except BaseException as exc:
        # Record every failure, interrupts included, so the controller sees a
        # result file instead of an unexplained exit code.
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["errors"].append(result["error"])
    finally:
        if widget is not None:
            try:
                widget.close()
            except (RuntimeError, OSError, ValueError) as exc:
                result.setdefault("cleanup_errors", []).append(
                    f"close: {type(exc).__name__}: {exc}"
                )
            workers = {
                name: bool(
                    getattr(widget, name, None) is not None
                    and getattr(widget, name).is_alive()
                )
                for name in ("_folder_watch_thread", "_folder_fill_thread")
            }
            result["workers_alive_after_close"] = workers
            if any(workers.values()):
                result.setdefault("cleanup_errors", []).append(
                    "a Show4DSTEM folder worker remained alive after close"
                )
            if widget._folder_acquisitions:
                result.setdefault("cleanup_errors", []).append(
                    "close() left folder acquisitions open"
                )
            result["memory_after_close"] = _memory_sample()
        result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        result["ended_at"] = _utc_now()
        if result.get("cleanup_errors") and result.get("status") == "pass":
            result["status"] = "fail"
            result["error"] = "; ".join(result["cleanup_errors"])
        _atomic_json(result_path, result)
    return 0 if result.get("status") == "pass" else 1


def _wait_for_idle(
    report: LiveReport,
    args: argparse.Namespace,
    *,
    phase: str,
) -> bool:
    deadline = time.monotonic() + args.wait_hours * 3600.0
    consecutive = 0
    while time.monotonic() < deadline:
        snapshot = _gpu_snapshot()
        idle, reasons = _idle_decision(
            snapshot,
            [args.device],
            max_utilization=args.max_idle_utilization,
            min_free_mib=args.min_free_mib,
            block_patterns=args.block_pattern,
        )
        consecutive = consecutive + 1 if idle else 0
        report.gpu(snapshot, phase=phase)
        report.update(
            status="waiting_for_gpu",
            current_phase=phase,
            selected_physical_gpu=args.device,
            idle_consecutive_samples=consecutive,
            idle_required_samples=args.idle_samples,
            idle_block_reasons=reasons,
            latest_gpu=snapshot,
        )
        report.event(
            "gpu_idle_sample",
            phase=phase,
            idle=idle,
            consecutive=consecutive,
            reasons=reasons,
        )
        if consecutive >= args.idle_samples:
            return True
        time.sleep(args.idle_sample_seconds)
    report.add_error(
        {
            "time": _utc_now(),
            "phase": phase,
            "error": f"GPU idle wait exceeded {args.wait_hours} hours",
        }
    )
    return False


def _run_case(
    report: LiveReport,
    args: argparse.Namespace,
    *,
    case_id: str,
    case_kind: str,
    cycles: int,
    deadline_epoch: float,
) -> bool:
    if report.completed(case_id):
        report.event("case_resume_skip", case_id=case_id)
        return True
    if not _wait_for_idle(report, args, phase=f"wait:{case_id}"):
        return False
    case_dir = report.artifact_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    command = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--child",
        "--case-id",
        case_id,
        "--case-kind",
        case_kind,
        "--source",
        str(args.source),
        "--result-path",
        str(result_path),
        "--pattern",
        args.pattern,
        "--columns",
        str(args.columns),
        "--page-size",
        str(args.page_size),
        "--min-ready",
        str(args.min_ready),
        "--cycles",
        str(cycles),
        "--deadline-epoch",
        str(deadline_epoch),
        "--fill-timeout",
        str(args.fill_timeout),
        "--watch-interval",
        str(args.watch_interval),
        "--max-memory-growth-mib",
        str(args.max_memory_growth_mib),
    ]
    env = dict(os.environ)
    tool_cache = report.artifact_dir / "cache"
    # Keep any inherited source override (for example a quantem.gpu checkout)
    # after this repository's src, so the child imports what the controller does.
    python_path = [str(args.repo / "src"), *filter(None, [env.get("PYTHONPATH")])]
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.device),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(python_path),
            "TMPDIR": str(report.artifact_dir / "tmp"),
            "XDG_CACHE_HOME": str(tool_cache / "xdg"),
            "CUPY_CACHE_DIR": str(tool_cache / "cupy"),
            "MPLCONFIGDIR": str(tool_cache / "matplotlib"),
        }
    )
    for key in ("TMPDIR", "XDG_CACHE_HOME", "CUPY_CACHE_DIR", "MPLCONFIGDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    report.update(status="running", current_phase=case_id, active_command=command)
    report.event(
        "case_start",
        case_id=case_id,
        kind=case_kind,
        physical_gpu=args.device,
        command=command,
    )
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        timed_out = False
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > args.case_timeout_hours * 3600.0:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                break
            snapshot = _gpu_snapshot()
            report.gpu(snapshot, phase=case_id)
            report.update(
                status="running",
                current_phase=case_id,
                active_pid=process.pid,
                active_elapsed_seconds=round(elapsed, 3),
                latest_gpu=snapshot,
            )
            time.sleep(args.heartbeat_seconds)
    try:
        case = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        case = {
            "id": case_id,
            "status": "fail",
            "error": f"missing/invalid child result: {type(exc).__name__}: {exc}",
        }
    case.update(
        {
            "id": case_id,
            "kind": case_kind,
            "physical_gpu": args.device,
            "child_exit_code": process.returncode,
            "timed_out": timed_out,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
    )
    if timed_out or process.returncode != 0:
        case["status"] = "fail"
    report.append_case(case)
    report.event("case_end", case_id=case_id, status=case.get("status"))
    if case.get("status") != "pass":
        report.add_error(
            {
                "time": _utc_now(),
                "phase": case_id,
                "error": case.get("error", f"child exited {process.returncode}"),
            }
        )
        return False
    return True


def _run_controller(args: argparse.Namespace) -> int:
    args.source = args.source.expanduser().resolve()
    args.artifact_dir = args.artifact_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    try:
        args.artifact_dir.relative_to(args.source)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"artifact directory must be outside the real-data source: {args.artifact_dir}"
        )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if not args.source.is_dir():
        raise FileNotFoundError(f"source folder not found: {args.source}")
    initial = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "starting",
        "started_at": _utc_now(),
        "heartbeat_at": _utc_now(),
        "current_phase": "preflight",
        "completed_cases": 0,
        "restart_count": 0,
        "provenance": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "command": sys.argv,
            "git": _git_snapshot(args.repo),
        },
        "data": {
            "source": str(args.source),
            "pattern": args.pattern,
            "masters_discovered": len(list(args.source.rglob(args.pattern))),
            "min_ready": args.min_ready,
            "physical_gpu": args.device,
        },
        "environment": {
            "source_filesystem": _filesystem_snapshot(args.source),
            "report_filesystem": _filesystem_snapshot(args.artifact_dir),
            "initial_gpu": _gpu_snapshot(),
        },
        "cases": [],
        "errors": [],
        "gates": [
            {"id": "S4D-20-backend", "status": "pending"},
            {
                "id": "S4D-14-live-arrival-browser",
                "status": "pending",
                "reason": "companion staged live-Jupyter/browser drive required",
            },
            {
                "id": "S4D-20-browser",
                "status": "pending",
                "reason": "actual browser paint/FPS evidence required",
            },
        ],
    }
    report = LiveReport(args.artifact_dir, initial)
    report.event("controller_start", run_id=args.run_id)
    if args.dry_run:
        report.update(status="planned", current_phase="dry_run", ended_at=_utc_now())
        return 0

    run_started = time.time()
    all_passed = True
    open_ids = [f"open-{idx}" for idx in range(1, args.opens + 1)]
    for case_id in open_ids:
        passed = _run_case(
            report,
            args,
            case_id=case_id,
            case_kind="open",
            cycles=1,
            deadline_epoch=0.0,
        )
        all_passed = all_passed and passed
        if not passed and not args.continue_on_failure:
            break
    if all_passed or args.continue_on_failure:
        passed = _run_case(
            report,
            args,
            case_id="endurance",
            case_kind="endurance",
            cycles=args.min_cycles,
            deadline_epoch=run_started + args.hours * 3600.0,
        )
        all_passed = all_passed and passed

    cases_by_id = {item.get("id"): item for item in report.data.get("cases", [])}
    opens = [cases_by_id.get(case_id, {}) for case_id in open_ids]
    endurance = cases_by_id.get("endurance", {})
    aggregates = {
        "first_viewer_seconds": [case.get("first_viewer_seconds") for case in opens],
        "folder_fill_seconds": [case.get("folder_fill_seconds") for case in opens],
        "open_page_seconds": [
            action.get("elapsed_seconds")
            for case in opens
            for action in case.get("page_actions", [])
        ],
        "endurance_cycles": endurance.get("cycles_completed"),
        "endurance_memory_growth_mib": endurance.get("correctness", {}).get(
            "torch_allocated_growth_mib"
        ),
        "note": (
            "backend completion timings only; browser paint thresholds remain "
            "a separate pending gate"
        ),
    }
    gates = list(report.data.get("gates", []))
    for gate in gates:
        if gate["id"] == "S4D-20-backend":
            final_state = endurance.get("final_state", {})
            gate["status"] = (
                "pass"
                if all(report.completed(case_id) for case_id in [*open_ids, "endurance"])
                else "fail"
            )
            gate["observed"] = {
                "datasets": final_state.get("n_frames"),
                "resident_bytes": final_state.get("resident_bytes"),
                "logical_bytes": final_state.get("logical_bytes"),
                "cycles": endurance.get("cycles_completed"),
                "torch_allocated_growth_mib": aggregates["endurance_memory_growth_mib"],
            }
    final_status = "backend_pass_browser_pending" if all_passed else "fail"
    report.update(
        status=final_status,
        current_phase="complete",
        gates=gates,
        aggregates=aggregates,
        ended_at=_utc_now(),
        active_pid=None,
    )
    report.event("controller_end", status=final_status)
    return 0 if all_passed else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "local-only real-data Show4DSTEM from_folder one-GPU endurance "
            "overnight signoff over encoded acquisitions"
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run-id", default=f"show4dstem-{datetime.now():%Y%m%d-%H%M%S}")
    parser.add_argument("--pattern", default="*_master.h5")
    parser.add_argument("--device", type=int, default=0, help="physical NVIDIA GPU index")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--min-ready", type=int, default=82)
    parser.add_argument("--opens", type=int, default=5, help="fresh-process open cases before endurance")
    parser.add_argument("--hours", type=float, default=4.0, help="clock budget for the whole run")
    parser.add_argument("--min-cycles", type=int, default=100)
    parser.add_argument("--max-memory-growth-mib", type=float, default=256.0)
    parser.add_argument("--case-timeout-hours", type=float, default=8.0)
    parser.add_argument("--wait-hours", type=float, default=24.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--idle-sample-seconds", type=float, default=60.0)
    parser.add_argument("--idle-samples", type=int, default=5)
    parser.add_argument("--max-idle-utilization", type=int, default=15)
    parser.add_argument("--min-free-mib", type=int, default=16384)
    parser.add_argument(
        "--block-pattern",
        action="append",
        default=list(DEFAULT_BLOCK_PATTERNS),
        help="case-insensitive foreign command substring that blocks launch",
    )
    parser.add_argument("--fill-timeout", type=float, default=3600.0)
    parser.add_argument("--watch-interval", type=float, default=2.0)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--case-kind", default="", help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cycles", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--deadline-epoch", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.page_size < 1 or args.columns < 1 or args.device < 0:
        raise ValueError("page-size and columns must be positive and device non-negative")
    if args.child:
        if args.result_path is None or not args.case_id or not args.case_kind:
            raise ValueError("child mode requires case id/kind and result path")
        return _run_child(args)
    if args.artifact_dir is None:
        raise ValueError("controller mode requires --artifact-dir")
    return _run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
