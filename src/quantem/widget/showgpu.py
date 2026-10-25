"""Live, read-only notebook resource monitoring."""

import math
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Literal

import anywidget
import psutil
import traitlets


def sample(*, users: bool = True) -> dict:
    """Collect driver telemetry and host memory, without allocating GPU arrays."""
    try:
        from quantem.gpu.device.telemetry import status
    except ImportError:
        snapshot = {
            "gpus": [],
            "error": "GPU telemetry needs a quantem.gpu version with device.status().",
            "timestamp": time.time(),
            "query_ms": 0.0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    else:
        snapshot = status(processes=users)
    if users and "processes" in snapshot:
        identities = {}
        groups = {}
        for process in snapshot.pop("processes"):
            pid = process["pid"]
            if pid not in identities:
                try:
                    identities[pid] = psutil.Process(pid).as_dict(
                        attrs=["username", "name"], ad_value=None
                    )
                except (psutil.Error, OSError):
                    identities[pid] = {"username": None, "name": None}
            identity = identities[pid]
            key = (process["gpu_uuid"], identity["username"])
            group = groups.setdefault(
                key,
                {
                    "name": identity["username"],
                    "used_bytes": 0,
                    "processes": 0,
                    "process_details": [],
                },
            )
            group["processes"] += 1
            group["process_details"].append(
                {
                    "pid": pid,
                    "name": identity["name"],
                    "used_bytes": process["used_bytes"],
                }
            )
            if process["used_bytes"] is None or group["used_bytes"] is None:
                group["used_bytes"] = None
            else:
                group["used_bytes"] += process["used_bytes"]
        for group in groups.values():
            group["process_details"].sort(key=lambda process: process["pid"])
        for gpu in snapshot["gpus"]:
            gpu["users"] = sorted(
                (group for (uuid, _), group in groups.items() if uuid == gpu["uuid"]),
                key=lambda group: (group["name"] is None, group["name"] or ""),
            )
    memory = psutil.virtual_memory()
    snapshot["system"] = {
        "total_bytes": memory.total,
        "available_bytes": memory.available,
        "used_bytes": memory.total - memory.available,
        "kernel_rss_bytes": psutil.Process().memory_info().rss,
    }
    return snapshot


def poll(reference: weakref.ReferenceType, stop: threading.Event) -> None:
    """Sample independently of the kernel execution loop, without retaining it."""
    while not stop.is_set():
        monitor = reference()
        if monitor is None:
            return
        interval = monitor.interval
        del monitor
        if stop.wait(interval):
            return
        monitor = reference()
        if monitor is None:
            return
        if monitor.running:
            monitor.refresh()
        del monitor


class ShowGPU(anywidget.AnyWidget):
    """Monitor live GPU and system memory on the notebook's server.

    Readings continue from a background thread while notebook code runs.
    This monitor never allocates GPU memory, synchronizes compute, releases
    caches or changes device selection. Driver rows include other processes.
    A timestamp and stale indicator distinguish live readings from saved output.

    Parameters
    ----------
    users : bool, default True
        Show compute users, their reported VRAM and process counts per GPU.
        Expand a user to inspect process names, IDs and individual VRAM use.
        Disable to avoid querying process ownership. Graphics and driver
        allocations remain included in device totals, not this breakdown.
    expand : bool, default True
        Start user process lists expanded. Each user row can still be opened
        or closed individually in the widget.
    size : {"small", "medium", "large", "xlarge"}, default "medium"
        Text and control size. Cards wrap to fit the notebook width. The
        widget's S/M/L/XL buttons change this without rerunning the cell.
    interval : float, default 1.0
        Seconds between driver samples, between 0.2 and 60. Polling is bounded
        and consecutive queries never overlap.
    running : bool, default True
        Start automatic sampling. Pause and Resume are also available in the
        widget; Refresh takes one sample while paused.

    Notes
    -----
    GPU telemetry currently supports NVIDIA drivers through quantem.gpu.
    System RAM remains available without NVIDIA. Physical driver indices may
    differ from CUDA logical indices; the compute visibility mask is shown.
    Close the widget when finished, or before replacing it in a rerun cell.
    A native extension holding Python's GIL can delay background sampling;
    stale readings are visibly marked. Browser commands may wait while the
    kernel is executing a cell.

    Examples
    --------
    >>> from quantem.widget import ShowGPU
    >>> monitor = ShowGPU(users=True, expand=True, size="medium")
    >>> monitor
    >>> monitor.close()  # stop sampling when finished
    """

    _esm = Path(__file__).parent / "static" / "showgpu.js"
    snapshot = traitlets.Dict().tag(sync=True)
    interval = traitlets.Float(1.0).tag(sync=True)
    running = traitlets.Bool(True).tag(sync=True)
    users = traitlets.Bool(True).tag(sync=True)
    expand = traitlets.Bool(True).tag(sync=True)
    size = traitlets.Enum(
        ["small", "medium", "large", "xlarge"], default_value="medium"
    ).tag(sync=True)

    def __init__(
        self,
        *,
        users: bool = True,
        expand: bool = True,
        size: Literal["small", "medium", "large", "xlarge"] = "medium",
        interval: float = 1.0,
        running: bool = True,
    ) -> None:
        self._stop = threading.Event()
        self._sample_lock = threading.Lock()
        self._sequence = 0
        self._worker = None
        super().__init__(
            users=users, expand=expand, size=size, interval=interval, running=running
        )
        self.on_msg(self.handle_message)
        self.refresh()
        self._worker = threading.Thread(
            target=poll,
            args=(weakref.ref(self), self._stop),
            name="ShowGPU telemetry",
            daemon=True,
        )
        weakref.finalize(self, self._stop.set)
        self._worker.start()

    @traitlets.validate("interval")
    def validate_interval(self, proposal):
        value = proposal["value"]
        if not math.isfinite(value) or not 0.2 <= value <= 60:
            raise traitlets.TraitError(
                "interval must be 0.2 to 60 seconds; use interval=1.0."
            )
        return value

    @traitlets.observe("users")
    def users_changed(self, change) -> None:
        """Refresh the ownership display even when automatic polling is paused."""
        if self._worker is not None:
            self.refresh()

    def refresh(self) -> dict:
        """Read one current sample, also when automatic sampling is paused.

        Returns
        -------
        dict
            Latest GPU and system-memory snapshot. Concurrent requests share
            the existing in-flight sample instead of starting another query.

        Examples
        --------
        >>> monitor.refresh()["system"]["available_bytes"]
        """
        if self._stop.is_set() or not self._sample_lock.acquire(blocking=False):
            return self.snapshot
        try:
            try:
                snapshot = sample(users=self.users)
            except (OSError, RuntimeError, psutil.Error) as exc:
                # Report a failed sample and retry on the next interval. Never
                # leave an old successful reading labelled as current.
                snapshot = {
                    "gpus": [],
                    "system": {},
                    "timestamp": time.time(),
                    "error": f"Telemetry unavailable: {exc}",
                    "query_ms": 0.0,
                }
            self._sequence += 1
            snapshot["sequence"] = self._sequence
            if not self._stop.is_set():
                self.snapshot = snapshot
            return self.snapshot
        finally:
            self._sample_lock.release()

    def handle_message(self, widget, content: dict, buffers) -> None:
        """Handle an explicit browser refresh."""
        if content.get("action") == "refresh":
            self.refresh()

    def close(self) -> None:
        """Stop automatic sampling and close the notebook display.

        Examples
        --------
        >>> monitor.close()
        """
        if hasattr(self, "_stop"):
            self._stop.set()
        worker = getattr(self, "_worker", None)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=4.5)
        super().close()
