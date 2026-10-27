"""ScanControl: run STEM scans from a notebook through an acquisition runner.

The widget is the window; the runner does the work. It accepts any object with the
``scan_runner.Runner`` interface (``start(step, *args)``, ``status()``, ``stop()``,
``close()``): connect to the detector and scan controller, set the detector up, run N scans,
and list, download or delete the detector's files. Every hardware call stays in the runner, so
this module knows no addresses and moves no beam by itself; the beam moves only when the
operator presses Go.
"""

from __future__ import annotations

import math
import re
import threading
import weakref
from collections import namedtuple
from pathlib import Path

import anywidget
import traitlets

#: The settings a runner's ``connect``, ``configure`` and ``run`` take. A runner that compares
#: settings by value (``scan_runner.Runner``) accepts any namedtuple with these fields.
ScanSettings = namedtuple("ScanSettings", "size binned output save")

SIZES = (256, 512, 1024)
FOREVER = 10_000


def _poll(ref, stop: threading.Event, interval: float) -> None:
    """Copy the runner's status into the widget until it closes or is collected."""
    while not stop.wait(interval):
        widget = ref()
        if widget is None:
            return
        widget.refresh()
        del widget


def _heartbeat(runner, stop: threading.Event, interval: float) -> None:
    """Re-read HV, detector position and microscope detectors every interval s (read only)."""
    while not stop.wait(interval):
        try:
            runner.read_hardware()
        except Exception:
            pass


def scan_name(file_name: str) -> str:
    """The scan a detector file belongs to: ``series_7_master.h5`` and ``series_7_data_000001.h5``
    are both ``series_7``; any other file is its own entry."""
    return re.sub(r"_(master|data_\d+)\.h5$", "", file_name)


def dwell_us(binned: bool) -> float:
    """Dwell per scan position: binned 120 kHz, unbinned 30 kHz."""
    return 1e6 / (120_000 if binned else 30_000)


class ScanControl(anywidget.AnyWidget):
    """Connect, set up and run scans, with the detector's files, in one panel.

    Parameters
    ----------
    runner : scan_runner.Runner or compatible
        Owns every detector and scan-controller call; the widget only starts its steps
        (``start("connect" | "configure" | "run" | "list_files" | "download" | "delete", ...)``),
        reads ``status()`` and calls ``stop()``.
    size : int, default 512
        Square scan side; one of ``sizes``.
    binned : bool, default True
        2x2 binning (120 kHz) or unbinned (30 kHz).
    output : {"rdma", "stream"}, default "rdma"
        RDMA to the processing PC, or the normal stream.
    save : bool, default False
        The detector also writes HDF5 files.
    count : int, default 1
        Scans per Go.
    folder : str, default ""
        Download folder for the detector's files.
    sizes : tuple of int
        The scan sizes offered.
    title : str, default "Scan control"
    heartbeat : float, default 5.0
        Seconds between read-only checks of HV, detector position and the microscope
        detectors (``runner.read_hardware()``); 0 turns it off.

    Notes
    -----
    Go moves the electron beam: press it only when the operator says go, with the beam on.
    Changing a setting while Ready sets the detector up again. Stop ends the run after the
    scan in progress. Deleting files asks for confirmation in the panel.
    """

    _esm = Path(__file__).parent / "static" / "scancontrol.js"

    status = traitlets.Dict().tag(sync=True)
    size = traitlets.Int(512).tag(sync=True)
    binned = traitlets.Bool(True).tag(sync=True)
    output = traitlets.Enum(["rdma", "stream"], default_value="rdma").tag(sync=True)
    save = traitlets.Bool(False).tag(sync=True)
    count = traitlets.Int(1).tag(sync=True)
    forever = traitlets.Bool(False).tag(sync=True)
    folder = traitlets.Unicode("").tag(sync=True)
    sizes = traitlets.List(traitlets.Int()).tag(sync=True)
    title = traitlets.Unicode("Scan control").tag(sync=True)

    def __init__(self, runner, *, size: int = 512, binned: bool = True, output: str = "rdma",
                 save: bool = False, count: int = 1, folder: str = "", sizes=SIZES,
                 title: str = "Scan control", interval: float = 0.25, heartbeat: float = 5.0) -> None:
        for name in ("start", "status", "stop", "close"):
            if not callable(getattr(runner, name, None)):
                raise TypeError(f"ScanControl needs a runner with {name}(); pass scan_runner.Runner(...).")
        if not math.isfinite(interval) or not 0.05 <= interval <= 5:
            raise ValueError("interval must be 0.05 to 5 seconds; use interval=0.25.")
        self.runner = runner
        self._stop = threading.Event()
        self._file_info = {}
        super().__init__(sizes=[int(side) for side in sizes], size=int(size), binned=bool(binned),
                         output=output, save=bool(save), count=int(count), folder=str(folder),
                         title=str(title))
        self.on_msg(self._handle_message)
        self.refresh()
        self._worker = threading.Thread(target=_poll, args=(weakref.ref(self), self._stop, interval),
                                        name="ScanControl status", daemon=True)
        weakref.finalize(self, self._stop.set)
        self._worker.start()
        if heartbeat and callable(getattr(runner, "read_hardware", None)):
            threading.Thread(target=_heartbeat, args=(runner, self._stop, float(heartbeat)),
                             name="ScanControl heartbeat", daemon=True).start()

    # --- settings ------------------------------------------------------------------------------

    @property
    def settings(self) -> ScanSettings:
        """The scan settings the panel shows now."""
        return ScanSettings(self.size, self.binned, self.output, self.save)

    @property
    def scans(self) -> int:
        """Scans per Go: ``count``, or the "run forever" count."""
        return FOREVER if self.forever else self.count

    @traitlets.validate("size")
    def _validate_size(self, proposal):
        value = int(proposal["value"])
        if self.sizes and value not in self.sizes:
            raise traitlets.TraitError(f"size must be one of {self.sizes}, got {value}.")
        return value

    @traitlets.validate("count")
    def _validate_count(self, proposal):
        value = int(proposal["value"])
        if not 1 <= value <= FOREVER:
            raise traitlets.TraitError(f"count must be 1 to {FOREVER}, got {value}.")
        return value

    @traitlets.observe("size", "binned", "output", "save")
    def _settings_changed(self, change) -> None:
        """A change while Ready sets the detector up again (as the scan window does)."""
        if self.status.get("state") == "READY":
            self.runner.start("configure", self.settings)

    # --- actions -------------------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to the detector and scan controller and set the detector up."""
        return self.runner.start("connect", self.settings)

    def go(self) -> bool:
        """*** Moves the beam. *** Run ``scans`` scans with the panel's settings."""
        return self.runner.start("run", self.settings, self.scans)

    def stop(self) -> None:
        """End the run after the scan in progress."""
        self.runner.stop()

    def set_high_voltage(self, on: bool) -> bool:
        """Detector high voltage on or off."""
        return self.runner.start("set_high_voltage", bool(on))

    def insert_sensor(self) -> bool:
        """Move the detector in."""
        return self.runner.start("insert_sensor")

    def retract_sensor(self) -> bool:
        return self.runner.start("retract_sensor")

    def check_microscope(self) -> bool:
        """Read the microscope detectors (Ceta, STEM) through the runner, read only."""
        read = getattr(self.runner, "read_microscope", None)
        if not callable(read):
            return False
        threading.Thread(target=read, name="ScanControl microscope", daemon=True).start()
        return True

    def refresh_files(self) -> bool:
        return self.runner.start("list_files")

    def files_of(self, scans) -> list:
        """The detector files of these scans (a master and its data files)."""
        wanted = set(scans)
        return [name for name in self.status.get("files", []) if scan_name(name) in wanted]

    def download(self, scans, format: str = "h5") -> bool:
        """Download whole scans. "h5": the detector's files as they are. "qem": download, then
        re-save ANS-encoded with quantem.gpu (one .qem per scan) and remove the downloaded .h5."""
        if not self.folder:
            raise ValueError("Set folder before downloading.")
        if format not in ("h5", "qem"):
            raise ValueError("format must be 'h5' or 'qem'.")
        scans, folder = list(scans), Path(self.folder)
        if format == "h5":
            return self.runner.start("download", self.files_of(scans), str(folder))
        staging = folder / ".h5-staging"
        if not self.runner.start("download", self.files_of(scans), str(staging)):
            return False
        threading.Thread(target=self._to_qem, args=(scans, staging, folder), name="ScanControl qem",
                         daemon=True).start()
        return True

    def _to_qem(self, scans, staging: Path, folder: Path) -> None:
        busy = getattr(self.runner, "busy", None)
        while callable(busy) and busy():
            self._stop.wait(0.2)
        if self.runner.status().state == "ERROR":
            return
        try:
            from quantem.gpu import io
            for index, scan in enumerate(scans, 1):
                self._notice(f"Saving {scan}.qem ({index} of {len(scans)}): ANS on the GPU...")
                target = folder / f"{scan}.qem"
                if target.exists():
                    raise FileExistsError(f"{target} exists; delete it or choose another folder.")
                data = io.load(staging / f"{scan}_master.h5", device="cuda:0")
                try:
                    io.save(target, data)
                finally:
                    data.close()
                for path in staging.glob(f"{scan}_*.h5"):
                    path.unlink()
            try:
                staging.rmdir()
            except OSError:
                pass
            self._notice(f"Saved {len(scans)} scan(s) as .qem in {folder}", error=False)
        except Exception as exc:
            self._notice(f".qem save failed: {exc} (the downloaded .h5 stays in {staging})")

    def _notice(self, text: str, error: bool = True) -> None:
        self.status = {**self.status, "notice": text, "notice_error": error}

    def delete(self, scans) -> bool:
        return self.runner.start("delete", self.files_of(scans))

    def _handle_message(self, _widget, content, _buffers) -> None:
        action = content.get("action") if isinstance(content, dict) else None
        try:
            if action == "connect":
                started = self.connect()
            elif action == "go":
                started = self.go()
            elif action == "stop":
                self.stop()
                started = True
            elif action == "high_voltage":
                started = self.set_high_voltage(content.get("on", False))
            elif action == "insert":
                started = self.insert_sensor()
            elif action == "retract":
                started = self.retract_sensor()
            elif action == "microscope":
                started = self.check_microscope()
            elif action == "refresh":
                started = self.refresh_files()
            elif action == "download":
                started = self.download(content.get("scans", []), content.get("format", "h5"))
            elif action == "delete":
                started = self.delete(content.get("scans", []))
            else:
                return
        except Exception as exc:            # shown in the panel; the runner keeps its own state
            self.status = {**self.status, "notice": str(exc), "notice_error": True}
            return
        if not started:
            self.status = {**self.status, "notice": "Busy: wait for the current step to finish.",
                           "notice_error": True}
        self.refresh()

    # --- status --------------------------------------------------------------------------------

    def refresh(self) -> None:
        """Copy the runner's status into the panel."""
        st = self.runner.status()
        fields = st._asdict() if hasattr(st, "_asdict") else dict(st)
        status = {key: fields.get(key) for key in ("state", "message", "dcu_ok", "revolon_ok",
                                                    "high_voltage", "done", "total", "dropped",
                                                    "files", "free_bytes", "sensor", "microscope")}
        status["files"] = list(status["files"] or [])
        status["scans"] = self._scans(status["files"])
        status["dwell_us"] = dwell_us(self.binned)
        busy = getattr(self.runner, "busy", None)
        status["busy"] = bool(busy()) if callable(busy) else status["state"] in ("BUSY", "SCANNING", "STOPPING")
        if self.status.get("notice") and status["state"] == self.status.get("state"):
            status["notice"] = self.status["notice"]
            status["notice_error"] = self.status.get("notice_error", False)
        if status != self.status:
            self.status = status

    def _scans(self, files) -> list:
        """One row per scan: name, file count, bytes and last modified, from the runner's
        optional ``file_info(names) -> {name: (bytes, modified)}`` (cached per file)."""
        info = getattr(self.runner, "file_info", None)
        unknown = [name for name in files if name not in self._file_info]
        if unknown and callable(info) and not self.status.get("busy"):
            try:
                self._file_info.update(info(unknown))
            except Exception:
                pass
        rows = {}
        for name in files:
            row = rows.setdefault(scan_name(name), {"name": scan_name(name), "files": 0, "bytes": 0,
                                                    "modified": ""})
            size, modified = self._file_info.get(name, (None, ""))
            row["files"] += 1
            row["bytes"] = None if size is None or row["bytes"] is None else row["bytes"] + size
            row["modified"] = max(row["modified"], modified or "")
        return sorted(rows.values(), key=lambda row: row["name"])

    def close(self) -> None:
        """Stop polling, stop after the scan in progress, and disconnect the scan controller."""
        stop = getattr(self, "_stop", None)        # absent when the constructor rejected its input
        if stop is None:
            super().close()
            return
        stop.set()
        try:
            self.runner.close()
        finally:
            super().close()
