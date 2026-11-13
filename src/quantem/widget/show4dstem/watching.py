"""A folder a detector writes acquisitions into, read as a recording whose newest scans stay loaded on the GPU."""

import threading
from pathlib import Path

from quantem.widget.adapters import gpu as gpu_adapter

# Newer files than the ones kept that are looked at in one pass: scans still being written sit there.
ARRIVING_LOOKAHEAD = 4


class WatchedFolder:
    """The newest complete acquisitions under one folder, loaded as they finish arriving.

    ``Show4DSTEM.stream`` follows it as a recording (``lock``, ``datasets``, ``receiving``, ``error``). One thread
    looks at the folder every ``refresh_s``. A file is read once it is complete and has stopped changing between
    two looks (a detector writes the master before its data files), so a scan in the middle of being written is
    never opened. The newest ``latest`` scans stay on the GPU, oldest first in ``datasets``; a scan that leaves
    that set is closed after the viewer has let go of it, so its memory is free before the next one loads. When
    the device cannot hold ``latest`` scans, the older ones are released and one line says how many it holds.
    """

    select_newest = True

    def __init__(self, folder, *, latest: int, pattern: str, recursive: bool, refresh_s: float,
                 device: str | None, hot_pixel_correction: str, scan_shape=None):
        self.folder = Path(folder).expanduser()
        if not self.folder.is_dir():
            raise FileNotFoundError(f"{self.folder} is not a folder to watch.")
        if int(latest) != latest or latest < 1:
            raise ValueError("latest is how many of the newest scans stay loaded: a whole number, 1 or more.")
        self.latest = int(latest)
        self.pattern, self.recursive, self.refresh_s = pattern, recursive, refresh_s
        self.device, self.hot_pixel_correction, self.scan_shape = device, hot_pixel_correction, scan_shape
        self.lock = threading.RLock()
        self.datasets: list = []
        self.paths: list[Path] = []
        self.receiving = True
        self.error = None
        self.viewer = None
        # what each candidate looked like at the previous look: a file is read once it stops changing
        self._fingerprints: dict[Path, tuple] = {}
        # files that could not be loaded, by fingerprint: tried again only after they change
        self._declined: dict[Path, tuple] = {}
        self._device_holds = self.latest
        # held during a look; a fit on a watched scan holds it too, so no scan loads or leaves under the fit
        self.busy = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def title(self) -> str:
        return f"Watching {self.folder.name} · {len(self.datasets)} of the latest {self.latest}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._follow, name="Show4DSTEM-watch", daemon=True)
        self._thread.start()

    def close(self, timeout: float = 120.0) -> None:
        """Stop looking, then close the scans this folder loaded. A load in progress finishes first."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("Show4DSTEM is still loading a scan from the watched folder. Wait for it to "
                                   "finish, then call close() again.")
        self.receiving = False
        with self.lock:
            datasets, self.datasets, self.paths = self.datasets, [], []
        for dataset in datasets:
            dataset.close()
        self.viewer = None

    def look(self) -> bool:
        """Look at the folder once: release the scans that left the newest ``latest``, load the ones that arrived.

        Returns whether the loaded scans changed.
        """
        with self.busy:
            target = self._settled_newest()
            changed = False
            for path in [path for path in self.paths if path not in target]:
                self._release(path)
                changed = True
            for path in target:
                if path not in self.paths and not self._stop.is_set():
                    changed |= self._load(path, newest=path == target[-1])
            return changed

    # ---

    def _follow(self) -> None:
        while not self._stop.is_set():
            try:
                self.look()
            except (OSError, ValueError, RuntimeError, MemoryError) as error:
                # the viewer shows it and stops following; the scans already loaded stay open until close()
                self.error = error
                return
            self._stop.wait(self.refresh_s)

    def _settled_newest(self) -> list[Path]:
        """The newest complete files, oldest first: at most as many as the device holds.

        Candidates are ordered by the time they last changed. A file counts once ``inspect`` reports it ready
        and it looks the same as at the previous look; a loaded file counts without being inspected again.
        """
        found = self.folder.rglob(self.pattern) if self.recursive else self.folder.glob(self.pattern)
        candidates = []
        for path in found:
            try:
                if path.is_file() and not path.name.startswith("."):
                    candidates.append((path.stat().st_mtime, path.name, path))
            except OSError:
                continue  # removed between the listing and the look
        candidates.sort()
        limit = min(self.latest, self._device_holds)
        fingerprints, settled = {}, []
        for _, _, path in reversed(candidates[-(limit + ARRIVING_LOOKAHEAD):]):
            if path in self.paths:
                settled.append(path)
            else:
                fingerprint = self._fingerprint(path)
                if fingerprint is not None:
                    fingerprints[path] = fingerprint
                    if self._fingerprints.get(path) == fingerprint and self._declined.get(path) != fingerprint:
                        settled.append(path)
            if len(settled) == limit:
                break
        self._fingerprints = fingerprints
        return settled[::-1]

    def _fingerprint(self, path: Path) -> tuple | None:
        """What identifies a complete file's present contents, or None while it is incomplete."""
        try:
            inspection = gpu_adapter.inspect_master(path)
            status = path.stat()
        except (OSError, ValueError, KeyError):
            return None  # still being written: a truncated envelope, a missing data file, an unreadable header
        if not inspection.ready:
            return None
        return (repr(inspection.source_signature), status.st_size, status.st_mtime_ns)

    def _load(self, path: Path, *, newest: bool) -> bool:
        """Load one file, releasing the oldest scans first when the device has no room for it."""
        fingerprint = self._fingerprints.get(path)
        released_for_room = False
        while True:
            try:
                dataset = gpu_adapter.load_acquisition(
                    str(path), self.device, scan_shape=self.scan_shape,
                    hot_pixel_correction=self.hot_pixel_correction)
                break
            except (MemoryError, RuntimeError, OSError, ValueError) as error:
                out_of_memory = isinstance(error, MemoryError) or "out of memory" in str(error).lower()
                if out_of_memory and self.paths:
                    self._release(self.paths[0])
                    released_for_room = True
                    continue
                self._declined[path] = fingerprint
                reason = ("does not fit on the device even alone" if out_of_memory
                          else " ".join(str(error).split()) or type(error).__name__)
                print(f"Show4DSTEM.watch: skipped {path.name}: {reason}.")
                return released_for_room
        if self.datasets and tuple(dataset.shape) != tuple(self.datasets[0].shape):
            if not newest:
                dataset.close()
                self._declined[path] = fingerprint
                return released_for_room
            print(f"Show4DSTEM.watch: {path.name} is {tuple(dataset.shape)}, the scans before it "
                  f"{tuple(self.datasets[0].shape)}; one comparison holds one scan and detector shape, so they "
                  "were released.")
            for other in list(self.paths):
                self._release(other)
        with self.lock:
            self.datasets.append(dataset)
            self.paths.append(path)
            held = len(self.paths)
        if released_for_room and held < self._device_holds:
            self._device_holds = held
            print(f"Show4DSTEM.watch: the device holds {held} of these scans, not latest={self.latest}; "
                  "the older one was released. Scans stay lossless: nothing is binned to make room.")
        return True

    def _release(self, path: Path) -> None:
        """Drop one scan: the viewer closes its sessions on it first, so closing it frees the device."""
        with self.lock:
            index = self.paths.index(path)
            dataset = self.datasets.pop(index)
            self.paths.pop(index)
        viewer = self.viewer
        if viewer is not None and getattr(viewer, "_stream_update", None) is not None:
            viewer.refresh_stream()
        dataset.close()
        gpu_adapter.gpu_release_cached_memory(verbose=False)
