"""Direct ptychography of the scans a live Show4DSTEM shows (``ShowPtycho.live``)."""

import threading
import time

import ipywidgets

from quantem.widget.adapters import gpu as gpu_adapter

# The side of the square scan region the aberrations are fitted on: the fit's sides are 128, 256, 512 or 1024.
FIT_REGION_SIDE = 256
FIT_REGION_SIDES = (128, 256, 512, 1024)


def central_region(scan_shape, side: int = FIT_REGION_SIDE) -> tuple[int, int, int, int] | None:
    """The square ``side`` x ``side`` scan region at the centre of ``scan_shape``, or None (the whole scan) for a scan
    no larger than it: (row_start, row_stop, col_start, col_stop)."""
    rows, cols = (int(value) for value in scan_shape)
    if rows <= side or cols <= side:
        return None
    row_start, col_start = (rows - side) // 2, (cols - side) // 2
    return (row_start, row_start + side, col_start, col_start + side)


class LivePtycho:
    """Direct ptychography of the scans a ``Show4DSTEM.live`` viewer shows, one ShowPtycho per scan.

    Each scan that arrives is fitted through fifth order on a square scan region (``scan_region``, the central
    256 x 256 unless set) and its whole phase is shown in a ShowPtycho below a one-line status. The fit holds the
    live folder, so no scan loads or leaves under it; when the next scan arrives the panel and its session close
    before the viewer lets the old scan go. ``scan_region = (row_start, row_stop, col_start, col_stop)`` fits the
    shown scan there, and every later scan; dragging the panel's fit box does the same. ShowPtycho.live returns
    it, already shown where the cell ran; it is not drawn again.
    """

    def __init__(self, viewer, *, scan_region=None, trials: int = 200):
        folder = getattr(viewer, "_live_folder", None)
        if folder is None:
            raise ValueError("ShowPtycho.live follows a live viewer: viewer = Show4DSTEM.live(folder, ...).")
        self.viewer, self.folder, self.trials = viewer, folder, int(trials)
        self._region = None if scan_region is None else self._checked(scan_region, None)
        self.ptycho = None  # the ShowPtycho of the scan shown
        self._direct = None  # its quantem.gpu.DirectPtycho session
        self._path = None  # the scan it shows
        self._rotation = None  # its fitted scan rotation, kept by refits of the same scan
        self._lock = threading.Lock()  # one fit at a time
        self.status = ipywidgets.HTML("<b>Direct ptycho</b> · waiting for the first scan")
        self.panel = ipywidgets.VBox([self.status])
        folder.listeners.append(self)
        with folder.lock:
            newest = folder.paths[-1] if folder.paths else None
        if newest is not None:
            self.scan_loaded(newest)

    @property
    def scan_region(self) -> tuple[int, int, int, int] | None:
        """The scan region the aberrations are fitted on, or None for the central 256 x 256 of each scan."""
        return self._region

    @scan_region.setter
    def scan_region(self, region) -> None:
        self._region = None if region is None else self._checked(region, self._scan_shape())
        if self._path is not None:
            self._start(self._path)

    def scan_loaded(self, path) -> None:
        """The live folder has loaded ``path``: fit it on a worker thread (it waits for the folder)."""
        self._start(path)

    def scan_released(self, path) -> None:
        """The live folder is about to let ``path`` go: close its panel and session first."""
        if path == self._path:
            self._close_panel("<b>Direct ptycho</b> · waiting for the next scan")

    def close(self) -> None:
        """Stop following the viewer and close the panel's session."""
        if self in self.folder.listeners:
            self.folder.listeners.remove(self)
        self._close_panel("<b>Direct ptycho</b> · closed")

    def _ipython_display_(self):
        """Nothing: ShowPtycho.live drew the panel where it ran, and its handle never draws it again."""

    # ---

    def _start(self, path) -> None:
        threading.Thread(target=self._fit, args=(path,), name="ShowPtycho-live", daemon=True).start()

    def _fit(self, path) -> None:
        """Fit ``path`` on the fit region, then show its whole phase; a failure is the status line."""
        from quantem.widget.showptycho.widget import ShowPtycho

        # the folder first, then this panel: the folder's release calls scan_released while it holds ``busy``
        with self.folder.busy, self._lock:
            with self.folder.lock:
                if path not in self.folder.paths:
                    return  # it left before the folder let this fit start
                dataset = self.folder.datasets[self.folder.paths.index(path)]
            # the viewer adopts the scan now, so its calibration is this scan's
            self.viewer.refresh_stream()
            region = self._region or central_region(dataset.shape[:2])
            name = path.name.removesuffix("_master.h5").removesuffix(".qem")
            self.status.value = f"<b>Direct ptycho</b> · {name} · fitting…"
            # a refit of the scan shown keeps its scan rotation, so moving the box cannot land on the other 180°
            # solution (defocus and tilt of the other sign, contrast inverted)
            rotation = self._rotation if path == self._path else None
            started = time.perf_counter()
            try:
                direct = self._fitted_session(dataset, region, rotation)
            except (ValueError, RuntimeError, MemoryError, OSError) as error:
                self.status.value = (f"<b>Direct ptycho</b> could not fit {name}: {' '.join(str(error).split())} "
                                     "The viewer still shows the scan.")
                return
            seconds = time.perf_counter() - started
            ptycho = ShowPtycho(direct, restore=False)
            ptycho._live = self
            # the viewer's panel size, controls beside the image: both fit one screen
            ptycho.panel_px, ptycho.controls_beside = 450, True
            # the box drawn on the phase: where the aberrations were fitted, dragged to fit elsewhere
            ptycho.fit_region = list(region or ())  # a scan fitted whole has no box to move
            previous = self.ptycho, self._direct
            self.ptycho, self._direct, self._path = ptycho, direct, path
            self._rotation = float(direct.scan_detector_rotation_deg)
            self.status.value = f"<b>Direct ptycho</b> · {name} · fitted in {seconds:.1f} s"
            self.panel.children = (self.status, ptycho)
            self._close(*previous)

    def _fitted_session(self, dataset, region, rotation=None):
        """A quantem.gpu.DirectPtycho session on ``dataset`` fitted through fifth order on ``region``; at the scan
        rotation ``rotation`` when given, else at the one the fit chooses."""
        direct = gpu_adapter.DirectPtycho(dataset, **self.viewer._direct_calibration(),
                                          **({} if rotation is None else {"scan_detector_rotation_deg": rotation}))
        # on a small device the fit moves the encoded scan to host memory; the viewer's detector sessions go first
        direct.on_park = self.viewer._let_go_of_the_scan
        try:
            direct.aberrations(**({} if region is None else {"scan_region": tuple(region)}), trials=self.trials,
                               verbose=False, show=False, **({} if rotation is None else {"check_rotation": False}))
        except BaseException:
            direct.close()
            raise
        return direct

    def _close_panel(self, status: str) -> None:
        with self._lock:
            previous = self.ptycho, self._direct
            self.ptycho = self._direct = self._path = None
            self.status.value = status
            self.panel.children = (self.status,)
            self._close(*previous)

    @staticmethod
    def _close(ptycho, direct) -> None:
        if ptycho is not None:
            ptycho.close()
        if direct is not None:
            direct.close()

    def _scan_shape(self):
        with self.folder.lock:
            return tuple(self.folder.datasets[-1].shape[:2]) if self.folder.datasets else None

    @staticmethod
    def _checked(region, scan_shape) -> tuple[int, int, int, int]:
        """``region`` as four ints: a square of a fit side (128, 256, 512 or 1024) inside ``scan_shape`` when known."""
        if len(region) != 4:
            raise ValueError("scan_region is (row_start, row_stop, col_start, col_stop) in scan pixels.")
        row_start, row_stop, col_start, col_stop = (int(value) for value in region)
        side = row_stop - row_start
        if side != col_stop - col_start or side not in FIT_REGION_SIDES:
            raise ValueError(f"scan_region={tuple(region)} must be square with 128, 256, 512 or 1024 scan pixels per "
                             "side; the aberrations are fitted on it and the whole scan is shown.")
        if scan_shape is not None and not (0 <= row_start and row_stop <= scan_shape[0]
                                           and 0 <= col_start and col_stop <= scan_shape[1]):
            raise ValueError(f"scan_region={tuple(region)} must lie inside the {scan_shape[0]} x {scan_shape[1]} scan.")
        return (row_start, row_stop, col_start, col_stop)
