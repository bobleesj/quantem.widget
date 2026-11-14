"""Direct ptychography of the scans a live Show4DSTEM shows (``ShowPtycho.live``)."""

import gc
import json
import pathlib
import tempfile
import threading
import time
import uuid

import ipywidgets
import numpy as np

from quantem.widget.adapters import gpu as gpu_adapter

# The side of the square scan region the aberrations are fitted on: the fit's sides are 128, 256, 512 or 1024.
FIT_REGION_SIDE = 256
FIT_REGION_SIDES = (128, 256, 512, 1024)
# The fits of one scan kept for the strip, newest last: each holds its whole phase (4 MB at 1024 x 1024).
FITS_KEPT = 6
# What a new panel of the same viewer keeps from the one before it: the reader's display choices.
KEPT_DISPLAY = ("cmap", "initial_fft_on", "method", "phase_damp", "amplitude_damp")


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
    shown scan there, and every later scan; dragging the panel's fit box does the same. Each region fitted on the
    shown scan is kept in ``fits`` with its defocus, astigmatism and a thumbnail of its phase, drawn as a strip
    under the controls once there are two: hovering one shows its phase, a click fits there again, its star marks
    it and its cross drops it. ``save()`` writes the shown fit and the kept regions to ``<scan>.json``;
    ``load()`` shows that saved fit again without fitting. ShowPtycho.live returns it, already shown
    where the cell ran; it is not drawn again.
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
        self.fits: list[dict] = []  # the regions fitted on the scan shown, oldest first
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

    def save(self, save_dir=None) -> pathlib.Path:
        """Write the calibration shown to ``<scan>.json``: the scan's own name, in ``save_dir`` (the live folder
        unless given).

        The file is the record ``ShowPtycho.calibration()`` returns for the panel as it stands (aberrations
        through fifth order, sample tilt, scan rotation, acquisition), with ``scan`` (the scan's file name),
        ``fit_region`` (where the aberrations were fitted) and ``fits``: every region kept in the strip with its
        defocus, astigmatism and star. Saving again replaces the file; ``load()`` reads it.

        Examples
        --------
        >>> ptycho.save()
        """
        from quantem.widget.showptycho.widget import utc_now

        with self._lock:
            if self.ptycho is None:
                raise ValueError("save() writes the fit shown; no scan is fitted yet.")
            name = self._path.name.removesuffix("_master.h5").removesuffix(".qem")
            path = pathlib.Path(self.folder.folder if save_dir is None else save_dir) / f"{name}.json"
            shown = tuple(self.ptycho.fit_region) or None
            payload = {**self.ptycho.calibration(), "id": uuid.uuid4().hex[:12], "generated_at_utc": utc_now(),
                       "scan": self._path.name, "fit_region": shown,
                       "fits": [{key: value for key, value in kept.items() if key != "phase"}
                                for kept in self.fits]}
            path.write_text(json.dumps(payload, indent=2))
        self.status.value = f"{self.status.value.split(' · saved')[0]} · saved {path.name}"
        return path

    def load(self, path=None) -> pathlib.Path:
        """Show the fit ``save()`` wrote for the scan shown, without fitting: ``<scan>.json`` in the live folder
        unless ``path`` is given.

        The saved aberrations, sample tilt, scan rotation and reconstruction choice are set on a new session of
        the scan and its whole phase is shown; the fit box sits where that fit was made, and the regions the file
        kept come back with their stars (their thumbnails return when a region is fitted again). A file saved
        for another scan, or for other acquisition values, is refused.

        Examples
        --------
        >>> ptycho.load()
        """
        from quantem.widget.showptycho.widget import CALIBRATION_FILENAME, ShowPtycho

        with self.folder.busy, self._lock:
            with self.folder.lock:
                if not self.folder.paths:
                    raise ValueError("load() shows a saved fit of the scan shown; no scan has arrived yet.")
                scan, dataset = self.folder.paths[-1], self.folder.datasets[-1]
            name = scan.name.removesuffix("_master.h5").removesuffix(".qem")
            path = pathlib.Path(self.folder.folder) / f"{name}.json" if path is None else pathlib.Path(path)
            saved = json.loads(path.read_text())
            if saved.get("scan") != scan.name:
                raise ValueError(f"{path.name} is the saved fit of {saved.get('scan')}, not of {scan.name}, the scan shown.")
            self.viewer.refresh_stream()
            # the shown session goes first: a small device has no room for two
            self._release_shown(f"<b>Direct ptycho</b> · {name} · loading {path.name}…")
            direct = gpu_adapter.DirectPtycho(dataset, **self.viewer._direct_calibration())
            direct.on_park = self.viewer._let_go_of_the_scan
            note = ""
            try:
                ptycho = self._restored(ShowPtycho, direct, saved, CALIBRATION_FILENAME)
                if "Regularized direct reconstruction" in ptycho.calibration_status:
                    # the saved damped method cannot run on this session (the scan is read in batches on a small
                    # device): the saved aberrations are shown as the phase of the mean wave, and the status says so
                    ptycho.close()
                    ptycho = self._restored(ShowPtycho, direct, {**saved, "reconstruction": {"method": "phase_of_mean"}},
                                            CALIBRATION_FILENAME)
                    note = f" · shown as phase of mean (saved as {saved['reconstruction']['method']}; no room for it here)"
            except BaseException:
                direct.close()
                raise
            if not ptycho.calibration_status.startswith("Restored"):
                reason = ptycho.calibration_status
                self._close(ptycho, direct)
                self.status.value = f"<b>Direct ptycho</b> · {name} · {path.name} not loaded"
                raise ValueError(f"{path.name}: {reason}")
            region = saved.get("fit_region")
            # the regions the file kept, without their phases: listed in the next save, drawn once fitted again
            self.fits = [{**kept, "scan_region": tuple(kept["scan_region"]), "phase": None} for kept in saved.get("fits", [])]
            self._show(ptycho, direct, scan, region, f"<b>Direct ptycho</b> · {name} · loaded {path.name}{note}", keep_display=False)
        return path

    @staticmethod
    def _restored(ShowPtycho, direct, saved: dict, filename: str):
        """A ShowPtycho on ``direct`` with the record ``saved`` restored: ShowPtycho restores the calibration file
        of its save folder, so the record is written under that name in a folder of its own."""
        with tempfile.TemporaryDirectory() as folder:
            (pathlib.Path(folder) / filename).write_text(json.dumps(saved))
            return ShowPtycho(direct, save_dir=folder, restore=True)

    def change_fit(self, action: str, scan_region) -> None:
        """Star (``"star"``, a toggle) or drop (``"delete"``) the kept fit of ``scan_region``; the shown fit stays."""
        scan_region = tuple(int(value) for value in scan_region)
        with self._lock:
            if self.ptycho is None:
                return
            if action == "star":
                for kept in self.fits:
                    if kept["scan_region"] == scan_region:
                        kept["starred"] = not kept["starred"]
            elif scan_region != tuple(self.ptycho.fit_region):
                self.fits = [kept for kept in self.fits if kept["scan_region"] != scan_region]
            self._send_fits(self.ptycho)

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
            region = self._region or central_region(dataset.shape[:2])
            name = path.name.removesuffix("_master.h5").removesuffix(".qem")
            self.status.value = f"<b>Direct ptycho</b> · {name} · fitting…"
            # a refit of the scan shown keeps its scan rotation, so moving the box cannot land on the other 180°
            # solution (defocus and tilt of the other sign, contrast inverted)
            rotation = self._rotation if path == self._path else None
            started = time.perf_counter()
            try:
                # the viewer adopts the scan now, so its calibration is this scan's
                self.viewer.refresh_stream()
                direct = None
                try:
                    direct = self._fitted_session(dataset, region, rotation)
                except (MemoryError, RuntimeError) as error:
                    # CUDA and cuBLAS report a full device as runtime errors; any other runtime error is not retried
                    if not isinstance(error, MemoryError) and not any(
                            word in str(error).lower() for word in ("out of memory", "alloc_failed")):
                        raise
                # Retried after the except block: until it ends, the caught error's traceback keeps the failed
                # fit's frames and their GPU buffers alive, and on a 4 GB card the second try fails again.
                if direct is None:
                    # A small device ran out. Two causes, both gone on a second try: the shown session beside the
                    # new one (it goes first now), and the encoded scan still held for a moment by the viewer's own
                    # drawing when the fit wanted its room (about every second fit at 3400 MiB free).
                    self._release_shown(f"<b>Direct ptycho</b> · {name} · fitting…")
                    self.viewer._let_go_of_the_scan()
                    gc.collect()
                    gpu_adapter.gpu_release_cached_memory()
                    direct = self._fitted_session(dataset, region, rotation)
            except (ValueError, RuntimeError, MemoryError, OSError) as error:
                self.status.value = (f"<b>Direct ptycho</b> could not fit {name}: {' '.join(str(error).split())} "
                                     "The viewer still shows the scan.")
                return
            seconds = time.perf_counter() - started
            try:
                ptycho, note = self._panel(ShowPtycho, direct, name)
            except (ValueError, RuntimeError, MemoryError) as error:
                direct.close()
                self.status.value = (f"<b>Direct ptycho</b> fitted {name} but could not show it: "
                                     f"{' '.join(str(error).split())}")
                return
            if path != self._path:
                self.fits = []  # a new scan starts its own strip
            self._show(ptycho, direct, path, region,
                       f"<b>Direct ptycho</b> · {name} · fitted in {seconds:.1f} s{note}", keep_display=True)

    def _panel(self, ShowPtycho, direct, name: str):
        """A ShowPtycho on the fitted session ``direct`` and a note for the status line ("" unless reduced).

        On a small device the upsampled phase may not fit beside the session shown: that one goes first; if it
        still does not fit, the phase is shown at the scan's own sampling and the note says so.
        """
        try:
            return ShowPtycho(direct, restore=False), ""
        except MemoryError:
            self._release_shown(f"<b>Direct ptycho</b> · {name} · fitting…")
            gc.collect()
            gpu_adapter.gpu_release_cached_memory()
        try:
            return ShowPtycho(direct, restore=False), ""
        except MemoryError:
            gc.collect()
            gpu_adapter.gpu_release_cached_memory()
        return (ShowPtycho(direct, restore=False, upsample=1),
                " · shown at the scan's own sampling: no room on this GPU for the upsampled phase")

    def _show(self, ptycho, direct, path, region, status: str, *, keep_display: bool) -> None:
        """Make ``ptycho`` (on session ``direct``, of scan ``path``, fitted on ``region``) the panel shown.

        With ``keep_display`` the reader's choices on the panel before stay chosen, and a first panel opens on the
        damped phase; a loaded fit keeps the choices its file recorded.
        """
        ptycho._live = self
        # the viewer's panel size, controls beside the image: both fit one screen
        ptycho.panel_px, ptycho.controls_beside = 450, True
        # the box drawn on the phase: where the aberrations were fitted, dragged to fit elsewhere
        ptycho.fit_region = list(region or ())  # a scan fitted whole has no box to move
        previous = self.ptycho, self._direct
        if keep_display and previous[0] is not None:
            # the colormap, FFT inset and reconstruction method chosen on the panel before stay chosen
            for choice in KEPT_DISPLAY:
                # a damped method chosen where it could run stays off where it cannot (a batched session)
                if choice != "method" or previous[0].method == "phase_of_mean" or ptycho.regularization_available:
                    setattr(ptycho, choice, getattr(previous[0], choice))
        elif keep_display and ptycho.regularization_available:
            ptycho.method = "phase"  # a live panel opens on the damped phase; Complex is its other choice
        self.ptycho, self._direct, self._path = ptycho, direct, path
        self._rotation = float(direct.scan_detector_rotation_deg)
        self.status.value = status
        self._record(ptycho, region)
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
            self._release_shown(status)
            self._path = None
            self.fits = []

    def _record(self, ptycho, region) -> None:
        """Keep this fit of the shown scan and hand the panel every fit kept: ``fits_json`` and ``fits_phase``.

        A scan fitted whole has no region to go back to and keeps none. A region fitted again replaces its earlier
        entry, so ``fits`` holds each region once, in the order first fitted; beyond ``FITS_KEPT`` the oldest goes.
        Each entry keeps the whole phase at its own pixels, so the panel can show it again without fitting.
        """
        if region is None:
            return
        region = tuple(int(value) for value in region)
        fit = {"scan_region": region, "C10_nm": float(ptycho.fitted_C10_nm), "C12_nm": float(ptycho.fitted_C12_nm),
               "phi12_deg": float(ptycho.fitted_phi12_deg), "starred": False,
               "phase": np.array(ptycho._last_phase, dtype=np.float32)}  # a copy: the panel goes on drawing its own
        earlier = [index for index, kept in enumerate(self.fits) if kept["scan_region"] == region]
        if earlier:
            self.fits[earlier[0]] = {**fit, "starred": self.fits[earlier[0]]["starred"]}
        else:
            self.fits = [*self.fits, fit][-FITS_KEPT:]
        self._send_fits(ptycho)

    def _send_fits(self, ptycho) -> None:
        """Hand ``ptycho`` the kept fits: their records as ``fits_json``, their phases as ``fits_phase``."""
        with ptycho.hold_sync():
            drawn = [kept for kept in self.fits if kept["phase"] is not None]  # a loaded region has none yet
            ptycho.fits_phase = b"".join(kept["phase"].tobytes() for kept in drawn)
            ptycho.fits_json = json.dumps([{**{name: value for name, value in kept.items() if name != "phase"},
                                            "shape": list(kept["phase"].shape)} for kept in drawn])

    def _release_shown(self, status: str) -> None:
        """Close the panel shown and its session, leaving the status line; the scan stays the one fitted."""
        previous = self.ptycho, self._direct
        self.ptycho = self._direct = None
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

