"""ShowPtycho.live: direct ptychography of each scan a live Show4DSTEM shows, fitted on a square scan region."""

import json
import os
import sys
import time

import ipywidgets
import numpy as np
import pytest
import traitlets

from quantem.widget import Show4DSTEM, ShowPtycho
from quantem.widget.show4dstem.live import LiveFolder
from quantem.widget.showptycho import live as live_module
from quantem.widget.showptycho.live import LivePtycho, central_region
from tests.show4dstem.test_show4dstem import _counts, _write_master, accelerator  # noqa: F401  (the fixture)


def test_the_fit_region_is_the_central_256_square_unless_the_scan_is_smaller():
    assert central_region((512, 512)) == (128, 384, 128, 384)
    assert central_region((512, 1024)) == (128, 384, 384, 640)
    assert central_region((256, 256)) is None and central_region((128, 512)) is None


def test_a_fit_region_is_a_square_of_a_fit_side_inside_the_scan():
    assert LivePtycho._checked((0, 256, 0, 256), (512, 512)) == (0, 256, 0, 256)
    with pytest.raises(ValueError, match="square with 128, 256, 512 or 1024"):
        LivePtycho._checked((0, 200, 0, 200), (512, 512))
    with pytest.raises(ValueError, match="square with 128, 256, 512 or 1024"):
        LivePtycho._checked((0, 256, 0, 128), (512, 512))
    with pytest.raises(ValueError, match="inside the 512 x 512 scan"):
        LivePtycho._checked((384, 640, 0, 256), (512, 512))


def test_live_ptycho_needs_a_live_viewer():
    with pytest.raises(ValueError, match="Show4DSTEM.live"):
        LivePtycho(object())


def _arrive(folder, stem, seed, age_s):
    master = _write_master(folder, stem, _counts(seed))
    changed = time.time() - age_s
    for path in (master, folder / f"{stem}_data_000001.h5"):
        os.utime(path, (changed, changed))


def test_each_new_scan_is_fitted_and_its_panel_closes_before_the_scan_goes(tmp_path, accelerator, monkeypatch):
    monkeypatch.setattr(LiveFolder, "start", lambda self: None)
    events = []

    class Session:
        scan_detector_rotation_deg = 12.0

        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(("session closed", self.name))

    class Panel(ipywidgets.HTML):
        """Stands in for ShowPtycho: a widget the panel holds, recording when it closes."""

        regularization_available = False
        cmap, initial_fft_on, method, phase_damp, amplitude_damp = "inferno", False, "phase_of_mean", 0.03, 0.03

        def __init__(self, direct, restore):
            super().__init__()
            self.direct = direct

        def close(self):
            if getattr(self, "comm", None) is not None:  # ipywidgets closes again when the object is collected
                events.append(("panel closed", self.direct.name))
            super().close()

    def fitted(self, dataset, region, rotation=None):
        name = os.path.basename(dataset.metadata.get("source_path", "?"))
        events.append(("fitted", name, region))
        return Session(name)

    monkeypatch.setattr(LivePtycho, "_fitted_session", fitted)
    monkeypatch.setattr("quantem.widget.showptycho.widget.ShowPtycho", Panel)
    viewer = Show4DSTEM.live(tmp_path, latest=1, verbose=False)
    folder = viewer._live_folder
    ptycho = ShowPtycho.live(viewer)
    try:
        assert "waiting for the first scan" in ptycho.status.value
        _arrive(tmp_path, "a", 0, age_s=30)
        folder.look()
        folder.look()
        _wait_for(lambda: ptycho.ptycho is not None)
        # the synthetic scan is smaller than 256 x 256: the whole scan is the fit region
        assert events == [("fitted", "a_master.h5", None)] and ptycho.ptycho.fit_region == []
        assert ptycho.panel.children == (ptycho.status, ptycho.ptycho) and "· a · fitted in" in ptycho.status.value
        _arrive(tmp_path, "b", 1, age_s=20)
        folder.look()
        folder.look()
        _wait_for(lambda: ptycho._path is not None and ptycho._path.name == "b_master.h5")
        # a's panel and session closed before a was let go, then b was fitted
        assert events[1:] == [("panel closed", "a_master.h5"), ("session closed", "a_master.h5"),
                              ("fitted", "b_master.h5", None)]
    finally:
        ptycho.close()
        viewer.close()
    assert events[-2:] == [("panel closed", "b_master.h5"), ("session closed", "b_master.h5")]


def test_a_refit_without_room_beside_the_shown_session_closes_it_and_fits_again(tmp_path, accelerator, monkeypatch):
    """A small device cannot hold the shown session and the new one together: the refit lets the shown one go
    and tries once more, so moving the fit box works there."""
    monkeypatch.setattr(LiveFolder, "start", lambda self: None)
    events, open_sessions = [], []

    class Session:
        scan_detector_rotation_deg = 12.0

        def close(self):
            open_sessions.remove(self)
            events.append("session closed")

    class Panel(ipywidgets.HTML):
        regularization_available = False
        cmap, initial_fft_on, method, phase_damp, amplitude_damp = "inferno", False, "phase_of_mean", 0.03, 0.03

        def __init__(self, direct, restore):
            super().__init__()

    def fitted(self, dataset, region, rotation=None):
        if open_sessions:
            events.append("no room")
            raise MemoryError("Out of memory allocating 426,747,904 bytes")
        # the retry runs after the except block: inside it, the caught error's traceback keeps the failed fit's
        # GPU buffers alive and the second try ran out again (ARINA PC, 4 GB, 11-14)
        assert sys.exc_info()[1] is None
        open_sessions.append(Session())
        events.append("fitted")
        return open_sessions[-1]

    monkeypatch.setattr(LivePtycho, "_fitted_session", fitted)
    monkeypatch.setattr("quantem.widget.showptycho.widget.ShowPtycho", Panel)
    viewer = Show4DSTEM.live(tmp_path, latest=1, verbose=False)
    ptycho = ShowPtycho.live(viewer)
    try:
        _arrive(tmp_path, "a", 0, age_s=30)
        viewer._live_folder.look()
        viewer._live_folder.look()
        _wait_for(lambda: ptycho.ptycho is not None)
        first = ptycho.ptycho
        ptycho.scan_region = None  # what dragging the fit box does: fit the shown scan again
        _wait_for(lambda: ptycho.ptycho is not None and ptycho.ptycho is not first)
        assert events == ["fitted", "no room", "session closed", "fitted"]
        assert "· a · fitted in" in ptycho.status.value and ptycho._path.name == "a_master.h5"
    finally:
        ptycho.close()
        viewer.close()


def test_each_region_fitted_on_the_shown_scan_is_kept_for_the_panel_strip(tmp_path, accelerator, monkeypatch):
    """Moving the fit box keeps the earlier fit: its region, values and whole phase go to the panel, which draws
    the strip and shows a kept phase again on hover; a region fitted twice keeps one entry, the oldest go past six."""
    monkeypatch.setattr(LiveFolder, "start", lambda self: None)
    viewer = Show4DSTEM.live(tmp_path, latest=1, verbose=False)
    ptycho = ShowPtycho.live(viewer)

    class Shown(ipywidgets.Widget):
        cmap = "inferno"
        fits_json = traitlets.Unicode("[]")
        fits_phase = traitlets.Bytes(b"")

        def __init__(self, C10_nm, level):
            super().__init__()
            self.fitted_C10_nm, self.fitted_C12_nm, self.fitted_phi12_deg = C10_nm, 9.8, 19.0
            self._last_phase = np.full((8, 8), level, dtype=np.float32)

    try:
        second = Shown(2.8, 2.0)
        ptycho._record(Shown(-1.6, 1.0), (128, 384, 128, 384))
        ptycho._record(second, (94, 222, 283, 411))
        sent = json.loads(second.fits_json)
        assert [fit["scan_region"] for fit in sent] == [[128, 384, 128, 384], [94, 222, 283, 411]]
        assert sent[0]["C10_nm"] == -1.6 and sent[1]["shape"] == [8, 8] and "phase" not in sent[0]
        phases = np.frombuffer(second.fits_phase, dtype=np.float32).reshape(2, 8, 8)
        assert (phases[0] == 1.0).all() and (phases[1] == 2.0).all()
        second._last_phase[:] = 5.0  # the panel redraws its own array: the kept phase is a copy
        assert (ptycho.fits[1]["phase"] == 2.0).all()
        ptycho._record(Shown(-1.7, 3.0), (128, 384, 128, 384))
        assert [fit["scan_region"] for fit in ptycho.fits] == [(128, 384, 128, 384), (94, 222, 283, 411)]
        assert ptycho.fits[0]["C10_nm"] == -1.7
        for start in range(0, 6 * 128, 128):
            ptycho._record(Shown(0.0, 4.0), (0, 128, start, start + 128))
        assert len(ptycho.fits) == live_module.FITS_KEPT and ptycho.fits[-1]["scan_region"] == (0, 128, 640, 768)
    finally:
        ptycho.close()
        viewer.close()
    assert ptycho.fits == []


def test_a_star_and_a_delete_change_the_kept_fits_and_save_writes_the_scan_named_json(tmp_path, accelerator, monkeypatch):
    """The third cell: ``ptycho.save()`` writes ``<scan>.json`` in the live folder, the panel's calibration record
    with the scan, the region fitted and every kept region with its star; the shown fit cannot be deleted."""
    monkeypatch.setattr(LiveFolder, "start", lambda self: None)
    viewer = Show4DSTEM.live(tmp_path, latest=1, verbose=False)
    ptycho = ShowPtycho.live(viewer)

    class Shown(ipywidgets.Widget):
        fits_json = traitlets.Unicode("[]")
        fits_phase = traitlets.Bytes(b"")
        fit_region = [94, 222, 283, 411]
        fitted_C10_nm, fitted_C12_nm, fitted_phi12_deg = 2.8, 1.5, 40.0
        _last_phase = np.zeros((8, 8), dtype=np.float32)

        def calibration(self):
            return {"schema": "quantem.direct-calibration/9", "aberrations": {"C10_nm": 2.8}}

    try:
        with pytest.raises(ValueError, match="no scan is fitted yet"):
            ptycho.save()
        shown = Shown()
        ptycho._record(shown, (128, 384, 128, 384))
        ptycho._record(shown, (94, 222, 283, 411))
        ptycho.ptycho, ptycho._path = shown, tmp_path / "scan_0013_master.h5"
        ptycho.change_fit("star", (128, 384, 128, 384))
        assert [fit["starred"] for fit in json.loads(shown.fits_json)] == [True, False]
        path = ptycho.save()
        saved = json.loads(path.read_text())
        assert path == tmp_path / "scan_0013.json" and "saved scan_0013.json" in ptycho.status.value
        assert saved["schema"] == "quantem.direct-calibration/9" and saved["aberrations"] == {"C10_nm": 2.8}
        assert saved["scan"] == "scan_0013_master.h5" and saved["fit_region"] == [94, 222, 283, 411]
        assert saved["fits"] == [
            {"scan_region": [128, 384, 128, 384], "C10_nm": 2.8, "C12_nm": 1.5, "phi12_deg": 40.0, "starred": True},
            {"scan_region": [94, 222, 283, 411], "C10_nm": 2.8, "C12_nm": 1.5, "phi12_deg": 40.0, "starred": False}]
        assert ptycho.save(tmp_path / "..").name == "scan_0013.json"  # another folder, the same name
        ptycho.change_fit("delete", (94, 222, 283, 411))  # the shown fit stays
        assert len(ptycho.fits) == 2
        ptycho.change_fit("delete", (128, 384, 128, 384))
        assert [fit["scan_region"] for fit in ptycho.fits] == [(94, 222, 283, 411)] and len(shown.fits_phase) == 8 * 8 * 4
        ptycho.ptycho = None
    finally:
        ptycho.close()
        viewer.close()


def _wait_for(condition, timeout_s: float = 20.0) -> None:
    started = time.monotonic()
    while not condition():
        assert time.monotonic() - started < timeout_s, "timed out"
        time.sleep(0.05)


def test_the_module_fit_side_is_256():
    assert live_module.FIT_REGION_SIDE == 256
