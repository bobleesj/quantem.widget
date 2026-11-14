"""ShowPtycho.live: direct ptychography of each scan a live Show4DSTEM shows, fitted on a square scan region."""

import os
import time

import ipywidgets
import pytest

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

        def __init__(self, direct, restore):
            super().__init__()
            self.direct = direct

        def close(self):
            if getattr(self, "comm", None) is not None:  # ipywidgets closes again when the object is collected
                events.append(("panel closed", self.direct.name))
            super().close()

    def fitted(self, dataset, region, rotation=None):
        name = dataset.metadata.get("source_path", "?").rsplit("/", 1)[-1]
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


def _wait_for(condition, timeout_s: float = 20.0) -> None:
    started = time.monotonic()
    while not condition():
        assert time.monotonic() - started < timeout_s, "timed out"
        time.sleep(0.05)


def test_the_module_fit_side_is_256():
    assert live_module.FIT_REGION_SIDE == 256
