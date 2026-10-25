"""Idle microscope listeners adopt arriving scans without owning their counts."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem import widget as widget_module
from quantem.widget.show4dstem.dense import DenseSession


class LiveScan:
    """A CPU source implementing the same borrowed-session contract as live counts."""

    def __init__(self, series, value=1):
        self.counts = torch.full((3, 4, 8, 10), value, dtype=torch.float32)
        self.metadata = {"series": series, "scan_sampling_A": [0.4, 0.6], "voltage_kV": 300.0}
        self.received = self.total = 12
        self.sessions = []

    def detector_session(self):
        session = DenseSession(self.counts)
        session.metadata = self.metadata
        self.sessions.append(session)
        return session


@pytest.fixture
def recording(monkeypatch):
    # Keep polling deterministic; refresh_stream is the notebook's public refresh.
    loop = SimpleNamespace(asyncio_loop=SimpleNamespace(is_running=lambda: True),
                           add_callback=lambda call: None, call_later=lambda delay, call: None)
    monkeypatch.setattr(widget_module.IOLoop, "current", lambda **kwargs: loop)
    return SimpleNamespace(lock=threading.RLock(), acquisitions=[], datasets=[], receiving=True, error=None)


def test_idle_listener_receives_unloads_and_receives_again(recording):
    viewer = Show4DSTEM.stream(recording)
    try:
        assert viewer.n_frames == 0 and viewer._data is None
        assert viewer.title == "Waiting for acquisition"
        assert np.isnan(np.frombuffer(viewer.frame_bytes, np.float32)).all()
        viewer.refresh_stream()
        assert viewer.stream_error is None

        first = LiveScan("first")
        first.received = 0
        recording.acquisitions.append(first)
        viewer.refresh_stream()
        assert viewer.n_frames == 0 and not first.sessions
        first.received = first.total
        viewer.refresh_stream()
        assert viewer.n_frames == 1 and viewer.frame_labels == ["first"]
        assert (viewer.shape_rows, viewer.shape_cols, viewer.det_rows, viewer.det_cols) == (3, 4, 8, 10)
        assert viewer.scan_sampling_A == [0.4, 0.6] and viewer.voltage_kV == 300.0
        np.testing.assert_array_equal(viewer.pattern(1, 2), first.counts[1, 2].numpy())
        viewer.apply_preset("adf")
        viewer.pos_row, viewer.pos_col = 1, 2
        detector = (viewer.roi_center, viewer.roi_radius_inner, viewer.roi_radius)

        recording.acquisitions.clear()
        viewer.refresh_stream()
        viewer.refresh_stream()
        assert viewer.n_frames == 0 and viewer._data is None and viewer.stream_error is None
        assert first.sessions[0]._flat_t is None
        assert np.isnan(np.frombuffer(viewer.virtual_image_bytes, np.float32)).all()

        second = LiveScan("second", value=2)
        recording.acquisitions.append(second)
        viewer.refresh_stream()
        assert viewer.n_frames == 1 and viewer.frame_labels == ["second"]
        assert (viewer.pos_row, viewer.pos_col) == (1, 2)
        assert (viewer.roi_center, viewer.roi_radius_inner, viewer.roi_radius) == detector
        expected = second.counts.numpy()[..., viewer._current_detector_mask()].sum(-1)
        np.testing.assert_array_equal(viewer.virtual_image(), expected)
        np.testing.assert_array_equal(viewer.pattern(1, 2), second.counts[1, 2].numpy())
    finally:
        viewer.close()
    assert second.sessions[0]._flat_t is None
    assert second.counts.sum() > 0


def test_idle_first_arrival_matches_opening_the_same_scan(recording):
    options = {"labels": ["stated label"], "bright_field_center_px": (3.0, 4.0),
               "bright_field_radius_px": 2.0, "title": "Microscope"}
    waiting = Show4DSTEM.stream(recording, **options)
    scan = LiveScan("metadata label")
    recording.acquisitions.append(scan)
    opened = Show4DSTEM.stream(recording, first_timeout_s=1, **options)
    try:
        waiting.refresh_stream()
        opened.refresh_stream()
        for name in ("frame_labels", "scan_sampling_A", "voltage_kV", "roi_center", "roi_radius",
                     "frame_dim_label", "view_mode", "compare_dp_mode", "title"):
            assert getattr(waiting, name) == getattr(opened, name)
        np.testing.assert_array_equal(waiting.virtual_image(), opened.virtual_image())
        np.testing.assert_array_equal(waiting.pattern(0, 0), opened.pattern(0, 0))
    finally:
        waiting.close()
        opened.close()


def test_idle_listener_reports_receiver_failure_and_closes(recording):
    viewer = Show4DSTEM.stream(recording)
    failure = RuntimeError("Receiver disconnected")
    recording.error = failure
    try:
        assert viewer._stream_update() is False
        assert viewer.stream_error is failure
        assert viewer.title == "Live acquisition stopped: Receiver disconnected"
        assert viewer.n_frames == 0
    finally:
        viewer.close()
    assert viewer._stream_stop.is_set()
