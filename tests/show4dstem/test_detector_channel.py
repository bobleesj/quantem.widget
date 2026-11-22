"""Custom detector requests preserve canonical state and exact detector products."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem.detector import preset_mask

CONTEXT = "frame_idx vi_source view_mode compare_page_idx roi_mode roi_radius roi_radius_inner roi_width roi_height"
RADIUS_CONTEXT = "frame_idx vi_source view_mode compare_page_idx roi_mode roi_center_row roi_center_col"


def request(viewer, center, seq=1):
    return {"type": "show4dstem_detector", "version": 1, "view": "first", "epoch": 0,
            "seq": seq, "center": center, "context": [getattr(viewer, key) for key in CONTEXT.split()]}


def radius_request(viewer, radii, seq=1):
    return {"type": "show4dstem_detector_radius", "version": 1, "view": "first", "epoch": 0,
            "seq": seq, "radii": radii, "context": [getattr(viewer, key) for key in RADIUS_CONTEXT.split()]}


def test_preset_preview_returns_exact_counts_without_committing_the_roi(viewer):
    widget, values_t, _ = viewer
    widget.view_mode = "single"
    state = (widget.vi_source, list(widget.roi_center), widget.roi_mode, bytes(widget.virtual_image_bytes))
    replies = []
    widget.send = lambda content, buffers=None: replies.append((content, buffers))
    content = {"type": "show4dstem_preset_preview", "view": "test", "request": 1,
               "preset": "adf", "frame": 0, "scan": None, "shape": [5, 7, 16, 16]}
    widget._handle_preset_preview(widget, content, [])
    reply, buffers = replies[-1]
    assert reply == {"type": "show4dstem_preset_preview_result", "view": "test", "request": 1}
    mask = preset_mask("adf", (16, 16), (7.5, 7.5), 3)
    expected = values_t[0, ..., torch.from_numpy(mask)].sum(-1)
    actual = torch.frombuffer(bytearray(buffers[0]), dtype=torch.float32).reshape(5, 7)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (widget.vi_source, list(widget.roi_center), widget.roi_mode, bytes(widget.virtual_image_bytes)) == state
    widget.frame_idx = 1
    widget._handle_preset_preview(widget, content, [])
    assert replies[-1][1] == []


@pytest.fixture
def viewer():
    values_t = torch.arange(2 * 5 * 7 * 16 * 16, dtype=torch.float32).reshape(2, 5, 7, 16, 16)
    widget = Show4DSTEM(values_t, device="cpu", view_mode="multiple", show_fft=False,
                       bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3)
    messages = []
    widget.send = lambda content, buffers=None: messages.append(content)
    try:
        yield widget, values_t, messages
    finally:
        widget.free()
        widget.close()


def test_center_request_acknowledges_before_one_exact_computation(viewer, monkeypatch):
    widget, values_t, messages = viewer
    compute = widget._compare_images
    queries = []

    def observed(indices, mask):
        assert messages[-1]["status"] == "accepted"
        queries.append(tuple(indices))
        return compute(indices, mask)

    monkeypatch.setattr(widget, "_compare_images", observed)
    widget._handle_detector_request(widget, request(widget, [5, 9]), [])
    assert [message["status"] for message in messages] == ["accepted", "complete"]
    assert len(queries) == 1 and widget.roi_center == [5, 9]
    assert (widget.roi_center_row, widget.roi_center_col) == (5, 9)
    rows_t, cols_t = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    mask_t = (rows_t - 5) ** 2 + (cols_t - 9) ** 2 <= widget.roi_radius ** 2
    expected_t = values_t[..., mask_t].sum(-1) / mask_t.sum()
    actual_t = torch.frombuffer(bytearray(widget.compare_virtual_image_bytes), dtype=torch.float32).reshape(2, 5, 7)
    torch.testing.assert_close(actual_t, expected_t, atol=0, rtol=0)


def test_center_request_accepts_first_pixel_in_a_higher_resolution_preview(viewer):
    widget, _, messages = viewer
    widget.view_mode = "single"
    widget.live_phase_preview = {"scan": "series_2", "detRows": 32, "detCols": 32}
    widget._handle_detector_request(widget, request(widget, [-0.25, -0.25]), [])
    assert [message["status"] for message in messages] == ["accepted", "complete"]
    assert widget.roi_center == [-0.25, -0.25]
    widget._handle_detector_request(widget, request(widget, [-0.5, 0], seq=2), [])
    assert messages[-1]["status"] == "rejected"


def test_center_request_recovers_same_position_after_compute_failure(viewer, monkeypatch):
    widget, _, messages = viewer
    compute = widget._compare_images

    def fail(indices, mask):
        raise MemoryError("injected bounded compute failure")

    monkeypatch.setattr(widget, "_compare_images", fail)
    content = request(widget, [5, 9])
    widget._handle_detector_request(widget, content, [])
    assert [message["status"] for message in messages] == ["accepted", "rejected"]
    assert not widget.compare_virtual_image_bytes
    monkeypatch.setattr(widget, "_compare_images", compute)
    widget._handle_detector_request(widget, dict(content, seq=2), [])
    assert messages[-1]["status"] == "complete" and widget.compare_virtual_image_bytes


def test_stale_view_settings_and_freed_source_reject_before_computing(viewer):
    widget, _, messages = viewer
    content = request(widget, [5, 9])
    widget.roi_radius = 4
    before = bytes(widget.compare_virtual_image_bytes)
    center = list(widget.roi_center)
    widget._handle_detector_request(widget, content, [])
    assert messages[-1]["status"] == "rejected"
    assert bytes(widget.compare_virtual_image_bytes) == before and widget.roi_center == center
    widget.free()
    widget._handle_detector_request(widget, request(widget, [5, 9], 2), [])
    assert [message["status"] for message in messages] == ["rejected", "rejected"]


def test_position_request_reads_once_and_recovers_after_failure(viewer, monkeypatch):
    widget, values_t, messages = viewer
    widget.compare_dp_mode = "selected"
    content = {"type": "show4dstem_position", "version": 1, "view": "first",
               "epoch": 0, "seq": 1, "center": [1, 3],
               "context": [widget.frame_idx, widget.vi_roi_mode, widget.compare_dp_mode]}
    compute = widget._update_frame
    calls = []

    def observed(change=None):
        assert messages[-1]["status"] == "accepted"
        calls.append((widget.pos_row, widget.pos_col))
        if len(calls) == 1:
            raise MemoryError("injected frame failure")
        compute(change)

    monkeypatch.setattr(widget, "_update_frame", observed)
    widget._handle_position_request(widget, content, [])
    widget._handle_position_request(widget, dict(content, seq=2), [])
    assert [message["status"] for message in messages] == ["accepted", "rejected", "accepted", "complete"]
    assert calls == [(1, 3), (1, 3)]
    actual = torch.frombuffer(bytearray(widget.frame_bytes), dtype=torch.float32).reshape(16, 16)
    torch.testing.assert_close(actual, values_t[0, 1, 3], atol=0, rtol=0)
    widget._handle_position_request(widget, dict(content, context=[1, "off", "selected"]), [])
    assert messages[-1]["status"] == "rejected" and len(calls) == 2


def test_interlaced_live_probe_reads_the_measured_visit_for_a_grid_position(viewer, monkeypatch):
    widget, _, replies = viewer
    positions = np.array([(1, 2), (0, 2), (1, 0), (0, 0), (1, 1), (0, 1)])
    patterns = np.arange(6 * 16 * 16, dtype=np.float32).reshape(6, 16, 16)
    query = SimpleNamespace(frame=lambda visit: patterns[visit], close=lambda: None)
    source = SimpleNamespace(shape=(1, 6, 16, 16), received=2, detector_session=lambda: query,
                             metadata={"scan_positions": positions, "scan_positions_shape": (2, 3), "series": "live_9"})
    recording = SimpleNamespace(detector_preview_source=lambda _: source, datasets=[source])
    live = {"scan": "live_9", "visit_path": True, "total": 2, "cols": 3, "detRows": 16, "detCols": 16}
    request = {"type": "show4dstem_live_probe", "scan": "live_9", "view": "test", "epoch": 0, "seq": 1}
    widget.send = lambda reply, buffers=None: replies.append((reply, buffers))
    with monkeypatch.context() as scoped:
        scoped.setattr(widget, "_stream_follower", SimpleNamespace(recording=recording))
        widget.live_phase_preview = live
        widget._handle_live_probe(widget, dict(request, center=[1, 2]), [])
        reply, buffers = replies[-1]
        assert reply["center"] == [1, 2] and reply["scan"] == "live_9"
        np.testing.assert_array_equal(np.frombuffer(buffers[0], np.float32), patterns[0].ravel())
        widget._handle_live_probe(widget, dict(request, center=[1, 1]), [])
        assert not replies[-1][1]  # This position belongs to a later pass.
        recording.detector_preview_source = lambda _: None
        source.received = len(positions)
        widget.live_phase_preview = dict(live, frozen=True)
        widget._handle_live_probe(widget, dict(request, center=[1, 1]), [])
        np.testing.assert_array_equal(np.frombuffer(replies[-1][1][0], np.float32), patterns[4].ravel())
        widget._handle_live_probe(widget, dict(request, scan="live_8", center=[1, 1]), [])
        assert not replies[-1][1]


def test_interlaced_probe_uses_preceding_measurement_then_current_visit(viewer, monkeypatch):
    widget, _, replies = viewer
    positions = np.array([(0, 0), (1, 1), (1, 0), (0, 1)])
    previous_positions = positions[::-1]
    shape = (1, 4, 16, 16)
    reads = []

    def source(name, order, received, base):
        def read(visit):
            reads.append((name, visit))
            return np.full((16, 16), base + visit, dtype=np.float32)

        query = SimpleNamespace(frame=read, close=lambda: None)
        return SimpleNamespace(shape=shape, received=received, detector_session=lambda: query,
                               metadata={"series": name, "continuous_series": "live_12",
                                         "scan_positions": order, "scan_positions_shape": (2, 2)})

    current = source("live_12000002", positions, 1, 20)
    previous = source("live_12000001", previous_positions, 4, 10)
    recording = SimpleNamespace(detector_preview_source=lambda _: current, datasets=[previous])
    preview = {"scan": "live_12000002", "visit_path": True, "total": 2, "cols": 2, "detRows": 16, "detCols": 16}
    request = {"type": "show4dstem_live_probe", "scan": preview["scan"], "center": [0, 1]}
    widget.send = lambda reply, buffers=None: replies.append((reply, buffers))

    monkeypatch.setattr(widget, "_stream_follower", SimpleNamespace(recording=recording))
    widget.live_phase_preview = preview
    widget._handle_live_probe(widget, request, [])
    reply, buffers = replies[-1]
    assert reply["scan"] == current.metadata["series"] and reply["source_scan"] == previous.metadata["series"]
    np.testing.assert_array_equal(np.frombuffer(buffers[0], np.float32), np.full(256, 10, dtype=np.float32))

    retry = {**request, "previous_scan": previous.metadata["series"]}
    widget._handle_live_probe(widget, retry, [])
    assert not replies[-1][1] and reads == [(previous.metadata["series"], 0)]

    previous.metadata["continuous_series"] = "live_11"
    widget._handle_live_probe(widget, request, [])
    assert not replies[-1][1]

    current.received = 4
    widget._handle_live_probe(widget, retry, [])
    reply, buffers = replies[-1]
    assert reply["source_scan"] == reply["scan"] == current.metadata["series"]
    np.testing.assert_array_equal(np.frombuffer(buffers[0], np.float32), np.full(256, 23, dtype=np.float32))


@pytest.mark.parametrize("mode, inner", [("circle", 0.0), ("annular", 2.25)])
def test_radius_request_acknowledges_before_one_exact_computation(viewer, monkeypatch, mode, inner):
    widget, values_t, messages = viewer
    widget.roi_mode = mode
    compute = widget._compare_images
    queries = []

    def observed(indices, mask):
        assert messages[-1]["status"] == "accepted"
        queries.append((widget.roi_radius, widget.roi_radius_inner))
        return compute(indices, mask)

    monkeypatch.setattr(widget, "_compare_images", observed)
    started = widget._last_interaction_at
    widget._msg_callbacks(widget, radius_request(widget, [5.75, inner]), [])
    assert [message["status"] for message in messages] == ["accepted", "complete"]
    assert all(message["type"] == "show4dstem_detector_radius_receipt" for message in messages)
    assert queries == [(5.75, inner)] and widget.roi_active
    assert widget._last_interaction_at >= started
    rows_t, cols_t = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    distance_t = torch.sqrt((rows_t - 7.5) ** 2 + (cols_t - 7.5) ** 2)
    mask_t = (distance_t >= inner) & (distance_t <= 5.75)
    expected_t = values_t[..., mask_t].sum(-1) / mask_t.sum()
    actual_t = torch.frombuffer(bytearray(widget.compare_virtual_image_bytes), dtype=torch.float32).reshape(2, 5, 7)
    torch.testing.assert_close(actual_t, expected_t, atol=0, rtol=0)


def test_radius_request_recovers_after_compute_failure(viewer, monkeypatch):
    widget, _, messages = viewer
    compute = widget._compare_images

    def fail(indices, mask):
        raise MemoryError("injected bounded compute failure")

    monkeypatch.setattr(widget, "_compare_images", fail)
    content = radius_request(widget, [5.75, 0])
    widget._handle_detector_radius_request(widget, content, [])
    assert [message["status"] for message in messages] == ["accepted", "rejected"]
    assert not widget.compare_virtual_image_bytes
    monkeypatch.setattr(widget, "_compare_images", compute)
    widget._handle_detector_radius_request(widget, dict(content, seq=2), [])
    assert messages[-1]["status"] == "complete" and widget.compare_virtual_image_bytes


def test_stale_or_invalid_radius_requests_leave_the_display_unchanged(viewer, monkeypatch):
    widget, _, messages = viewer
    stale = radius_request(widget, [5.75, 0])
    widget.roi_center = [5, 9]
    before = bytes(widget.compare_virtual_image_bytes)
    radii = widget.roi_radius, widget.roi_radius_inner
    queries = []
    monkeypatch.setattr(widget, "_compare_images", lambda *args: queries.append(args))
    invalid = ([0, 0], [float("inf"), 0], [5, float("nan")], [5, -1], [5, 5])
    requests = [stale, *[radius_request(widget, pair) for pair in invalid]]
    for content in requests:
        widget._handle_detector_radius_request(widget, content, [])
        assert messages[-1]["status"] == "rejected"
        assert bytes(widget.compare_virtual_image_bytes) == before
        assert (widget.roi_radius, widget.roi_radius_inner) == radii
    assert not queries


def test_a_position_click_before_the_first_frame_says_so_once_not_stale(viewer, monkeypatch, capfd):
    """11-21, microscope: a fresh viewer's first Live had no complete frame and 53 clicks were each logged as
    "stale; reopen the viewer". Now the receipt says there is no frame yet, and the log says it once."""
    widget, _, messages = viewer
    monkeypatch.setenv("SHOW4DSTEM_LOG_ACTIONS", "1")
    monkeypatch.setattr(type(widget), "comm", SimpleNamespace(comm_id="test"), raising=False)
    monkeypatch.setattr(widget, "_data", None)
    content = {"type": "show4dstem_position", "version": 1, "view": "v", "epoch": 0, "seq": 1, "center": [1, 3],
               "context": widget._interaction_context("frame_idx vi_roi_mode compare_dp_mode")}
    for seq in (1, 2, 3):
        widget._handle_position_request(widget, dict(content, seq=seq), [])
    rejected = [message for message in messages if message.get("status") == "rejected"]
    assert len(rejected) == 3 and all("No complete frame yet" in message["error"] for message in rejected)
    assert "stale" not in " ".join(message["error"] for message in rejected)
    assert capfd.readouterr().err.count("REFUSED") == 1      # the log goes to the kernel's stderr


def test_a_disk_too_faint_to_fit_leaves_the_detector_at_its_middle(monkeypatch):
    """11-21, the operator: no bright field found (another camera length, almost no beam) is not a bug: the BF detector stays
    centred on the detector."""
    from quantem.widget.show4dstem import detector

    def no_disk(pattern):
        raise ValueError("No bright-field pixels found with threshold 0.00.")

    monkeypatch.setattr(detector, "bright_field_disk", no_disk)
    pattern = np.zeros((16, 20)); pattern[3, 4] = 50.0                # a few stray counts, no disk
    assert detector.fit_disk(pattern, (16, 20), 0.25) == (8.0, 10.0, 4.0)
