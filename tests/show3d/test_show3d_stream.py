"""Show3D.stream: append-only transfers, the acknowledgment window and background-thread appends."""

import threading

import numpy as np
import pytest

from quantem.widget import Show3D


def _capture(widget):
    """Record every comm message the widget sends as (method, content-or-state, buffer sizes)."""
    sent = []
    original = widget._send

    def record(msg, buffers=None):
        body = msg.get("content") if msg.get("method") == "custom" else msg.get("state")
        sent.append((msg.get("method"), body, [memoryview(buffer).nbytes for buffer in buffers or []]))
        return original(msg, buffers)

    widget._send = record
    return sent


def _frames(sent):
    return [(body, sizes) for method, body, sizes in sent if method == "custom" and body["type"] == "show3d_stream_frames"]


def _ready(widget, frames=None):
    widget._handle_stream_msg(widget, {"type": "show3d_frontend_ready", "version": 1,
                                       "frames": int(widget.n_slices) if frames is None else frames,
                                       "revision": int(widget.frame_seq)}, [])


def test_each_append_sends_only_the_new_frame():
    view = Show3D.stream(title="live")
    assert (view.n_slices, view.folder_waiting) == (0, True)
    sent = _capture(view)
    rows, cols = 16, 24
    frames = [np.random.default_rng(seed).random((rows, cols)).astype(np.float32) for seed in range(4)]
    for index, frame in enumerate(frames):
        view.append(frame, label=f"scan {index}")
        assert view._stream_wait_sent(5)
    transfers = _frames(sent)
    assert [body["start"] for body, _ in transfers] == [0, 1, 2, 3]
    # one frame's bytes per append, and the stack trait never travels
    assert all(sizes == [rows * cols * 4] for _, sizes in transfers)
    state_buffers = [sizes for method, _, sizes in sent if method == "update"]
    assert all(not sizes for sizes in state_buffers)
    assert all("_offline_float_stack" not in body for method, body, _ in sent if method == "update")
    assert (view.n_slices, view.height, view.width, view.folder_waiting) == (4, rows, cols, False)
    assert view.labels == [f"scan {index}" for index in range(4)]
    assert view.slice_idx == 3
    np.testing.assert_array_equal(view._data, np.stack(frames))
    # a full snapshot (a view that mounts later) still gets the whole stack
    state = view.get_state(key="_offline_float_stack")
    assert memoryview(state["_offline_float_stack"]).nbytes == 4 * rows * cols * 4
    assert len(view.auto_vmins) == 4 and view.data_max == pytest.approx(float(np.max(frames)))
    view.close()


def test_appended_frame_is_copied_before_append_returns():
    view = Show3D.stream()
    frame = np.ones((4, 4), dtype=np.float32)
    view.append(frame)
    frame[:] = 7  # the caller reuses its buffer at once
    assert view._stream_wait_sent(5)
    np.testing.assert_array_equal(view._data[0], np.ones((4, 4)))
    with pytest.raises(ValueError, match="does not match"):
        view.append(np.ones((5, 4)))
    view.close()


def test_two_transfers_await_acknowledgment_and_later_frames_join_one_transfer():
    view = Show3D.stream()
    sent = _capture(view)
    _ready(view)
    view._stream_credits.stalled_ack_s = 60.0
    for value in range(2):
        view.append(np.full((4, 4), value, dtype=np.float32))
        assert view._stream_wait_sent(5)
    for value in range(2, 5):
        view.append(np.full((4, 4), value, dtype=np.float32))
    assert not view._stream_wait_sent(0.3)  # no credit: frames wait in Python, none dropped
    assert len(_frames(sent)) == 2 and view.n_slices == 2
    view._handle_stream_msg(view, {"type": "show3d_stream_ack", "version": 1, "seq": 1}, [])
    assert view._stream_wait_sent(5)
    body, sizes = _frames(sent)[-1]
    assert (body["start"], body["count"], sizes) == (2, 3, [64, 64, 64])
    assert [float(view._data[index, 0, 0]) for index in range(5)] == [0, 1, 2, 3, 4]
    view.close()


def test_view_stays_on_an_earlier_frame_the_user_chose():
    view = Show3D.stream(np.zeros((3, 4, 4), dtype=np.float32))
    assert view.slice_idx == 2
    view.slice_idx = 0
    view.append(np.ones((4, 4)))
    assert view._stream_wait_sent(5)
    assert (view.n_slices, view.slice_idx) == (4, 0)
    view.slice_idx = 3
    view.append(np.ones((4, 4)))
    assert view._stream_wait_sent(5)
    assert view.slice_idx == 4
    view.close()


def test_late_view_receives_the_whole_stack_once():
    view = Show3D.stream()
    for _ in range(3):
        view.append(np.ones((4, 4)))
    assert view._stream_wait_sent(5)
    sent = _capture(view)
    seq = view.frame_seq
    _ready(view, frames=0)
    # binary traits leave the state dict as buffers; the stack is the only one
    full = [sizes for method, _, sizes in sent if method == "update" and sizes]
    assert full == [[3 * 4 * 4 * 4]] and view.frame_seq == seq + 1
    view.close()


def test_appends_from_a_background_thread_while_the_user_interacts():
    view = Show3D.stream()
    sent = _capture(view)

    def acquire():
        for index in range(20):
            view.append(np.full((8, 8), index, dtype=np.float32), label=f"t{index}")

    worker = threading.Thread(target=acquire)
    worker.start()
    for _ in range(50):
        view.set_state({"cmap": "gray"})
    worker.join()
    assert view._stream_wait_sent(5)
    assert view.n_slices == 20 and view.labels == [f"t{index}" for index in range(20)]
    assert [float(view._data[index, 0, 0]) for index in range(20)] == list(range(20))
    starts = [(body["start"], body["count"]) for body, _ in _frames(sent)]
    assert sum(count for _, count in starts) == 20
    assert all(start == sum(count for _, count in starts[:position]) for position, (start, _) in enumerate(starts))
    view.close()


def test_stream_requires_native_pixels_and_append_requires_a_stream():
    with pytest.raises(ValueError, match="native pixels"):
        Show3D.stream(display_bin=2)
    plain = Show3D(np.zeros((2, 4, 4), dtype=np.float32), verbose=False)
    with pytest.raises(RuntimeError, match="Show3D.stream"):
        plain.append(np.zeros((4, 4)))
    plain.close()


def test_find_drift_in_a_live_stream_does_not_stop_it():
    """A Find drift result in a live stream keeps its one stack: frames appended after it still arrive."""
    import numpy as np

    from quantem.widget import Show3D

    viewer = Show3D.stream(np.zeros((3, 8, 8), np.float32))
    viewer.drift = {"shift_px": [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], "velocity": [0.1, 0.1], "velocity_unit": "nm/s"}
    assert int(viewer.n_panels) == 1                                    # no drift-corrected panel in a live stream
    viewer.append(np.ones((8, 8), np.float32))
    assert viewer._stream_wait_sent(5) and int(viewer.n_slices) == 4    # the stream goes on


def test_find_drift_after_the_stream_ends_adds_the_drift_corrected_panel():
    """scan(n) in spectra: frames stream in, then end_stream(); Find drift then shows the corrected panel as for any
    viewer of frames already taken (11-13 on the site it did not, the viewer still counted as live)."""
    import numpy as np

    from quantem.widget import Show3D

    viewer = Show3D.stream(np.zeros((1, 8, 8), np.float32))
    for _ in range(3):
        viewer.append(np.ones((8, 8), np.float32))
    assert viewer.end_stream(5) and int(viewer.n_slices) == 4          # what was queued arrived first
    assert viewer.end_stream()                                          # once is enough; again is harmless
    viewer.drift = {"shift_px": [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]], "velocity": [0.1, 0.1],
                    "velocity_px_per_frame": [0.5, 0.5], "frames": [0, 1, 2, 3], "velocity_unit": "nm/s"}
    assert int(viewer.n_panels) == 2                                    # frames | drift corrected
    import pytest

    with pytest.raises(RuntimeError, match="running viewer"):
        viewer.append(np.ones((8, 8), np.float32))                      # ended: no more frames
