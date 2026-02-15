import numpy as np
import pytest
import torch

from quantem.widget import Show3D


def test_show3d_numpy():
    """Create widget from numpy array."""
    data = np.random.rand(10, 32, 32).astype(np.float32)
    widget = Show3D(data)
    assert widget.n_slices == 10
    assert widget.height == 32
    assert widget.width == 32
    assert len(widget.frame_bytes) > 0


def test_show3d_torch():
    """Create widget from PyTorch tensor."""
    data = torch.rand(10, 32, 32)
    widget = Show3D(data)
    assert widget.n_slices == 10
    assert widget.height == 32
    assert widget.width == 32


def test_show3d_initial_slice():
    """Initial slice is at middle."""
    data = np.random.rand(20, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.slice_idx == 10


def test_show3d_stats():
    """Statistics are computed for current slice."""
    data = np.zeros((5, 16, 16), dtype=np.float32)
    data[2, :, :] = 42.0
    widget = Show3D(data)
    widget.slice_idx = 2
    assert widget.stats_mean == pytest.approx(42.0)
    assert widget.stats_std == pytest.approx(0.0)


def test_show3d_labels():
    """Labels are set correctly."""
    data = np.random.rand(3, 16, 16).astype(np.float32)
    labels = ["Frame A", "Frame B", "Frame C"]
    widget = Show3D(data, labels=labels)
    assert widget.labels == labels


def test_show3d_default_labels():
    """Default labels are indices."""
    data = np.random.rand(3, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.labels == ["0", "1", "2"]


def test_show3d_playback():
    """Playback methods work."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.play()
    assert widget.playing is True
    widget.pause()
    assert widget.playing is False
    widget.stop()
    assert widget.playing is False
    assert widget.slice_idx == 0


def test_show3d_roi():
    """ROI can be set and mean is computed."""
    data = np.ones((5, 32, 32), dtype=np.float32) * 10.0
    widget = Show3D(data)
    widget.set_roi(16, 16, radius=5)
    assert widget.roi_active is True
    assert widget.roi_mean == pytest.approx(10.0)


def test_show3d_roi_shapes():
    """Different ROI shapes work."""
    data = np.ones((5, 32, 32), dtype=np.float32) * 10.0
    widget = Show3D(data)
    for shape in ["circle", "square", "rectangle"]:
        widget.roi_shape = shape
        widget.set_roi(16, 16, radius=5)
        assert widget.roi_mean == pytest.approx(10.0)


def test_show3d_colormap():
    """Colormap option is applied."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, cmap="viridis")
    assert widget.cmap == "viridis"


def test_show3d_rejects_2d():
    """Raises error for 2D input."""
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="3D"):
        Show3D(data)


def test_show3d_timestamps():
    """Timestamps are stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    widget = Show3D(data, timestamps=times, timestamp_unit="ms")
    assert widget.timestamps == times
    assert widget.timestamp_unit == "ms"


def test_show3d_display_options():
    """Log scale and auto contrast options work."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, log_scale=True, auto_contrast=True)
    assert widget.log_scale is True
    assert widget.auto_contrast is True


def test_show3d_boomerang():
    """Boomerang mode can be set."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.boomerang = True
    assert widget.boomerang is True


def test_show3d_bookmarks():
    """Bookmarked frames can be set and retrieved."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.bookmarked_frames = [0, 5, 9]
    assert widget.bookmarked_frames == [0, 5, 9]


def test_show3d_title():
    """Title parameter is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, title="My Stack")
    assert widget.title == "My Stack"


def test_show3d_vmin_vmax():
    """Custom vmin/vmax stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, vmin=0.2, vmax=0.8)
    assert widget._vmin == pytest.approx(0.2)
    assert widget._vmax == pytest.approx(0.8)


def test_show3d_fps():
    """FPS parameter is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, fps=30.0)
    assert widget.fps == pytest.approx(30.0)


def test_show3d_pixel_size():
    """Pixel size parameter is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, pixel_size=0.5)
    assert widget.pixel_size == pytest.approx(0.5)


def test_show3d_scale_bar():
    """Scale bar visibility parameter is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, scale_bar_visible=False)
    assert widget.scale_bar_visible is False


def test_show3d_loop_range():
    """Loop range parameters are stored."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 2
    widget.loop_end = 7
    assert widget.loop_start == 2
    assert widget.loop_end == 7


def test_show3d_slice_change_updates_stats():
    """Changing slice_idx updates statistics."""
    data = np.zeros((5, 16, 16), dtype=np.float32)
    data[0] = 10.0
    data[4] = 50.0
    widget = Show3D(data)
    widget.slice_idx = 0
    assert widget.stats_mean == pytest.approx(10.0)
    widget.slice_idx = 4
    assert widget.stats_mean == pytest.approx(50.0)


def test_show3d_roi_rectangle():
    """Rectangle ROI with roi_width/height computes mean."""
    data = np.ones((5, 32, 32), dtype=np.float32) * 7.0
    widget = Show3D(data)
    widget.roi_shape = "rectangle"
    widget.roi_width = 10
    widget.roi_height = 6
    widget.set_roi(16, 16, radius=5)
    assert widget.roi_mean == pytest.approx(7.0)


def test_show3d_roi_at_edge():
    """ROI at image edge doesn't crash."""
    data = np.ones((5, 32, 32), dtype=np.float32) * 5.0
    widget = Show3D(data)
    widget.set_roi(0, 0, radius=3)
    # Should not crash, and roi_mean should be finite
    assert np.isfinite(widget.roi_mean)


def test_show3d_constant_data():
    """Constant data doesn't crash."""
    data = np.ones((5, 16, 16), dtype=np.float32) * 42.0
    widget = Show3D(data)
    assert widget.stats_mean == pytest.approx(42.0)
    assert len(widget.frame_bytes) > 0


def test_show3d_single_slice():
    """Single slice works."""
    data = np.random.rand(1, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.n_slices == 1
    assert widget.slice_idx == 0


def test_show3d_image_width():
    """image_width_px parameter is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, image_width_px=400)
    assert widget.image_width_px == 400


def test_show3d_current_timestamp():
    """Current timestamp updates with slice_idx."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    widget = Show3D(data, timestamps=times)
    widget.slice_idx = 3
    assert widget.current_timestamp == pytest.approx(1.5)


def test_show3d_reverse_trait():
    """Reverse playback trait is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.reverse = True
    assert widget.reverse is True


# --- Playback control tests ---


def test_show3d_loop_default_on():
    """Loop defaults to True."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.loop is True


def test_show3d_boomerang_default_off():
    """Boomerang defaults to False."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.boomerang is False


def test_show3d_loop_off_preserves_boomerang_off():
    """Turning loop off when boomerang is already off keeps it off."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop = False
    assert widget.boomerang is False


def test_show3d_boomerang_requires_loop_conceptually():
    """Boomerang can be set independently at trait level (JS enforces coupling)."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.boomerang = True
    assert widget.boomerang is True
    widget.loop = False
    # At Python trait level, boomerang stays set (JS toggle handles coupling)
    # This test documents that Python traits are independent
    assert widget.loop is False


def test_show3d_play_sets_playing():
    """play() sets playing to True."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.play()
    assert widget.playing is True


def test_show3d_pause_clears_playing():
    """pause() sets playing to False."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.play()
    widget.pause()
    assert widget.playing is False


def test_show3d_stop_resets_to_start():
    """stop() sets playing to False and resets to first frame."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.slice_idx = 7
    widget.play()
    widget.stop()
    assert widget.playing is False
    assert widget.slice_idx == 0


def test_show3d_stop_resets_to_zero():
    """stop() always resets to frame 0 (JS stop button respects loop_start)."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 3
    widget.slice_idx = 7
    widget.stop()
    assert widget.slice_idx == 0


def test_show3d_loop_range_set_and_get():
    """Loop range can be set and read back."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 2
    widget.loop_end = 8
    assert widget.loop_start == 2
    assert widget.loop_end == 8


def test_show3d_loop_range_default_full():
    """Default loop range covers entire stack."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.loop_start == 0
    assert widget.loop_end == -1  # -1 means last frame


def test_show3d_loop_range_reset():
    """Loop range can be reset to full."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 3
    widget.loop_end = 7
    # Reset
    widget.loop_start = 0
    widget.loop_end = -1
    assert widget.loop_start == 0
    assert widget.loop_end == -1


def test_show3d_loop_range_clamp():
    """Loop range values are stored even if out of bounds (JS clamps)."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 5
    widget.loop_end = 5
    assert widget.loop_start == 5
    assert widget.loop_end == 5


def test_show3d_play_pause_toggle():
    """Repeated play/pause toggles work."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.play()
    assert widget.playing is True
    widget.pause()
    assert widget.playing is False
    widget.play()
    assert widget.playing is True
    widget.pause()
    assert widget.playing is False


def test_show3d_reverse_with_play():
    """Reverse can be set before or during playback."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.reverse = True
    widget.play()
    assert widget.playing is True
    assert widget.reverse is True


def test_show3d_fps_range():
    """FPS can be set to various valid values."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.fps = 1
    assert widget.fps == pytest.approx(1.0)
    widget.fps = 30
    assert widget.fps == pytest.approx(30.0)


def test_show3d_playing_default_false():
    """Widget starts with playing=False."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.playing is False


def test_show3d_boomerang_with_loop():
    """Boomerang can be enabled alongside loop."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop = True
    widget.boomerang = True
    assert widget.loop is True
    assert widget.boomerang is True


def test_show3d_loop_range_with_boomerang():
    """Loop range works with boomerang mode."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop = True
    widget.boomerang = True
    widget.loop_start = 2
    widget.loop_end = 6
    assert widget.loop_start == 2
    assert widget.loop_end == 6
    assert widget.boomerang is True


def test_show3d_frame_bytes_float32_size():
    """frame_bytes has correct size (height * width * 4 bytes for float32)."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert len(widget.frame_bytes) == 16 * 16 * 4


def test_show3d_data_range():
    """data_min and data_max reflect global range across all frames."""
    data = np.zeros((5, 16, 16), dtype=np.float32)
    data[0] = -1.0
    data[4] = 10.0
    widget = Show3D(data)
    assert widget.data_min == pytest.approx(-1.0)
    assert widget.data_max == pytest.approx(10.0)


# =========================================================================
# Playback Buffer (sliding prefetch)
# =========================================================================


def test_show3d_default_buffer_size():
    """Default buffer_size is 64."""
    data = np.random.rand(100, 8, 8).astype(np.float32)
    widget = Show3D(data)
    assert widget._buffer_size == 64


def test_show3d_buffer_size_param():
    """buffer_size parameter is respected."""
    data = np.random.rand(100, 8, 8).astype(np.float32)
    widget = Show3D(data, buffer_size=32)
    assert widget._buffer_size == 32


def test_show3d_buffer_small_stack():
    """Stack smaller than buffer_size clamps to n_slices."""
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data, buffer_size=64)
    assert widget._buffer_size == 5


def test_show3d_buffer_sent_on_play():
    """Buffer bytes are sent when playback starts."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.slice_idx = 3
    widget.playing = True
    assert len(widget._buffer_bytes) > 0
    assert widget._buffer_start == 3
    assert widget._buffer_count > 0


def test_show3d_buffer_data_correct():
    """Buffer contains correct float32 frame data."""
    data = np.zeros((10, 8, 8), dtype=np.float32)
    for i in range(10):
        data[i] = float(i)
    widget = Show3D(data)
    widget.slice_idx = 0
    widget.playing = True
    buf = np.frombuffer(widget._buffer_bytes, dtype=np.float32)
    frame_size = 8 * 8
    assert buf[:frame_size].mean() == pytest.approx(0.0)
    assert buf[frame_size : 2 * frame_size].mean() == pytest.approx(1.0)


def test_show3d_slice_change_skipped_during_playback():
    """Changing slice_idx during playback does NOT trigger _update_all."""
    data = np.zeros((10, 8, 8), dtype=np.float32)
    data[0] = 10.0
    data[5] = 50.0
    widget = Show3D(data)
    widget.slice_idx = 0
    assert widget.stats_mean == pytest.approx(10.0)
    widget.playing = True
    widget.slice_idx = 5
    # Stats should NOT have updated (still 10.0 from frame 0)
    assert widget.stats_mean == pytest.approx(10.0)


def test_show3d_stats_correct_after_stop():
    """After playback stops and slice_idx is set, stats are recomputed."""
    data = np.zeros((10, 8, 8), dtype=np.float32)
    data[5] = 42.0
    widget = Show3D(data)
    widget.slice_idx = 0
    widget.playing = True
    widget.playing = False
    widget.slice_idx = 5
    assert widget.stats_mean == pytest.approx(42.0)


def test_show3d_prefetch_triggers_buffer():
    """Setting _prefetch_request triggers new buffer send."""
    data = np.random.rand(100, 8, 8).astype(np.float32)
    widget = Show3D(data, buffer_size=32)
    widget.slice_idx = 0
    widget.playing = True
    assert widget._buffer_start == 0
    widget._prefetch_request = 32
    assert widget._buffer_start == 32
    assert widget._buffer_count == 32


def test_show3d_prefetch_ignored_when_not_playing():
    """Prefetch request is ignored when not playing."""
    data = np.random.rand(100, 8, 8).astype(np.float32)
    widget = Show3D(data, buffer_size=32)
    widget._prefetch_request = 50
    assert widget._buffer_start == 0
    assert widget._buffer_count == 0


def test_show3d_buffer_wraparound():
    """Buffer wraps around when start_idx + buffer_size > n_slices."""
    data = np.zeros((10, 4, 4), dtype=np.float32)
    for i in range(10):
        data[i] = float(i)
    widget = Show3D(data, buffer_size=8)
    widget.slice_idx = 5
    widget.playing = True
    assert widget._buffer_start == 5
    assert widget._buffer_count == 8
    buf = np.frombuffer(widget._buffer_bytes, dtype=np.float32)
    frame_size = 4 * 4
    # Frame at index 5 (first in buffer)
    assert buf[:frame_size].mean() == pytest.approx(5.0)
    # Frame at index 9 (5th in buffer, last before wrap)
    assert buf[4 * frame_size : 5 * frame_size].mean() == pytest.approx(9.0)
    # Frame at index 0 (6th in buffer, wrapped)
    assert buf[5 * frame_size : 6 * frame_size].mean() == pytest.approx(0.0)


