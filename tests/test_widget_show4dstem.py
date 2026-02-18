import numpy as np
import quantem.widget
from quantem.widget import Show4DSTEM


def test_version_exists():
    assert hasattr(quantem.widget, "__version__")


def test_version_is_string():
    assert isinstance(quantem.widget.__version__, str)


def test_show4dstem_loads():
    """Widget can be created from mock 4D data."""
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget is not None


def test_show4dstem_flattened_scan_shape_mapping():
    """Test flattened 3D data with explicit scan shape."""
    data = np.zeros((6, 2, 2), dtype=np.float32)
    for idx in range(data.shape[0]):
        data[idx] = idx
    widget = Show4DSTEM(data, scan_shape=(2, 3))
    assert (widget.shape_rows, widget.shape_cols) == (2, 3)
    assert (widget.det_rows, widget.det_cols) == (2, 2)
    frame = widget._get_frame(1, 2)
    assert np.array_equal(frame, np.full((2, 2), 5, dtype=np.float32))


def test_show4dstem_log_scale():
    """Test that log scale changes frame bytes."""
    data = np.random.rand(2, 2, 8, 8).astype(np.float32) * 100 + 1
    widget = Show4DSTEM(data, log_scale=True)
    log_bytes = bytes(widget.frame_bytes)
    widget.log_scale = False
    widget._update_frame()
    linear_bytes = bytes(widget.frame_bytes)
    assert log_bytes != linear_bytes


def test_show4dstem_auto_detect_center():
    """Test automatic center spot detection using centroid."""
    data = np.zeros((2, 2, 7, 7), dtype=np.float32)
    for i in range(7):
        for j in range(7):
            dist = np.sqrt((i - 3) ** 2 + (j - 3) ** 2)
            if dist <= 1.5:
                data[:, :, i, j] = 100.0
    widget = Show4DSTEM(data, precompute_virtual_images=False)
    widget.auto_detect_center()
    assert abs(widget.center_col - 3.0) < 0.5
    assert abs(widget.center_row - 3.0) < 0.5
    assert widget.bf_radius > 0


def test_show4dstem_adf_preset_cache():
    """Test that ADF preset cache works when precompute is enabled."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, center=(8, 8), bf_radius=2, precompute_virtual_images=True)
    assert widget._cached_adf_virtual is not None
    widget.roi_mode = "annular"
    widget.roi_center_col = 8
    widget.roi_center_row = 8
    widget.roi_radius_inner = 2
    widget.roi_radius = 8
    cached = widget._get_cached_preset()
    assert cached == widget._cached_adf_virtual


def test_show4dstem_rectangular_scan_shape():
    """Test that rectangular (non-square) scans work correctly."""
    data = np.random.rand(4, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.shape_rows == 4
    assert widget.shape_cols == 8
    assert widget.det_rows == 16
    assert widget.det_cols == 16
    frame_00 = widget._get_frame(0, 0)
    frame_37 = widget._get_frame(3, 7)
    assert frame_00.shape == (16, 16)
    assert frame_37.shape == (16, 16)


def test_show4dstem_hot_pixel_removal_uint16():
    """Test that saturated uint16 hot pixels are removed at init."""
    data = np.zeros((4, 4, 8, 8), dtype=np.uint16)
    data[:, :, :, :] = 100
    data[:, :, 3, 5] = 65535
    data[:, :, 1, 2] = 65535
    widget = Show4DSTEM(data)
    assert widget.dp_global_max < 65535
    assert widget.dp_global_max == 100.0
    frame = widget._get_frame(0, 0)
    assert frame[3, 5] == 0
    assert frame[1, 2] == 0
    assert frame[0, 0] == 100


def test_show4dstem_hot_pixel_removal_uint8():
    """Test that saturated uint8 hot pixels are removed at init."""
    data = np.zeros((4, 4, 8, 8), dtype=np.uint8)
    data[:, :, :, :] = 50
    data[:, :, 2, 3] = 255
    widget = Show4DSTEM(data)
    assert widget.dp_global_max == 50.0
    frame = widget._get_frame(0, 0)
    assert frame[2, 3] == 0


def test_show4dstem_no_hot_pixel_removal_float32():
    """Test that float32 data is not modified (no saturated value)."""
    data = np.ones((4, 4, 8, 8), dtype=np.float32) * 1000
    widget = Show4DSTEM(data)
    assert widget.dp_global_max == 1000.0


def test_show4dstem_roi_modes():
    """Test all ROI modes compute virtual images correctly."""
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, center=(8, 8), bf_radius=3)
    for mode in ["point", "circle", "square", "annular", "rect"]:
        widget.roi_mode = mode
        widget.roi_active = True
        assert len(widget.vi_stats) == 4
        assert widget.vi_stats[2] >= widget.vi_stats[1]


def test_show4dstem_virtual_image_excludes_hot_pixels():
    """Test that virtual images don't include hot pixel contributions."""
    data = np.ones((4, 4, 8, 8), dtype=np.uint16) * 10
    data[:, :, 4, 4] = 65535
    widget = Show4DSTEM(data, center=(4, 4), bf_radius=2)
    widget.roi_mode = "circle"
    widget.roi_center_col = 4
    widget.roi_center_row = 4
    widget.roi_radius = 3
    assert widget.vi_stats[2] < 1000


def test_show4dstem_torch_input():
    """PyTorch tensor input works."""
    import torch
    data = torch.rand(4, 4, 8, 8)
    widget = Show4DSTEM(data)
    assert widget.shape_rows == 4
    assert widget.shape_cols == 4
    assert widget.det_rows == 8
    assert widget.det_cols == 8


def test_show4dstem_position_property():
    """Position property get/set works."""
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.position = (2, 3)
    assert widget.position == (2, 3)
    assert widget.pos_row == 2
    assert widget.pos_col == 3


def test_show4dstem_scan_shape_property():
    """scan_shape property returns correct tuple."""
    data = np.random.rand(4, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.scan_shape == (4, 8)


def test_show4dstem_detector_shape_property():
    """detector_shape property returns correct tuple."""
    data = np.random.rand(4, 4, 12, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.detector_shape == (12, 16)


def test_show4dstem_initial_position():
    """Initial position is at scan center."""
    data = np.random.rand(8, 10, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.pos_row == 4
    assert widget.pos_col == 5


def test_show4dstem_frame_bytes_nonzero():
    """frame_bytes is non-empty after init."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4DSTEM(data)
    assert len(widget.frame_bytes) > 0
    assert len(widget.frame_bytes) == 8 * 8 * 4  # float32


def test_show4dstem_roi_circle_method():
    """roi_circle() sets mode and optional radius."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_circle(5.0)
    assert widget.roi_mode == "circle"
    assert widget.roi_radius == 5.0


def test_show4dstem_roi_square_method():
    """roi_square() sets mode and optional half_size."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_square(7.0)
    assert widget.roi_mode == "square"
    assert widget.roi_radius == 7.0


def test_show4dstem_roi_annular_method():
    """roi_annular() sets mode and inner/outer radii."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_annular(3.0, 10.0)
    assert widget.roi_mode == "annular"
    assert widget.roi_radius_inner == 3.0
    assert widget.roi_radius == 10.0


def test_show4dstem_roi_rect_method():
    """roi_rect() sets mode and width/height."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_rect(20.0, 15.0)
    assert widget.roi_mode == "rect"
    assert widget.roi_width == 20.0
    assert widget.roi_height == 15.0


def test_show4dstem_roi_point_method():
    """roi_point() sets mode to point."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_point()
    assert widget.roi_mode == "point"


def test_show4dstem_method_chaining():
    """All ROI and playback methods return self for chaining."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.roi_circle(5) is widget
    assert widget.roi_point() is widget
    assert widget.roi_square(3) is widget
    assert widget.roi_annular(2, 8) is widget
    assert widget.roi_rect(10, 5) is widget
    assert widget.auto_detect_center() is widget
    assert widget.pause() is widget
    assert widget.stop() is widget


def test_show4dstem_path_animation():
    """set_path, play, pause, stop, goto work."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4DSTEM(data)
    points = [(0, 0), (1, 1), (2, 2), (3, 3)]
    widget.set_path(points, interval_ms=50, loop=True, autoplay=False)
    assert widget.path_length == 4
    assert widget.path_playing is False
    widget.play()
    assert widget.path_playing is True
    widget.pause()
    assert widget.path_playing is False
    widget.goto(2)
    assert widget.path_index == 2
    widget.stop()
    assert widget.path_index == 0


def test_show4dstem_raster():
    """raster() creates a path covering the scan."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.raster(step=2, interval_ms=50, loop=False)
    assert widget.path_length > 0
    assert widget.path_playing is True  # autoplay by default


def test_show4dstem_calibration():
    """Calibration parameters are stored."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)
    assert widget.pixel_size == 2.39
    assert widget.k_pixel_size == 0.46
    assert widget.k_calibrated is True


def test_show4dstem_dp_stats():
    """dp_stats has 4 values (mean, min, max, std)."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4DSTEM(data)
    assert len(widget.dp_stats) == 4


def test_show4dstem_vi_stats():
    """vi_stats has 4 values (mean, min, max, std)."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4DSTEM(data)
    assert len(widget.vi_stats) == 4


def test_show4dstem_rejects_2d():
    """2D input raises ValueError."""
    data = np.random.rand(16, 16).astype(np.float32)
    try:
        Show4DSTEM(data)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_show4dstem_rejects_6d():
    """6D input raises ValueError."""
    data = np.random.rand(2, 2, 2, 2, 8, 8).astype(np.float32)
    try:
        Show4DSTEM(data)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_show4dstem_non_square_flattened():
    """Non-perfect-square N without scan_shape raises ValueError."""
    data = np.random.rand(7, 8, 8).astype(np.float32)
    try:
        Show4DSTEM(data)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_show4dstem_repr():
    """__repr__ returns useful string."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    r = repr(widget)
    assert "Show4DSTEM" in r
    assert "4" in r
    assert "16" in r


def test_show4dstem_virtual_image_bytes_nonzero():
    """Virtual image bytes are populated after ROI setup."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_circle(5)
    assert len(widget.virtual_image_bytes) > 0


def test_show4dstem_center_explicit():
    """Explicit center and bf_radius are used."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, center=(5.0, 6.0), bf_radius=3.0)
    assert widget.center_row == 5.0
    assert widget.center_col == 6.0
    assert widget.bf_radius == 3.0


# ── State Protocol ────────────────────────────────────────────────────────


def test_show4dstem_state_dict_roundtrip():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, log_scale=True, center=(5.0, 6.0), bf_radius=3.0)
    sd = w.state_dict()
    assert sd["log_scale"] is True
    assert sd["center_row"] == 5.0
    assert sd["center_col"] == 6.0
    assert sd["bf_radius"] == 3.0
    w2 = Show4DSTEM(data, state=sd)
    assert w2.log_scale is True
    assert w2.bf_radius == 3.0


def test_show4dstem_save_load_file(tmp_path):
    import json
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, log_scale=True)
    path = tmp_path / "stem_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["log_scale"] is True
    w2 = Show4DSTEM(data, state=str(path))
    assert w2.log_scale is True


def test_show4dstem_summary(capsys):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)
    w.summary()
    out = capsys.readouterr().out
    assert "Show4DSTEM" in out
    assert "4×4" in out
    assert "16×16" in out
    assert "2.39" in out


def test_show4dstem_set_image():
    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.shape_rows == 4

    new_data = np.random.rand(8, 8, 64, 64).astype(np.float32)
    widget.set_image(new_data)
    assert widget.shape_rows == 8
    assert widget.shape_cols == 8
    assert widget.det_rows == 64
    assert widget.det_cols == 64


# ── Line Profile ─────────────────────────────────────────────────────────


def test_show4dstem_profile_defaults():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    assert w.profile_line == []
    assert w.profile_width == 1
    assert w.profile == []
    assert w.profile_values is None
    assert w.profile_distance == 0.0


def test_show4dstem_set_profile():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    result = w.set_profile((0, 0), (15, 15))
    assert result is w
    assert len(w.profile_line) == 2
    assert w.profile_line[0] == {"row": 0.0, "col": 0.0}
    assert w.profile_line[1] == {"row": 15.0, "col": 15.0}


def test_show4dstem_clear_profile():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_profile((0, 0), (15, 15))
    assert len(w.profile_line) == 2
    result = w.clear_profile()
    assert result is w
    assert w.profile_line == []


def test_show4dstem_profile_property():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_profile((2.0, 3.0), (12.0, 8.0))
    pts = w.profile
    assert len(pts) == 2
    assert pts[0] == (2.0, 3.0)
    assert pts[1] == (12.0, 8.0)


def test_show4dstem_profile_values():
    data = np.ones((4, 4, 16, 16), dtype=np.float32) * 3.0
    w = Show4DSTEM(data)
    w.set_profile((0, 0), (15, 0))
    vals = w.profile_values
    assert vals is not None
    assert len(vals) >= 2
    assert np.allclose(vals, 3.0, atol=0.01)


def test_show4dstem_profile_distance_calibrated():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, k_pixel_size=0.5)
    w.set_profile((0, 0), (3, 4))
    # pixel distance = 5, k-calibrated = 5 * 0.5 = 2.5
    assert abs(w.profile_distance - 2.5) < 0.01


def test_show4dstem_profile_distance_uncalibrated():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_profile((0, 0), (3, 4))
    # No k_pixel_size calibration → pixel distance = 5
    assert abs(w.profile_distance - 5.0) < 0.01


def test_show4dstem_profile_in_state_dict():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_profile((1, 2), (10, 12))
    w.profile_width = 5
    sd = w.state_dict()
    assert "profile_line" in sd
    assert "profile_width" in sd
    assert sd["profile_width"] == 5
    assert len(sd["profile_line"]) == 2


def test_show4dstem_profile_in_summary(capsys):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_profile((0, 0), (15, 15))
    w.summary()
    out = capsys.readouterr().out
    assert "Profile:" in out


# ── GIF Export ──────────────────────────────────────────────────────────────

def test_show4dstem_gif_export_defaults():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    assert w._gif_export_requested is False
    assert w._gif_data == b""


def test_show4dstem_gif_generation_with_path():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_path([(0, 0), (1, 1), (2, 2)], autoplay=False)
    w._generate_gif()
    assert len(w._gif_data) > 0
    assert w._gif_data[:3] == b"GIF"


def test_show4dstem_gif_generation_no_path():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w._generate_gif()
    assert w._gif_data == b""


def test_show4dstem_normalize_frame():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    frame = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    result = w._normalize_frame(frame)
    assert result.dtype == np.uint8
    assert result.shape == (2, 2)


# ── 5D Time/Tilt Series ────────────────────────────────────────────────────


def test_show4dstem_5d_basic():
    """5D array creates widget with n_frames > 1."""
    data = np.random.rand(5, 4, 4, 8, 8).astype(np.float32)
    w = Show4DSTEM(data)
    assert w.n_frames == 5
    assert w.shape_rows == 4
    assert w.shape_cols == 4
    assert w.det_rows == 8
    assert w.det_cols == 8
    assert w.frame_idx == 0


def test_show4dstem_5d_frame_navigation():
    """Changing frame_idx updates the displayed frame."""
    data = np.zeros((3, 2, 2, 4, 4), dtype=np.float32)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    w = Show4DSTEM(data)
    frame0 = w._get_frame(0, 0)
    assert np.allclose(frame0, 1.0)
    w.frame_idx = 1
    frame1 = w._get_frame(0, 0)
    assert np.allclose(frame1, 2.0)
    w.frame_idx = 2
    frame2 = w._get_frame(0, 0)
    assert np.allclose(frame2, 3.0)


def test_show4dstem_5d_frame_dim_label():
    """frame_dim_label is set from constructor param."""
    data = np.random.rand(3, 2, 2, 4, 4).astype(np.float32)
    w = Show4DSTEM(data, frame_dim_label="Tilt")
    assert w.frame_dim_label == "Tilt"
    w2 = Show4DSTEM(data)
    assert w2.frame_dim_label == "Frame"


def test_show4dstem_5d_virtual_image_per_frame():
    """Virtual image changes when frame_idx changes."""
    data = np.zeros((2, 4, 4, 8, 8), dtype=np.float32)
    data[0, :, :, 3:5, 3:5] = 10.0
    data[1, :, :, 3:5, 3:5] = 50.0
    w = Show4DSTEM(data, center=(4, 4), bf_radius=3)
    w.roi_mode = "circle"
    w.roi_center_row = 4.0
    w.roi_center_col = 4.0
    w.roi_radius = 3.0
    vi_bytes_0 = bytes(w.virtual_image_bytes)
    w.frame_idx = 1
    vi_bytes_1 = bytes(w.virtual_image_bytes)
    assert vi_bytes_0 != vi_bytes_1


def test_show4dstem_5d_global_range():
    """dp_global_min/max spans all frames."""
    data = np.zeros((3, 2, 2, 4, 4), dtype=np.float32)
    data[0] = 1.0
    data[1] = 5.0
    data[2] = 10.0
    w = Show4DSTEM(data)
    assert w.dp_global_min <= 1.0
    assert w.dp_global_max >= 10.0


def test_show4dstem_5d_set_image():
    """set_image works with 5D data."""
    data_4d = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4DSTEM(data_4d)
    assert w.n_frames == 1
    data_5d = np.random.rand(3, 4, 4, 8, 8).astype(np.float32)
    w.set_image(data_5d)
    assert w.n_frames == 3
    assert w.frame_idx == 0


def test_show4dstem_5d_state_dict():
    """state_dict includes frame traits."""
    data = np.random.rand(3, 2, 2, 4, 4).astype(np.float32)
    w = Show4DSTEM(data, frame_dim_label="Time")
    w.frame_idx = 1
    sd = w.state_dict()
    assert sd["frame_idx"] == 1
    assert sd["frame_dim_label"] == "Time"
    assert "frame_loop" in sd
    assert "frame_interval_ms" in sd


def test_show4dstem_5d_state_roundtrip():
    """State can be saved and restored for 5D widget."""
    data = np.random.rand(3, 2, 2, 4, 4).astype(np.float32)
    w = Show4DSTEM(data, frame_dim_label="Tilt")
    w.frame_idx = 2
    w.frame_interval_ms = 100
    sd = w.state_dict()
    w2 = Show4DSTEM(data, state=sd)
    assert w2.frame_idx == 2
    assert w2.frame_dim_label == "Tilt"
    assert w2.frame_interval_ms == 100


def test_show4dstem_5d_summary(capsys):
    """summary() shows frame info for 5D data."""
    data = np.random.rand(5, 2, 2, 4, 4).astype(np.float32)
    w = Show4DSTEM(data, frame_dim_label="Tilt")
    w.frame_idx = 2
    w.summary()
    out = capsys.readouterr().out
    assert "Frames:" in out
    assert "Tilt" in out


def test_show4dstem_5d_repr():
    """__repr__ includes frame info for 5D data."""
    data = np.random.rand(3, 2, 2, 4, 4).astype(np.float32)
    w = Show4DSTEM(data, frame_dim_label="Focus")
    r = repr(w)
    assert "3," in r  # n_frames in shape
    assert "focus=" in r  # frame_dim_label.lower()


def test_show4dstem_4d_no_frame_traits():
    """4D data keeps n_frames=1, no frame info in repr."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4DSTEM(data)
    assert w.n_frames == 1
    assert w.frame_idx == 0
    r = repr(w)
    assert "frame" not in r.lower() or "frame" not in r


def test_show4dstem_5d_torch_input():
    """5D PyTorch tensor input works."""
    import torch
    data = torch.rand(3, 4, 4, 8, 8)
    w = Show4DSTEM(data)
    assert w.n_frames == 3
    assert w.shape_rows == 4
    assert w.det_rows == 8
