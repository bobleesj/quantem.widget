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
    assert (widget.shape_x, widget.shape_y) == (2, 3)
    assert (widget.det_x, widget.det_y) == (2, 2)
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
    assert abs(widget.center_x - 3.0) < 0.5
    assert abs(widget.center_y - 3.0) < 0.5
    assert widget.bf_radius > 0


def test_show4dstem_adf_preset_cache():
    """Test that ADF preset cache works when precompute is enabled."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, center=(8, 8), bf_radius=2, precompute_virtual_images=True)
    assert widget._cached_adf_virtual is not None
    widget.roi_mode = "annular"
    widget.roi_center_x = 8
    widget.roi_center_y = 8
    widget.roi_radius_inner = 2
    widget.roi_radius = 8
    cached = widget._get_cached_preset()
    assert cached == widget._cached_adf_virtual


def test_show4dstem_rectangular_scan_shape():
    """Test that rectangular (non-square) scans work correctly."""
    data = np.random.rand(4, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.shape_x == 4
    assert widget.shape_y == 8
    assert widget.det_x == 16
    assert widget.det_y == 16
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
    widget.roi_center_x = 4
    widget.roi_center_y = 4
    widget.roi_radius = 3
    assert widget.vi_stats[2] < 1000


def test_show4dstem_torch_input():
    """PyTorch tensor input works."""
    import torch
    data = torch.rand(4, 4, 8, 8)
    widget = Show4DSTEM(data)
    assert widget.shape_x == 4
    assert widget.shape_y == 4
    assert widget.det_x == 8
    assert widget.det_y == 8


def test_show4dstem_position_property():
    """Position property get/set works."""
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.position = (2, 3)
    assert widget.position == (2, 3)
    assert widget.pos_x == 2
    assert widget.pos_y == 3


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
    assert widget.pos_x == 4
    assert widget.pos_y == 5


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


def test_show4dstem_rejects_5d():
    """5D input raises ValueError."""
    data = np.random.rand(2, 2, 2, 8, 8).astype(np.float32)
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
    assert widget.center_x == 5.0
    assert widget.center_y == 6.0
    assert widget.bf_radius == 3.0
