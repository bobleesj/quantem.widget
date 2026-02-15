import numpy as np
import pytest
import quantem.widget
from quantem.widget import Show4D


def test_version_exists():
    assert hasattr(quantem.widget, "__version__")


def test_show4d_loads():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    assert widget is not None


def test_show4d_shape_traits():
    data = np.random.rand(4, 8, 12, 16).astype(np.float32)
    widget = Show4D(data)
    assert widget.nav_x == 4
    assert widget.nav_y == 8
    assert widget.sig_x == 12
    assert widget.sig_y == 16


def test_show4d_initial_position():
    data = np.random.rand(8, 10, 16, 16).astype(np.float32)
    widget = Show4D(data)
    assert widget.pos_x == 4
    assert widget.pos_y == 5


def test_show4d_frame_bytes_nonzero():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert len(widget.frame_bytes) == 8 * 8 * 4  # float32


def test_show4d_nav_image_bytes_nonzero():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert len(widget.nav_image_bytes) == 4 * 4 * 4  # float32


def test_show4d_default_nav_image_is_mean():
    data = np.ones((4, 4, 8, 8), dtype=np.float32) * 42
    widget = Show4D(data)
    nav_arr = np.frombuffer(widget.nav_image_bytes, dtype=np.float32)
    assert np.allclose(nav_arr, 42.0)


def test_show4d_nav_image_override():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    nav = np.ones((4, 4), dtype=np.float32) * 99
    widget = Show4D(data, nav_image=nav)
    nav_arr = np.frombuffer(widget.nav_image_bytes, dtype=np.float32)
    assert np.allclose(nav_arr, 99.0)


def test_show4d_position_change_updates_frame():
    data = np.zeros((4, 4, 8, 8), dtype=np.float32)
    data[0, 0] = 1.0
    data[1, 1] = 2.0
    widget = Show4D(data)
    widget.pos_x = 0
    widget.pos_y = 0
    frame0 = bytes(widget.frame_bytes)
    widget.pos_x = 1
    widget.pos_y = 1
    frame1 = bytes(widget.frame_bytes)
    assert frame0 != frame1


def test_show4d_position_property():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.position = (2, 3)
    assert widget.position == (2, 3)
    assert widget.pos_x == 2
    assert widget.pos_y == 3


def test_show4d_roi_circle():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "circle"
    widget.roi_center_x = 4
    widget.roi_center_y = 4
    widget.roi_radius = 3
    assert len(widget.frame_bytes) == 16 * 16 * 4
    assert len(widget.sig_stats) == 4


def test_show4d_roi_square():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "square"
    widget.roi_center_x = 4
    widget.roi_center_y = 4
    widget.roi_radius = 2
    assert len(widget.frame_bytes) > 0


def test_show4d_roi_rect():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "rect"
    widget.roi_center_x = 4
    widget.roi_center_y = 4
    widget.roi_width = 4
    widget.roi_height = 6
    assert len(widget.frame_bytes) > 0


def test_show4d_roi_off_falls_back():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "circle"
    widget.roi_mode = "off"
    assert len(widget.frame_bytes) == 16 * 16 * 4


def test_show4d_nav_stats():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert len(widget.nav_stats) == 4
    assert widget.nav_stats[2] >= widget.nav_stats[1]  # max >= min


def test_show4d_sig_stats():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert len(widget.sig_stats) == 4
    assert widget.sig_stats[2] >= widget.sig_stats[1]  # max >= min


def test_show4d_pixel_size():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data, nav_pixel_size=2.5, sig_pixel_size=0.5, nav_pixel_unit="Å", sig_pixel_unit="mrad")
    assert widget.nav_pixel_size == 2.5
    assert widget.sig_pixel_size == 0.5
    assert widget.nav_pixel_unit == "Å"
    assert widget.sig_pixel_unit == "mrad"


def test_show4d_torch_input():
    import torch
    data = torch.rand(4, 4, 8, 8)
    widget = Show4D(data)
    assert widget.nav_x == 4
    assert widget.sig_x == 8


def test_show4d_rejects_2d():
    with pytest.raises(ValueError):
        Show4D(np.random.rand(16, 16).astype(np.float32))


def test_show4d_rejects_3d():
    with pytest.raises(ValueError):
        Show4D(np.random.rand(4, 16, 16).astype(np.float32))


def test_show4d_rejects_5d():
    with pytest.raises(ValueError):
        Show4D(np.random.rand(2, 2, 2, 8, 8).astype(np.float32))


def test_show4d_title():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data, title="Test 4D")
    assert widget.title == "Test 4D"


def test_show4d_repr():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4D(data)
    r = repr(widget)
    assert "Show4D" in r
    assert "4" in r
    assert "16" in r


def test_show4d_nav_shape_property():
    data = np.random.rand(4, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    assert widget.nav_shape == (4, 8)


def test_show4d_sig_shape_property():
    data = np.random.rand(4, 4, 12, 16).astype(np.float32)
    widget = Show4D(data)
    assert widget.sig_shape == (12, 16)


def test_show4d_data_ranges():
    data = np.ones((4, 4, 8, 8), dtype=np.float32) * 42
    widget = Show4D(data)
    assert widget.nav_data_min > 0
    assert widget.sig_data_min == 42.0
    assert widget.sig_data_max == 42.0


def test_show4d_roi_empty_mask_falls_back():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "circle"
    widget.roi_center_x = -100
    widget.roi_center_y = -100
    widget.roi_radius = 1
    # Should fall back to single position frame
    assert len(widget.frame_bytes) == 16 * 16 * 4


def test_show4d_roi_reduce_default():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert widget.roi_reduce == "mean"


def test_show4d_roi_reduce_mean():
    data = np.zeros((4, 4, 8, 8), dtype=np.float32)
    data[0, 0] = 2.0
    data[0, 1] = 4.0
    widget = Show4D(data)
    widget.roi_mode = "square"
    widget.roi_center_x = 0
    widget.roi_center_y = 0
    widget.roi_radius = 1
    widget.roi_reduce = "mean"
    frame = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    # Mean of positions within radius 1 of (0,0) should be between 0 and 4
    assert frame.max() <= 4.0


def test_show4d_roi_reduce_max():
    data = np.zeros((4, 4, 8, 8), dtype=np.float32)
    data[1, 1] = 10.0
    data[2, 2] = 5.0
    widget = Show4D(data)
    widget.roi_mode = "circle"
    widget.roi_center_x = 1.5
    widget.roi_center_y = 1.5
    widget.roi_radius = 2
    widget.roi_reduce = "max"
    frame = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    assert frame.max() == 10.0


def test_show4d_roi_reduce_min():
    data = np.ones((4, 4, 8, 8), dtype=np.float32) * 5.0
    data[1, 1] = 1.0
    widget = Show4D(data)
    widget.roi_mode = "square"
    widget.roi_center_x = 1
    widget.roi_center_y = 1
    widget.roi_radius = 1
    widget.roi_reduce = "min"
    frame = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    assert frame.min() == 1.0


def test_show4d_roi_reduce_sum():
    data = np.ones((4, 4, 8, 8), dtype=np.float32)
    widget = Show4D(data)
    widget.roi_mode = "square"
    widget.roi_center_x = 1
    widget.roi_center_y = 1
    widget.roi_radius = 1
    widget.roi_reduce = "sum"
    frame = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    # Sum of N positions, each with value 1.0 → frame values should be > 1
    assert frame.max() > 1.0


def test_show4d_roi_reduce_change_updates_frame():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "circle"
    widget.roi_center_x = 2
    widget.roi_center_y = 2
    widget.roi_radius = 2
    widget.roi_reduce = "mean"
    frame_mean = bytes(widget.frame_bytes)
    widget.roi_reduce = "max"
    frame_max = bytes(widget.frame_bytes)
    assert frame_mean != frame_max
