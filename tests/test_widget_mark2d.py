import numpy as np
import pytest
import torch

from quantem.widget import Mark2D


def test_mark2d_numpy():
    """Create widget from numpy array."""
    data = np.random.rand(64, 64).astype(np.float32)
    widget = Mark2D(data)
    assert widget.width == 64
    assert widget.height == 64
    assert widget.n_images == 1
    assert len(widget.frame_bytes) == 1 * 64 * 64 * 4


def test_mark2d_torch():
    """Create widget from PyTorch tensor."""
    data = torch.rand(32, 32)
    widget = Mark2D(data)
    assert widget.width == 32
    assert widget.height == 32
    assert widget.n_images == 1
    assert len(widget.frame_bytes) == 1 * 32 * 32 * 4


def test_mark2d_scale():
    """Scale parameter is applied."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Mark2D(data, scale=2.0)
    assert widget.scale == 2.0


def test_mark2d_dot_size():
    """Dot size parameter is applied."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Mark2D(data, dot_size=20)
    assert widget.dot_size == 20


def test_mark2d_max_points():
    """Max points parameter is applied."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Mark2D(data, max_points=5)
    assert widget.max_points == 5


def test_mark2d_min_max():
    """Min/max values are computed correctly (per-image lists)."""
    data = np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32)
    widget = Mark2D(data)
    assert widget.img_min[0] == pytest.approx(0.0)
    assert widget.img_max[0] == pytest.approx(15.0)


def test_mark2d_constant_image():
    """Constant image doesn't cause division by zero."""
    data = np.ones((16, 16), dtype=np.float32) * 42.0
    widget = Mark2D(data)
    assert widget.img_min[0] == pytest.approx(42.0)
    assert widget.img_max[0] == pytest.approx(42.0)
    assert len(widget.frame_bytes) == 1 * 16 * 16 * 4


def test_mark2d_selected_points_init():
    """Selected points start empty."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Mark2D(data)
    assert widget.selected_points == []


def test_mark2d_set_image():
    """set_image replaces image and resets points."""
    data1 = np.random.rand(16, 16).astype(np.float32)
    data2 = np.random.rand(32, 32).astype(np.float32)
    widget = Mark2D(data1)
    widget.selected_points = [{"row": 5, "col": 5}]
    widget.set_image(data2)
    assert widget.width == 32
    assert widget.height == 32
    assert widget.selected_points == []


def test_mark2d_float32_bytes():
    """Raw float32 bytes are stored correctly in frame_bytes."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    widget = Mark2D(data)
    f32 = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    np.testing.assert_array_almost_equal(f32, [1.0, 2.0, 3.0, 4.0])


def test_mark2d_default_params():
    """Default constructor parameters."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Mark2D(data)
    assert widget.scale == 1.0
    assert widget.dot_size == 12
    assert widget.max_points == 10
    assert widget.ncols == 3
    assert widget.marker_border == 2
    assert widget.marker_opacity == 1.0
    assert widget.label_size == 0
    assert widget.label_color == ""


def test_mark2d_marker_styling():
    """Advanced marker styling parameters."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Mark2D(data, marker_border=0, marker_opacity=0.5,
                     label_size=14, label_color="yellow")
    assert widget.marker_border == 0
    assert widget.marker_opacity == 0.5
    assert widget.label_size == 14
    assert widget.label_color == "yellow"


def test_mark2d_large_image():
    """512x512 image works."""
    data = np.random.rand(512, 512).astype(np.float32)
    widget = Mark2D(data)
    assert widget.width == 512
    assert widget.height == 512
    assert len(widget.frame_bytes) == 1 * 512 * 512 * 4


def test_mark2d_non_square():
    """Non-square image works."""
    data = np.random.rand(64, 128).astype(np.float32)
    widget = Mark2D(data)
    assert widget.height == 64
    assert widget.width == 128


def test_mark2d_negative_values():
    """Image with negative values stores correct min/max."""
    data = np.array([[-10.0, 0.0], [5.0, 10.0]], dtype=np.float32)
    widget = Mark2D(data)
    assert widget.img_min[0] == pytest.approx(-10.0)
    assert widget.img_max[0] == pytest.approx(10.0)
    f32 = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    np.testing.assert_array_almost_equal(f32, [-10.0, 0.0, 5.0, 10.0])


def test_mark2d_set_image_torch():
    """set_image with torch tensor works."""
    data1 = np.random.rand(16, 16).astype(np.float32)
    data2 = torch.rand(24, 24)
    widget = Mark2D(data1)
    widget.set_image(data2)
    assert widget.width == 24
    assert widget.height == 24


def test_mark2d_frame_bytes_size():
    """frame_bytes has exactly n_images * height * width * 4 bytes."""
    data = np.random.rand(30, 50).astype(np.float32)
    widget = Mark2D(data)
    assert len(widget.frame_bytes) == 1 * 30 * 50 * 4


# ============================================================================
# Gallery mode tests
# ============================================================================


def test_mark2d_gallery_numpy_list():
    """Gallery from list of numpy arrays."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Mark2D(images)
    assert widget.n_images == 3
    assert widget.width == 16
    assert widget.height == 16
    assert len(widget.frame_bytes) == 3 * 16 * 16 * 4
    assert widget.selected_points == [[], [], []]
    assert len(widget.labels) == 3


def test_mark2d_gallery_torch_list():
    """Gallery from list of torch tensors."""
    images = [torch.rand(16, 16) for _ in range(3)]
    widget = Mark2D(images)
    assert widget.n_images == 3
    assert widget.selected_points == [[], [], []]


def test_mark2d_gallery_3d_array():
    """Gallery from 3D numpy array."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Mark2D(data)
    assert widget.n_images == 5
    assert widget.selected_points == [[], [], [], [], []]


def test_mark2d_gallery_ncols():
    """ncols parameter is applied."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(6)]
    widget = Mark2D(images, ncols=2)
    assert widget.ncols == 2


def test_mark2d_gallery_labels():
    """Custom labels are stored."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Mark2D(images, labels=["A", "B", "C"])
    assert widget.labels == ["A", "B", "C"]


def test_mark2d_gallery_default_labels():
    """Default labels are generated."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Mark2D(images)
    assert widget.labels == ["Image 1", "Image 2", "Image 3"]


def test_mark2d_gallery_per_image_minmax():
    """Per-image min/max are computed."""
    img1 = np.ones((8, 8), dtype=np.float32) * 5.0
    img2 = np.ones((8, 8), dtype=np.float32) * 10.0
    widget = Mark2D([img1, img2])
    assert widget.img_min[0] == pytest.approx(5.0)
    assert widget.img_min[1] == pytest.approx(10.0)
    assert widget.img_max[0] == pytest.approx(5.0)
    assert widget.img_max[1] == pytest.approx(10.0)


def test_mark2d_gallery_different_sizes():
    """Different-sized images are resized to max dims."""
    img1 = np.random.rand(16, 16).astype(np.float32)
    img2 = np.random.rand(32, 24).astype(np.float32)
    widget = Mark2D([img1, img2])
    assert widget.height == 32
    assert widget.width == 24


def test_mark2d_single_backward_compat():
    """Single image still returns flat point list."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Mark2D(data)
    assert widget.n_images == 1
    assert widget.selected_points == []  # flat, not [[]]


def test_mark2d_set_image_to_gallery():
    """set_image can switch from single to gallery."""
    data1 = np.random.rand(16, 16).astype(np.float32)
    widget = Mark2D(data1)
    assert widget.n_images == 1
    data2 = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget.set_image(data2)
    assert widget.n_images == 3
    assert widget.selected_points == [[], [], []]


def test_mark2d_gallery_selected_idx():
    """selected_idx starts at 0."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Mark2D(images)
    assert widget.selected_idx == 0


def test_mark2d_rejects_4d():
    """4D input raises ValueError."""
    data = np.random.rand(2, 3, 16, 16).astype(np.float32)
    with pytest.raises(ValueError):
        Mark2D(data)


def test_mark2d_gallery_frame_bytes_size():
    """frame_bytes size matches n_images * height * width * 4."""
    images = [np.random.rand(8, 12).astype(np.float32) for _ in range(4)]
    widget = Mark2D(images)
    assert len(widget.frame_bytes) == 4 * 8 * 12 * 4


def test_mark2d_gallery_set_image_resets():
    """set_image resets selected_points in gallery mode."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Mark2D(images)
    widget.selected_points = [[{"row": 1, "col": 2}], [], []]
    widget.set_image(images)
    assert widget.selected_points == [[], [], []]


def test_mark2d_repr():
    """repr includes class name."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Mark2D(data)
    assert "Mark2D" in repr(widget)


# ============================================================================
# Import/load points tests
# ============================================================================


def test_mark2d_points_tuples():
    """Initialize with list of tuples."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, points=[(5, 10), (15, 7)])
    assert len(w.selected_points) == 2
    assert w.selected_points[0]["row"] == 5
    assert w.selected_points[0]["col"] == 10
    assert "shape" in w.selected_points[0]
    assert "color" in w.selected_points[0]


def test_mark2d_points_dicts():
    """Initialize with list of dicts."""
    data = np.random.rand(16, 16).astype(np.float32)
    pts = [{"row": 10, "col": 12, "shape": "star", "color": "#ff0000"}]
    w = Mark2D(data, points=pts)
    assert w.selected_points[0]["shape"] == "star"
    assert w.selected_points[0]["color"] == "#ff0000"


def test_mark2d_points_numpy():
    """Initialize with numpy array of shape (N, 2)."""
    data = np.random.rand(16, 16).astype(np.float32)
    coords = np.array([[3, 4], [7, 8]], dtype=np.float64)
    w = Mark2D(data, points=coords)
    assert len(w.selected_points) == 2
    assert w.selected_points[0]["row"] == 3
    assert w.selected_points[0]["col"] == 4


def test_mark2d_points_gallery():
    """Initialize gallery with per-image points."""
    imgs = [np.random.rand(16, 16).astype(np.float32) for _ in range(2)]
    pts = [[(1, 2)], [(3, 4), (5, 6)]]
    w = Mark2D(imgs, points=pts)
    assert len(w.selected_points[0]) == 1
    assert len(w.selected_points[1]) == 2
    assert w.selected_points[1][0]["row"] == 3


def test_mark2d_points_auto_shape_color():
    """Points without shape/color get auto-assigned cycling values."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, points=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])
    shapes = [p["shape"] for p in w.selected_points]
    assert shapes[0] == "circle"
    assert shapes[1] == "triangle"
    assert shapes[5] == "circle"  # wraps after 5 shapes


def test_mark2d_points_none_default():
    """No points parameter gives empty list."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.selected_points == []


# ============================================================================
# ROI API tests
# ============================================================================


def test_mark2d_roi_list_default():
    """roi_list starts empty."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.roi_list == []


def test_mark2d_add_roi():
    """add_roi convenience method."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data)
    w.add_roi(32, 32, mode="circle", radius=15)
    assert len(w.roi_list) == 1
    assert w.roi_list[0]["row"] == 32
    assert w.roi_list[0]["mode"] == "circle"
    assert w.roi_list[0]["radius"] == 15


def test_mark2d_add_multiple_rois():
    """Multiple ROIs get unique IDs."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data)
    w.add_roi(10, 10)
    w.add_roi(20, 20, mode="square")
    w.add_roi(30, 30, mode="rectangle", rect_w=40, rect_h=20)
    assert len(w.roi_list) == 3
    ids = [r["id"] for r in w.roi_list]
    assert len(set(ids)) == 3  # unique IDs


def test_mark2d_clear_rois():
    """clear_rois removes all ROI overlays."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data)
    w.add_roi(10, 10)
    w.add_roi(20, 20)
    assert len(w.roi_list) == 2
    w.clear_rois()
    assert w.roi_list == []


def test_mark2d_clear_points():
    """clear_points removes all placed points."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, points=[(5, 5), (10, 10)])
    assert len(w.selected_points) == 2
    w.clear_points()
    assert w.selected_points == []


def test_mark2d_clear_points_gallery():
    """clear_points in gallery mode resets per-image lists."""
    imgs = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Mark2D(imgs, points=[[(1, 2)], [(3, 4)], []])
    assert len(w.selected_points[0]) == 1
    w.clear_points()
    assert w.selected_points == [[], [], []]


# ============================================================================
# State portability tests
# ============================================================================


def test_mark2d_marker_shape_default():
    """Default marker_shape is circle."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.marker_shape == "circle"


def test_mark2d_marker_shape_custom():
    """marker_shape can be set via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, marker_shape="star")
    assert w.marker_shape == "star"


def test_mark2d_marker_color_default():
    """Default marker_color is red."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.marker_color == "#f44336"


def test_mark2d_marker_color_custom():
    """marker_color can be set via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, marker_color="#00ff00")
    assert w.marker_color == "#00ff00"


def test_mark2d_snap_defaults():
    """Snap is disabled by default with radius 5."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.snap_enabled is False
    assert w.snap_radius == 5


def test_mark2d_snap_custom():
    """Snap can be enabled and configured via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, snap_enabled=True, snap_radius=10)
    assert w.snap_enabled is True
    assert w.snap_radius == 10


def test_mark2d_state_portability_all():
    """All portable state can be set at construction for state sharing."""
    data = np.random.rand(32, 32).astype(np.float32)
    w = Mark2D(
        data,
        points=[(5, 5), (10, 10)],
        marker_shape="diamond",
        marker_color="#9c27b0",
        snap_enabled=True,
        snap_radius=8,
    )
    w.add_roi(16, 16, mode="circle", radius=10)
    # Verify all state is accessible
    assert len(w.selected_points) == 2
    assert w.marker_shape == "diamond"
    assert w.marker_color == "#9c27b0"
    assert w.snap_enabled is True
    assert w.snap_radius == 8
    assert len(w.roi_list) == 1


def test_mark2d_colormap_default():
    """Colormap defaults to gray."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.colormap == "gray"


def test_mark2d_colormap_custom():
    """Colormap can be set via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, colormap="viridis")
    assert w.colormap == "viridis"
    w.colormap = "plasma"
    assert w.colormap == "plasma"


def test_mark2d_auto_contrast_default():
    """Auto-contrast defaults to True."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.auto_contrast is True


def test_mark2d_auto_contrast_custom():
    """Auto-contrast can be disabled."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, auto_contrast=False)
    assert w.auto_contrast is False


def test_mark2d_log_scale_default():
    """Log scale defaults to False."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.log_scale is False


def test_mark2d_log_scale_custom():
    """Log scale can be enabled."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, log_scale=True)
    assert w.log_scale is True


def test_mark2d_show_fft_default():
    """FFT display defaults to False."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.show_fft is False


def test_mark2d_show_fft_custom():
    """FFT display can be enabled."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, show_fft=True)
    assert w.show_fft is True


def test_mark2d_imaging_traits_portability():
    """All imaging traits (colormap, contrast, fft) round-trip correctly."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, colormap="inferno", auto_contrast=False, log_scale=True, show_fft=True)
    assert w.colormap == "inferno"
    assert w.auto_contrast is False
    assert w.log_scale is True
    assert w.show_fft is True


# ============================================================================
# Title and show_stats tests
# ============================================================================


def test_mark2d_title_default():
    """Default title is empty string."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.title == ""


def test_mark2d_title_custom():
    """Title can be set via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, title="HAADF-STEM")
    assert w.title == "HAADF-STEM"


def test_mark2d_title_in_repr():
    """Custom title appears in repr."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, title="My Image")
    assert "My Image" in repr(w)


def test_mark2d_title_default_repr():
    """Default repr shows Mark2D when title is empty."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert repr(w).startswith("Mark2D(")


def test_mark2d_show_stats_default():
    """show_stats defaults to True."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.show_stats is True


def test_mark2d_show_stats_custom():
    """show_stats can be disabled."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, show_stats=False)
    assert w.show_stats is False


def test_mark2d_points_as_array():
    """points_as_array returns (N, 2) array of [row, col]."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data, points=[(10, 20), (30, 40)])
    arr = w.points_as_array()
    assert arr.shape == (2, 2)
    assert arr[0, 0] == 10
    assert arr[0, 1] == 20
    assert arr[1, 0] == 30
    assert arr[1, 1] == 40


def test_mark2d_points_as_array_empty():
    """points_as_array returns empty (0, 2) array when no points."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data)
    arr = w.points_as_array()
    assert arr.shape == (0, 2)


def test_mark2d_points_as_array_gallery():
    """points_as_array returns list of arrays in gallery mode."""
    imgs = [np.random.rand(32, 32).astype(np.float32) for _ in range(3)]
    pts = [[(5, 10)], [(15, 20), (25, 30)], []]
    w = Mark2D(imgs, points=pts)
    result = w.points_as_array()
    assert len(result) == 3
    assert result[0].shape == (1, 2)
    assert result[1].shape == (2, 2)
    assert result[2].shape == (0, 2)


def test_mark2d_points_as_dict():
    """points_as_dict returns list of {row, col} dicts."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data, points=[(10, 20), (30, 40)])
    result = w.points_as_dict()
    assert len(result) == 2
    assert result[0] == {"row": 10, "col": 20}
    assert result[1] == {"row": 30, "col": 40}


def test_mark2d_points_as_dict_gallery():
    """points_as_dict returns list of lists in gallery mode."""
    imgs = [np.random.rand(32, 32).astype(np.float32) for _ in range(2)]
    pts = [[(5, 10)], []]
    w = Mark2D(imgs, points=pts)
    result = w.points_as_dict()
    assert len(result) == 2
    assert result[0] == [{"row": 5, "col": 10}]
    assert result[1] == []


def test_mark2d_set_profile():
    """set_profile sets profile_line trait."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data)
    w.set_profile((10, 20), (50, 60))
    assert len(w.profile_line) == 2
    assert w.profile_line[0] == {"row": 10.0, "col": 20.0}
    assert w.profile_line[1] == {"row": 50.0, "col": 60.0}


def test_mark2d_clear_profile():
    """clear_profile empties profile_line."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data)
    w.set_profile((10, 20), (50, 60))
    w.clear_profile()
    assert w.profile_line == []


def test_mark2d_profile_property():
    """profile property returns list of (row, col) tuples."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data)
    assert w.profile == []
    w.set_profile((10, 20), (50, 60))
    assert w.profile == [(10.0, 20.0), (50.0, 60.0)]


def test_mark2d_profile_values():
    """profile_values samples along the line."""
    data = np.ones((64, 64), dtype=np.float32) * 5.0
    w = Mark2D(data)
    w.set_profile((0, 0), (63, 63))
    vals = w.profile_values
    assert vals is not None
    assert len(vals) > 2
    assert np.allclose(vals, 5.0)


def test_mark2d_profile_values_none():
    """profile_values returns None when no profile."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data)
    assert w.profile_values is None


def test_mark2d_profile_distance():
    """profile_distance returns distance in pixels or calibrated units."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data)
    assert w.profile_distance is None
    w.set_profile((0, 0), (30, 40))
    assert w.profile_distance == pytest.approx(50.0)


def test_mark2d_profile_distance_calibrated():
    """profile_distance uses pixel_size_angstrom when set."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data, pixel_size_angstrom=2.0)
    w.set_profile((0, 0), (30, 40))
    assert w.profile_distance == pytest.approx(100.0)


# ============================================================================
# New trait tests (image_width_px, show_controls, percentile, stats)
# ============================================================================


def test_mark2d_image_width_px_default():
    """image_width_px defaults to 0 (auto-size)."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.image_width_px == 0


def test_mark2d_image_width_px_custom():
    """image_width_px can be set via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, image_width_px=400)
    assert w.image_width_px == 400


def test_mark2d_show_controls_default():
    """show_controls defaults to True."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.show_controls is True


def test_mark2d_show_controls_custom():
    """show_controls can be disabled."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, show_controls=False)
    assert w.show_controls is False


def test_mark2d_percentile_defaults():
    """Percentile defaults are 2.0 and 98.0."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.percentile_low == pytest.approx(2.0)
    assert w.percentile_high == pytest.approx(98.0)


def test_mark2d_percentile_custom():
    """Percentile can be set via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, percentile_low=5.0, percentile_high=95.0)
    assert w.percentile_low == pytest.approx(5.0)
    assert w.percentile_high == pytest.approx(95.0)


def test_mark2d_stats_computed():
    """Per-image stats are computed in _set_data."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    w = Mark2D(data)
    assert len(w.stats_mean) == 1
    assert len(w.stats_min) == 1
    assert len(w.stats_max) == 1
    assert len(w.stats_std) == 1
    assert w.stats_mean[0] == pytest.approx(2.5)
    assert w.stats_min[0] == pytest.approx(1.0)
    assert w.stats_max[0] == pytest.approx(4.0)
    assert w.stats_std[0] == pytest.approx(np.array([1.0, 2.0, 3.0, 4.0]).std())


def test_mark2d_stats_gallery():
    """Per-image stats for gallery mode."""
    img1 = np.ones((8, 8), dtype=np.float32) * 5.0
    img2 = np.ones((8, 8), dtype=np.float32) * 10.0
    w = Mark2D([img1, img2])
    assert len(w.stats_mean) == 2
    assert w.stats_mean[0] == pytest.approx(5.0)
    assert w.stats_mean[1] == pytest.approx(10.0)
    assert w.stats_std[0] == pytest.approx(0.0)
    assert w.stats_std[1] == pytest.approx(0.0)


def test_mark2d_stats_updated_on_set_image():
    """Stats are recomputed when set_image is called."""
    data1 = np.ones((8, 8), dtype=np.float32) * 5.0
    w = Mark2D(data1)
    assert w.stats_mean[0] == pytest.approx(5.0)
    data2 = np.ones((8, 8), dtype=np.float32) * 20.0
    w.set_image(data2)
    assert w.stats_mean[0] == pytest.approx(20.0)


def test_mark2d_dataset_reextraction_on_set_image():
    """set_image with a Dataset re-extracts metadata."""
    class FakeDataset:
        def __init__(self, arr, name, sampling, units):
            self.array = arr
            self.name = name
            self.sampling = sampling
            self.units = units

    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    assert w.title == ""
    assert w.pixel_size_angstrom == 0.0

    ds = FakeDataset(np.random.rand(16, 16).astype(np.float32),
                     "HAADF", [0.5, 0.5], ["Å", "Å"])
    w.set_image(ds)
    assert w.title == "HAADF"
    assert w.pixel_size_angstrom == pytest.approx(0.5)


def test_mark2d_explicit_overrides_dataset():
    """Explicit title/pixel_size override Dataset metadata."""
    class FakeDataset:
        def __init__(self, arr, name, sampling, units):
            self.array = arr
            self.name = name
            self.sampling = sampling
            self.units = units

    ds = FakeDataset(np.random.rand(16, 16).astype(np.float32),
                     "HAADF", [0.5, 0.5], ["Å", "Å"])
    w = Mark2D(ds, title="Custom", pixel_size_angstrom=1.0)
    assert w.title == "Custom"
    assert w.pixel_size_angstrom == pytest.approx(1.0)


def test_mark2d_repr_gallery_idx():
    """Gallery repr includes idx=N."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Mark2D(images)
    r = repr(w)
    assert "idx=0" in r


def test_mark2d_summary_roi_area():
    """summary() includes calibrated ROI area."""
    import io
    import sys
    data = np.random.rand(64, 64).astype(np.float32)
    w = Mark2D(data, pixel_size_angstrom=1.0)
    w.add_roi(32, 32, mode="circle", radius=10)
    # Capture summary output
    captured = io.StringIO()
    sys.stdout = captured
    w.summary()
    sys.stdout = sys.__stdout__
    output = captured.getvalue()
    assert "area=" in output


# ============================================================================
# State persistence tests (state_dict / load_state_dict / state param)
# ============================================================================


def test_mark2d_state_dict_keys():
    """state_dict returns all expected keys."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    sd = w.state_dict()
    expected = {
        "selected_points", "roi_list", "profile_line", "selected_idx",
        "marker_shape", "marker_color", "dot_size", "max_points",
        "marker_border", "marker_opacity", "label_size", "label_color",
        "snap_enabled", "snap_radius", "colormap", "auto_contrast",
        "log_scale", "show_fft", "show_stats", "show_controls",
        "percentile_low", "percentile_high", "title",
        "pixel_size_angstrom", "scale", "image_width_px",
    }
    assert set(sd.keys()) == expected


def test_mark2d_state_dict_roundtrip():
    """state_dict captures current state and restores via state param."""
    data = np.random.rand(32, 32).astype(np.float32)
    w = Mark2D(data, colormap="viridis", auto_contrast=False, log_scale=True,
                marker_shape="star", marker_color="#00ff00", snap_enabled=True,
                snap_radius=12, title="Test", pixel_size_angstrom=2.5,
                percentile_low=5.0, percentile_high=90.0, show_controls=False)
    w.add_roi(16, 16, mode="circle", radius=8)
    w.set_profile((0, 0), (31, 31))

    sd = w.state_dict()

    w2 = Mark2D(data, state=sd)
    assert w2.colormap == "viridis"
    assert w2.auto_contrast is False
    assert w2.log_scale is True
    assert w2.marker_shape == "star"
    assert w2.marker_color == "#00ff00"
    assert w2.snap_enabled is True
    assert w2.snap_radius == 12
    assert w2.title == "Test"
    assert w2.pixel_size_angstrom == pytest.approx(2.5)
    assert w2.percentile_low == pytest.approx(5.0)
    assert w2.percentile_high == pytest.approx(90.0)
    assert w2.show_controls is False
    assert len(w2.roi_list) == 1
    assert len(w2.profile_line) == 2


def test_mark2d_state_dict_with_points():
    """state_dict preserves selected_points."""
    data = np.random.rand(32, 32).astype(np.float32)
    w = Mark2D(data, points=[(5, 10), (15, 20)])
    sd = w.state_dict()

    w2 = Mark2D(data, state=sd)
    assert len(w2.selected_points) == 2
    assert w2.selected_points[0]["row"] == 5
    assert w2.selected_points[0]["col"] == 10


def test_mark2d_load_state_dict():
    """load_state_dict restores state on existing widget."""
    data = np.random.rand(16, 16).astype(np.float32)
    w1 = Mark2D(data, colormap="plasma", dot_size=20)
    sd = w1.state_dict()

    w2 = Mark2D(data)
    assert w2.colormap == "gray"
    assert w2.dot_size == 12
    w2.load_state_dict(sd)
    assert w2.colormap == "plasma"
    assert w2.dot_size == 20


def test_mark2d_load_state_dict_ignores_unknown():
    """load_state_dict silently skips unknown keys."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data)
    w.load_state_dict({"colormap": "inferno", "nonexistent_key": 42})
    assert w.colormap == "inferno"


def test_mark2d_state_param_overrides_defaults():
    """state param overrides default trait values but explicit params win."""
    data = np.random.rand(16, 16).astype(np.float32)
    sd = {"colormap": "viridis", "title": "From State", "dot_size": 15}
    # state is applied last, so it overrides defaults and explicit params
    w = Mark2D(data, state=sd)
    assert w.colormap == "viridis"
    assert w.title == "From State"
    assert w.dot_size == 15


def test_mark2d_save_and_load_file(tmp_path):
    """save() writes JSON, state param loads from file path."""
    data = np.random.rand(32, 32).astype(np.float32)
    w = Mark2D(data, colormap="viridis", marker_shape="star",
                title="Saved", pixel_size_angstrom=1.5)
    w.add_roi(16, 16, mode="circle", radius=8)

    path = str(tmp_path / "state.json")
    w.save(path)

    w2 = Mark2D(data, state=path)
    assert w2.colormap == "viridis"
    assert w2.marker_shape == "star"
    assert w2.title == "Saved"
    assert w2.pixel_size_angstrom == pytest.approx(1.5)
    assert len(w2.roi_list) == 1


def test_mark2d_save_json_readable(tmp_path):
    """Saved file is valid, human-readable JSON."""
    import json
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, points=[(5, 10)], colormap="plasma")
    path = tmp_path / "state.json"
    w.save(str(path))

    content = json.loads(path.read_text())
    assert content["colormap"] == "plasma"
    assert len(content["selected_points"]) == 1
    assert content["selected_points"][0]["row"] == 5


def test_mark2d_save_load_roundtrip_points(tmp_path):
    """Points survive save/load roundtrip via JSON file."""
    data = np.random.rand(32, 32).astype(np.float32)
    w = Mark2D(data, points=[(1, 2), (10, 20), (30, 5)])
    w.set_profile((0, 0), (31, 31))

    path = str(tmp_path / "pts.json")
    w.save(path)

    w2 = Mark2D(data, state=path)
    assert len(w2.selected_points) == 3
    assert w2.selected_points[0]["row"] == 1
    assert w2.selected_points[0]["col"] == 2
    assert w2.selected_points[2]["row"] == 30
    assert len(w2.profile_line) == 2


def test_mark2d_state_pathlib_path(tmp_path):
    """state param accepts pathlib.Path as well as str."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Mark2D(data, colormap="inferno")
    path = tmp_path / "state.json"
    w.save(str(path))

    w2 = Mark2D(data, state=path)  # pathlib.Path, not str
    assert w2.colormap == "inferno"
