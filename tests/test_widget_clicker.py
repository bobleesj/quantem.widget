import numpy as np
import pytest
import torch

from quantem.widget import Clicker


def test_clicker_numpy():
    """Create widget from numpy array."""
    data = np.random.rand(64, 64).astype(np.float32)
    widget = Clicker(data)
    assert widget.width == 64
    assert widget.height == 64
    assert widget.n_images == 1
    assert len(widget.frame_bytes) == 1 * 64 * 64 * 4


def test_clicker_torch():
    """Create widget from PyTorch tensor."""
    data = torch.rand(32, 32)
    widget = Clicker(data)
    assert widget.width == 32
    assert widget.height == 32
    assert widget.n_images == 1
    assert len(widget.frame_bytes) == 1 * 32 * 32 * 4


def test_clicker_scale():
    """Scale parameter is applied."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Clicker(data, scale=2.0)
    assert widget.scale == 2.0


def test_clicker_dot_size():
    """Dot size parameter is applied."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Clicker(data, dot_size=20)
    assert widget.dot_size == 20


def test_clicker_max_points():
    """Max points parameter is applied."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Clicker(data, max_points=5)
    assert widget.max_points == 5


def test_clicker_min_max():
    """Min/max values are computed correctly (per-image lists)."""
    data = np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32)
    widget = Clicker(data)
    assert widget.img_min[0] == pytest.approx(0.0)
    assert widget.img_max[0] == pytest.approx(15.0)


def test_clicker_constant_image():
    """Constant image doesn't cause division by zero."""
    data = np.ones((16, 16), dtype=np.float32) * 42.0
    widget = Clicker(data)
    assert widget.img_min[0] == pytest.approx(42.0)
    assert widget.img_max[0] == pytest.approx(42.0)
    assert len(widget.frame_bytes) == 1 * 16 * 16 * 4


def test_clicker_selected_points_init():
    """Selected points start empty."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Clicker(data)
    assert widget.selected_points == []


def test_clicker_set_image():
    """set_image replaces image and resets points."""
    data1 = np.random.rand(16, 16).astype(np.float32)
    data2 = np.random.rand(32, 32).astype(np.float32)
    widget = Clicker(data1)
    widget.selected_points = [{"row": 5, "col": 5}]
    widget.set_image(data2)
    assert widget.width == 32
    assert widget.height == 32
    assert widget.selected_points == []


def test_clicker_float32_bytes():
    """Raw float32 bytes are stored correctly in frame_bytes."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    widget = Clicker(data)
    f32 = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    np.testing.assert_array_almost_equal(f32, [1.0, 2.0, 3.0, 4.0])


def test_clicker_default_params():
    """Default constructor parameters."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Clicker(data)
    assert widget.scale == 1.0
    assert widget.dot_size == 12
    assert widget.max_points == 10
    assert widget.ncols == 3
    assert widget.marker_border == 2
    assert widget.marker_opacity == 1.0
    assert widget.label_size == 0
    assert widget.label_color == ""


def test_clicker_marker_styling():
    """Advanced marker styling parameters."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Clicker(data, marker_border=0, marker_opacity=0.5,
                     label_size=14, label_color="yellow")
    assert widget.marker_border == 0
    assert widget.marker_opacity == 0.5
    assert widget.label_size == 14
    assert widget.label_color == "yellow"


def test_clicker_large_image():
    """512x512 image works."""
    data = np.random.rand(512, 512).astype(np.float32)
    widget = Clicker(data)
    assert widget.width == 512
    assert widget.height == 512
    assert len(widget.frame_bytes) == 1 * 512 * 512 * 4


def test_clicker_non_square():
    """Non-square image works."""
    data = np.random.rand(64, 128).astype(np.float32)
    widget = Clicker(data)
    assert widget.height == 64
    assert widget.width == 128


def test_clicker_negative_values():
    """Image with negative values stores correct min/max."""
    data = np.array([[-10.0, 0.0], [5.0, 10.0]], dtype=np.float32)
    widget = Clicker(data)
    assert widget.img_min[0] == pytest.approx(-10.0)
    assert widget.img_max[0] == pytest.approx(10.0)
    f32 = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    np.testing.assert_array_almost_equal(f32, [-10.0, 0.0, 5.0, 10.0])


def test_clicker_set_image_torch():
    """set_image with torch tensor works."""
    data1 = np.random.rand(16, 16).astype(np.float32)
    data2 = torch.rand(24, 24)
    widget = Clicker(data1)
    widget.set_image(data2)
    assert widget.width == 24
    assert widget.height == 24


def test_clicker_frame_bytes_size():
    """frame_bytes has exactly n_images * height * width * 4 bytes."""
    data = np.random.rand(30, 50).astype(np.float32)
    widget = Clicker(data)
    assert len(widget.frame_bytes) == 1 * 30 * 50 * 4


# ============================================================================
# Gallery mode tests
# ============================================================================


def test_clicker_gallery_numpy_list():
    """Gallery from list of numpy arrays."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Clicker(images)
    assert widget.n_images == 3
    assert widget.width == 16
    assert widget.height == 16
    assert len(widget.frame_bytes) == 3 * 16 * 16 * 4
    assert widget.selected_points == [[], [], []]
    assert len(widget.labels) == 3


def test_clicker_gallery_torch_list():
    """Gallery from list of torch tensors."""
    images = [torch.rand(16, 16) for _ in range(3)]
    widget = Clicker(images)
    assert widget.n_images == 3
    assert widget.selected_points == [[], [], []]


def test_clicker_gallery_3d_array():
    """Gallery from 3D numpy array."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Clicker(data)
    assert widget.n_images == 5
    assert widget.selected_points == [[], [], [], [], []]


def test_clicker_gallery_ncols():
    """ncols parameter is applied."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(6)]
    widget = Clicker(images, ncols=2)
    assert widget.ncols == 2


def test_clicker_gallery_labels():
    """Custom labels are stored."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Clicker(images, labels=["A", "B", "C"])
    assert widget.labels == ["A", "B", "C"]


def test_clicker_gallery_default_labels():
    """Default labels are generated."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Clicker(images)
    assert widget.labels == ["Image 1", "Image 2", "Image 3"]


def test_clicker_gallery_per_image_minmax():
    """Per-image min/max are computed."""
    img1 = np.ones((8, 8), dtype=np.float32) * 5.0
    img2 = np.ones((8, 8), dtype=np.float32) * 10.0
    widget = Clicker([img1, img2])
    assert widget.img_min[0] == pytest.approx(5.0)
    assert widget.img_min[1] == pytest.approx(10.0)
    assert widget.img_max[0] == pytest.approx(5.0)
    assert widget.img_max[1] == pytest.approx(10.0)


def test_clicker_gallery_different_sizes():
    """Different-sized images are resized to max dims."""
    img1 = np.random.rand(16, 16).astype(np.float32)
    img2 = np.random.rand(32, 24).astype(np.float32)
    widget = Clicker([img1, img2])
    assert widget.height == 32
    assert widget.width == 24


def test_clicker_single_backward_compat():
    """Single image still returns flat point list."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Clicker(data)
    assert widget.n_images == 1
    assert widget.selected_points == []  # flat, not [[]]


def test_clicker_set_image_to_gallery():
    """set_image can switch from single to gallery."""
    data1 = np.random.rand(16, 16).astype(np.float32)
    widget = Clicker(data1)
    assert widget.n_images == 1
    data2 = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget.set_image(data2)
    assert widget.n_images == 3
    assert widget.selected_points == [[], [], []]


def test_clicker_gallery_selected_idx():
    """selected_idx starts at 0."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Clicker(images)
    assert widget.selected_idx == 0


def test_clicker_rejects_4d():
    """4D input raises ValueError."""
    data = np.random.rand(2, 3, 16, 16).astype(np.float32)
    with pytest.raises(ValueError):
        Clicker(data)


def test_clicker_gallery_frame_bytes_size():
    """frame_bytes size matches n_images * height * width * 4."""
    images = [np.random.rand(8, 12).astype(np.float32) for _ in range(4)]
    widget = Clicker(images)
    assert len(widget.frame_bytes) == 4 * 8 * 12 * 4


def test_clicker_gallery_set_image_resets():
    """set_image resets selected_points in gallery mode."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Clicker(images)
    widget.selected_points = [[{"row": 1, "col": 2}], [], []]
    widget.set_image(images)
    assert widget.selected_points == [[], [], []]


def test_clicker_repr():
    """repr includes class name."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Clicker(data)
    assert "Clicker" in repr(widget)


# ============================================================================
# Import/load points tests
# ============================================================================


def test_clicker_points_tuples():
    """Initialize with list of tuples."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, points=[(5, 10), (15, 7)])
    assert len(w.selected_points) == 2
    assert w.selected_points[0]["row"] == 5
    assert w.selected_points[0]["col"] == 10
    assert "shape" in w.selected_points[0]
    assert "color" in w.selected_points[0]


def test_clicker_points_dicts():
    """Initialize with list of dicts."""
    data = np.random.rand(16, 16).astype(np.float32)
    pts = [{"row": 10, "col": 12, "shape": "star", "color": "#ff0000"}]
    w = Clicker(data, points=pts)
    assert w.selected_points[0]["shape"] == "star"
    assert w.selected_points[0]["color"] == "#ff0000"


def test_clicker_points_numpy():
    """Initialize with numpy array of shape (N, 2)."""
    data = np.random.rand(16, 16).astype(np.float32)
    coords = np.array([[3, 4], [7, 8]], dtype=np.float64)
    w = Clicker(data, points=coords)
    assert len(w.selected_points) == 2
    assert w.selected_points[0]["row"] == 3
    assert w.selected_points[0]["col"] == 4


def test_clicker_points_gallery():
    """Initialize gallery with per-image points."""
    imgs = [np.random.rand(16, 16).astype(np.float32) for _ in range(2)]
    pts = [[(1, 2)], [(3, 4), (5, 6)]]
    w = Clicker(imgs, points=pts)
    assert len(w.selected_points[0]) == 1
    assert len(w.selected_points[1]) == 2
    assert w.selected_points[1][0]["row"] == 3


def test_clicker_points_auto_shape_color():
    """Points without shape/color get auto-assigned cycling values."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, points=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])
    shapes = [p["shape"] for p in w.selected_points]
    assert shapes[0] == "circle"
    assert shapes[1] == "triangle"
    assert shapes[5] == "circle"  # wraps after 5 shapes


def test_clicker_points_none_default():
    """No points parameter gives empty list."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.selected_points == []


# ============================================================================
# ROI API tests
# ============================================================================


def test_clicker_roi_list_default():
    """roi_list starts empty."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.roi_list == []


def test_clicker_add_roi():
    """add_roi convenience method."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Clicker(data)
    w.add_roi(32, 32, mode="circle", radius=15)
    assert len(w.roi_list) == 1
    assert w.roi_list[0]["row"] == 32
    assert w.roi_list[0]["mode"] == "circle"
    assert w.roi_list[0]["radius"] == 15


def test_clicker_add_multiple_rois():
    """Multiple ROIs get unique IDs."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Clicker(data)
    w.add_roi(10, 10)
    w.add_roi(20, 20, mode="square")
    w.add_roi(30, 30, mode="rectangle", rect_w=40, rect_h=20)
    assert len(w.roi_list) == 3
    ids = [r["id"] for r in w.roi_list]
    assert len(set(ids)) == 3  # unique IDs


def test_clicker_clear_rois():
    """clear_rois removes all ROI overlays."""
    data = np.random.rand(64, 64).astype(np.float32)
    w = Clicker(data)
    w.add_roi(10, 10)
    w.add_roi(20, 20)
    assert len(w.roi_list) == 2
    w.clear_rois()
    assert w.roi_list == []


def test_clicker_clear_points():
    """clear_points removes all placed points."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, points=[(5, 5), (10, 10)])
    assert len(w.selected_points) == 2
    w.clear_points()
    assert w.selected_points == []


def test_clicker_clear_points_gallery():
    """clear_points in gallery mode resets per-image lists."""
    imgs = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Clicker(imgs, points=[[(1, 2)], [(3, 4)], []])
    assert len(w.selected_points[0]) == 1
    w.clear_points()
    assert w.selected_points == [[], [], []]


# ============================================================================
# State portability tests
# ============================================================================


def test_clicker_marker_shape_default():
    """Default marker_shape is circle."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.marker_shape == "circle"


def test_clicker_marker_shape_custom():
    """marker_shape can be set via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, marker_shape="star")
    assert w.marker_shape == "star"


def test_clicker_marker_color_default():
    """Default marker_color is red."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.marker_color == "#f44336"


def test_clicker_marker_color_custom():
    """marker_color can be set via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, marker_color="#00ff00")
    assert w.marker_color == "#00ff00"


def test_clicker_snap_defaults():
    """Snap is disabled by default with radius 5."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.snap_enabled is False
    assert w.snap_radius == 5


def test_clicker_snap_custom():
    """Snap can be enabled and configured via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, snap_enabled=True, snap_radius=10)
    assert w.snap_enabled is True
    assert w.snap_radius == 10


def test_clicker_state_portability_all():
    """All portable state can be set at construction for state sharing."""
    data = np.random.rand(32, 32).astype(np.float32)
    w = Clicker(
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


def test_clicker_colormap_default():
    """Colormap defaults to gray."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.colormap == "gray"


def test_clicker_colormap_custom():
    """Colormap can be set via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, colormap="viridis")
    assert w.colormap == "viridis"
    w.colormap = "plasma"
    assert w.colormap == "plasma"


def test_clicker_auto_contrast_default():
    """Auto-contrast defaults to True."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.auto_contrast is True


def test_clicker_auto_contrast_custom():
    """Auto-contrast can be disabled."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, auto_contrast=False)
    assert w.auto_contrast is False


def test_clicker_log_scale_default():
    """Log scale defaults to False."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.log_scale is False


def test_clicker_log_scale_custom():
    """Log scale can be enabled."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, log_scale=True)
    assert w.log_scale is True


def test_clicker_show_fft_default():
    """FFT display defaults to False."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.show_fft is False


def test_clicker_show_fft_custom():
    """FFT display can be enabled."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, show_fft=True)
    assert w.show_fft is True


def test_clicker_imaging_traits_portability():
    """All imaging traits (colormap, contrast, fft) round-trip correctly."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, colormap="inferno", auto_contrast=False, log_scale=True, show_fft=True)
    assert w.colormap == "inferno"
    assert w.auto_contrast is False
    assert w.log_scale is True
    assert w.show_fft is True


# ============================================================================
# Title and show_stats tests
# ============================================================================


def test_clicker_title_default():
    """Default title is empty string."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.title == ""


def test_clicker_title_custom():
    """Title can be set via constructor."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, title="HAADF-STEM")
    assert w.title == "HAADF-STEM"


def test_clicker_title_in_repr():
    """Custom title appears in repr."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, title="My Image")
    assert "My Image" in repr(w)


def test_clicker_title_default_repr():
    """Default repr shows Clicker when title is empty."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert repr(w).startswith("Clicker(")


def test_clicker_show_stats_default():
    """show_stats defaults to True."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data)
    assert w.show_stats is True


def test_clicker_show_stats_custom():
    """show_stats can be disabled."""
    data = np.random.rand(16, 16).astype(np.float32)
    w = Clicker(data, show_stats=False)
    assert w.show_stats is False
