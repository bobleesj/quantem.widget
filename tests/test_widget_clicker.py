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
    widget.selected_points = [{"x": 5, "y": 5}]
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
    widget.selected_points = [[{"x": 1, "y": 2}], [], []]
    widget.set_image(images)
    assert widget.selected_points == [[], [], []]


def test_clicker_repr():
    """repr includes class name."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Clicker(data)
    assert "Clicker" in repr(widget)
