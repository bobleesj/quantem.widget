import numpy as np
import pytest
import torch

from quantem.widget import Show2D


def test_show2d_single_numpy():
    """Single image from numpy array."""
    data = np.random.rand(32, 32).astype(np.float32)
    widget = Show2D(data)
    assert widget.n_images == 1
    assert widget.height == 32
    assert widget.width == 32
    assert len(widget.frame_bytes) > 0


def test_show2d_single_torch():
    """Single image from PyTorch tensor."""
    data = torch.rand(32, 32)
    widget = Show2D(data)
    assert widget.n_images == 1
    assert widget.height == 32
    assert widget.width == 32


def test_show2d_multiple_numpy():
    """Gallery mode from list of numpy arrays."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Show2D(images, labels=["A", "B", "C"])
    assert widget.n_images == 3
    assert widget.labels == ["A", "B", "C"]


def test_show2d_multiple_torch():
    """Gallery mode from list of PyTorch tensors."""
    images = [torch.rand(16, 16) for _ in range(3)]
    widget = Show2D(images)
    assert widget.n_images == 3


def test_show2d_3d_array():
    """3D array treated as multiple images."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.n_images == 5


def test_show2d_stats():
    """Statistics are computed correctly."""
    data = np.ones((16, 16), dtype=np.float32) * 42.0
    widget = Show2D(data)
    assert widget.stats_mean[0] == pytest.approx(42.0)
    assert widget.stats_min[0] == pytest.approx(42.0)
    assert widget.stats_max[0] == pytest.approx(42.0)
    assert widget.stats_std[0] == pytest.approx(0.0)


def test_show2d_colormap():
    """Colormap option is applied."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, cmap="viridis")
    assert widget.cmap == "viridis"


def test_show2d_different_sizes():
    """Gallery with different sized images resizes to largest."""
    images = [
        np.random.rand(16, 16).astype(np.float32),
        np.random.rand(32, 32).astype(np.float32),
    ]
    widget = Show2D(images)
    assert widget.height == 32
    assert widget.width == 32


def test_show2d_display_options():
    """Log scale and auto contrast options are accepted."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, log_scale=True, auto_contrast=True)
    assert widget.log_scale is True
    assert widget.auto_contrast is True


def test_show2d_title():
    """Title parameter is stored."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, title="Test Title")
    assert widget.title == "Test Title"


def test_show2d_default_labels():
    """Labels default to 'Image 1', 'Image 2', etc."""
    data = np.random.rand(3, 16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.labels == ["Image 1", "Image 2", "Image 3"]


def test_show2d_scale_bar():
    """Scale bar parameters are stored."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, pixel_size_angstrom=1.5, scale_bar_visible=False)
    assert widget.pixel_size_angstrom == pytest.approx(1.5)
    assert widget.scale_bar_visible is False


def test_show2d_ncols():
    """ncols parameter is stored."""
    data = np.random.rand(6, 16, 16).astype(np.float32)
    widget = Show2D(data, ncols=2)
    assert widget.ncols == 2


def test_show2d_image_width_px():
    """image_width_px parameter is stored."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, image_width_px=500)
    assert widget.image_width_px == 500


def test_show2d_show_controls():
    """show_controls can be toggled."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, show_controls=False)
    assert widget.show_controls is False


def test_show2d_show_stats():
    """show_stats can be toggled."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, show_stats=False)
    assert widget.show_stats is False


def test_show2d_constant_image_stats():
    """Constant image doesn't crash stats computation."""
    data = np.zeros((16, 16), dtype=np.float32)
    widget = Show2D(data)
    assert widget.stats_mean[0] == pytest.approx(0.0)
    assert widget.stats_std[0] == pytest.approx(0.0)


def test_show2d_single_image_is_3d_internally():
    """2D input is wrapped to 3D (1, H, W) internally."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.n_images == 1
    # frame_bytes contains exactly 1 * 16 * 16 float32 values
    assert len(widget.frame_bytes) == 1 * 16 * 16 * 4


def test_show2d_large_gallery():
    """Large gallery (20 images) works."""
    data = np.random.rand(20, 8, 8).astype(np.float32)
    widget = Show2D(data)
    assert widget.n_images == 20
    assert len(widget.stats_mean) == 20
    assert len(widget.stats_min) == 20
    assert len(widget.stats_max) == 20
    assert len(widget.stats_std) == 20


def test_show2d_gallery_stats_per_image():
    """Stats are computed per image in gallery."""
    img1 = np.ones((8, 8), dtype=np.float32) * 10.0
    img2 = np.ones((8, 8), dtype=np.float32) * 20.0
    widget = Show2D([img1, img2])
    assert widget.stats_mean[0] == pytest.approx(10.0)
    assert widget.stats_mean[1] == pytest.approx(20.0)
