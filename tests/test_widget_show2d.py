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


def test_show2d_roi_stats():
    """ROI computes mean/min/max/std for selected region."""
    data = np.ones((32, 32), dtype=np.float32) * 5.0
    widget = Show2D(data)
    widget.roi_list = [{"shape": "rectangle", "row": 16, "col": 16, "width": 10, "height": 10}]
    widget.roi_selected_idx = 0
    widget.roi_active = True
    assert widget.roi_stats["mean"] == pytest.approx(5.0)
    assert widget.roi_stats["min"] == pytest.approx(5.0)
    assert widget.roi_stats["max"] == pytest.approx(5.0)
    assert widget.roi_stats["std"] == pytest.approx(0.0)


def test_show2d_roi_shapes():
    """ROI supports circle, square, rectangle, annular shapes."""
    data = np.ones((32, 32), dtype=np.float32) * 3.0
    widget = Show2D(data)
    widget.roi_active = True
    # Circle
    widget.roi_list = [{"shape": "circle", "row": 16, "col": 16, "radius": 5}]
    widget.roi_selected_idx = 0
    assert widget.roi_stats["mean"] == pytest.approx(3.0)
    # Square
    widget.roi_list = [{"shape": "square", "row": 16, "col": 16, "radius": 5}]
    assert widget.roi_stats["mean"] == pytest.approx(3.0)
    # Annular
    widget.roi_list = [{"shape": "annular", "row": 16, "col": 16, "radius": 8, "radius_inner": 3}]
    assert widget.roi_stats["mean"] == pytest.approx(3.0)


def test_show2d_roi_inactive():
    """ROI stats not computed when inactive."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.roi_stats == {}


# ── State Protocol ────────────────────────────────────────────────────────


def test_show2d_state_dict_roundtrip():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, cmap="viridis", log_scale=True, auto_contrast=True,
               title="Test", pixel_size_angstrom=2.5, show_fft=True)
    w.roi_active = True
    w.roi_list = [{"shape": "circle", "row": 10, "col": 15, "radius": 5}]
    w.roi_selected_idx = 0
    sd = w.state_dict()
    w2 = Show2D(data, state=sd)
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.auto_contrast is True
    assert w2.title == "Test"
    assert w2.pixel_size_angstrom == pytest.approx(2.5)
    assert w2.show_fft is True
    assert w2.roi_active is True
    assert len(w2.roi_list) == 1
    assert w2.roi_list[0]["row"] == 10
    assert w2.roi_list[0]["col"] == 15


def test_show2d_save_load_file(tmp_path):
    import json
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data, cmap="magma", title="Saved")
    path = tmp_path / "show2d_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["cmap"] == "magma"
    assert saved["title"] == "Saved"
    w2 = Show2D(data, state=str(path))
    assert w2.cmap == "magma"
    assert w2.title == "Saved"


def test_show2d_summary(capsys):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, title="My Image", cmap="viridis")
    w.summary()
    out = capsys.readouterr().out
    assert "My Image" in out
    assert "32×32" in out
    assert "viridis" in out


def test_show2d_repr():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, cmap="inferno")
    r = repr(w)
    assert "Show2D" in r
    assert "32×32" in r
    assert "inferno" in r


def test_show2d_repr_gallery():
    data = np.random.rand(3, 16, 16).astype(np.float32)
    w = Show2D(data)
    r = repr(w)
    assert "3×16×16" in r
    assert "idx=0" in r


def test_show2d_set_image():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, cmap="viridis", log_scale=True)
    assert widget.height == 16
    assert widget.width == 16

    new_data = np.random.rand(32, 24).astype(np.float32)
    widget.set_image(new_data)
    assert widget.height == 32
    assert widget.width == 24
    assert widget.n_images == 1
    assert widget.cmap == "viridis"
    assert widget.log_scale is True
    assert len(widget.frame_bytes) == 32 * 24 * 4


def test_show2d_set_image_gallery():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    new_data = [np.random.rand(20, 20).astype(np.float32) for _ in range(3)]
    widget.set_image(new_data, labels=["A", "B", "C"])
    assert widget.n_images == 3
    assert widget.height == 20
    assert widget.width == 20
    assert widget.labels == ["A", "B", "C"]
    assert widget.selected_idx == 0
