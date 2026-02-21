import numpy as np
import pytest
import torch

from quantem.widget import Show2D

try:
    import h5py  # type: ignore

    _HAS_H5PY = True
except Exception:
    h5py = None
    _HAS_H5PY = False


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
    widget = Show2D(data, pixel_size=1.5, scale_bar_visible=False)
    assert widget.pixel_size == pytest.approx(1.5)
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


def test_show2d_disabled_tools_default():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.disabled_tools == []


def test_show2d_disabled_tools_custom():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, disabled_tools=["display", "ROI", "profile"])
    assert widget.disabled_tools == ["display", "roi", "profile"]


def test_show2d_disabled_tools_flags():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, disable_display=True, disable_navigation=True, disable_view=True)
    assert widget.disabled_tools == ["display", "navigation", "view"]


def test_show2d_disabled_tools_disable_all():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, disable_all=True, disable_display=True)
    assert widget.disabled_tools == ["all"]


def test_show2d_disabled_tools_unknown_raises():
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show2D(data, disabled_tools=["not_real"])


def test_show2d_disabled_tools_trait_assignment_normalizes():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    widget.disabled_tools = ["DISPLAY", "display", "roi"]
    assert widget.disabled_tools == ["display", "roi"]


def test_show2d_hidden_tools_default():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.hidden_tools == []


def test_show2d_hidden_tools_custom():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, hidden_tools=["display", "ROI", "profile"])
    assert widget.hidden_tools == ["display", "roi", "profile"]


def test_show2d_hidden_tools_flags():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, hide_display=True, hide_navigation=True, hide_view=True)
    assert widget.hidden_tools == ["display", "navigation", "view"]


def test_show2d_hidden_tools_hide_all():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, hide_all=True, hide_display=True)
    assert widget.hidden_tools == ["all"]


def test_show2d_hidden_tools_unknown_raises():
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show2D(data, hidden_tools=["not_real"])


def test_show2d_hidden_tools_trait_assignment_normalizes():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    widget.hidden_tools = ["DISPLAY", "display", "roi"]
    assert widget.hidden_tools == ["display", "roi"]


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
               title="Test", pixel_size=2.5, show_fft=True,
               disabled_tools=["display", "view"], hidden_tools=["stats"])
    w.roi_active = True
    w.roi_list = [{"shape": "circle", "row": 10, "col": 15, "radius": 5}]
    w.roi_selected_idx = 0
    sd = w.state_dict()
    w2 = Show2D(data, state=sd)
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.auto_contrast is True
    assert w2.title == "Test"
    assert w2.pixel_size == pytest.approx(2.5)
    assert w2.show_fft is True
    assert w2.disabled_tools == ["display", "view"]
    assert w2.hidden_tools == ["stats"]
    assert w2.roi_active is True
    assert len(w2.roi_list) == 1
    assert w2.roi_list[0]["row"] == 10
    assert w2.roi_list[0]["col"] == 15


def test_show2d_state_dict_keys():
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data)
    keys = set(w.state_dict().keys())
    assert "disabled_tools" in keys
    assert "hidden_tools" in keys
    assert "show_stats" in keys
    assert "show_fft" in keys


def test_show2d_save_load_file(tmp_path):
    import json
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data, cmap="magma", title="Saved")
    path = tmp_path / "show2d_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "Show2D"
    assert isinstance(saved["widget_version"], str)
    assert saved["state"]["cmap"] == "magma"
    assert saved["state"]["title"] == "Saved"
    w2 = Show2D(data, state=str(path))
    assert w2.cmap == "magma"
    assert w2.title == "Saved"


def test_show2d_rejects_legacy_flat_state_file(tmp_path):
    import json

    data = np.random.rand(16, 16).astype(np.float32)
    path = tmp_path / "legacy_show2d_state.json"
    path.write_text(json.dumps({"cmap": "magma", "title": "Legacy"}, indent=2))

    with pytest.raises(ValueError, match="versioned envelope"):
        Show2D(data, state=str(path))


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


def test_show2d_from_png_file(tmp_path):
    from PIL import Image

    path = tmp_path / "img.png"
    Image.fromarray((np.ones((8, 6), dtype=np.uint8) * 33)).save(path)

    widget = Show2D.from_png(path)
    assert widget.n_images == 1
    assert widget.height == 8
    assert widget.width == 6
    assert widget.title == "img"


def test_show2d_from_png_folder_gallery(tmp_path):
    from PIL import Image

    folder = tmp_path / "png_stack"
    folder.mkdir()
    for i in range(3):
        Image.fromarray((np.ones((8, 6), dtype=np.uint8) * (10 + i))).save(folder / f"slice_{i:02d}.png")

    widget = Show2D.from_folder(folder, file_type="png")
    assert widget.n_images == 3
    assert widget.labels[0] == "slice_00.png"

    widget2 = Show2D.from_folder(folder, file_type="png", mode="mean")
    assert widget2.n_images == 1


def test_show2d_from_path_folder_requires_file_type(tmp_path):
    from PIL import Image

    folder = tmp_path / "png_stack"
    folder.mkdir()
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(folder / "a.png")

    with pytest.raises(ValueError, match="file_type is required"):
        Show2D.from_path(folder)


def test_show2d_from_tiff_file_gallery_and_reduce(tmp_path):
    from PIL import Image

    tiff_path = tmp_path / "stack.tiff"
    frames = [
        Image.fromarray((np.ones((7, 5), dtype=np.uint8) * 20)),
        Image.fromarray((np.ones((7, 5), dtype=np.uint8) * 22)),
    ]
    frames[0].save(tiff_path, save_all=True, append_images=frames[1:])

    gallery = Show2D.from_tiff(tiff_path)
    assert gallery.n_images == 2
    assert gallery.labels[0].startswith("stack[0]")

    reduced = Show2D.from_tiff(tiff_path, mode="mean")
    assert reduced.n_images == 1
    assert reduced.stats_mean[0] == pytest.approx(21.0)


def test_show2d_from_mixed_folder_explicit_type(tmp_path):
    from PIL import Image

    folder = tmp_path / "mixed_stack"
    folder.mkdir()
    Image.fromarray((np.ones((6, 4), dtype=np.uint8) * 7)).save(folder / "a.png")
    frames = [
        Image.fromarray((np.ones((6, 4), dtype=np.uint8) * 21)),
        Image.fromarray((np.ones((6, 4), dtype=np.uint8) * 22)),
    ]
    frames[0].save(folder / "b.tiff", save_all=True, append_images=frames[1:])

    png_widget = Show2D.from_folder(folder, file_type="png")
    assert png_widget.n_images == 1
    assert png_widget.labels == ["a.png"]

    tiff_widget = Show2D.from_folder(folder, file_type="tiff")
    assert tiff_widget.n_images == 2
    assert tiff_widget.labels[0].startswith("b[0]")


def test_show2d_rejects_dataset_path_for_non_emd(tmp_path):
    from PIL import Image

    path = tmp_path / "img.png"
    Image.fromarray((np.ones((8, 6), dtype=np.uint8) * 33)).save(path)

    with pytest.raises(ValueError, match="dataset_path is only supported"):
        Show2D.from_path(path, dataset_path="/data/signal")


def test_show2d_from_path_rejects_file_type_for_file(tmp_path):
    from PIL import Image

    path = tmp_path / "img.png"
    Image.fromarray((np.ones((8, 6), dtype=np.uint8) * 33)).save(path)

    with pytest.raises(ValueError, match="file_type is only used for folder"):
        Show2D.from_path(path, file_type="png")


def test_show2d_invalid_reduce_mode(tmp_path):
    from PIL import Image

    folder = tmp_path / "png_stack"
    folder.mkdir()
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(folder / "a.png")

    with pytest.raises(ValueError, match="Unknown reduce mode"):
        Show2D.from_folder(folder, file_type="png", mode="median")


@pytest.mark.skipif(not _HAS_H5PY or h5py is None, reason="h5py not available")
def test_show2d_from_emd_with_dataset_path_and_reduce(tmp_path):
    emd_path = tmp_path / "stack.emd"
    with h5py.File(emd_path, "w") as h5f:  # type: ignore[arg-type]
        h5f.create_dataset("preview/thumb", data=np.ones((5, 5), dtype=np.float32) * 2.0)
        h5f.create_dataset("data/signal", data=np.ones((3, 7, 5), dtype=np.float32) * 13.0)

    gallery = Show2D.from_emd(emd_path, dataset_path="/data/signal")
    assert gallery.n_images == 3
    assert gallery.height == 7
    assert gallery.width == 5
    assert gallery.labels[0].startswith("stack[0]")

    reduced = Show2D.from_emd(emd_path, dataset_path="/data/signal", mode="max")
    assert reduced.n_images == 1
    assert reduced.stats_mean[0] == pytest.approx(13.0)


@pytest.mark.skipif(not _HAS_H5PY or h5py is None, reason="h5py not available")
def test_show2d_from_emd_highdim_default_and_reduction(tmp_path):
    emd_path = tmp_path / "highdim.emd"
    arr = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    with h5py.File(emd_path, "w") as h5f:  # type: ignore[arg-type]
        h5f.create_dataset("data/signal", data=arr)

    gallery = Show2D.from_emd(emd_path, dataset_path="/data/signal")
    assert gallery.n_images == 6
    assert gallery.height == 4
    assert gallery.width == 4

    reduced = Show2D.from_emd(emd_path, dataset_path="/data/signal", mode="mean")
    assert reduced.n_images == 1
    assert reduced.height == 4
    assert reduced.width == 4


# ── save_image ───────────────────────────────────────────────────────────


def test_show2d_save_image_png(tmp_path):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, cmap="viridis")
    out = w.save_image(tmp_path / "out.png")
    assert out.exists()
    assert out.stat().st_size > 0
    from PIL import Image
    img = Image.open(out)
    assert img.size == (32, 32)


def test_show2d_save_image_pdf(tmp_path):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, cmap="inferno")
    out = w.save_image(tmp_path / "out.pdf")
    assert out.exists()
    assert out.stat().st_size > 0


def test_show2d_save_image_gallery_idx(tmp_path):
    imgs = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Show2D(imgs)
    out0 = w.save_image(tmp_path / "img0.png", idx=0)
    out2 = w.save_image(tmp_path / "img2.png", idx=2)
    assert out0.exists()
    assert out2.exists()


def test_show2d_save_image_log_auto(tmp_path):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, log_scale=True, auto_contrast=True)
    out = w.save_image(tmp_path / "out.png")
    assert out.exists()


def test_show2d_save_image_bad_format(tmp_path):
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data)
    with pytest.raises(ValueError, match="Unsupported format"):
        w.save_image(tmp_path / "out.bmp")


def test_show2d_save_image_bad_idx(tmp_path):
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data)
    with pytest.raises(IndexError):
        w.save_image(tmp_path / "out.png", idx=5)


def test_show2d_widget_version_is_set():
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data)
    assert w.widget_version != "unknown"
