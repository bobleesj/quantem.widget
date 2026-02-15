import numpy as np
import pytest
import torch

from quantem.widget import Show3DVolume


def test_show3dvolume_numpy():
    """Create widget from numpy array."""
    data = np.random.rand(16, 16, 16).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.nz == 16
    assert widget.ny == 16
    assert widget.nx == 16
    assert len(widget.volume_bytes) > 0


def test_show3dvolume_torch():
    """Create widget from PyTorch tensor."""
    data = torch.rand(16, 16, 16)
    widget = Show3DVolume(data)
    assert widget.nz == 16
    assert widget.ny == 16
    assert widget.nx == 16


def test_show3dvolume_initial_slices():
    """Initial slices are at middle."""
    data = np.random.rand(20, 30, 40).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.slice_z == 10
    assert widget.slice_y == 15
    assert widget.slice_x == 20


def test_show3dvolume_stats():
    """Statistics are computed for 3 orthogonal slices."""
    data = np.ones((16, 16, 16), dtype=np.float32) * 5.0
    widget = Show3DVolume(data)
    assert len(widget.stats_mean) == 3
    assert len(widget.stats_min) == 3
    assert len(widget.stats_max) == 3
    assert len(widget.stats_std) == 3
    for mean in widget.stats_mean:
        assert mean == pytest.approx(5.0)


def test_show3dvolume_rejects_2d():
    """Raises error for 2D input."""
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="3D"):
        Show3DVolume(data)


def test_show3dvolume_options():
    """Display options are applied."""
    data = np.random.rand(16, 16, 16).astype(np.float32)
    widget = Show3DVolume(
        data,
        title="Test Volume",
        cmap="viridis",
        log_scale=True,
        auto_contrast=True,
    )
    assert widget.title == "Test Volume"
    assert widget.cmap == "viridis"
    assert widget.log_scale is True
    assert widget.auto_contrast is True


def test_show3dvolume_non_cubic():
    """Non-cubic volumes work correctly."""
    data = np.random.rand(10, 20, 30).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.nz == 10
    assert widget.ny == 20
    assert widget.nx == 30


def test_show3dvolume_rejects_4d():
    """Raises error for 4D input."""
    data = np.random.rand(8, 8, 8, 8).astype(np.float32)
    with pytest.raises(ValueError, match="3D"):
        Show3DVolume(data)


def test_show3dvolume_slice_change_updates_stats():
    """Changing slice positions recomputes stats."""
    data = np.zeros((16, 16, 16), dtype=np.float32)
    data[0, :, :] = 10.0
    data[15, :, :] = 50.0
    widget = Show3DVolume(data)
    # Move Z slice to first plane (all 10s)
    widget.slice_z = 0
    assert widget.stats_mean[0] == pytest.approx(10.0)
    # Move Z slice to last plane (all 50s)
    widget.slice_z = 15
    assert widget.stats_mean[0] == pytest.approx(50.0)


def test_show3dvolume_stats_per_plane():
    """Stats are computed from correct orthogonal planes."""
    data = np.zeros((10, 20, 30), dtype=np.float32)
    # XY plane at slice_z=5: set to 1.0
    data[5, :, :] = 1.0
    # XZ plane at slice_y=10: set to 2.0
    data[:, 10, :] = 2.0
    # YZ plane at slice_x=15: set to 3.0
    data[:, :, 15] = 3.0
    widget = Show3DVolume(data)
    widget.slice_z = 5
    widget.slice_y = 10
    widget.slice_x = 15
    # XY plane mean should reflect data[5, :, :]
    # XZ plane mean should reflect data[:, 10, :]
    # YZ plane mean should reflect data[:, :, 15]
    assert widget.stats_mean[0] != widget.stats_mean[1]  # Different planes have different stats


def test_show3dvolume_crosshair():
    """show_crosshair default True, can be toggled."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.show_crosshair is True
    widget2 = Show3DVolume(data, show_crosshair=False)
    assert widget2.show_crosshair is False


def test_show3dvolume_show_controls():
    """show_controls default True."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.show_controls is True
    widget2 = Show3DVolume(data, show_controls=False)
    assert widget2.show_controls is False


def test_show3dvolume_show_fft():
    """show_fft default False."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.show_fft is False
    widget2 = Show3DVolume(data, show_fft=True)
    assert widget2.show_fft is True


def test_show3dvolume_scale_bar():
    """Scale bar parameters are stored."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data, pixel_size_angstrom=2.5, scale_bar_visible=False)
    assert widget.pixel_size_angstrom == pytest.approx(2.5)
    assert widget.scale_bar_visible is False


def test_show3dvolume_volume_bytes_size():
    """volume_bytes has correct size (nz * ny * nx * 4 bytes for float32)."""
    data = np.random.rand(8, 10, 12).astype(np.float32)
    widget = Show3DVolume(data)
    assert len(widget.volume_bytes) == 8 * 10 * 12 * 4


def test_show3dvolume_constant_data():
    """Constant volume doesn't crash."""
    data = np.ones((8, 8, 8), dtype=np.float32) * 42.0
    widget = Show3DVolume(data)
    for mean in widget.stats_mean:
        assert mean == pytest.approx(42.0)
    for std in widget.stats_std:
        assert std == pytest.approx(0.0)


def test_show3dvolume_single_voxel():
    """(1, 1, 1) volume works."""
    data = np.array([[[5.0]]], dtype=np.float32)
    widget = Show3DVolume(data)
    assert widget.nz == 1
    assert widget.ny == 1
    assert widget.nx == 1
    assert widget.slice_z == 0
    assert widget.slice_y == 0
    assert widget.slice_x == 0


def test_show3dvolume_asymmetric():
    """Very asymmetric dimensions work."""
    data = np.random.rand(5, 100, 10).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.nz == 5
    assert widget.ny == 100
    assert widget.nx == 10
    assert widget.slice_z == 2
    assert widget.slice_y == 50
    assert widget.slice_x == 5


def test_show3dvolume_playback_defaults():
    """Playback defaults match Show3D."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.playing is False
    assert widget.reverse is False
    assert widget.boomerang is False
    assert widget.fps == pytest.approx(5.0)
    assert widget.loop is True
    assert widget.play_axis == 0


def test_show3dvolume_play_pause_stop():
    """play/pause/stop methods work."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    widget.play()
    assert widget.playing is True
    widget.pause()
    assert widget.playing is False
    widget.slice_z = 5
    widget.stop()
    assert widget.playing is False
    assert widget.slice_z == 0


def test_show3dvolume_fps_parameter():
    """fps constructor parameter is applied."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data, fps=15.0)
    assert widget.fps == pytest.approx(15.0)


def test_show3dvolume_play_axis():
    """play_axis can be set."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    widget.play_axis = 2
    assert widget.play_axis == 2
    widget.play_axis = 3  # "All"
    assert widget.play_axis == 3
