import numpy as np
import pytest
import torch

from quantem.widget import Align2D


# === Basic construction ===

def test_align2d_numpy():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    assert widget.height == 32
    assert widget.width == 32
    assert len(widget.image_a_bytes) > 0
    assert len(widget.image_b_bytes) > 0


def test_align2d_torch():
    a = torch.rand(32, 32)
    b = torch.rand(32, 32)
    widget = Align2D(a, b, auto_align=False)
    assert widget.height == 32
    assert widget.width == 32


def test_align2d_mixed_numpy_torch():
    a = np.random.rand(32, 32).astype(np.float32)
    b = torch.rand(32, 32)
    widget = Align2D(a, b, auto_align=False)
    assert widget.height == 32


# === Padding ===

def test_align2d_default_padding():
    a = np.random.rand(100, 100).astype(np.float32)
    b = np.random.rand(100, 100).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    assert widget.padding == pytest.approx(0.2)
    assert widget.height == 100
    assert widget.width == 100


def test_align2d_custom_padding():
    a = np.random.rand(100, 100).astype(np.float32)
    b = np.random.rand(100, 100).astype(np.float32)
    widget = Align2D(a, b, padding=0.5, auto_align=False)
    assert widget.padding == pytest.approx(0.5)


def test_align2d_zero_padding():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, padding=0.0, auto_align=False)
    assert widget.padding == pytest.approx(0.0)
    assert widget.height == 32
    assert widget.width == 32


def test_align2d_unpadded_bytes_size():
    """Images are sent unpadded — bytes = height * width * 4."""
    a = np.random.rand(50, 60).astype(np.float32)
    b = np.random.rand(50, 60).astype(np.float32)
    widget = Align2D(a, b, padding=0.2, auto_align=False)
    expected_size = 50 * 60 * 4
    assert len(widget.image_a_bytes) == expected_size
    assert len(widget.image_b_bytes) == expected_size


def test_align2d_median_values():
    a = np.ones((10, 10), dtype=np.float32) * 5.0
    b = np.ones((10, 10), dtype=np.float32) * 15.0
    widget = Align2D(a, b, padding=0.5, auto_align=False)
    assert widget.median_a == pytest.approx(5.0)
    assert widget.median_b == pytest.approx(15.0)


# === Alignment offset ===

def test_align2d_initial_offset_zero():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    assert widget.dx == pytest.approx(0.0)
    assert widget.dy == pytest.approx(0.0)


def test_align2d_offset_property():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    widget.dx = 5.5
    widget.dy = -3.2
    assert widget.offset == (pytest.approx(5.5), pytest.approx(-3.2))


def test_align2d_reset_alignment():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    widget.dx = 10.0
    widget.dy = -5.0
    widget.reset_alignment()
    assert widget.dx == pytest.approx(0.0)
    assert widget.dy == pytest.approx(0.0)


# === Auto-alignment ===

def test_align2d_auto_align_shifted_image():
    """Auto-alignment should find the known offset."""
    rng = np.random.default_rng(42)
    a = rng.random((64, 64)).astype(np.float32)
    b = np.roll(a, (3, -5), axis=(0, 1))
    widget = Align2D(a, b, padding=0.3, auto_align=True)
    # roll(3, -5) shifts B down 3 and left 5; xcorr finds the undo: dx=+5, dy=-3
    assert widget.dx == pytest.approx(5.0, abs=1.5)
    assert widget.dy == pytest.approx(-3.0, abs=1.5)


def test_align2d_auto_align_sets_xcorr_zero():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    assert isinstance(widget.xcorr_zero, float)


def test_align2d_auto_align_disabled():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    assert widget.dx == pytest.approx(0.0)
    assert widget.dy == pytest.approx(0.0)


# === Different sizes ===

def test_align2d_different_sizes():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(64, 48).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    assert widget.height == 64
    assert widget.width == 48


# === Display options ===

def test_align2d_colormap():
    a = np.random.rand(16, 16).astype(np.float32)
    b = np.random.rand(16, 16).astype(np.float32)
    widget = Align2D(a, b, cmap="viridis", auto_align=False)
    assert widget.cmap == "viridis"


def test_align2d_opacity():
    a = np.random.rand(16, 16).astype(np.float32)
    b = np.random.rand(16, 16).astype(np.float32)
    widget = Align2D(a, b, opacity=0.7, auto_align=False)
    assert widget.opacity == pytest.approx(0.7)


def test_align2d_title():
    a = np.random.rand(16, 16).astype(np.float32)
    b = np.random.rand(16, 16).astype(np.float32)
    widget = Align2D(a, b, title="Alignment Test", auto_align=False)
    assert widget.title == "Alignment Test"


def test_align2d_labels():
    a = np.random.rand(16, 16).astype(np.float32)
    b = np.random.rand(16, 16).astype(np.float32)
    widget = Align2D(a, b, label_a="0 deg", label_b="90 deg", auto_align=False)
    assert widget.label_a == "0 deg"
    assert widget.label_b == "90 deg"


def test_align2d_pixel_size():
    a = np.random.rand(16, 16).astype(np.float32)
    b = np.random.rand(16, 16).astype(np.float32)
    widget = Align2D(a, b, pixel_size=0.5, auto_align=False)
    assert widget.pixel_size == pytest.approx(0.5)


def test_align2d_canvas_size():
    a = np.random.rand(16, 16).astype(np.float32)
    b = np.random.rand(16, 16).astype(np.float32)
    widget = Align2D(a, b, canvas_size=600, auto_align=False)
    assert widget.canvas_size == 600


# === Edge cases ===

def test_align2d_non_square():
    a = np.random.rand(32, 64).astype(np.float32)
    b = np.random.rand(32, 64).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    assert widget.height == 32
    assert widget.width == 64


def test_align2d_rejects_3d():
    a = np.random.rand(5, 16, 16).astype(np.float32)
    b = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="2D"):
        Align2D(a, b)


def test_align2d_rejects_3d_image_b():
    a = np.random.rand(16, 16).astype(np.float32)
    b = np.random.rand(5, 16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="2D"):
        Align2D(a, b)


# === Matrix DFT sub-pixel refinement ===

def test_dft_upsample_recovers_integer_peak():
    from quantem.widget.align2d import _dft_upsample, _cross_correlate_fft
    rng = np.random.default_rng(42)
    a = rng.random((64, 64)).astype(np.float32)
    b = np.roll(a, (3, -5), axis=(0, 1))
    dx, dy = _cross_correlate_fft(a, b)
    assert abs(dx - 5.0) < 0.02
    assert abs(dy - (-3.0)) < 0.02


def test_cross_correlate_fft_subpixel():
    """Matrix DFT should recover sub-pixel shifts with high accuracy."""
    from quantem.widget.align2d import _cross_correlate_fft
    from scipy.ndimage import shift as ndi_shift
    rng = np.random.default_rng(99)
    a = rng.random((64, 64)).astype(np.float32)
    b = ndi_shift(a, (2.3, -1.7), order=3, mode="wrap").astype(np.float32)
    dx, dy = _cross_correlate_fft(a, b)
    # ndi_shift cubic interpolation introduces ~0.1px error
    assert abs(dx - 1.7) < 0.15, f"dx={dx}, expected ~1.7"
    assert abs(dy - (-2.3)) < 0.15, f"dy={dy}, expected ~-2.3"


def test_align2d_ncc_aligned_exists():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=True)
    assert isinstance(widget.ncc_aligned, float)


# === max_shift ===

def test_align2d_max_shift_default():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    assert widget.max_shift == pytest.approx(0.0)


def test_align2d_max_shift_clamps_auto_align():
    """Auto-aligned dx/dy should be clamped within max_shift."""
    rng = np.random.default_rng(42)
    a = rng.random((64, 64)).astype(np.float32)
    b = np.roll(a, (3, -5), axis=(0, 1))
    widget = Align2D(a, b, auto_align=True, max_shift=2.0, padding=0.3)
    assert abs(widget.dx) <= 2.0 + 0.01
    assert abs(widget.dy) <= 2.0 + 0.01


def test_align2d_max_shift_traitlet():
    a = np.random.rand(16, 16).astype(np.float32)
    b = np.random.rand(16, 16).astype(np.float32)
    widget = Align2D(a, b, max_shift=10.0, auto_align=False)
    assert widget.max_shift == pytest.approx(10.0)


# === Rotation ===

def test_align2d_rotation_default():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    assert widget.rotation == pytest.approx(0.0)


def test_align2d_rotation_constructor():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, rotation=45.0, auto_align=False)
    assert widget.rotation == pytest.approx(45.0)


def test_align2d_rotation_negative():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, rotation=-90.0, auto_align=False)
    assert widget.rotation == pytest.approx(-90.0)


def test_align2d_rotation_mutable():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    widget.rotation = 12.5
    assert widget.rotation == pytest.approx(12.5)


def test_align2d_reset_includes_rotation():
    a = np.random.rand(32, 32).astype(np.float32)
    b = np.random.rand(32, 32).astype(np.float32)
    widget = Align2D(a, b, auto_align=False)
    widget.dx = 10.0
    widget.dy = -5.0
    widget.rotation = 30.0
    widget.reset_alignment()
    assert widget.dx == pytest.approx(0.0)
    assert widget.dy == pytest.approx(0.0)
    assert widget.rotation == pytest.approx(0.0)
