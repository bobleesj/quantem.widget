"""Tests for quantem Dataset integration across all widgets.

Verifies that widgets correctly accept Dataset objects and
auto-extract metadata (title, pixel_size, units) via duck typing.
"""

import numpy as np
import pytest

from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4d, Dataset4dstem
from quantem.widget import Align2D, Edit2D, Mark2D, Show2D, Show3D, Show3DVolume, Show4D, Show4DSTEM, ShowComplex2D


# =========================================================================
# Show2D + Dataset2d
# =========================================================================


def test_show2d_dataset2d_extracts_title():
    ds = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="HRTEM Image",
        sampling=(0.5, 0.5),
        units=("Å", "Å"),
    )
    w = Show2D(ds)
    assert w.title == "HRTEM Image"
    assert w.n_images == 1
    assert w.height == 32
    assert w.width == 32


def test_show2d_dataset2d_extracts_pixel_size():
    ds = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="test",
        sampling=(1.5, 1.5),
        units=("Å", "Å"),
    )
    w = Show2D(ds)
    assert w.pixel_size == pytest.approx(1.5)


def test_show2d_dataset2d_converts_nm_to_angstrom():
    ds = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="test",
        sampling=(0.2, 0.2),
        units=("nm", "nm"),
    )
    w = Show2D(ds)
    assert w.pixel_size == pytest.approx(2.0)  # 0.2 nm = 2.0 Å


def test_show2d_dataset2d_explicit_overrides():
    ds = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="Dataset Title",
        sampling=(1.0, 1.0),
        units=("Å", "Å"),
    )
    w = Show2D(ds, title="My Title", pixel_size=5.0)
    assert w.title == "My Title"
    assert w.pixel_size == pytest.approx(5.0)


# =========================================================================
# Show3D + Dataset3d
# =========================================================================


def test_show3d_dataset3d_extracts_title():
    ds = Dataset3d.from_array(
        array=np.random.rand(5, 32, 32).astype(np.float32),
        name="Focal Series",
        sampling=(1.0, 0.25, 0.25),
        units=("nm", "Å", "Å"),
    )
    w = Show3D(ds)
    assert w.title == "Focal Series"
    assert w.n_slices == 5


def test_show3d_dataset3d_extracts_pixel_size_units():
    ds = Dataset3d.from_array(
        array=np.random.rand(5, 32, 32).astype(np.float32),
        name="test",
        sampling=(1.0, 2.5, 2.5),
        units=("nm", "Å", "Å"),
    )
    w = Show3D(ds)
    # pixel_size is in Å; sampling in Å → passthrough
    assert w.pixel_size == pytest.approx(2.5)


def test_show3d_dataset3d_nm_units_passthrough():
    ds = Dataset3d.from_array(
        array=np.random.rand(5, 32, 32).astype(np.float32),
        name="test",
        sampling=(1.0, 0.5, 0.5),
        units=("nm", "nm", "nm"),
    )
    w = Show3D(ds)
    # pixel_size in Å, sampling in nm → convert: 0.5 nm * 10 = 5.0 Å
    assert w.pixel_size == pytest.approx(5.0)


def test_show3d_dataset3d_explicit_overrides():
    ds = Dataset3d.from_array(
        array=np.random.rand(5, 32, 32).astype(np.float32),
        name="Dataset Title",
        sampling=(1.0, 0.25, 0.25),
        units=("nm", "nm", "nm"),
    )
    w = Show3D(ds, title="Override", pixel_size=10.0)
    assert w.title == "Override"
    assert w.pixel_size == pytest.approx(10.0)


# =========================================================================
# Show3DVolume + Dataset3d
# =========================================================================


def test_show3dvolume_dataset3d_extracts_title():
    ds = Dataset3d.from_array(
        array=np.random.rand(16, 16, 16).astype(np.float32),
        name="Tomogram",
        sampling=(0.5, 0.5, 0.5),
        units=("Å", "Å", "Å"),
    )
    w = Show3DVolume(ds)
    assert w.title == "Tomogram"
    assert w.nz == 16


def test_show3dvolume_dataset3d_extracts_pixel_size():
    ds = Dataset3d.from_array(
        array=np.random.rand(16, 16, 16).astype(np.float32),
        name="test",
        sampling=(1.5, 1.5, 1.5),
        units=("Å", "Å", "Å"),
    )
    w = Show3DVolume(ds)
    assert w.pixel_size == pytest.approx(1.5)


def test_show3dvolume_dataset3d_converts_nm():
    ds = Dataset3d.from_array(
        array=np.random.rand(16, 16, 16).astype(np.float32),
        name="test",
        sampling=(0.3, 0.3, 0.3),
        units=("nm", "nm", "nm"),
    )
    w = Show3DVolume(ds)
    assert w.pixel_size == pytest.approx(3.0)  # 0.3 nm = 3.0 Å


def test_show3dvolume_dataset3d_explicit_overrides():
    ds = Dataset3d.from_array(
        array=np.random.rand(16, 16, 16).astype(np.float32),
        name="Dataset Title",
        sampling=(1.0, 1.0, 1.0),
        units=("Å", "Å", "Å"),
    )
    w = Show3DVolume(ds, title="Override", pixel_size=5.0)
    assert w.title == "Override"
    assert w.pixel_size == pytest.approx(5.0)


# =========================================================================
# Show4DSTEM + Dataset4dstem
# =========================================================================


def test_show4dstem_dataset4dstem_extracts_calibration():
    ds = Dataset4dstem.from_array(
        array=np.random.rand(8, 8, 16, 16).astype(np.float32),
        name="test",
        sampling=(2.39, 2.39, 0.46, 0.46),
        units=("Å", "Å", "mrad", "mrad"),
    )
    w = Show4DSTEM(ds)
    assert w.pixel_size == pytest.approx(2.39)
    assert w.k_pixel_size == pytest.approx(0.46)
    assert w.k_calibrated is True


def test_show4dstem_dataset4dstem_nm_to_angstrom():
    ds = Dataset4dstem.from_array(
        array=np.random.rand(8, 8, 16, 16).astype(np.float32),
        name="test",
        sampling=(0.239, 0.239, 0.46, 0.46),
        units=("nm", "nm", "mrad", "mrad"),
    )
    w = Show4DSTEM(ds)
    assert w.pixel_size == pytest.approx(2.39)  # 0.239 nm = 2.39 Å


def test_show4dstem_dataset4dstem_explicit_overrides():
    ds = Dataset4dstem.from_array(
        array=np.random.rand(8, 8, 16, 16).astype(np.float32),
        name="test",
        sampling=(2.39, 2.39, 0.46, 0.46),
        units=("Å", "Å", "mrad", "mrad"),
    )
    w = Show4DSTEM(ds, pixel_size=5.0, k_pixel_size=1.0)
    assert w.pixel_size == pytest.approx(5.0)
    assert w.k_pixel_size == pytest.approx(1.0)


# =========================================================================
# Show4D + Dataset4d
# =========================================================================


def test_show4d_dataset4d_extracts_calibration():
    ds = Dataset4d.from_array(
        array=np.random.rand(8, 8, 16, 16).astype(np.float32),
        name="PDF Map",
        sampling=(2.5, 2.5, 0.1, 0.1),
        units=("Å", "Å", "Å", "Å"),
    )
    w = Show4D(ds)
    assert w.title == "PDF Map"
    assert w.nav_pixel_size == pytest.approx(2.5)
    assert w.nav_pixel_unit == "Å"
    assert w.sig_pixel_size == pytest.approx(0.1)
    assert w.sig_pixel_unit == "Å"


def test_show4d_dataset4d_nm_to_angstrom():
    ds = Dataset4d.from_array(
        array=np.random.rand(8, 8, 16, 16).astype(np.float32),
        name="test",
        sampling=(0.239, 0.239, 0.05, 0.05),
        units=("nm", "nm", "nm", "nm"),
    )
    w = Show4D(ds)
    assert w.nav_pixel_size == pytest.approx(2.39)  # 0.239 nm → 2.39 Å
    assert w.nav_pixel_unit == "Å"
    assert w.sig_pixel_size == pytest.approx(0.5)  # 0.05 nm → 0.5 Å
    assert w.sig_pixel_unit == "Å"


def test_show4d_dataset4d_mrad_units():
    ds = Dataset4d.from_array(
        array=np.random.rand(8, 8, 16, 16).astype(np.float32),
        name="test",
        sampling=(2.39, 2.39, 0.46, 0.46),
        units=("Å", "Å", "mrad", "mrad"),
    )
    w = Show4D(ds)
    assert w.sig_pixel_size == pytest.approx(0.46)
    assert w.sig_pixel_unit == "mrad"


def test_show4d_dataset4dstem_also_works():
    ds = Dataset4dstem.from_array(
        array=np.random.rand(8, 8, 16, 16).astype(np.float32),
        name="4D-STEM via Show4D",
        sampling=(2.39, 2.39, 0.46, 0.46),
        units=("Å", "Å", "mrad", "mrad"),
    )
    w = Show4D(ds)
    assert w.title == "4D-STEM via Show4D"
    assert w.nav_pixel_size == pytest.approx(2.39)
    assert w.sig_pixel_size == pytest.approx(0.46)


def test_show4d_dataset4d_explicit_overrides():
    ds = Dataset4d.from_array(
        array=np.random.rand(8, 8, 16, 16).astype(np.float32),
        name="Dataset Title",
        sampling=(2.5, 2.5, 0.1, 0.1),
        units=("Å", "Å", "Å", "Å"),
    )
    w = Show4D(ds, title="Override", nav_pixel_size=5.0, sig_pixel_size=1.0)
    assert w.title == "Override"
    assert w.nav_pixel_size == pytest.approx(5.0)
    assert w.sig_pixel_size == pytest.approx(1.0)


# =========================================================================
# Mark2D + Dataset2d
# =========================================================================


def test_mark2d_dataset2d_accepts():
    ds = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="test",
        sampling=(0.5, 0.5),
        units=("Å", "Å"),
    )
    w = Mark2D(ds)
    assert w.n_images == 1
    assert w.height == 32
    assert w.width == 32
    assert len(w.frame_bytes) > 0


# =========================================================================
# Align2D + Dataset2d
# =========================================================================


def test_align2d_dataset2d_accepts():
    ds_a = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="Image A",
        sampling=(0.5, 0.5),
        units=("Å", "Å"),
    )
    ds_b = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="Image B",
        sampling=(0.5, 0.5),
        units=("Å", "Å"),
    )
    w = Align2D(ds_a, ds_b, auto_align=False)
    assert w.height == 32
    assert w.width == 32
    assert len(w.image_a_bytes) > 0
    assert len(w.image_b_bytes) > 0


def test_align2d_dataset2d_extracts_pixel_size():
    ds_a = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="test",
        sampling=(2.5, 2.5),
        units=("Å", "Å"),
    )
    ds_b = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="test",
        sampling=(2.5, 2.5),
        units=("Å", "Å"),
    )
    w = Align2D(ds_a, ds_b, auto_align=False)
    assert w.pixel_size == pytest.approx(2.5)  # 2.5 Å passthrough


def test_align2d_dataset2d_mixed_with_array():
    ds = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="test",
        sampling=(1.0, 1.0),
        units=("Å", "Å"),
    )
    raw = np.random.rand(32, 32).astype(np.float32)
    w = Align2D(ds, raw, auto_align=False)
    assert w.height == 32
    assert w.pixel_size == pytest.approx(1.0)  # 1.0 Å passthrough


def test_align2d_dataset2d_explicit_overrides():
    ds_a = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="test",
        sampling=(2.0, 2.0),
        units=("Å", "Å"),
    )
    ds_b = Dataset2d.from_array(
        array=np.random.rand(32, 32).astype(np.float32),
        name="test",
        sampling=(2.0, 2.0),
        units=("Å", "Å"),
    )
    w = Align2D(ds_a, ds_b, pixel_size=5.0, auto_align=False)
    assert w.pixel_size == pytest.approx(5.0)


# ── Edit2D + Dataset2d ──────────────────────────────────────────────────────


class MockDataset2dForEdit2D:
    """Duck-typed Dataset2d for Edit2D tests."""
    def __init__(self, array, name="", sampling=(1.0, 1.0), units=("Å", "Å")):
        self.array = array
        self.name = name
        self.sampling = sampling
        self.units = units


def test_edit2d_dataset_title():
    arr = np.random.rand(32, 32).astype(np.float32)
    ds = MockDataset2dForEdit2D(arr, name="HAADF Crop")
    widget = Edit2D(ds)
    assert widget.title == "HAADF Crop"


def test_edit2d_dataset_pixel_size():
    arr = np.random.rand(32, 32).astype(np.float32)
    ds = MockDataset2dForEdit2D(arr, name="Test", sampling=(2.5, 2.5), units=("Å", "Å"))
    widget = Edit2D(ds)
    assert widget.pixel_size == pytest.approx(2.5)


def test_edit2d_dataset_nm_conversion():
    arr = np.random.rand(32, 32).astype(np.float32)
    ds = MockDataset2dForEdit2D(arr, name="Test", sampling=(0.25, 0.25), units=("nm", "nm"))
    widget = Edit2D(ds)
    assert widget.pixel_size == pytest.approx(2.5)  # 0.25 nm = 2.5 Å


def test_edit2d_dataset_explicit_override():
    arr = np.random.rand(32, 32).astype(np.float32)
    ds = MockDataset2dForEdit2D(arr, name="Auto Title", sampling=(2.0, 2.0), units=("Å", "Å"))
    widget = Edit2D(ds, title="My Title", pixel_size=5.0)
    assert widget.title == "My Title"
    assert widget.pixel_size == pytest.approx(5.0)


# ── ShowComplex2D + Dataset2d ───────────────────────────────────────────────


class MockDataset2dForShowComplex2D:
    """Duck-typed Dataset2d for ShowComplex2D tests (complex data)."""
    def __init__(self, array, name="", sampling=(1.0, 1.0), units=("Å", "Å")):
        self.array = array
        self.name = name
        self.sampling = sampling
        self.units = units


def test_showcomplex2d_dataset_title():
    arr = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    ds = MockDataset2dForShowComplex2D(arr, name="Exit Wave")
    widget = ShowComplex2D(ds)
    assert widget.title == "Exit Wave"


def test_showcomplex2d_dataset_pixel_size():
    arr = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    ds = MockDataset2dForShowComplex2D(arr, name="Test", sampling=(1.5, 1.5), units=("Å", "Å"))
    widget = ShowComplex2D(ds)
    assert widget.pixel_size == pytest.approx(1.5)


def test_showcomplex2d_dataset_nm_conversion():
    arr = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    ds = MockDataset2dForShowComplex2D(arr, name="Test", sampling=(0.15, 0.15), units=("nm", "nm"))
    widget = ShowComplex2D(ds)
    assert widget.pixel_size == pytest.approx(1.5)  # 0.15 nm = 1.5 Å


def test_showcomplex2d_dataset_explicit_override():
    arr = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    ds = MockDataset2dForShowComplex2D(arr, name="Auto Title", sampling=(2.0, 2.0), units=("Å", "Å"))
    widget = ShowComplex2D(ds, title="My Title", pixel_size=3.0)
    assert widget.title == "My Title"
    assert widget.pixel_size == pytest.approx(3.0)
