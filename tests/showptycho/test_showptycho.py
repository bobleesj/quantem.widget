"""ShowPtycho: a reconstruction array on any computer, and a real quantem.gpu SSB session.

Arrays (NumPy or torch on any device, real phase or complex object wave) must show exactly the
values given, to float32, with no quantem.gpu. For the session, a synthetic 128 x 128 scan of a
bright-field disk that shifts with a smooth phase object is fitted nowhere; the session carries
starting aberrations and the widget must show exactly the phase ``SSB.preview`` returns for them,
follow slider requests, write the calibration and star files, and export the browser folder with
exact bright-field counts.
"""

import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from quantem.widget import ShowPtycho
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.device import INSTALL_CUDA, INSTALL_MPS, NO_GPU_PATH
from quantem.widget.showptycho import widget as showptycho_module

SCAN, DET = 128, 16
START = {"C10_nm": 3.0, "C12_nm": 1.0, "phi12_deg": math.degrees(0.2)}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])


def _object_wave(dtype=np.complex64) -> np.ndarray:
    """A weak-phase lattice (0.3 rad peaks) with 10 % amplitude contrast, 96 x 80 so rows and columns differ."""
    rows, cols = np.mgrid[0:96, 0:80].astype(np.float64)
    phase = 0.3 * np.cos(2 * np.pi * rows / 12.0) * np.cos(2 * np.pi * cols / 10.0)
    amplitude = 1.0 - 0.1 * (np.sin(2 * np.pi * rows / 24.0) ** 2)
    return (amplitude * np.exp(1j * phase)).astype(dtype)


def _shown(widget, trait: str = "phase_bytes") -> np.ndarray:
    """The float32 image the browser receives in ``trait``, shaped like the displayed phase."""
    return np.frombuffer(getattr(widget, trait), np.float32).reshape(widget.phase_height, widget.phase_width)


# --- Arrays: every computer ---
def test_a_phase_array_is_shown_as_given_with_the_session_controls_explained():
    phase = np.angle(_object_wave())
    widget = ShowPtycho(phase, sampling=0.05, units="nm")
    np.testing.assert_array_equal(_shown(widget), phase)
    assert widget.amplitude_bytes == b"" and (widget.phase_height, widget.phase_width) == (96, 80)
    assert widget.object_sampling_A == pytest.approx([0.5, 0.5])            # 0.05 nm drawn in Angstrom
    assert not widget.ssb_session and "needs an SSB session" in widget.session_note
    assert repr(widget) == "ShowPtycho(phase 96x80, 0.5 Å/px)"
    assert ShowPtycho(phase, sampling=(0.4, 0.5)).object_sampling_A == [0.4, 0.5]  # [row, col]; the column step sets the scale bar
    with pytest.raises(ValueError, match="export"):
        widget.export()


@pytest.mark.parametrize("installed, hint, says, never", [
    # quantem.gpu here: open a session (the note inspects quantem.gpu's session type, so the package must be present)
    pytest.param(True, INSTALL_CUDA, "quantem.gpu.SSB.open(", "pip install",
                 marks=pytest.mark.skipif(importlib.util.find_spec("quantem.gpu") is None, reason="needs quantem.gpu installed")),
    (False, INSTALL_CUDA, INSTALL_CUDA, INSTALL_MPS),                      # NVIDIA machine without it
    (False, INSTALL_MPS, INSTALL_MPS, INSTALL_CUDA),                       # Apple silicon without it
    (False, None, NO_GPU_PATH, "pip install"),                             # no GPU route on this machine
])
def test_the_session_note_names_what_this_machine_needs(monkeypatch, installed, hint, says, never):
    monkeypatch.setattr(gpu_adapter, "available", lambda: installed)
    monkeypatch.setattr(showptycho_module, "gpu_path_hint", lambda: hint, raising=False)
    note = ShowPtycho(np.angle(_object_wave())).session_note
    assert says in note and never not in note


def test_a_complex_object_wave_shows_its_phase_and_amplitude(capsys):
    wave = _object_wave(np.complex64)
    widget = ShowPtycho(wave)
    np.testing.assert_array_equal(_shown(widget), np.angle(wave))
    np.testing.assert_array_equal(_shown(widget, "amplitude_bytes"), np.abs(wave))
    assert capsys.readouterr().out == ""                                    # complex64 is drawn as it is
    wide = _object_wave(np.complex128)
    widget = ShowPtycho(wide)
    np.testing.assert_array_equal(_shown(widget), np.angle(wide).astype(np.float32))
    np.testing.assert_array_equal(_shown(widget, "amplitude_bytes"), np.abs(wide).astype(np.float32))
    printed = capsys.readouterr().out
    assert printed.startswith("ShowPtycho: complex128 input is drawn in float32; largest change: phase ")
    assert repr(widget) == "ShowPtycho(object wave 96x80, no sampling)"


@pytest.mark.parametrize("device", DEVICES)
def test_a_torch_tensor_on_any_device_shows_the_same_numbers(device):
    wave = _object_wave(np.complex64)
    widget = ShowPtycho(torch.from_numpy(wave).to(device), sampling=0.2)
    expected = torch.from_numpy(wave).to(device)
    np.testing.assert_array_equal(_shown(widget), expected.angle().cpu().numpy())
    np.testing.assert_array_equal(_shown(widget, "amplitude_bytes"), expected.abs().cpu().numpy())
    phase = torch.from_numpy(np.angle(wave)).to(device)
    np.testing.assert_array_equal(_shown(ShowPtycho(phase)), np.angle(wave))


def test_any_2d_array_like_opens_and_other_shapes_are_refused():
    with pytest.raises(ValueError, match="one 2D reconstruction"):
        ShowPtycho(np.zeros((3, 16, 16), np.float32))
    with pytest.raises(ValueError, match="Unknown unit 'inch'"):
        ShowPtycho(np.zeros((16, 16), np.float32), sampling=1.0, units="inch")
    # C-1: a list (or a CuPy array) raised TypeError although the widgets accept any array-like
    widget = ShowPtycho([[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_array_equal(np.frombuffer(widget.phase_bytes, np.float32), [0.0, 1.0, 2.0, 3.0])
    cupy = pytest.importorskip("cupy")
    phase = np.linspace(-1, 1, 64, dtype=np.float32).reshape(8, 8)
    np.testing.assert_array_equal(np.frombuffer(ShowPtycho(cupy.asarray(phase)).phase_bytes, np.float32).reshape(8, 8), phase)


# --- SSB session: quantem.gpu with CUDA or Metal ---


def _counts(scan: int = SCAN) -> np.ndarray:
    """uint16 counts of a bright-field disk shifted per scan position by a smooth phase object, plus 0-2 counts of noise."""
    rows, cols = np.indices((DET, DET))
    disk = ((rows - 7.5) ** 2 + (cols - 7.5) ** 2) <= 4.0**2
    phase_object = np.sin(np.linspace(0, 6 * np.pi, scan))[:, None] * np.cos(np.linspace(0, 4 * np.pi, scan))[None, :]
    counts = np.zeros((scan, scan, DET, DET), dtype=np.uint16)
    counts[..., disk] = 40
    shift = np.round(3 * phase_object).astype(int)
    for row in range(scan):
        for col in range(scan):
            counts[row, col] = np.roll(counts[row, col], shift[row, col], axis=1)
    return counts + np.random.default_rng(0).integers(0, 3, counts.shape, dtype=np.uint16)


def _session(scan: int = SCAN):
    """A real quantem.gpu SSB session at the START aberrations, or a skip on machines without CUDA or MPS."""
    gpu = pytest.importorskip("quantem.gpu", reason="an SSB session needs quantem.gpu: pip install quantem.widget[cuda] or [mps]")
    try:
        pytest.importorskip("quantem.gpu.device").detect()
    except RuntimeError as exc:
        pytest.skip(f"SSB needs CUDA or MPS: {exc}")
    return gpu.SSB(_counts(scan), voltage_kV=300.0, semiangle_mrad=20.0, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5.0, 5.0),
                   aberrations=dict(START), scan_detector_rotation_deg=10.0)


@pytest.fixture(scope="module")
def ssb():
    """One session shared by the module: building it dominates the test time."""
    return _session()


@pytest.fixture
def widget(ssb, tmp_path):
    """A fresh widget over the shared session, which the previous test may have rotated or re-aberrated."""
    ssb.set_rotation(10.0)
    ssb.aberrations, ssb.best_loss = dict(START), float("inf")
    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"master")
    ssb.source_path = str(master)  # the session names its file; the widget reads it from there
    return ShowPtycho(ssb, show_fft=True, save_dir=tmp_path)


def test_opens_on_the_session_fit_and_shows_its_preview(widget, ssb):
    assert (widget.fitted_C10_nm, widget.fitted_C12_nm, widget.fitted_phi12_deg) == (3.0, 1.0, pytest.approx(math.degrees(0.2)))
    assert widget.scan_detector_rotation_deg == widget.fitted_scan_detector_rotation_deg == 10.0
    assert (widget.scan_sampling_A, widget.object_sampling_A, widget.scan_rows, widget.scan_cols) == ([0.5, 0.5], [], SCAN, SCAN)
    assert widget.num_bright_field_pixels == widget.num_drag_bright_field_pixels == ssb.num_bright_field_pixels
    assert widget.initial_fft_on and widget.crop_refit_available and widget.trials_json == ""
    # Python reconstructs every live frame: no browser SSB runs, so there is no browser status to show
    assert widget.webgpu_preview_status == "" and not widget.webgpu_preview_enabled
    expected, loss = ssb.preview(dict(START), compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), expected)
    result = json.loads(widget.result_json)
    assert result == {"id": 0, "C10_nm": 3.0, "upsampling_factor": 1, "C12_nm": 1.0, "phi12_deg": pytest.approx(math.degrees(0.2)), "loss": pytest.approx(loss)}
    assert repr(widget) == f"ShowPtycho(0 pinned, {ssb.num_bright_field_pixels} bright-field pixels while dragging)"


def test_requests_flip_and_rotation_follow_the_session(widget, ssb):
    widget.request_json = json.dumps({"id": 7, "C10_nm": 2.0, "C12_nm": 0.5, "phi12_deg": 30.0, "committed": True})
    expected, loss = ssb.preview({"C10_nm": 2.0, "C12_nm": 0.5, "phi12_deg": 30.0}, compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), expected)
    assert json.loads(widget.result_json)["loss"] == pytest.approx(loss)
    widget.request_json = json.dumps({"id": 8, "C10_nm": 2.5, "C12_nm": 0.5, "phi12_deg": 30.0, "committed": False})
    assert json.loads(widget.result_json)["loss"] is None                     # drags skip the variance pass
    widget.flip_phase = True
    np.testing.assert_array_equal(_shown(widget), -ssb.preview({"C10_nm": 2.5, "C12_nm": 0.5, "phi12_deg": 30.0}, compute_loss=False)[0])
    widget.scan_detector_rotation_deg = 25.0
    assert ssb.scan_detector_rotation_deg == 25.0
    expected, _ = ssb.preview({"C10_nm": 2.5, "C12_nm": 0.5, "phi12_deg": 30.0}, compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), -expected)
    assert json.loads(widget.result_json)["C10_nm"] == 2.5                 # the rotation keeps the displayed aberrations
    # the request keys of the earlier release are not read under a second spelling
    with pytest.raises(KeyError, match="C10_nm"):
        widget.request_json = json.dumps({"id": 9, "c10": 2.0, "c12": 0.5, "phi12_deg": 30.0, "committed": True})


def test_higher_order_panel_switches_to_the_full_kernel():
    """The CUDA higher-order loss kernel starts at 256 x 256 scans."""
    ssb = _session(256)
    widget = ShowPtycho(ssb)
    widget.higher_order_aberrations_json = json.dumps({"C21_nm": 5.0, "phi21_deg": 45.0, "C30_nm": 2.0})
    expected, _ = ssb.preview({**START, "C21_nm": 5.0, "phi21_deg": 45.0, "C30_nm": 2.0}, compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), expected)
    with pytest.raises(ValueError, match="upsampling_factor=1"):
        widget.upsampling_factor = 2
    widget.higher_order_aberrations_json = "{}"
    # a panel left at zero keeps the two-term phase: a named term, even at zero, makes quantem.gpu hold C10, C12 and
    # phi12 in float32, which changed the last bits of most pixels for coefficients float32 cannot hold exactly
    widget.higher_order_aberrations_json = json.dumps({"C21_nm": 0.0, "phi21_deg": 45.0})
    displayed = {"C10_nm": -26.259025251007465, "C12_nm": 6.41328169139375, "phi12_deg": 63.473910926518215}
    widget.request_json = json.dumps({"id": 3, **displayed, "committed": True})
    np.testing.assert_array_equal(_shown(widget), ssb.preview(displayed, compute_loss=True)[0])
    if ssb.backend == "cuda":
        widget.upsampling_factor = 2
        assert _shown(widget).shape == (512, 512)
        with pytest.raises(ValueError, match="Set upsampling_factor=1"):
            widget.higher_order_aberrations_json = json.dumps({"C30_nm": 1.0})
    with pytest.raises(ValueError, match="1, 2, 3, 4, or 8"):
        widget.upsampling_factor = 5


def test_sample_tilt_panel_uses_the_thick_sample_model(widget, ssb):
    assert widget.sample_available
    widget.sample_json = json.dumps({"sample_tilt_mrad": [3.0, -4.0], "depth_spread_nm": 12.0})
    expected, _ = ssb.preview(dict(START), compute_loss=True, sample_tilt_mrad=(3.0, -4.0), depth_spread_nm=12.0)
    np.testing.assert_array_equal(_shown(widget), expected)
    widget.save_trigger += 1
    saved = json.loads(Path(widget.calibration_path).read_text())
    # the panel's depth is the model's depth spread: it is stored under that name and never as a thickness
    assert saved["sample_tilt_mrad"] == [3.0, -4.0] and saved["depth_spread_nm"] == 12.0 and "thickness_nm" not in saved
    rot = math.radians(10.0)
    assert saved["propagator_tilt_mrad"] == pytest.approx([3.0 * math.cos(rot) + (-4.0) * math.sin(rot), -3.0 * math.sin(rot) + (-4.0) * math.cos(rot)])


def test_save_writes_the_calibration_and_mirrors_it_onto_the_session(widget, ssb, tmp_path):
    widget.request_json = json.dumps({"id": 1, "C10_nm": -4.0, "C12_nm": 0.25, "phi12_deg": 12.0, "committed": True})
    widget.notes = "lamella A"
    widget.save_trigger += 1
    assert widget.calibration_path == str(tmp_path / "calibration.json") and widget.calibration_saved_at
    saved = json.loads((tmp_path / "calibration.json").read_text())
    assert saved["schema"] == "quantem.ssb-calibration/4"
    # nm and degrees under names that say so: phi12 was radians beside higher-order angles in degrees
    assert saved["aberrations"] == {"C10_nm": -4.0, "C12_nm": 0.25, "phi12_deg": 12.0}
    assert (saved["scan_detector_rotation_deg"], saved["detector_transposed"], saved["flip_phase"]) == (10.0, False, False)
    assert (saved["notes"], saved["scan_region"]) == ("lamella A", [0, SCAN, 0, SCAN])
    # one [row, col] pair for the scan step and one spelling for the detector step, filled from the session
    assert (saved["voltage_kV"], saved["semiangle_mrad"], saved["scan_sampling_A"], saved["detector_sampling_mrad"]) == (300.0, 20.0, [0.5, 0.5], [5.0, 5.0])
    assert saved["source_path"] == str(tmp_path / "scan_master.h5") and saved["loss"] == json.loads(widget.result_json)["loss"]
    assert saved["sample_tilt_mrad"] is None and saved["generated_at_utc"] == widget.calibration_saved_at and saved["generated_at_utc"].endswith("Z")
    assert saved["upsampling_factor"] == widget.upsampling_factor
    retired = {"schema_version", "version", "aberration_unit", "rotation_angle_deg", "higher_order", "det_sampling_mrad_px",
               "source_file", "source_stem", "label", "timestamp", "tilt_mrad", "tilt_object_mrad"}
    assert not retired & set(saved)
    assert ssb.aberrations == {"C10_nm": -4.0, "C12_nm": 0.25, "phi12_deg": 12.0} and ssb.best_loss == saved["loss"]


def test_freezing_keeps_the_saved_phase_without_owning_the_session(widget, ssb):
    widget.upsampling_factor = 2
    phase = _shown(widget).copy()
    widget.freeze()
    assert not widget.ssb_session and not widget.crop_refit_available
    assert widget.object_sampling_A == [0.25, 0.25]
    np.testing.assert_array_equal(_shown(widget), phase)
    widget.flip_phase = True
    np.testing.assert_array_equal(_shown(widget), -phase)
    # A second consumer can still use the session: freeze detaches the viewer.
    expected, _ = ssb.preview(dict(ssb.aberrations), upsampling_factor=2)
    np.testing.assert_array_equal(phase, expected)


def test_pins_star_view_and_unpin(widget, tmp_path):
    first = _shown(widget).copy()
    widget.pin_json = json.dumps({"action": "pin", "id": "a", "C10_nm": 3.0, "C12_nm": 1.0, "phi12_deg": 11.0, "loss": 0.1})
    widget.request_json = json.dumps({"id": 2, "C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0, "committed": True})
    widget.pin_json = json.dumps({"action": "pin", "id": "b", "C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0, "loss": 0.2})
    assert repr(widget).startswith("ShowPtycho(2 pinned")
    widget.pin_json = json.dumps({"action": "star", "id": "a"})
    stars = json.loads((tmp_path / "showptycho_stars.json").read_text())
    assert stars["schema"] == "quantem.ssb-stars/1" and [star["id"] for star in stars["stars"]] == ["a"]
    star = stars["stars"][0]
    # a star is the calibration file's record: the same keys, units and physical rotation
    assert star["aberrations"] == {"C10_nm": 3.0, "C12_nm": 1.0, "phi12_deg": 11.0}
    assert (star["scan_detector_rotation_deg"], star["loss"], star["source_path"]) == (10.0, 0.1, str(tmp_path / "scan_master.h5"))
    assert (star["scan_sampling_A"], star["detector_sampling_mrad"]) == ([0.5, 0.5], [5.0, 5.0])
    widget.pin_json = json.dumps({"action": "view", "id": "a"})
    np.testing.assert_array_equal(_shown(widget), first)
    assert json.loads(widget.result_json)["C10_nm"] == 3.0
    widget.pin_json = json.dumps({"action": "unpin", "id": "a"})
    assert json.loads((tmp_path / "showptycho_stars.json").read_text()) == {"schema": "quantem.ssb-stars/1", "stars": []}
    assert repr(widget).startswith("ShowPtycho(1 pinned")


def test_export_writes_the_browser_folder_from_exact_bf_counts(widget, ssb, tmp_path):
    out = widget.export(tmp_path / "review", title="my sample SSB")
    assert {p.name for p in out.iterdir()} == {"index.html", "source", "snapshots", ".viewer", "ShowPtycho.command"}
    suffix = "qem" if ssb.backend == "cuda" else "u16"
    assert sorted(p.name for p in (out / "source").iterdir()) == [f"bf_columns.{suffix}"]
    cal = json.loads((out / "snapshots" / "cal.json").read_text())
    assert cal["schema"] == "quantem.ssb-browser-calibration/2" and cal["source_path"] == "redacted_local_source"
    assert cal["precision"] == {"real_dtype": "float32", "complex_dtype": "complex64"}  # the browser kernels refuse any other
    assert (cal["num_bright_field_pixels"], cal["g_shape"], cal["phase_shape"], cal["detector_shape"]) == (ssb.num_bright_field_pixels, [ssb.num_bright_field_pixels, SCAN, SCAN], [SCAN, SCAN], [DET, DET])
    # each value once, in the vocabulary's unit: no second copy in radians, no folded angle
    assert cal["aberrations"] == {"C10_nm": 3.0, "C12_nm": 1.0, "phi12_deg": pytest.approx(math.degrees(0.2))}
    assert (cal["scan_detector_rotation_deg"], cal["detector_transposed"], cal["scan_sampling_A"], cal["voltage_kV"], cal["gpu_memory_gb"]) == (10.0, False, [0.5, 0.5], 300.0, 4.5)
    assert cal["detector_sampling_mrad"] == pytest.approx([5.0, 5.0]) and cal["scan_region"] == [0, SCAN, 0, SCAN]
    state = ssb.browser_state()
    # the reciprocal arrays keep their values under axis names: quantem.gpu's x is the scan row axis
    np.testing.assert_array_equal(np.asarray(cal["bright_field_k_row"], np.float32), np.asarray(state.kx_bf, np.float32))
    np.testing.assert_array_equal(np.asarray(cal["scan_q_col"], np.float32), np.asarray(state.qy_1d, np.float32))
    retired = {"schema_version", "kind", "source_file", "num_bf", "bf_center", "bf_radius_px", "kx_bf", "ky_bf", "qx_1d", "qy_1d",
               "semiangle_rad", "sampling_A", "angular_sampling_rad", "det_sampling_mrad_px", "rotation_angle_deg", "rotation_angle_rad"}
    assert not retired & set(cal)
    assert cal["bf_column_companion_path"] == f"source/bf_columns.{suffix}" and cal["persistent_bf_cache"] is False
    manifest = json.loads((out / "snapshots" / "manifest.json").read_text())
    assert manifest["schema"] == "quantem.ssb-browser-folder/3" and manifest["title"] == "my sample SSB"
    assert manifest["source"]["bf_columns"]["scan_shape"] == [SCAN, SCAN]
    assert json.loads((out / "snapshots" / "snapshots.json").read_text()) == {"schema": "quantem.ssb-snapshots/1", "snapshots": []}
    html = (out / "index.html").read_text()
    embedded = json.loads(re.search(r'<script type="application/vnd.jupyter.widget-state\+json">(.*?)</script>', html, re.S).group(1))
    exported = next(model["state"] for model in embedded["state"].values() if "webgpu_standalone" in model["state"])
    assert exported["webgpu_standalone"] and exported["webgpu_preview_enabled"] and not exported["crop_refit_available"]
    assert exported["auto_upsampling_factor"] == 4 and widget.auto_upsampling_factor == 3
    assert exported["webgpu_preview_status"].startswith("WebGPU folder ready") and widget.webgpu_preview_status == ""
    source = json.loads(exported["webgpu_h5_source_json"])
    assert source["kind"] == "bf_columns" and exported["stars_path"] == "snapshots/snapshots.json"
    assert (source["num_bright_field_pixels"], source["scan_shape"]) == (ssb.num_bright_field_pixels, [SCAN, SCAN]) and "numBf" not in source
    assert str(tmp_path) not in html and "scan_master.h5" not in html
    assert widget.crop_refit_available and not widget.webgpu_standalone and len(widget.phase_bytes) == 4 * SCAN * SCAN
    counts_file = (out / "source" / f"bf_columns.{suffix}").read_bytes()
    widget.export(out)                                                     # a re-export reuses its own counts
    assert (out / "source" / f"bf_columns.{suffix}").read_bytes() == counts_file


def test_crop_refit_validates_the_region_and_a_session_keeps_its_own_sampling(widget, ssb):
    widget.crop_refit_request_json = json.dumps({"scan_region": [0, 16, 0, 16], "num_trials": 1})
    assert widget.crop_refit_status.startswith("Crop refit failed: SSB refit crops must be at least 32x32")
    widget.crop_refit_request_json = json.dumps({"scan_region": [0, 64, 100, 164], "num_trials": 1})
    assert "outside 128x128 data" in widget.crop_refit_status
    assert widget.ssb_session and widget.session_note == "" and widget.amplitude_bytes == b""
    with pytest.raises(ValueError, match="scan_sampling_A"):
        ShowPtycho(ssb, sampling=1.0)
    # the session is the source of its file: the keyword that named it again is gone
    with pytest.raises(TypeError, match="source_file"):
        ShowPtycho(ssb, source_file="scan_master.h5")


def test_session_scale_bar_uses_the_column_scan_step():
    # D-1 (ShowPtycho): an SSB session gave the scale bar its row step; the array path uses the column.
    gpu = pytest.importorskip("quantem.gpu")
    try:
        pytest.importorskip("quantem.gpu.device").detect()
    except RuntimeError as exc:
        pytest.skip(f"SSB needs CUDA or MPS: {exc}")
    session = gpu.SSB(_counts(SCAN), voltage_kV=300.0, semiangle_mrad=20.0, scan_sampling_A=(0.5, 0.25), detector_sampling_mrad=(5.0, 5.0),
                      aberrations=dict(START), scan_detector_rotation_deg=10.0)
    # the trait holds both steps; the browser's horizontal bar reads the column one
    assert ShowPtycho(session).scan_sampling_A == [0.5, 0.25]


def test_a_dataset_in_the_format_unit_draws_its_scale_bar():
    # a Dataset2d whose unit is "angstrom" (what a .qem file writes) opened with no scale bar and no message
    from quantem.widget.adapters.core import make_dataset

    phase = np.zeros((16, 16), np.float32)
    assert ShowPtycho(make_dataset(phase, sampling=(0.2, 0.3), units=("angstrom", "angstrom"))).object_sampling_A == pytest.approx([0.2, 0.3])
    assert ShowPtycho(make_dataset(phase, sampling=(0.05, 0.05), units=("nm", "nm"))).object_sampling_A == pytest.approx([0.5, 0.5])
    assert ShowPtycho(make_dataset(phase)).object_sampling_A == []  # an uncalibrated dataset is absent
    # the dataset is the source: a keyword may repeat its pixel size, never contradict it
    assert ShowPtycho(make_dataset(phase, sampling=(0.2, 0.2), units=("angstrom", "angstrom")), sampling=0.02, units="nm").object_sampling_A == pytest.approx([0.2, 0.2])
    with pytest.raises(ValueError, match="contradicts the value the input records"):
        ShowPtycho(make_dataset(phase, sampling=(0.2, 0.2), units=("angstrom", "angstrom")), sampling=0.3)


def test_the_rewrite_script_carries_earlier_calibration_files_over(tmp_path):
    # the package reads one layout; a calibration a person chose is carried over once by scripts/rewrite_ssb_calibration.py
    script = Path(__file__).resolve().parents[2] / "scripts" / "rewrite_ssb_calibration.py"
    saved = tmp_path / "calibration.json"
    saved.write_text(json.dumps({
        "schema_version": 3, "version": "2.0", "aberration_unit": "nm", "rotation_angle_deg": 345.5,
        "aberrations": {"C10": -4.0, "C12": 0.25, "phi12": math.radians(12.0), "C21_mag": 5.0, "C21_angle": 45.0, "C30": 2.0},
        "flip_phase": True, "voltage_kV": 300.0, "semiangle_mrad": 20.0, "scan_sampling_A": 0.5, "loss": 0.1,
        "source_file": "scan_master.h5", "scan_region": [0, 128, 0, 128], "notes": "lamella A", "id": "abc",
        "tilt_mrad": [3.0, -4.0], "tilt_object_mrad": [2.3, -4.5], "depth_spread_nm": 12.0, "timestamp": "2026-10-01T10:00:00",
    }))
    fitted = tmp_path / "scan_128x128x16x16_ssb_calibration.json"
    fitted.write_text(json.dumps({
        "schema": "quantem.ssb.calibration.v1", "created_utc": "2026-10-01T10:00:00Z",
        "source": {"widget": "Show4DSTEM", "title": "scan", "frame_idx": 0, "scan_shape": [128, 128], "detector_shape": [16, 16]},
        "aberrations": {"C10": 12.0, "C12": 3.0, "phi12": math.pi / 6}, "aberration_units": {"C10": "nm", "C12": "nm", "phi12": "rad"},
        "rotation_angle_deg": 165.5, "com_reversed": True,
        "calibration": {"voltage_kV": 300.0, "semiangle_mrad": 20.0, "scan_sampling_A": [0.4, 0.6], "det_sampling_mrad": [1.5, 1.5],
                        "bf_radius": None, "bf_intensity_threshold": 0.5, "bf_pixels": 52},
        "run": {"n_trials": 200, "refine": True, "manual_locked": False, "seed": 42, "loss": 0.25, "elapsed_s": 1.5},
    }))
    before = saved.read_text()
    dry = subprocess.run([sys.executable, str(script), "--dry-run", str(saved)], capture_output=True, text=True, check=True)
    assert saved.read_text() == before and "would be rewritten" in dry.stdout and "row step 0.5" in dry.stdout
    subprocess.run([sys.executable, str(script), str(saved), str(fitted)], capture_output=True, text=True, check=True)
    current = json.loads(saved.read_text())
    assert current["schema"] == "quantem.ssb-calibration/4" and (tmp_path / "calibration.json.before").read_text() == before
    assert current["aberrations"] == pytest.approx({"C10_nm": -4.0, "C12_nm": 0.25, "phi12_deg": 12.0, "C21_nm": 5.0, "phi21_deg": 45.0, "C30_nm": 2.0})
    assert (current["scan_detector_rotation_deg"], current["scan_sampling_A"], current["source_path"]) == (345.5, [0.5, 0.5], "scan_master.h5")
    assert (current["sample_tilt_mrad"], current["propagator_tilt_mrad"], current["depth_spread_nm"]) == ([3.0, -4.0], [2.3, -4.5], 12.0)
    current = json.loads(fitted.read_text())
    # the angle below 180 degrees and its reversal flag become the one physical angle
    assert (current["scan_detector_rotation_deg"], current["detector_sampling_mrad"], current["num_bright_field_pixels"]) == (345.5, [1.5, 1.5], 52)
    assert current["aberrations"] == pytest.approx({"C10_nm": 12.0, "C12_nm": 3.0, "phi12_deg": 30.0}) and current["source"]["acquisition_index"] == 0
    again = subprocess.run([sys.executable, str(script), str(saved)], capture_output=True, text=True, check=True)
    assert "already in the current layout" in again.stdout
    # before the nm marker the numbers were Angstrom under an nm label: such a file is refused, not guessed
    unmarked = tmp_path / "older.json"
    unmarked.write_text(json.dumps({"schema_version": 3, "rotation_angle_deg": 10.0, "aberrations": {"C10": -40.0}}))
    refused = subprocess.run([sys.executable, str(script), str(unmarked)], capture_output=True, text=True)
    assert refused.returncode != 0 and "unit of C10 and C12 in this file is not known" in refused.stderr


def test_auto_sampling_matches_core_and_save_keeps_resolved_factor(widget, ssb, monkeypatch):
    """Native -> Auto -> native preserves optics, physical extent and saved sampling."""
    from quantem.gpu.optics.physics import ssb_upsampling_factor

    factor = ssb_upsampling_factor(voltage_kV=ssb.voltage_kV,
                                   semiangle_mrad=ssb.semiangle_mrad,
                                   scan_sampling_A=ssb.scan_sampling_A)
    widget.sample_json = json.dumps({"sample_tilt_mrad": [3.0, -4.0], "depth_spread_nm": 12.0})
    native = _shown(widget).copy()
    packets = []
    send_state = widget.send_state

    def record_state(key=None):
        packets.append({key} if isinstance(key, str) else set(key or ()))
        return send_state(key)

    monkeypatch.setattr(widget, "send_state", record_state)
    widget.upsampling_factor = "auto"
    image_keys = {"phase_height", "phase_width", "phase_bytes"}
    assert any(image_keys <= keys for keys in packets)
    assert not any(keys & image_keys and not image_keys <= keys for keys in packets)
    assert widget.auto_upsampling_factor == factor == 3
    expected, _ = ssb.preview(dict(START), upsampling_factor=factor,
                              sample_tilt_mrad=(3.0, -4.0), depth_spread_nm=12.0,
                              compute_loss=False)
    np.testing.assert_array_equal(_shown(widget), expected)
    assert _shown(widget).shape == tuple(value * factor for value in native.shape)
    widget.save_trigger += 1
    saved = json.loads(Path(widget.calibration_path).read_text())
    assert saved["upsampling_factor"] == factor and saved["upsampling_mode"] == "auto"
    assert saved["scan_sampling_A"] == list(ssb.scan_sampling_A)
    widget.upsampling_factor = 1
    np.testing.assert_array_equal(_shown(widget), native)
    widget.save_trigger += 1
    saved = json.loads(Path(widget.calibration_path).read_text())
    assert saved["upsampling_factor"] == 1 and saved["upsampling_mode"] == "explicit"


@pytest.mark.parametrize("estimator", ["mean_phase", "phase_of_mean"])
def test_explicit_estimator_is_displayed_and_saved(ssb, tmp_path, estimator):
    """A saved initialization must describe the numerical phase reviewed on either GPU."""
    view = ShowPtycho(ssb, save_dir=tmp_path, phase_estimator=estimator)
    try:
        view.upsampling_factor = "auto"
        expected, _ = ssb.preview(
            dict(ssb.aberrations), upsampling_factor=view.auto_upsampling_factor,
            phase_estimator=estimator, compute_loss=True,
        )
        np.testing.assert_array_equal(_shown(view), expected)
        view.flip_phase = True
        view.save_trigger += 1
        record = json.loads(Path(view.calibration_path).read_text())
        with pytest.raises(ValueError, match="does not yet preserve"):
            view.export(tmp_path / "browser")
        assert not (tmp_path / "browser").exists()
        assert record["phase_estimator"] == estimator
        assert record["upsampling_factor"] == view.auto_upsampling_factor
        np.testing.assert_array_equal(_shown(view), -expected)
    finally:
        view.close()


def test_estimator_requires_a_supported_name_and_an_ssb_session():
    with pytest.raises(ValueError, match="phase_estimator must"):
        ShowPtycho(np.ones((8, 8)), phase_estimator="unknown")
    with pytest.raises(ValueError, match="needs an SSB session"):
        ShowPtycho(np.ones((8, 8)), phase_estimator="mean_phase")
