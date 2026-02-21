import json
import struct

import numpy as np
import pytest

from quantem.widget import Merge4DSTEM

def _make_sources(n=3, shape=(4, 4, 8, 8)):
    return [np.random.rand(*shape).astype(np.float32) for _ in range(n)]

# --- Construction and validation ---

def test_merge4dstem_constructs_from_arrays():
    sources = _make_sources(3)
    w = Merge4DSTEM(sources)
    assert w.n_sources == 3
    assert (w.scan_rows, w.scan_cols) == (4, 4)
    assert (w.det_rows, w.det_cols) == (8, 8)
    assert w.merged is False
    assert w.device in ("cpu", "mps", "cuda")

def test_merge4dstem_rejects_single_source():
    with pytest.raises(ValueError, match="at least 2"):
        Merge4DSTEM([np.random.rand(4, 4, 8, 8).astype(np.float32)])

def test_merge4dstem_rejects_empty_sources():
    with pytest.raises(ValueError, match="at least 2"):
        Merge4DSTEM([])

def test_merge4dstem_rejects_shape_mismatch():
    a = np.random.rand(4, 4, 8, 8).astype(np.float32)
    b = np.random.rand(4, 4, 16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="shape"):
        Merge4DSTEM([a, b])

def test_merge4dstem_rejects_non_4d():
    a = np.random.rand(4, 4, 8, 8).astype(np.float32)
    b = np.random.rand(4, 8, 8).astype(np.float32)
    with pytest.raises(ValueError, match="4D"):
        Merge4DSTEM([a, b])

def test_merge4dstem_requires_torch():
    import unittest.mock as mock

    sources = _make_sources(2)
    with mock.patch.dict("quantem.widget.merge4dstem.__dict__", {"_HAS_TORCH": False}):
        # Need to also patch the module-level variable
        import quantem.widget.merge4dstem as m

        orig = m._HAS_TORCH
        try:
            m._HAS_TORCH = False
            with pytest.raises(ImportError, match="torch"):
                Merge4DSTEM(sources)
        finally:
            m._HAS_TORCH = orig

# --- Merge ---

def test_merge4dstem_merge_produces_5d():
    sources = _make_sources(3, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources, bin_factor=1)
    assert w.merged is False

    result = w.merge()
    assert result is w
    assert w.merged is True

    arr = w.result_array
    assert arr is not None
    assert arr.shape == (3, 2, 2, 4, 4)

def test_merge4dstem_result_is_none_before_merge():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)
    assert w.result_array is None

def test_merge4dstem_result_array_values_correct():
    a = np.ones((2, 2, 4, 4), dtype=np.float32) * 1.0
    b = np.ones((2, 2, 4, 4), dtype=np.float32) * 2.0
    w = Merge4DSTEM([a, b], bin_factor=1)
    w.merge()

    arr = w.result_array
    assert arr is not None
    np.testing.assert_allclose(arr[0], 1.0)
    np.testing.assert_allclose(arr[1], 2.0)

def test_merge4dstem_output_shape_json_updated():
    sources = _make_sources(3, shape=(2, 3, 4, 5))
    w = Merge4DSTEM(sources, bin_factor=1)
    w.merge()

    shape = json.loads(w.output_shape_json)
    assert shape == [3, 2, 3, 4, 5]

# --- Binning ---

def test_merge4dstem_bin_factor_default():
    sources = _make_sources(2, shape=(4, 4, 8, 8))
    w = Merge4DSTEM(sources)
    assert w.bin_factor == 2

def test_merge4dstem_bin_factor_1_no_binning():
    sources = _make_sources(2, shape=(2, 2, 8, 8))
    w = Merge4DSTEM(sources, bin_factor=1)
    w.merge()
    arr = w.result_array
    assert arr.shape == (2, 2, 2, 8, 8)

def test_merge4dstem_bin_factor_2():
    a = np.ones((2, 2, 8, 8), dtype=np.float32) * 4.0
    b = np.ones((2, 2, 8, 8), dtype=np.float32) * 8.0
    w = Merge4DSTEM([a, b], bin_factor=2)
    w.merge()
    arr = w.result_array
    assert arr.shape == (2, 2, 2, 4, 4)
    np.testing.assert_allclose(arr[0], 4.0)
    np.testing.assert_allclose(arr[1], 8.0)

def test_merge4dstem_bin_factor_4():
    sources = _make_sources(2, shape=(2, 2, 16, 16))
    w = Merge4DSTEM(sources, bin_factor=4)
    w.merge()
    arr = w.result_array
    assert arr.shape == (2, 2, 2, 4, 4)

def test_merge4dstem_output_shape_accounts_for_binning():
    sources = _make_sources(3, shape=(4, 4, 8, 8))
    w = Merge4DSTEM(sources, bin_factor=2)
    shape = json.loads(w.output_shape_json)
    assert shape == [3, 4, 4, 4, 4]

def test_merge4dstem_bin_factor_in_state_dict():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources, bin_factor=4)
    sd = w.state_dict()
    assert sd["bin_factor"] == 4

    w2 = Merge4DSTEM(_make_sources(2, shape=(2, 2, 4, 4)))
    w2.load_state_dict(sd)
    assert w2.bin_factor == 4

# --- Preview index ---

def test_merge4dstem_preview_index_default():
    sources = _make_sources(3, shape=(4, 4, 8, 8))
    w = Merge4DSTEM(sources)
    assert w.preview_index == 0

def test_merge4dstem_preview_index_changes_preview():
    a = np.ones((2, 2, 4, 4), dtype=np.float32) * 1.0
    b = np.ones((2, 2, 4, 4), dtype=np.float32) * 100.0
    w = Merge4DSTEM([a, b])

    # Preview of source 0 should be ~1.0
    import struct
    first_val = struct.unpack("f", w.preview_bytes[:4])[0]
    assert first_val == pytest.approx(1.0, abs=0.01)

    # Switch to source 1
    w.preview_index = 1
    first_val_2 = struct.unpack("f", w.preview_bytes[:4])[0]
    assert first_val_2 == pytest.approx(100.0, abs=0.01)

# --- Merge trigger via trait ---

def test_merge4dstem_merge_triggered_by_trait():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)
    assert w.merged is False

    w._merge_requested = True
    assert w.merged is True
    assert w._merge_requested is False

# --- Preview ---

def test_merge4dstem_preview_bytes_populated():
    sources = _make_sources(2, shape=(4, 4, 8, 8))
    w = Merge4DSTEM(sources)
    assert len(w.preview_bytes) == 8 * 8 * 4  # float32
    assert w.preview_rows == 8
    assert w.preview_cols == 8

# --- Source info ---

def test_merge4dstem_source_info_json():
    sources = _make_sources(3, shape=(4, 4, 8, 8))
    w = Merge4DSTEM(sources)

    info = json.loads(w.source_info_json)
    assert len(info) == 3
    for entry in info:
        assert entry["valid"] is True
        assert entry["shape"] == [4, 4, 8, 8]
        assert entry["message"] == "OK"

# --- Calibration ---

def test_merge4dstem_explicit_calibration():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources, pixel_size=2.5, k_pixel_size=0.5)

    assert w.pixel_size == 2.5
    assert w.pixel_unit == "Å"
    assert w.pixel_calibrated is True
    assert w.k_pixel_size == 0.5
    assert w.k_unit == "mrad"
    assert w.k_calibrated is True

def test_merge4dstem_uncalibrated_defaults():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)

    assert w.pixel_size == 1.0
    assert w.pixel_unit == "px"
    assert w.pixel_calibrated is False
    assert w.k_pixel_size == 1.0
    assert w.k_unit == "px"
    assert w.k_calibrated is False

# --- Display traits ---

def test_merge4dstem_display_traits():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources, cmap="viridis", log_scale=True, title="Test")
    assert w.cmap == "viridis"
    assert w.log_scale is True
    assert w.title == "Test"

def test_merge4dstem_frame_dim_label():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources, frame_dim_label="Tilt")
    assert w.frame_dim_label == "Tilt"

# --- State persistence ---

def test_merge4dstem_state_dict_roundtrip():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources, cmap="viridis", log_scale=True, frame_dim_label="Tilt")

    sd = w.state_dict()
    assert sd["cmap"] == "viridis"
    assert sd["log_scale"] is True
    assert sd["frame_dim_label"] == "Tilt"

    w2 = Merge4DSTEM(_make_sources(2, shape=(2, 2, 4, 4)))
    w2.load_state_dict(sd)
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.frame_dim_label == "Tilt"

def test_merge4dstem_save_load_file(tmp_path):
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources, cmap="magma", log_scale=True)

    path = tmp_path / "merge4dstem_state.json"
    w.save(str(path))

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["metadata_version"] == "1.0"
    assert data["widget_name"] == "Merge4DSTEM"
    assert isinstance(data["widget_version"], str)
    assert data["state"]["cmap"] == "magma"

    w2 = Merge4DSTEM(_make_sources(2, shape=(2, 2, 4, 4)), state=str(path))
    assert w2.cmap == "magma"
    assert w2.log_scale is True

def test_merge4dstem_summary(capsys):
    sources = _make_sources(3, shape=(4, 4, 8, 8))
    w = Merge4DSTEM(sources, pixel_size=2.0)
    w.summary()

    output = capsys.readouterr().out
    assert "Merge4DSTEM" in output
    assert "Sources" in output
    assert "3" in output
    assert "scan" in output

# --- save_result ---

def test_merge4dstem_save_result_raises_before_merge(tmp_path):
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)

    with pytest.raises(RuntimeError, match="merge"):
        w.save_result(tmp_path / "out.npz")

def test_merge4dstem_save_result_after_merge(tmp_path):
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)
    w.merge()

    path = tmp_path / "merged.npz"
    result = w.save_result(path)
    assert result == path
    assert path.exists()

# --- to_show4dstem ---

def test_merge4dstem_to_show4dstem_raises_before_merge():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)

    with pytest.raises(RuntimeError, match="merge"):
        w.to_show4dstem()

def test_merge4dstem_to_show4dstem_after_merge():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)
    w.merge()

    from quantem.widget import Show4DSTEM

    viewer = w.to_show4dstem()
    assert isinstance(viewer, Show4DSTEM)

# --- Tool lock/hide ---

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_merge4dstem_tool_default_empty(trait_name):
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)
    assert getattr(w, trait_name) == []

@pytest.mark.parametrize(
    ("trait_name", "kwargs", "expected"),
    [
        ("disabled_tools", {"disabled_tools": ["display", "Sources"]}, ["display", "sources"]),
        ("hidden_tools", {"hidden_tools": ["display", "Sources"]}, ["display", "sources"]),
        ("disabled_tools", {"disable_display": True, "disable_sources": True}, ["display", "sources"]),
        ("hidden_tools", {"hide_display": True, "hide_sources": True}, ["display", "sources"]),
        ("disabled_tools", {"disable_all": True}, ["all"]),
        ("hidden_tools", {"hide_all": True}, ["all"]),
    ],
)
def test_merge4dstem_tool_lock_hide(trait_name, kwargs, expected):
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources, **kwargs)
    assert getattr(w, trait_name) == expected

@pytest.mark.parametrize("kwargs", [{"disabled_tools": ["not_real"]}, {"hidden_tools": ["not_real"]}])
def test_merge4dstem_tool_invalid_key_raises(kwargs):
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    with pytest.raises(ValueError):
        Merge4DSTEM(sources, **kwargs)

def test_merge4dstem_tool_runtime_api():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)

    assert w.lock_tool("display") is w
    assert "display" in w.disabled_tools
    assert w.unlock_tool("display") is w
    assert "display" not in w.disabled_tools

    assert w.hide_tool("sources") is w
    assert "sources" in w.hidden_tools
    assert w.show_tool("sources") is w
    assert "sources" not in w.hidden_tools

def test_merge4dstem_state_dict_includes_tools():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources, disabled_tools=["display"], hidden_tools=["stats"])

    sd = w.state_dict()
    assert sd["disabled_tools"] == ["display"]
    assert sd["hidden_tools"] == ["stats"]

    w2 = Merge4DSTEM(_make_sources(2, shape=(2, 2, 4, 4)))
    w2.load_state_dict(sd)
    assert w2.disabled_tools == ["display"]
    assert w2.hidden_tools == ["stats"]

# --- repr ---

def test_merge4dstem_repr():
    sources = _make_sources(2, shape=(4, 4, 8, 8))
    w = Merge4DSTEM(sources)
    r = repr(w)
    assert "Merge4DSTEM" in r
    assert "sources=2" in r
    assert "bin=2x" in r
    assert "merged=False" in r

# --- Status ---

def test_merge4dstem_status_ready():
    sources = _make_sources(3, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)
    assert w.status_level == "ok"
    assert "3 compatible" in w.status_message

def test_merge4dstem_status_after_merge():
    sources = _make_sources(2, shape=(2, 2, 4, 4))
    w = Merge4DSTEM(sources)
    w.merge()
    assert w.status_level == "ok"
    assert "Merged" in w.status_message
