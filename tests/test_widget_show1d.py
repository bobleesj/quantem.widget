import json

import numpy as np
import pytest
import torch

from quantem.widget import Show1D


# =========================================================================
# Basic construction
# =========================================================================
def test_show1d_single_numpy():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w = Show1D(data)
    assert w.n_traces == 1
    assert w.n_points == 4
    assert len(w.y_bytes) == 4 * 4


def test_show1d_single_torch():
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    w = Show1D(data)
    assert w.n_traces == 1
    assert w.n_points == 3


def test_show1d_multiple_traces_2d():
    data = np.random.randn(3, 100).astype(np.float32)
    w = Show1D(data)
    assert w.n_traces == 3
    assert w.n_points == 100


def test_show1d_multiple_traces_list():
    t1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    w = Show1D([t1, t2], labels=["A", "B"])
    assert w.n_traces == 2
    assert w.n_points == 3
    assert w.labels == ["A", "B"]


def test_show1d_with_x_data():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    w = Show1D(y, x=x)
    assert len(w.x_bytes) == 3 * 4


def test_show1d_x_length_mismatch_raises():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="points"):
        Show1D(y, x=x)


def test_show1d_list_length_mismatch_raises():
    t1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t2 = np.array([4.0, 5.0], dtype=np.float32)
    with pytest.raises(ValueError, match="same length"):
        Show1D([t1, t2])


def test_show1d_3d_raises():
    data = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="1D or 2D"):
        Show1D(data)


# =========================================================================
# Statistics
# =========================================================================
def test_show1d_stats_single():
    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    w = Show1D(data)
    assert w.stats_mean[0] == pytest.approx(20.0)
    assert w.stats_min[0] == pytest.approx(10.0)
    assert w.stats_max[0] == pytest.approx(30.0)


def test_show1d_stats_multiple():
    t1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t2 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    w = Show1D([t1, t2])
    assert len(w.stats_mean) == 2
    assert w.stats_mean[0] == pytest.approx(2.0)
    assert w.stats_mean[1] == pytest.approx(20.0)


# =========================================================================
# Display options
# =========================================================================
def test_show1d_title():
    w = Show1D(np.ones(5, dtype=np.float32), title="My Plot")
    assert w.title == "My Plot"


def test_show1d_labels_default():
    w = Show1D(np.ones(5, dtype=np.float32))
    assert w.labels == ["Trace"]


def test_show1d_labels_multi_default():
    w = Show1D(np.ones((3, 5), dtype=np.float32))
    assert w.labels == ["Trace 1", "Trace 2", "Trace 3"]


def test_show1d_colors():
    w = Show1D(np.ones(5, dtype=np.float32), colors=["#ff0000"])
    assert w.colors == ["#ff0000"]


def test_show1d_colors_default():
    w = Show1D(np.ones((2, 5), dtype=np.float32))
    assert len(w.colors) == 2
    assert w.colors[0] == "#4fc3f7"
    assert w.colors[1] == "#81c784"


def test_show1d_axis_labels():
    w = Show1D(np.ones(5, dtype=np.float32), x_label="Energy", y_label="Counts", x_unit="eV", y_unit="a.u.")
    assert w.x_label == "Energy"
    assert w.y_label == "Counts"
    assert w.x_unit == "eV"
    assert w.y_unit == "a.u."


def test_show1d_log_scale():
    w = Show1D(np.ones(5, dtype=np.float32), log_scale=True)
    assert w.log_scale is True


def test_show1d_show_grid():
    w = Show1D(np.ones(5, dtype=np.float32), show_grid=False)
    assert w.show_grid is False


def test_show1d_show_legend():
    w = Show1D(np.ones(5, dtype=np.float32), show_legend=False)
    assert w.show_legend is False


def test_show1d_show_stats():
    w = Show1D(np.ones(5, dtype=np.float32), show_stats=False)
    assert w.show_stats is False


def test_show1d_line_width():
    w = Show1D(np.ones(5, dtype=np.float32), line_width=2.5)
    assert w.line_width == pytest.approx(2.5)


# =========================================================================
# Mutation methods
# =========================================================================
def test_show1d_set_data():
    w = Show1D(np.array([1.0, 2.0, 3.0], dtype=np.float32), title="Keep", log_scale=True)
    w.set_data(np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
    assert w.n_points == 4
    assert w.n_traces == 1
    assert w.title == "Keep"
    assert w.log_scale is True


def test_show1d_set_data_with_x():
    w = Show1D(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    w.set_data(np.array([4.0, 5.0, 6.0], dtype=np.float32), x=x)
    assert len(w.x_bytes) == 3 * 4


def test_show1d_add_trace():
    w = Show1D(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert w.n_traces == 1
    w.add_trace(np.array([4.0, 5.0, 6.0], dtype=np.float32), label="New")
    assert w.n_traces == 2
    assert w.labels[-1] == "New"


def test_show1d_add_trace_length_mismatch():
    w = Show1D(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(ValueError, match="points"):
        w.add_trace(np.array([4.0, 5.0], dtype=np.float32))


def test_show1d_remove_trace():
    w = Show1D(np.ones((3, 5), dtype=np.float32), labels=["A", "B", "C"])
    w.remove_trace(1)
    assert w.n_traces == 2
    assert w.labels == ["A", "C"]


def test_show1d_remove_trace_out_of_range():
    w = Show1D(np.ones(5, dtype=np.float32))
    with pytest.raises(IndexError):
        w.remove_trace(5)


def test_show1d_clear():
    w = Show1D(np.ones((3, 5), dtype=np.float32))
    w.clear()
    assert w.n_traces == 0
    assert w.n_points == 0
    assert w.labels == []
    assert w.colors == []
    assert w.y_bytes == b""


# =========================================================================
# State persistence (required 3 tests)
# =========================================================================
def test_show1d_state_dict_roundtrip():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(
        data,
        title="Test",
        x_label="Energy",
        y_label="Counts",
        x_unit="eV",
        log_scale=True,
        show_grid=False,
        show_legend=False,
        line_width=2.0,
    )
    sd = w.state_dict()
    w2 = Show1D(data, state=sd)
    assert w2.title == "Test"
    assert w2.x_label == "Energy"
    assert w2.y_label == "Counts"
    assert w2.x_unit == "eV"
    assert w2.log_scale is True
    assert w2.show_grid is False
    assert w2.show_legend is False
    assert w2.line_width == pytest.approx(2.0)


def test_show1d_save_load_file(tmp_path):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, title="Save Test", log_scale=True)
    path = tmp_path / "show1d_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "Show1D"
    assert isinstance(saved["widget_version"], str)
    assert saved["state"]["title"] == "Save Test"
    w2 = Show1D(data, state=str(path))
    assert w2.title == "Save Test"
    assert w2.log_scale is True


def test_show1d_summary(capsys):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, title="My Plot", x_label="Distance", x_unit="nm")
    w.summary()
    out = capsys.readouterr().out
    assert "My Plot" in out
    assert "3" in out
    assert "Distance" in out


# =========================================================================
# Repr
# =========================================================================
def test_show1d_repr_single():
    w = Show1D(np.ones(100, dtype=np.float32))
    r = repr(w)
    assert "Show1D" in r
    assert "100" in r


def test_show1d_repr_multi():
    w = Show1D(np.ones((3, 50), dtype=np.float32))
    r = repr(w)
    assert "Show1D" in r
    assert "3" in r
    assert "50" in r


# =========================================================================
# Edge cases
# =========================================================================
def test_show1d_constant_trace():
    data = np.full(100, 5.0, dtype=np.float32)
    w = Show1D(data)
    assert w.stats_mean[0] == pytest.approx(5.0)
    assert w.stats_std[0] == pytest.approx(0.0)


def test_show1d_single_point():
    data = np.array([42.0], dtype=np.float32)
    w = Show1D(data)
    assert w.n_points == 1
    assert w.stats_mean[0] == pytest.approx(42.0)


def test_show1d_dataset_duck_typing():
    class MockDataset:
        def __init__(self):
            self.array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            self.name = "Test Spectrum"

    ds = MockDataset()
    w = Show1D(ds)
    assert w.title == "Test Spectrum"
    assert w.n_points == 3


def test_show1d_set_data_returns_self():
    w = Show1D(np.ones(5, dtype=np.float32))
    result = w.set_data(np.ones(3, dtype=np.float32))
    assert result is w


def test_show1d_add_trace_returns_self():
    w = Show1D(np.ones(5, dtype=np.float32))
    result = w.add_trace(np.ones(5, dtype=np.float32))
    assert result is w


def test_show1d_remove_trace_returns_self():
    w = Show1D(np.ones((2, 5), dtype=np.float32))
    result = w.remove_trace(0)
    assert result is w


def test_show1d_clear_returns_self():
    w = Show1D(np.ones(5, dtype=np.float32))
    result = w.clear()
    assert result is w


def test_show1d_data_bytes_match():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    recovered = np.frombuffer(w.y_bytes, dtype=np.float32)
    np.testing.assert_array_equal(recovered, data)


# =========================================================================
# Widget version / show_controls defaults
# =========================================================================
def test_show1d_widget_version_is_set():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.widget_version != "unknown"


def test_show1d_show_controls_default():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.show_controls is True
