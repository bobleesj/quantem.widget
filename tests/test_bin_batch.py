import json
import numpy as np
import pytest

import quantem.widget.bin_batch as bin_batch
from quantem.widget.bin_batch import BinPreset, apply_preset_to_array, load_preset, run_batch, save_preset

def test_apply_preset_to_4d_array_shape_and_values():
    data = np.ones((4, 4, 8, 8), dtype=np.float32)
    preset = BinPreset(scan_bin_row=2, scan_bin_col=2, det_bin_row=4, det_bin_col=2, bin_mode="sum", edge_mode="crop")

    out = apply_preset_to_array(data, preset)

    assert out.shape == (2, 2, 2, 4)
    # Each block sums 2*2*4*2 = 32 ones
    assert np.allclose(out, 32.0)

def test_apply_preset_to_5d_array_keeps_time_axis_first():
    data = np.ones((3, 4, 4, 8, 8), dtype=np.float32)
    preset = BinPreset(scan_bin_row=2, scan_bin_col=2, det_bin_row=2, det_bin_col=2, bin_mode="mean", edge_mode="crop", time_axis=0)

    out = apply_preset_to_array(data, preset)

    assert out.shape == (3, 2, 2, 4, 4)
    assert np.allclose(out, 1.0)

def test_preset_rejects_non_torch_backend():
    with pytest.raises(ValueError):
        BinPreset(backend="numpy").validate()

def test_apply_preset_rejects_non_torch_backend_without_manual_validate():
    data = np.ones((4, 4, 4, 4), dtype=np.float32)
    preset = BinPreset(backend="numpy")
    with pytest.raises(ValueError):
        apply_preset_to_array(data, preset)

def test_apply_preset_torch_backend_cpu():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    preset = BinPreset(
        scan_bin_row=2,
        scan_bin_col=2,
        det_bin_row=2,
        det_bin_col=2,
        backend="torch",
        device="cpu",
    )
    out = apply_preset_to_array(data, preset)
    assert out.shape == (2, 2, 4, 4)

def test_preset_save_and_load_roundtrip(tmp_path):
    preset = BinPreset(scan_bin_row=2, scan_bin_col=3, det_bin_row=4, det_bin_col=5, bin_mode="mean", edge_mode="pad", scan_shape=(8, 8), npz_key="arr")

    path = tmp_path / "preset.json"
    save_preset(path, preset)
    loaded = load_preset(path)

    assert loaded == preset

def test_load_preset_from_versioned_widget_envelope(tmp_path):
    preset = BinPreset(scan_bin_row=2, scan_bin_col=3, det_bin_row=4, det_bin_col=5, bin_mode="mean", edge_mode="pad")
    payload = {
        "metadata_version": "1.0",
        "widget_name": "Bin",
        "widget_version": "test",
        "state": preset.to_dict(),
    }
    path = tmp_path / "preset_envelope.json"
    path.write_text(json.dumps(payload, indent=2))

    loaded = load_preset(path)
    assert loaded == preset

def test_run_batch_over_folder_npy(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    a = np.random.rand(4, 4, 8, 8).astype(np.float32)
    b = np.random.rand(2, 6, 10, 10).astype(np.float32)
    np.save(input_dir / "a.npy", a)
    np.save(input_dir / "b.npy", b)

    preset = BinPreset(scan_bin_row=2, scan_bin_col=2, det_bin_row=2, det_bin_col=2, bin_mode="mean", edge_mode="crop")
    results = run_batch(input_dir, output_dir, preset, pattern="*.npy", recursive=False)

    ok = [r for r in results if r.get("status") == "ok"]
    assert len(ok) == 2

    out_a = np.load(output_dir / "a_binned.npy")
    out_b = np.load(output_dir / "b_binned.npy")

    assert out_a.shape == (2, 2, 4, 4)
    assert out_b.shape == (1, 3, 5, 5)

    manifest = output_dir / "bin_batch_manifest.jsonl"
    assert manifest.exists()
    lines = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert len(lines) == 2

def test_run_batch_over_folder_npz(tmp_path):
    input_dir = tmp_path / "in_npz"
    output_dir = tmp_path / "out_npz"
    input_dir.mkdir()

    arr = np.ones((4, 4, 4, 4), dtype=np.float32)
    np.savez_compressed(input_dir / "sample.npz", data=arr)

    preset = BinPreset(scan_bin_row=2, scan_bin_col=2, det_bin_row=2, det_bin_col=2, npz_key="data")
    results = run_batch(input_dir, output_dir, preset, pattern="*.npz")

    assert len(results) == 1
    assert results[0]["status"] == "ok"

    out = np.load(output_dir / "sample_binned.npz")
    assert "data" in out.files
    assert out["data"].shape == (2, 2, 2, 2)

def test_run_batch_retries_and_progress_fields(tmp_path, monkeypatch):
    input_dir = tmp_path / "in_retry"
    output_dir = tmp_path / "out_retry"
    input_dir.mkdir()

    arr = np.random.rand(4, 4, 8, 8).astype(np.float32)
    np.save(input_dir / "first.npy", arr)
    np.save(input_dir / "second.npy", arr)

    preset = BinPreset(device="cpu")

    attempts = {"first.npy": 0, "second.npy": 0}
    original_process_file = bin_batch.process_file

    def flaky_process_file(input_path, output_path, preset_obj, **kwargs):
        name = input_path.name
        attempts[name] += 1
        if name == "first.npy" and attempts[name] == 1:
            raise RuntimeError("transient failure")
        return original_process_file(input_path, output_path, preset_obj, **kwargs)

    monkeypatch.setattr(bin_batch, "process_file", flaky_process_file)

    seen = []
    results = run_batch(
        input_dir,
        output_dir,
        preset,
        pattern="*.npy",
        max_retries=1,
        progress_callback=lambda row: seen.append((row["job_index"], row["status"])),
    )

    assert len(results) == 2
    assert all(r.get("job_index") in {1, 2} for r in results)
    assert all("eta_sec" in r for r in results)
    assert all("progress_pct" in r for r in results)
    assert all("runtime" in r for r in results)
    first_row = next(r for r in results if r["input"].endswith("first.npy"))
    assert first_row["status"] == "ok"
    assert first_row["attempt"] == 2
    assert first_row["retries_used"] == 1
    assert seen == [(1, "ok"), (2, "ok")]

def test_run_batch_fail_fast_stops_queue(tmp_path, monkeypatch):
    input_dir = tmp_path / "in_failfast"
    output_dir = tmp_path / "out_failfast"
    input_dir.mkdir()

    arr = np.random.rand(4, 4, 8, 8).astype(np.float32)
    np.save(input_dir / "first.npy", arr)
    np.save(input_dir / "second.npy", arr)

    preset = BinPreset(device="cpu")

    def always_fail(*args, **kwargs):
        raise RuntimeError("hard failure")

    monkeypatch.setattr(bin_batch, "process_file", always_fail)

    results = run_batch(
        input_dir,
        output_dir,
        preset,
        pattern="*.npy",
        max_retries=0,
        fail_fast=True,
    )

    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert results[0]["job_index"] == 1

# --- HDF5/EMD support tests ---

def test_process_file_h5_4d(tmp_path):
    h5py = pytest.importorskip("h5py")

    input_path = tmp_path / "data.h5"
    output_path = tmp_path / "data_binned.npy"
    data = np.random.rand(4, 6, 8, 10).astype(np.float32)

    with h5py.File(input_path, "w") as f:
        f.create_dataset("data", data=data)

    preset = BinPreset(scan_bin_row=2, scan_bin_col=2, det_bin_row=2, det_bin_col=2, device="cpu")
    result = bin_batch.process_file(input_path, output_path, preset)

    assert result["status"] == "ok"
    out = np.load(output_path)
    assert out.shape == (2, 3, 4, 5)

def test_process_file_h5_explicit_dataset_path(tmp_path):
    h5py = pytest.importorskip("h5py")

    input_path = tmp_path / "multi.h5"
    output_path = tmp_path / "multi_binned.npy"

    with h5py.File(input_path, "w") as f:
        f.create_dataset("preview/thumb", data=np.zeros((10, 10), dtype=np.float32))
        f.create_dataset("experiment/diffraction", data=np.ones((4, 4, 8, 8), dtype=np.float32))

    preset = BinPreset(
        scan_bin_row=2, scan_bin_col=2, det_bin_row=2, det_bin_col=2,
        h5_dataset_path="experiment/diffraction", device="cpu",
    )
    result = bin_batch.process_file(input_path, output_path, preset)

    assert result["status"] == "ok"
    out = np.load(output_path)
    assert out.shape == (2, 2, 4, 4)
    assert np.allclose(out, 1.0)

def test_process_file_emd_extension(tmp_path):
    h5py = pytest.importorskip("h5py")

    input_path = tmp_path / "scan.emd"
    output_path = tmp_path / "scan_binned.npy"
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)

    with h5py.File(input_path, "w") as f:
        f.create_dataset("data/frames", data=data)

    preset = BinPreset(scan_bin_row=2, scan_bin_col=2, det_bin_row=2, det_bin_col=2, device="cpu")
    result = bin_batch.process_file(input_path, output_path, preset)

    assert result["status"] == "ok"
    out = np.load(output_path)
    assert out.shape == (2, 2, 4, 4)

def test_run_batch_h5_folder(tmp_path):
    h5py = pytest.importorskip("h5py")

    input_dir = tmp_path / "in_h5"
    output_dir = tmp_path / "out_h5"
    input_dir.mkdir()

    for name in ["a.h5", "b.h5"]:
        data = np.random.rand(4, 4, 8, 8).astype(np.float32)
        with h5py.File(input_dir / name, "w") as f:
            f.create_dataset("data", data=data)

    preset = BinPreset(scan_bin_row=2, scan_bin_col=2, det_bin_row=2, det_bin_col=2, device="cpu")
    results = run_batch(input_dir, output_dir, preset, pattern="*.h5")

    ok = [r for r in results if r.get("status") == "ok"]
    assert len(ok) == 2
    assert (output_dir / "a_binned.npy").exists()
    assert (output_dir / "b_binned.npy").exists()

def test_preset_h5_dataset_path_roundtrip(tmp_path):
    preset = BinPreset(
        scan_bin_row=2, det_bin_row=4, h5_dataset_path="experiment/stem_data",
    )
    path = tmp_path / "preset_h5.json"
    save_preset(path, preset)
    loaded = load_preset(path)
    assert loaded.h5_dataset_path == "experiment/stem_data"
    assert loaded == preset

def test_find_best_h5_dataset_prefers_4d(tmp_path):
    h5py = pytest.importorskip("h5py")
    from quantem.widget.bin_batch import _find_best_h5_dataset

    input_path = tmp_path / "multi_dim.h5"
    with h5py.File(input_path, "w") as f:
        f.create_dataset("spec_1d", data=np.zeros((100,), dtype=np.float32))
        f.create_dataset("image_3d", data=np.zeros((10, 64, 64), dtype=np.float32))
        f.create_dataset("stem_4d", data=np.zeros((4, 4, 32, 32), dtype=np.float32))

    with h5py.File(input_path, "r") as f:
        ds_path, ds = _find_best_h5_dataset(f)

    assert ds_path == "stem_4d"
    assert ds is not None
