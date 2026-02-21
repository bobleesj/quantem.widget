import json
import pathlib

import numpy as np

from quantem.widget.show4dstem_batch import main

def test_show4dstem_batch_cli_single(tmp_path):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    input_path = tmp_path / "sample.npy"
    np.save(input_path, data)

    output_dir = tmp_path / "out_single"
    rc = main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "single",
            "--view",
            "diffraction",
            "--format",
            "png",
            "--quiet-progress",
        ]
    )
    assert rc == 0

    image_path = output_dir / "sample.png"
    assert image_path.exists()

    batch_manifest = output_dir / "show4dstem_batch_manifest.jsonl"
    assert batch_manifest.exists()
    rows = [json.loads(line) for line in batch_manifest.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert pathlib.Path(rows[0]["report_path"]).exists()

def test_show4dstem_batch_cli_frames_sequence(tmp_path):
    data = np.random.rand(3, 4, 4, 16, 16).astype(np.float32)
    input_path = tmp_path / "time.npy"
    np.save(input_path, data)

    output_dir = tmp_path / "out_frames"
    rc = main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "frames",
            "--view",
            "virtual",
            "--format",
            "png",
            "--frame-range",
            "0:2",
            "--quiet-progress",
        ]
    )
    assert rc == 0

    sequence_manifest = output_dir / "save_sequence_manifest.json"
    assert sequence_manifest.exists()
    payload = json.loads(sequence_manifest.read_text())
    assert payload["export_kind"] == "sequence_batch"
    assert payload["n_exports"] == 3
    for row in payload["exports"]:
        assert pathlib.Path(row["path"]).exists()

def test_show4dstem_batch_cli_adaptive(tmp_path):
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    input_path = tmp_path / "adaptive.npy"
    np.save(input_path, data)

    output_dir = tmp_path / "out_adaptive"
    rc = main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "adaptive",
            "--view",
            "all",
            "--format",
            "png",
            "--adaptive-coarse-step",
            "4",
            "--adaptive-target-fraction",
            "0.3",
            "--quiet-progress",
        ]
    )
    assert rc == 0

    sequence_manifest = output_dir / "save_sequence_manifest.json"
    assert sequence_manifest.exists()

    batch_manifest = output_dir / "show4dstem_batch_manifest.jsonl"
    rows = [json.loads(line) for line in batch_manifest.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert "adaptive" in rows[0]
    assert rows[0]["adaptive"]["path_count"] > 0
