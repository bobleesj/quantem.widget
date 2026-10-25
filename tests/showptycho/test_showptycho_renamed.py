"""ShowPtycho's retired spellings: phase_regularization, amplitude_regularization, upsampling_factor and calibration /6.

The strengths are now phase_damping and amplitude_damping and the output factor is upsampling. Old keywords
warn and reach the same reconstruction bit for bit; both spellings at once are refused; a calibration.json
saved in the /6 layout (old key names) restores exactly what its /7 rewrite restores.
"""

import json

import numpy as np
import pytest
import torch

from quantem.widget import ShowPtycho
from tests.showptycho.test_showptycho import START, _counts, _shown

OPTIONS = dict(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5),
               aberrations=START, scan_detector_rotation_deg=10)
RETIRED = {"upsampling": "upsampling_factor", "phase_damping": "phase_regularization",
           "amplitude_damping": "amplitude_regularization"}


def _retired_keys(value):
    """``value`` with every current key under its retired name: the record an earlier release wrote."""
    if isinstance(value, dict):
        return {RETIRED.get(key, key): _retired_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_retired_keys(item) for item in value]
    return value


@pytest.fixture
def cuda_source(tmp_path):
    gpu = pytest.importorskip("quantem.gpu")
    if not torch.cuda.is_available():
        pytest.skip("Direct reconstruction methods currently require CUDA")
    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    return gpu, source


@pytest.mark.parametrize(("current", "retired"), [
    ({"method": "complex", "phase_damping": 0.07, "amplitude_damping": 0.15},
     {"method": "complex", "phase_regularization": 0.07, "amplitude_regularization": 0.15}),
    ({"upsampling": 2}, {"upsampling_factor": 2}),
])
def test_retired_keywords_show_the_identical_reconstruction(cuda_source, tmp_path, current, retired):
    gpu, source = cuda_source
    with gpu.DirectPtycho.open(source, **OPTIONS) as ptycho:
        new = ShowPtycho(ptycho, save_dir=tmp_path, restore=False, **current)
        expected = _shown(new).copy()
        state = {name: getattr(new, name) for name in ("method", "phase_damping", "amplitude_damping", "upsampling")}
        new.close()
        with pytest.warns(DeprecationWarning, match="is deprecated; use"):
            old = ShowPtycho(ptycho, save_dir=tmp_path, restore=False, **retired)
        np.testing.assert_array_equal(_shown(old), expected)
        assert {name: getattr(old, name) for name in state} == state
        old.close()


def test_both_spellings_of_one_keyword_are_refused():
    with pytest.raises(TypeError, match="both phase_regularization= and phase_damping="):
        ShowPtycho(np.zeros((8, 8), np.float32), phase_damping=0.1, phase_regularization=0.1)
    with pytest.raises(TypeError, match="both upsampling_factor= and upsampling="):
        ShowPtycho(np.zeros((8, 8), np.float32), upsampling=2, upsampling_factor=2)


def test_a_calibration_saved_in_the_previous_layout_restores_the_same_review(cuda_source, tmp_path):
    """A quantem.ssb-calibration/6 file (upsampling_factor, *_regularization) restores as its /7 twin does."""
    gpu, source = cuda_source
    with gpu.DirectPtycho.open(source, **OPTIONS) as ptycho:
        viewer = ShowPtycho(ptycho, method="complex", phase_damping=0.07, amplitude_damping=0.15, upsampling=2,
                            save_dir=tmp_path, restore=False)
        viewer.save_calibration()
        viewer.close()
    path = tmp_path / "calibration.json"
    current = json.loads(path.read_text())
    assert current["schema"] == "quantem.ssb-calibration/7" and current["upsampling"] == 2
    assert {"phase_damping", "amplitude_damping"} <= set(current["reconstruction"])
    restored = {}
    for layout, record in (("/7", current), ("/6", {**_retired_keys(current), "schema": "quantem.ssb-calibration/6"})):
        path.write_text(json.dumps(record))
        with gpu.DirectPtycho.open(source, **OPTIONS) as reopened:
            shown = ShowPtycho(reopened, save_dir=tmp_path)
            assert shown.calibration_status.startswith("Restored saved"), (layout, shown.calibration_status)
            restored[layout] = ({name: getattr(shown, name) for name in
                                 ("method", "phase_damping", "amplitude_damping", "upsampling")}, _shown(shown).copy())
            shown.close()
    assert restored["/6"][0] == restored["/7"][0] == {
        "method": "complex", "phase_damping": 0.07, "amplitude_damping": 0.15, "upsampling": 2}
    np.testing.assert_array_equal(restored["/6"][1], restored["/7"][1])
