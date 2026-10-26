"""Read the complete saved SSB correction before the widget's first preview."""

import json
import math
from pathlib import Path

from quantem.widget.adapters.gpu import DirectSession
from quantem.widget.showptycho._renamed import current_keys

# the one earlier layout each current one still reads: /6 named upsampling and the damping strengths otherwise
PREVIOUS_SCHEMAS = {"quantem.ssb-calibration/7": "quantem.ssb-calibration/6"}

# quantem.gpu retired these SSB phase estimators for the phase of the mean wave, "phase_of_mean". A saved fit that
# names one still restores its optics; the widget reconstructs them with phase_of_mean and says so once.
RETIRED_PHASE_ESTIMATORS = ("mean_phase", "complex_wave")


def read_calibration(path: Path, identity: dict | None, scan_region: tuple[int, int, int, int], schema: str) -> dict:
    """Require matching evidence and a complete, finite model of the session's whole scan.

    ``scan_region`` is ``(row_start, row_stop, col_start, col_stop)`` of the
    acquisition the session holds, in full-scan positions with exclusive
    stops: the whole scan, or the ``scan_region`` the session was opened on.
    """
    saved = json.loads(path.read_text(encoding="utf-8"))
    found = saved.get("schema") if isinstance(saved, dict) else None
    if found == PREVIOUS_SCHEMAS.get(schema):
        # the same record with upsampling_factor and the *_regularization strengths under their current names
        saved, found = current_keys(saved), schema
    if found != schema:
        rewrite = ("; rewrite it once with scripts/rewrite_ssb_calibration.py from the quantem.widget repository"
                   if isinstance(found, str) and found.startswith("quantem.ssb-calibration/") else "")
        raise ValueError(f"the saved calibration format {found!r} is not ShowPtycho's {schema!r}{rewrite}")
    if identity is None:
        raise ValueError("this input has no persistent acquisition identity")
    if saved.get("session_identity") is None:
        raise ValueError("this older save has no acquisition identity; save the fitted session once")
    if not DirectSession.same_calibration_identity(saved["session_identity"], identity):
        raise ValueError("the acquisition or launch calibration has changed")
    if saved.get("scan_region") != [int(value) for value in scan_region]:
        raise ValueError(f"the saved fit covers scan region {saved.get('scan_region')}, "
                         f"not this session's {[int(value) for value in scan_region]}")
    estimator = saved["phase_estimator"]
    if estimator not in ("phase_of_mean", *RETIRED_PHASE_ESTIMATORS):
        raise ValueError("the saved phase estimator is not supported")
    reconstruction = dict(saved.get("reconstruction") or saved.get("reconstruction_options") or {})
    method = reconstruction.setdefault("method", saved.get("reconstruction_method", "ssb"))
    if method not in ("ssb", "phase", "complex"):
        raise ValueError("the saved reconstruction method is not supported")
    if "reconstruction_method" in saved and saved["reconstruction_method"] != method:
        raise ValueError("the saved reconstruction method and options disagree")
    if method != "ssb":
        allowed = {"method", "phase_damping", "smoothness_inv_A", "detector_weighting"}
        if method == "complex":
            allowed.add("amplitude_damping")
        if set(reconstruction) - allowed:
            raise ValueError("the saved reconstruction contains unsupported options")
        for name in allowed - {"method", "detector_weighting"}:
            value = reconstruction.setdefault(name, 1.0 if name == "smoothness_inv_A" else 0.03)
            if not math.isfinite(float(value)) or (float(value) < 0 if name == "smoothness_inv_A" else float(value) <= 0):
                bound = "nonnegative" if name == "smoothness_inv_A" else "positive"
                raise ValueError(f"the saved {name} must be finite and {bound}")
        if reconstruction.setdefault("detector_weighting", "uniform") not in ("uniform", "inverse_mean"):
            raise ValueError("the saved detector weighting is not supported")
    saved["reconstruction"] = reconstruction
    if saved["upsampling_mode"] not in ("auto", "explicit") or saved["upsampling"] not in (1, 2, 3, 4, 8):
        raise ValueError("the saved output sampling is not supported")
    aberrations = saved["aberrations"]
    numbers = [aberrations[name] for name in ("C10_nm", "C12_nm", "phi12_deg")]
    numbers.extend(aberrations.values())
    numbers.append(saved["scan_detector_rotation_deg"])
    if saved["loss"] is not None:
        numbers.append(saved["loss"])
    tilt, depth = saved["sample_tilt_mrad"], saved["depth_spread_nm"]
    if tilt is not None or depth is not None:
        if not isinstance(tilt, list) or len(tilt) != 2 or depth is None or depth <= 0:
            raise ValueError("the saved tilt and depth must form one complete model")
        numbers.extend([*tilt, depth])
    if not all(math.isfinite(float(value)) for value in numbers):
        raise ValueError("the saved correction contains non-finite values")
    if not isinstance(saved["display_negate_phase"], bool) or not isinstance(saved["display"]["tilt_correction"], bool):
        raise ValueError("the saved display settings are incomplete")
    if saved["display"].get("max_columns", 3) not in range(1, 9):
        raise ValueError("the saved maximum columns must be between 1 and 8")
    return saved
