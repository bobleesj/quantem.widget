"""Rewrite SSB calibration files of an earlier quantem.widget release into the current layout, once.

quantem.widget reads and writes one layout per file kind and converts nothing
inside the package. A calibration a person chose by eye cannot be recomputed,
so this script carries the three earlier file kinds over:

- ``calibration.json`` from ShowPtycho (``schema_version`` 3, ``aberration_unit`` "nm")
- ``showptycho_stars.json`` from ShowPtycho (a list of stars)
- ``*_ssb_calibration.json`` from Show4DSTEM (``schema`` "quantem.ssb.calibration.v1")

    python scripts/rewrite_ssb_calibration.py --dry-run path/to/calibration.json
    python scripts/rewrite_ssb_calibration.py path/to/calibration.json path/to/showptycho_stars.json

``--dry-run`` prints what would be written and changes nothing. Without it each
file is replaced and the earlier one is kept beside it as ``<name>.before``.
A file already in the current layout is left alone. A ShowPtycho file without
``"aberration_unit": "nm"`` is refused: before that marker quantem.gpu reported
Angstrom under the nm label, so the unit of its numbers is not known.
"""

import argparse
import json
import math
import pathlib

CALIBRATION_SCHEMA = "quantem.ssb-calibration/4"
STARS_SCHEMA = "quantem.ssb-stars/1"


def aberrations(earlier: dict, *, higher_order_radians: bool) -> dict:
    """``{C10, C12 (nm), phi12 (rad), ...}`` as ``{C10_nm, C12_nm, phi12_deg, C21_nm, phi21_deg, ...}``.

    ``calibration.json`` held higher orders as ``C21_mag`` (nm) and
    ``C21_angle`` (degrees), the stars file as ``C21`` (nm) and ``phi21``
    (radians); ``higher_order_radians`` says which file this is.
    """
    current = {"C10_nm": float(earlier.get("C10", 0.0)), "C12_nm": float(earlier.get("C12", 0.0)),
               "phi12_deg": math.degrees(float(earlier.get("phi12", 0.0)))}
    for key, value in earlier.items():
        if key in ("C10", "C12", "phi12"):
            continue
        if key.endswith("_mag"):
            current[f"{key[:-4]}_nm"] = float(value)
        elif key.endswith("_angle"):
            current[f"phi{key[1:-6]}_deg"] = float(value)
        elif key.startswith("phi"):
            current[f"{key}_deg"] = math.degrees(float(value)) if higher_order_radians else float(value)
        else:
            current[f"{key}_nm"] = float(value)
    return current


def showptycho_record(earlier: dict, *, higher_order_radians: bool) -> tuple[dict, list[str]]:
    """One ShowPtycho calibration or star in the current keys, with the notes a person should read."""
    if earlier.get("aberration_unit") != "nm":
        raise ValueError('no "aberration_unit": "nm": the unit of C10 and C12 in this file is not known')
    notes = []
    step = earlier.get("scan_sampling_A")
    if isinstance(step, (int, float)):
        notes.append(f"scan_sampling_A held one number, the row step {step}; written as [{step}, {step}]: correct the column step if it differs")
        step = [float(step), float(step)]
    if earlier.get("timestamp"):
        notes.append(f"timestamp {earlier['timestamp']!r} is a local time without a zone and is not carried into generated_at_utc")
    current = {
        "id": earlier.get("id"), "generated_at_utc": None,
        "aberrations": aberrations(earlier["aberrations"], higher_order_radians=higher_order_radians),
        "scan_detector_rotation_deg": float(earlier["rotation_angle_deg"]) % 360.0, "detector_transposed": False,
        "flip_phase": bool(earlier.get("flip_phase", False)), "voltage_kV": earlier.get("voltage_kV"),
        "semiangle_mrad": earlier.get("semiangle_mrad"), "scan_sampling_A": step, "detector_sampling_mrad": None,
        "loss": earlier.get("loss"), "source_path": earlier.get("source_file"),
    }
    if "scan_region" in earlier:
        current["scan_region"] = list(earlier["scan_region"])
    return current, notes


def rewrite(earlier) -> tuple[dict | None, list[str]]:
    """The current layout of one earlier file's content, or None when it is current already."""
    if isinstance(earlier, list):
        records = [showptycho_record(star, higher_order_radians=True) for star in earlier]
        return {"schema": STARS_SCHEMA, "stars": [record for record, _ in records]}, [note for _, notes in records for note in notes]
    if earlier.get("schema") in (CALIBRATION_SCHEMA, STARS_SCHEMA):
        return None, []
    if earlier.get("schema") == "quantem.ssb.calibration.v1":
        calibration, run, source = earlier["calibration"], earlier["run"], dict(earlier["source"])
        source["acquisition_index"] = source.pop("frame_idx")
        held = earlier["aberrations"]
        return {
            "schema": CALIBRATION_SCHEMA, "generated_at_utc": earlier["created_utc"], "source": source,
            "aberrations": {"C10_nm": float(held["C10"]), "C12_nm": float(held["C12"]), "phi12_deg": math.degrees(float(held["phi12"]))},
            # the earlier file held the angle below 180 degrees and the reversal beside it
            "scan_detector_rotation_deg": (float(earlier["rotation_angle_deg"]) + (180.0 if earlier["com_reversed"] else 0.0)) % 360.0,
            "detector_transposed": False, "voltage_kV": calibration["voltage_kV"], "semiangle_mrad": calibration["semiangle_mrad"],
            "scan_sampling_A": calibration["scan_sampling_A"], "detector_sampling_mrad": calibration["det_sampling_mrad"],
            "bright_field_radius_px": calibration["bf_radius"], "bright_field_intensity_threshold": calibration["bf_intensity_threshold"],
            "num_bright_field_pixels": calibration["bf_pixels"], "num_trials": run["n_trials"], "refine": run["refine"],
            "manual_locked": run["manual_locked"], "seed": run["seed"], "loss": run["loss"], "elapsed_s": run["elapsed_s"],
        }, []
    if earlier.get("schema_version") == 3:
        current, notes = showptycho_record(earlier, higher_order_radians=False)
        tilt = earlier.get("tilt_mrad")
        current.update(schema=CALIBRATION_SCHEMA, notes=earlier.get("notes"), sample_tilt_mrad=None if tilt is None else list(tilt),
                       propagator_tilt_mrad=earlier.get("tilt_object_mrad"), depth_spread_nm=earlier.get("depth_spread_nm"))
        return current, notes
    raise ValueError("not an SSB calibration file this script knows")


def main() -> None:
    """Rewrite each path given on the command line, or print the result with ``--dry-run``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true", help="print the rewritten content and change nothing")
    args = parser.parse_args()
    for path in args.paths:
        current, notes = rewrite(json.loads(path.read_text(encoding="utf-8")))
        if current is None:
            print(f"{path}: already in the current layout, left alone")
            continue
        for note in notes:
            print(f"{path}: {note}")
        text = json.dumps(current, indent=2)
        if args.dry_run:
            print(f"{path}: would be rewritten as\n{text}")
            continue
        path.with_name(path.name + ".before").write_bytes(path.read_bytes())
        path.write_text(text + "\n", encoding="utf-8")
        print(f"{path}: rewritten; the earlier file is {path.name}.before")


if __name__ == "__main__":
    main()
