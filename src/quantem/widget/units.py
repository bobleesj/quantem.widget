"""The one unit table of the widgets: what the unit string of a calibrated axis means.

A ``.qem`` file writes three unit strings (``angstrom`` on the scan axes,
``mrad`` or ``1/angstrom`` on the detector axes) and quantem core datasets
carry the text their author typed (``"A"``, ``"nm"``, ``"pixels"``). Every
widget reads an axis through ``axis_step``: a unit in ``UNIT_TABLE`` gives its
kind and the step in that kind's reference unit, a unit in
``UNCALIBRATED_UNITS`` (or a missing step or unit) gives None, and any other
string raises, so an axis is never drawn in a unit the widget only guessed.
``js/units.ts`` holds the same table for the frontends; keep the two in step.
"""

import math

# unit string -> (kind, size of one unit in the kind's reference unit, symbol a label prints).
# Reference units: angstrom (length), 1/angstrom (reciprocal_length), mrad (angle), second (time).
UNIT_TABLE: dict[str, tuple[str, float, str]] = {
    "angstrom": ("length", 1.0, "Å"), "Å": ("length", 1.0, "Å"), "A": ("length", 1.0, "Å"),
    "nm": ("length", 10.0, "nm"), "pm": ("length", 0.01, "pm"),
    "µm": ("length", 1e4, "µm"), "μm": ("length", 1e4, "µm"), "um": ("length", 1e4, "µm"), "mm": ("length", 1e7, "mm"), "m": ("length", 1e10, "m"),
    "1/angstrom": ("reciprocal_length", 1.0, "Å⁻¹"), "1/Å": ("reciprocal_length", 1.0, "Å⁻¹"),
    "1/A": ("reciprocal_length", 1.0, "Å⁻¹"), "A^-1": ("reciprocal_length", 1.0, "Å⁻¹"),
    "1/nm": ("reciprocal_length", 0.1, "nm⁻¹"), "nm^-1": ("reciprocal_length", 0.1, "nm⁻¹"),
    "mrad": ("angle", 1.0, "mrad"), "rad": ("angle", 1e3, "rad"), "µrad": ("angle", 1e-3, "µrad"),
    "urad": ("angle", 1e-3, "µrad"), "deg": ("angle", 1e3 * math.pi / 180.0, "°"),
    "s": ("time", 1.0, "s"), "ms": ("time", 1e-3, "ms"), "µs": ("time", 1e-6, "µs"), "us": ("time", 1e-6, "µs"),
    "ns": ("time", 1e-9, "ns"), "ps": ("time", 1e-12, "ps"), "fs": ("time", 1e-15, "fs"),
}
# What an axis without a calibration is called: quantem core's default ("pixels", "index"), the
# first axis of a stack ("frame", "slice") and the widgets' own label ("px").
UNCALIBRATED_UNITS = frozenset({"", "pixels", "pixel", "px", "index", "frame", "slice"})
# Length units a label may switch to, largest first, each as its size in angstrom: a value
# below 1 in one unit (0.5 nm) reads as a whole number in a smaller one (5 Å).
LENGTH_LADDER: tuple[tuple[str, float], ...] = (("mm", 1e7), ("µm", 1e4), ("nm", 10.0), ("Å", 1.0), ("pm", 0.01))


def unit_entry(unit) -> tuple[str, float, str]:
    """``(kind, size in the reference unit, symbol)`` of a unit string; an uncalibrated one is ``("pixels", 1.0, "px")``.

    A string in neither table raises ValueError. Without this check a dataset
    in a unit the widgets do not know would open with no scale bar and no
    message, or with a bar in the wrong unit.
    """
    if unit is None or str(unit) in UNCALIBRATED_UNITS:
        return ("pixels", 1.0, "px")
    if str(unit) not in UNIT_TABLE:
        raise ValueError(f"Unknown unit {str(unit)!r}. A calibrated axis uses one of: {', '.join(sorted(UNIT_TABLE))}; "
                         "an uncalibrated axis uses 'pixels'.")
    return UNIT_TABLE[str(unit)]


def axis_step(sampling, unit) -> tuple[str, float] | None:
    """Kind and step of one axis in the kind's reference unit, or None for an uncalibrated axis.

    ``sampling`` is the step in ``unit``. A missing step or unit, an
    uncalibrated unit or a step that is not positive means the axis carries no
    calibration: unknown is absent, never ``1.0`` or ``"px"``.
    """
    kind, size, _ = unit_entry(unit)
    if sampling is None or kind == "pixels" or float(sampling) <= 0:
        return None
    return kind, float(sampling) * size


def axis_pair(sampling, units, kinds: tuple[str, ...], axes: str) -> tuple[str, list[float]] | None:
    """Kind and ``[row, col]`` steps of two axes in the reference unit, or None when either is uncalibrated.

    ``kinds`` are the kinds the two axes may have (a scan is a ``length``, a
    detector an ``angle`` or a ``reciprocal_length``) and ``axes`` names them
    in the error. Both axes must share one kind; a scan in mrad or a detector
    calibrated in two kinds has no geometry a widget can draw, so it raises.
    """
    steps = [axis_step(step, unit) for step, unit in zip(sampling, units, strict=True)]
    if None in steps:
        return None
    (row_kind, row_step), (col_kind, col_step) = steps
    if row_kind != col_kind or row_kind not in kinds:
        raise ValueError(f"The {axes} axes are in {units[0]!r} and {units[1]!r}; both must be one "
                         f"{' or '.join(kinds)}.")
    return row_kind, [row_step, col_step]


def acquisition_calibration(sampling, units, metadata) -> dict:
    """The calibration a 4D acquisition records, under the ``.qem`` format's flat names; an unknown value is None.

    ``sampling`` and ``units`` are the four axis values of a
    ``quantem.gpu.io.Dataset4dstemGPU`` (an unknown axis is None) or of a
    quantem core ``Dataset4dstem`` (an unknown axis is ``1.0 "pixels"``), and
    ``metadata`` its metadata dict. The scan step is ``scan_sampling_A`` and
    the detector step ``detector_sampling_mrad`` or ``detector_sampling_inv_A``,
    each ``[row, col]``; ``voltage_kV`` and ``semiangle_mrad`` are the flat
    metadata keys. Every 4D widget reads a dataset here, so one file opens
    with one calibration whichever dataset type this machine read it into.
    """
    scan = axis_pair(sampling[:2], units[:2], ("length",), "scan")
    detector = axis_pair(sampling[2:4], units[2:4], ("angle", "reciprocal_length"), "detector")
    return {
        "scan_sampling_A": None if scan is None else scan[1],
        "detector_sampling_mrad": detector[1] if detector is not None and detector[0] == "angle" else None,
        "detector_sampling_inv_A": detector[1] if detector is not None and detector[0] == "reciprocal_length" else None,
        "voltage_kV": metadata.get("voltage_kV"),
        "semiangle_mrad": metadata.get("semiangle_mrad"),
    }


def image_axes(sampling, units, dataset) -> tuple:
    """``(sampling, units)`` of an image widget opened on ``dataset``: the dataset's, which the keywords may repeat.

    ``sampling`` and ``units`` are the constructor keywords, each None when
    omitted and then taken from the last two axes of the dataset. When the
    dataset records a calibrated column step and the keywords give another one,
    ValueError says so: the dataset is the source of its calibration, and
    before this check the keyword replaced it without a message.
    """
    held_sampling, held_units = tuple(float(step) for step in dataset.sampling[-2:]), [str(unit) for unit in dataset.units[-2:]]
    sampling = held_sampling if sampling is None else sampling
    units = held_units if units is None else units
    recorded = axis_step(held_sampling[-1], held_units[-1])
    given = axis_step(sampling if isinstance(sampling, (int, float)) else sampling[-1], units if isinstance(units, str) else units[-1])
    if recorded is not None and (given is None or given[0] != recorded[0] or not math.isclose(given[1], recorded[1], rel_tol=1e-6)):
        raise ValueError(f"sampling={sampling!r}, units={units!r} contradict the calibration the dataset records "
                         f"({held_sampling[-1]:g} {held_units[-1]} per pixel). Leave them out: the dataset is the source of its calibration.")
    return sampling, units


def stated(name: str, keyword, recorded):
    """The value of one calibration: ``recorded`` by the input, else the ``keyword`` the caller passed.

    A keyword equal to the recorded value is accepted. One that differs raises
    ValueError naming both: the dataset or file is the source of its own
    calibration, and a second value would leave two answers for one quantity.
    """
    if keyword is None or recorded is None:
        return recorded if keyword is None else keyword
    given = list(keyword) if isinstance(keyword, (list, tuple)) else [keyword] * (len(recorded) if isinstance(recorded, (list, tuple)) else 1)
    held = list(recorded) if isinstance(recorded, (list, tuple)) else [recorded]
    if len(given) != len(held) or not all(math.isclose(passed, kept, rel_tol=1e-6) for passed, kept in zip(given, held)):
        raise ValueError(f"{name}={keyword!r} contradicts the value the input records ({recorded!r}). "
                         f"Leave {name} out: the dataset or file is the source of its calibration.")
    return recorded
