"""Scale-bar label helpers: the one Python port of js/figure.ts.

Every static render (Show2D PNG fallback and save_image, Show3D GIF
frames, Show4DSTEM save_image) labels its scale bar through these three
functions so the label is character-identical to the live widget's canvas
label. Keep the thresholds and unit tables in step with js/figure.ts.
"""

import math

from matplotlib import font_manager

# Length-unit ladder, each as its size in Angstrom (js/figure.ts LENGTH_LADDER):
# a sub-1 value in one unit (e.g. 0.5 nm) displays as a clean integer in a
# smaller unit (5 Å), because microscopists read "5 Å", not "0.50 nm".
LENGTH_LADDER: tuple[tuple[str, float], ...] = (
    ("mm", 1e7), ("µm", 1e4), ("nm", 10.0), ("Å", 1.0), ("pm", 0.01),
)
# Size of one length unit in Angstrom, by every lower-case spelling a unit trait
# accepts (js/figure.ts ANGSTROMS_PER_UNIT, the same numbers so both round alike).
# Units that are not lengths (px, mrad, ps, ...) are absent and keep their own unit.
ANGSTROMS_PER_UNIT: dict[str, float] = {
    "å": 1.0, "a": 1.0, "ang": 1.0, "angstrom": 1.0, "angstroms": 1.0,
    "nm": 10.0, "nanometer": 10.0, "nanometers": 10.0,
    "pm": 0.01, "picometer": 0.01, "picometers": 0.01,
    "µm": 1e4, "μm": 1e4, "um": 1e4, "micron": 1e4, "microns": 1e4, "micrometer": 1e4, "micrometers": 1e4,
    "mm": 1e7, "millimeter": 1e7, "millimeters": 1e7,
    "m": 1e10, "meter": 1e10, "meters": 1e10,
}


def round_to_nice(value: float) -> float:
    """Snap a physical length to 1, 2, 5 or 10 times a power of ten.

    Port of js/figure.ts roundToNiceValue. A scale bar whose length is a
    round number is what readers expect on a figure; the thresholds 1.5, 3.5
    and 7.5 are the geometric midpoints between the candidates.
    """
    if value <= 0:
        return 1.0
    base = 10 ** math.floor(math.log10(value))
    mantissa = value / base
    if mantissa < 1.5:
        return base
    if mantissa < 3.5:
        return 2 * base
    if mantissa < 7.5:
        return 5 * base
    return 10 * base


def unit_symbol(unit: str) -> str:
    """Display symbol for a unit string (port of js/figure.ts unitSymbol).

    Users pass units like "micron" or "A" on a Dataset; the widget renders
    the conventional glyph (µm, Å) so labels read like a journal figure.
    Unknown strings pass through unchanged.
    """
    stripped = (unit or "").strip()
    lowered = stripped.lower()
    if lowered in ("micron", "microns", "um") or stripped in ("μm", "µm"):
        return "µm"
    if lowered in ("angstrom", "angstroms", "ang", "a") or stripped == "Å":
        return "Å"
    if lowered in ("nanometer", "nanometers", "nm"):
        return "nm"
    if lowered in ("picometer", "picometers", "pm"):
        return "pm"
    if lowered in ("millimeter", "millimeters", "mm"):
        return "mm"
    if lowered in ("picosecond", "picoseconds", "ps"):
        return "ps"
    if lowered in ("femtosecond", "femtoseconds", "fs"):
        return "fs"
    if lowered in ("nanosecond", "nanoseconds", "ns"):
        return "ns"
    if lowered in ("pixel", "pixels"):  # the unit quantem datasets give an uncalibrated axis
        return "px"
    return stripped


def format_scale_label(value: float, unit: str) -> str:
    """Scale bar label (port of js/figure.ts formatScaleLabel).

    Length values auto-pick the unit that reads as a clean integer:
    0.5 nm -> "5 Å", 0.005 nm -> "5 pm". Non-length units (mrad, ps, px)
    keep their unit. ``round_to_nice`` gives n*10^k and every ladder step
    is a power of 10, so the rescaled number is always exact. Rounding is
    JS ``Math.round`` (half away from zero); Python's banker's rounding
    would format 2.5 as "2" where the widget shows "3".
    """
    nice = round_to_nice(value)
    unit_angstroms = ANGSTROMS_PER_UNIT.get((unit or "").strip().lower())
    if unit_angstroms is None:
        symbol = unit_symbol(unit)
        return f"{math.floor(nice + 0.5)} {symbol}" if nice >= 1 else f"{nice:.2f} {symbol}"
    angstroms = nice * unit_angstroms
    # largest ladder unit where the value is >= 1 -> the fewest-digit integer
    for symbol, ladder_angstroms in LENGTH_LADDER:
        if angstroms / ladder_angstroms >= 1:
            return f"{math.floor(angstroms / ladder_angstroms + 0.5)} {symbol}"
    symbol, ladder_angstroms = LENGTH_LADDER[-1]
    return f"{math.floor(angstroms / ladder_angstroms + 0.5)} {symbol}"


OVERLAY_FONT: list[str] | None = None


def static_overlay_font() -> list[str]:
    """Closest installed match to the widget's ``-apple-system, BlinkMacSystemFont,
    'Segoe UI', sans-serif`` stack, resolved once. Filtering to installed fonts
    avoids matplotlib findfont warnings on machines without Helvetica/Arial."""
    global OVERLAY_FONT
    if OVERLAY_FONT is None:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        preferred = ("Helvetica Neue", "Segoe UI", "Arial", "Liberation Sans")
        OVERLAY_FONT = [name for name in preferred if name in installed] + ["DejaVu Sans"]
    return OVERLAY_FONT
