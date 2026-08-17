from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "widget_browser_smoke.py"
SPEC = importlib.util.spec_from_file_location("widget_browser_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
_image_nonblank = MODULE._image_nonblank


def _png(pixels: list[tuple[int, int, int]], size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size)
    image.putdata(pixels)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_scientific_output_pixel_gate_rejects_black_and_flat_images() -> None:
    black = _png([(0, 0, 0)] * (32 * 32), (32, 32))
    flat_white = _png([(255, 255, 255)] * (32 * 32), (32, 32))

    black_passed, black_stats = _image_nonblank(black)
    flat_passed, flat_stats = _image_nonblank(flat_white)

    assert black_passed is False
    assert black_stats["nonblack_fraction"] == 0.0
    assert flat_passed is False
    assert flat_stats["max_channel_span"] == 0


def test_scientific_output_pixel_gate_accepts_color_and_grayscale_range() -> None:
    color = [(x * 8, y * 8, (x + y) * 4) for y in range(32) for x in range(32)]
    grayscale = [((x + y) * 4,) * 3 for y in range(32) for x in range(32)]

    color_passed, color_stats = _image_nonblank(_png(color, (32, 32)))
    grayscale_passed, grayscale_stats = _image_nonblank(_png(grayscale, (32, 32)))

    assert color_passed is True
    assert color_stats["unique_colors"] >= 8
    assert grayscale_passed is True
    assert grayscale_stats["max_channel_span"] >= 8


def test_scientific_output_pixel_gate_rejects_nearly_empty_white_canvas() -> None:
    pixels = [(255, 255, 255)] * (64 * 64)
    for index in range(8):
        pixels[index] = (0, index * 20, 255 - index * 20)

    passed, stats = _image_nonblank(_png(pixels, (64, 64)))

    assert passed is False
    assert stats["unique_colors"] >= 8
    assert stats["max_channel_span"] >= 8
    assert stats["nonwhite_fraction"] < 0.005
