"""An integer stack in Show3D draws the pixels of its float32 copy, on the hardware adapter and on the CPU path.

Show3D keeps uint8, int8, uint16 and int16 stacks in their own type in the kernel and in the page (one or
two bytes per pixel instead of four) and converts one frame at a time. Every value is exact, so each frame
must be pixel-identical to the same stack sent as float32, also after stepping to another frame.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

IMAGE = 'canvas[data-quantem-scientific-output="show3d-image"]'
HASH = f"""() => {{
  const canvas = document.querySelector('{IMAGE}');
  const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  let hash = 2166136261, lit = 0;
  for (let i = 0; i < data.length; i++) {{ hash = Math.imul(hash ^ data[i], 16777619) >>> 0; if (i % 4 < 3 && data[i] > 0) lit++; }}
  return {{ size: `${{canvas.width}}x${{canvas.height}}`, hash, lit }};
}}"""
FOCUS_FRAME = """() => {
  const thumbs = [...document.querySelectorAll('input[type="range"]')].filter(input => /and current frame/i.test(input.getAttribute("aria-label") ?? ""));
  thumbs[1].focus();
}"""


@pytest.mark.parametrize("dtype", ["uint8", "int16"])
def test_integer_stack_draws_the_float32_pixels(dtype, serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show3D

    info = np.iinfo(dtype)
    rows, cols = np.indices((70, 90))
    frames = np.stack([np.sin(rows / (5.0 + z)) * np.cos(cols / 7.0) + 0.05 * z for z in range(8)])
    stack = (info.min + (frames - frames.min()) / np.ptp(frames) * (int(info.max) - int(info.min))).round().astype(dtype)
    pages = {}
    for name, data in (("native", stack), ("float32", stack.astype(np.float32))):
        widget = Show3D(data, title=f"{dtype} {name}", verbose=False)
        assert widget._stack_dtype == (dtype if name == "native" else "float32")
        widget.export_html(browser_out / f"show3d_{dtype}_{name}.html")
        pages[name] = f"show3d_{dtype}_{name}.html"
    drawn = {}
    for name, html in pages.items():
        for webgpu in (True, False):
            page = open_page(f"{serve(browser_out)}/{html}", webgpu=webgpu)
            page.wait_for_selector(IMAGE, timeout=60_000)
            page.wait_for_timeout(4_000)
            if webgpu:
                hardware_adapter(page.adapter())
            first = page.evaluate(HASH)
            page.evaluate(FOCUS_FRAME)
            for _ in range(3):
                page.keyboard.press("ArrowRight")
            page.wait_for_timeout(1_500)
            later = page.evaluate(HASH)
            assert first["lit"] > 0 and later["lit"] > 0 and first != later, (first, later)
            assert page.errors == [], page.errors
            drawn[name, webgpu] = (first, later)
            page.screenshot(path=str(browser_out / f"show3d_{dtype}_{name}_{'webgpu' if webgpu else 'cpu'}.png"))
    assert drawn["native", True] == drawn["float32", True], drawn
    assert drawn["native", False] == drawn["float32", False], drawn
