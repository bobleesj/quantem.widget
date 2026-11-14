"""Show3DSlices keeps Smooth while the slice moves on a zoomed-in panel.

Moving the slice paints each slice straight from the GPU, outside React. That paint drew the panel with the
nearest source pixel whatever Smooth said, so a zoomed-in panel turned blocky on the first slice step and
stayed so. With Smooth on, a step must look like the panel at rest; with Smooth off, it stays sharp.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

SLICE = 'canvas[data-quantem-scientific-output="show3dslices-slice-0"]'
# Share of horizontally adjacent pixels with the same color: high for an enlarged nearest-pixel image,
# low once bilinear smoothing blends them.
FLAT = f"""() => {{
  const c = document.querySelector('{SLICE}');
  const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  let same = 0, n = 0;
  for (let y = 0; y < c.height; y++) for (let x = 1; x < c.width; x++) {{
    const i = (y * c.width + x) * 4; n++;
    if (d[i] === d[i - 4] && d[i + 1] === d[i - 3] && d[i + 2] === d[i - 2]) same++;
  }}
  return same / n;
}}"""
ZOOM_IN = f"""async () => {{
  const c = document.querySelector('{SLICE}'); const r = c.getBoundingClientRect();
  for (let k = 0; k < 8; k++) {{
    c.dispatchEvent(new WheelEvent("wheel", {{deltaY: -120, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2, bubbles: true, cancelable: true}}));
    await new Promise(resolve => setTimeout(resolve, 40));
  }}
}}"""
FOCUS_SLICE = """() => {
  const thumbs = [...document.querySelectorAll('input[type="range"]')].filter(input => /current top slice/i.test(input.getAttribute("aria-label") ?? ""));
  thumbs[1].focus();
}"""


def test_smooth_holds_while_the_slice_moves_zoomed_in(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show3DSlices

    rows, cols = np.indices((120, 100))
    volume = np.stack([np.sin(rows / (3.0 + z * 0.3)) * np.cos(cols / 4.0) for z in range(10)]).astype(np.float32)
    Show3DSlices(volume, title="smooth").export_html(browser_out / "show3dslices_smooth.html")
    page = open_page(f"{serve(browser_out)}/show3dslices_smooth.html", webgpu=True)
    page.wait_for_selector(SLICE, timeout=60_000)
    page.wait_for_timeout(3_000)
    hardware_adapter(page.adapter())
    page.locator('input[aria-label="Toggle bilinear smoothing"]').check()
    page.evaluate(ZOOM_IN)
    page.wait_for_timeout(1_000)
    at_rest = page.evaluate(FLAT)
    stepped = []
    for _ in range(3):
        page.evaluate(FOCUS_SLICE)
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(150)
        stepped.append(page.evaluate(FLAT))
    page.screenshot(path=str(browser_out / "show3dslices_smooth_stepped.png"))
    page.locator('input[aria-label="Toggle bilinear smoothing"]').uncheck()
    page.wait_for_timeout(800)
    page.evaluate(FOCUS_SLICE)
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(150)
    sharp = page.evaluate(FLAT)
    assert at_rest < 0.25, at_rest
    assert all(value < 0.25 for value in stepped), (at_rest, stepped)
    assert sharp > 0.5, sharp  # Smooth off still shows the enlarged source pixels
    assert page.errors == [], page.errors
