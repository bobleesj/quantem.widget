"""Show2D tools driven like a user, on exported pages with WebGPU on.

An unlinked gallery histogram is computed once per panel frame, not on every
hover.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

COUNT_LOG1P = """
window.__log1pCalls = 0;
const log1p = Math.log1p;
Math.log1p = (value) => { window.__log1pCalls += 1; return log1p(value); };
"""
# One entry per histogram (its range slider has two inputs): the min and max labels.
HISTOGRAM_LABELS = """
() => Array.from(new Set(Array.from(document.querySelectorAll('input[aria-label="Histogram intensity clip range"]'))
  .map(input => input.closest('.MuiBox-root').parentElement)))
  .map(histogram => histogram.innerText.split('\\n').filter(Boolean).slice(0, 2))
"""


def open_export(page_name, serve, open_page, hardware_adapter, browser_out, ready, init_script=None):
    page = open_page(f"{serve(browser_out)}/{page_name}", webgpu=True)
    if init_script:
        page.add_init_script(init_script)
        page.reload()
    page.wait_for_selector(ready, timeout=120_000)
    page.wait_for_timeout(5_000)
    hardware_adapter(page.adapter())
    return page


def test_unlinked_gallery_hover_reuses_each_panel_histogram(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D

    rng = np.random.default_rng(2)
    panels = [(rng.standard_normal((256, 256)) * 40 * (k + 1)).astype(np.float32) for k in range(4)]
    Show2D(panels, link_contrast=False, log_scale=True).export_html(browser_out / "show2d_unlinked_log.html")
    page = open_export("show2d_unlinked_log.html", serve, open_page, hardware_adapter, browser_out,
                       'canvas[data-quantem-scientific-output="show2d-image-3"]', COUNT_LOG1P)
    box = page.locator('canvas[data-quantem-scientific-output="show2d-image-0"]').bounding_box()
    page.evaluate("() => { window.__log1pCalls = 0; }")
    for step in range(8):
        page.mouse.move(box["x"] + box["width"] * (0.2 + 0.08 * step), box["y"] + box["height"] * 0.5)
        page.wait_for_timeout(80)
    page.screenshot(path=str(browser_out / "show2d_unlinked_log_hover.png"))
    # A hover changes no pixel: it must not log-transform any panel again.
    assert page.evaluate("() => window.__log1pCalls") == 0
    # Each histogram spans its own panel's signed log1p range (float32, as the browser stores it).
    expected = []
    for panel in panels:
        logged = (np.sign(panel) * np.log1p(np.abs(panel).astype(np.float64))).astype(np.float32)
        expected.append([f"{float(logged.min()):.1f}", f"{float(logged.max()):.1f}"])
    assert page.evaluate(HISTOGRAM_LABELS) == expected
    assert page.errors == [], page.errors
