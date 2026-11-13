"""A standalone Show3DSlices page saves itself from Export, as a Show3D page does.

The page has no kernel, so Export downloads the page with its embedded volume, exact in its own type,
and the current view. The saved page must open at the slice it was saved on and draw it pixel for pixel
as a page exported from Python at that slice.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

SLICE = 'canvas[data-quantem-scientific-output="show3dslices-slice-0"]'
HASH = f"""() => {{
  const c = document.querySelector('{SLICE}');
  const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  let h = 2166136261;
  for (let i = 0; i < d.length; i++) h = Math.imul(h ^ d[i], 16777619) >>> 0;
  return h;
}}"""
TOP = """() => {
  const thumbs = [...document.querySelectorAll('input[type="range"]')].filter(i => /current top slice|^top slice/i.test(i.getAttribute("aria-label") ?? ""));
  return Number(thumbs[thumbs.length > 1 ? 1 : 0].value);
}"""
FOCUS_SLICE = """() => {
  const thumbs = [...document.querySelectorAll('input[type="range"]')].filter(input => /current top slice/i.test(input.getAttribute("aria-label") ?? ""));
  thumbs[thumbs.length > 1 ? 1 : 0].focus();
}"""


def open_slices(page_url, open_page, hardware_adapter):
    page = open_page(page_url, webgpu=True)
    page.wait_for_selector(SLICE, timeout=60_000)
    page.wait_for_timeout(3_000)
    hardware_adapter(page.adapter())
    return page


def test_standalone_page_exports_itself_exact(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show3DSlices

    rows, cols = np.indices((80, 70))
    volume = np.stack([np.sin(rows / (3.0 + z * 0.4)) * np.cos(cols / 5.0) + 1e-4 * z for z in range(12)]).astype(np.float32)
    Show3DSlices(volume, title="saved").export_html(browser_out / "show3dslices_saved.html")
    page = open_slices(f"{serve(browser_out)}/show3dslices_saved.html", open_page, hardware_adapter)
    page.evaluate(FOCUS_SLICE)
    for _ in range(3):
        page.keyboard.press("ArrowRight")
    page.wait_for_timeout(1_500)
    shown = page.evaluate(HASH), page.evaluate(TOP)
    page.screenshot(path=str(browser_out / "show3dslices_saved_before.png"))
    page.locator('button[aria-label="Export standalone HTML"]').click()
    item = page.get_by_role("menuitem", name="HTML exact float32")
    assert item.is_visible(), page.locator('[aria-label="Export standalone HTML options"]').inner_text()
    with page.expect_download() as download:
        item.click()
    saved = browser_out / "show3dslices_saved_again.html"
    download.value.save_as(saved)
    assert download.value.suggested_filename == "saved_12x80x70_exact.html"
    page.screenshot(path=str(browser_out / "show3dslices_saved_menu.png"))
    assert page.errors == [], page.errors
    again = open_slices(f"{serve(browser_out)}/{saved.name}", open_page, hardware_adapter)
    again.wait_for_timeout(1_000)
    again.screenshot(path=str(browser_out / "show3dslices_saved_again.png"))
    assert again.evaluate(TOP) == shown[1]
    at_slice = Show3DSlices(volume, title="saved")
    at_slice.slice_index = shown[1]
    at_slice.export_html(browser_out / "show3dslices_at_slice.html")
    reference = open_slices(f"{serve(browser_out)}/show3dslices_at_slice.html", open_page, hardware_adapter)
    reference.wait_for_timeout(1_000)
    assert again.evaluate(HASH) == reference.evaluate(HASH)
    assert again.errors == [], again.errors
