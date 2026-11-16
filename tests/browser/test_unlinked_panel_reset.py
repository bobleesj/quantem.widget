"""Reset remains available when the zoomed image is a different panel."""

import numpy as np
import pytest

pytestmark = pytest.mark.browser


@pytest.mark.parametrize("widget", ["Show2D", "Show3D"])
def test_reset_unlinked_panel(widget, serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D, Show3D

    image = np.random.default_rng(16).normal(size=(48, 64)).astype(np.float32)
    if widget == "Show2D":
        viewer = Show2D([image, image], link_zoom=False, link_pan=False, panel_width_px=180)
    else:
        stack = np.stack([image, image])
        viewer = Show3D(stack, stack, panel_width_px=181, panel_gap_px=3,
                        link_contrast=True, auto_contrast=False, vmin=-2, vmax=2)
        viewer.link_panels = False
    path = browser_out / f"{widget}-unlinked-reset.html"
    viewer.export_html(path)
    page = open_page(f"{serve(browser_out)}/{path.name}", webgpu=True)
    hardware_adapter(page.adapter())

    def sample_panel(canvas):
        return canvas.evaluate("""el => {
            const ctx = el.getContext("2d");
            return Array.from(ctx.getImageData(el.width * .67, el.height * .33, 32, 32).data);
        }""")

    if widget == "Show2D":
        first = page.locator('[data-quantem-scientific-output="show2d-image-0"]')
        first.click()
        first.hover()
        reset = page.get_by_role("button", name="Reset", exact=True).first
    else:
        canvas = page.locator('[data-quantem-scientific-output="show3d-image"]')
        page.locator('[data-show3d-gpu-residency="ready"]').wait_for()
        canvas.wait_for()
        bounds = canvas.bounding_box()
        before = sample_panel(canvas)
        canvas.hover(position={"x": bounds["width"] * .75, "y": bounds["height"] * .5})
        reset = page.get_by_role("button", name="Reset zoom and pan")
    page.mouse.wheel(0, -240)
    if widget == "Show2D":
        page.locator('[data-quantem-scientific-output="show2d-image-1"]').click()
    reset.wait_for()
    if widget == "Show3D":
        page.wait_for_timeout(350)
        assert sample_panel(canvas) != before
    reset.click(timeout=3000)
    assert reset.is_disabled()
    if widget == "Show3D":
        page.wait_for_timeout(350)
        assert sample_panel(canvas) == before
    assert not page.errors
