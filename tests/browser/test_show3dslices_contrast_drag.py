"""A held histogram range stays visible while playback advances."""

import numpy as np
import pytest

pytestmark = pytest.mark.browser


@pytest.mark.parametrize("webgpu", [False, True], ids=["cpu", "webgpu"])
def test_contrast_holds_during_zoomed_playback(webgpu, serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show3DSlices

    plane = np.linspace(0, 1, 96, dtype=np.float32)[None, :].repeat(96, axis=0)
    viewer = Show3DSlices(np.stack([plane] * 10))
    viewer.fps = 8
    path = browser_out / "contrast-drag.html"
    viewer.export_html(path)
    page = open_page(f"{serve(browser_out)}/{path.name}", webgpu=webgpu)
    canvas = page.locator('[data-quantem-scientific-output="show3dslices-slice-0"]')
    canvas.wait_for()
    if webgpu:
        hardware_adapter(page.adapter())
    page.get_by_role("button", name="Play", exact=True).click()
    canvas.hover()
    page.mouse.wheel(0, -160)
    page.wait_for_timeout(400)
    pixel = "() => Array.from(document.querySelector('[data-quantem-scientific-output=\"show3dslices-slice-0\"]').getContext('2d').getImageData(45,45,1,1).data).join(',')"
    before = page.evaluate(pixel)
    track = page.locator('.MuiSlider-root').filter(has=page.locator('input')).all()
    histogram = next(slider for slider in track if slider.locator('input').count() == 2)
    thumb = histogram.locator(".MuiSlider-thumb").last
    thumb.scroll_into_view_if_needed()
    box = thumb.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x - histogram.bounding_box()["width"] * 0.32, y, steps=10)
    samples = []
    labels = []
    for _ in range(10):
        page.wait_for_timeout(80)
        samples.append(page.evaluate(pixel))
        labels.append(canvas.get_attribute('aria-label'))
    page.mouse.up()
    page.wait_for_timeout(300)
    settled = page.evaluate(pixel)
    assert len(set(labels)) > 1
    assert settled != before and set(samples) == {settled}, (before, samples, settled)
    assert not page.errors
