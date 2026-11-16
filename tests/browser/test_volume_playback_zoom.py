"""Zoom during playback must keep showing the live slice between ticks."""

import numpy as np
import pytest

pytestmark = pytest.mark.browser


@pytest.mark.parametrize('webgpu', [False, True])
def test_zoom_paints_the_playing_slice(webgpu, serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show3DSlices

    # A uniform plane keeps its center color at every zoom. Each depth has a distinct color.
    volume = np.broadcast_to(np.arange(8, dtype=np.float32)[:, None, None], (8, 64, 64)).copy()
    path = browser_out / 'playing-slices.html'
    viewer = Show3DSlices(volume)
    viewer.fps = 1
    viewer.export_html(path)
    page = open_page(f'{serve(browser_out)}/{path.name}', webgpu=webgpu)
    canvas = page.locator('[data-quantem-scientific-output="show3dslices-slice-0"]')
    canvas.wait_for()
    if webgpu:
        hardware_adapter(page.adapter())
    page.wait_for_timeout(1000)
    page.get_by_role('button', name='Play', exact=True).click()
    colors = canvas.evaluate('''async canvas => {
        const color = () => Array.from(canvas.getContext('2d').getImageData(canvas.width >> 1, canvas.height >> 1, 1, 1).data).join(',');
        const initial = color();
        const start = performance.now();
        while (color() === initial && performance.now() - start < 4000) await new Promise(requestAnimationFrame);
        const live = color();
        const rect = canvas.getBoundingClientRect();
        canvas.dispatchEvent(new WheelEvent('wheel', {deltaY: -120, clientX: rect.left + rect.width/2, clientY: rect.top + rect.height/2, bubbles: true, cancelable: true}));
        await new Promise(resolve => setTimeout(resolve, 50));
        const during = color();
        await new Promise(resolve => setTimeout(resolve, 250));
        return [initial, live, during, color()];
    }''')
    initial, live, during, settled = colors
    assert initial != live and during == settled == live, colors
    assert not page.errors


@pytest.mark.parametrize('webgpu', [False, True])
def test_show3d_zoom_during_playback(webgpu, serve, open_page, hardware_adapter, browser_out):
    from io import BytesIO

    from PIL import Image
    from quantem.widget import Show3D

    volume = np.broadcast_to(np.arange(12, dtype=np.float32)[:, None, None], (12, 64, 64)).copy()
    viewer = Show3D(volume, cmap="gray", fps=1, panel_width_px=256, vmin=0, vmax=11)
    path = browser_out / 'playing-stack.html'
    viewer.export_html(path)
    page = open_page(f'{serve(browser_out)}/{path.name}', webgpu=webgpu)
    if webgpu:
        hardware_adapter(page.adapter())
    canvas = page.locator('canvas[data-quantem-scientific-output="show3d-image"]')
    canvas.wait_for()
    box = canvas.bounding_box()
    x, y = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2

    def center_value():
        shot = page.screenshot(clip=dict(x=x, y=y, width=4, height=4))
        return int(np.median(np.asarray(Image.open(BytesIO(shot)))[..., 0]))

    page.get_by_label('Play', exact=True).first.click()
    page.wait_for_function("document.querySelector('[data-show3d-playback-count]').textContent.trim() === '2/12'")
    live = center_value()
    page.mouse.move(x, y)
    page.mouse.wheel(0, -240)
    page.wait_for_timeout(180)
    assert center_value() == live
    page.wait_for_function("document.querySelector('[data-show3d-playback-count]').textContent.trim() === '3/12'")
    assert center_value() != live
    assert not page.errors


@pytest.mark.parametrize("webgpu", [False, True], ids=["cpu", "webgpu"])
def test_histogram_plot_drag_on_linked_stacks(webgpu, serve, open_page, hardware_adapter, browser_out):
    from io import BytesIO

    from PIL import Image
    from quantem.widget import Show3D

    # Same center intensity, different extrema. A shared scale must render both centers alike.
    stacks = np.full((2, 3, 24, 24), 0.4, dtype=np.float32)
    stacks[:, :, :, 0] = 0
    stacks[0, :, :, -1] = 1
    stacks[1, :, :, -1] = 2
    viewer = Show3D(*stacks, cmap="gray", vmin=0, vmax=1, auto_contrast=False,
                    link_contrast=True, panel_width_px=160, show_scale_bar=False)
    path = browser_out / "linked-histogram.html"
    viewer.export_html(path)
    page = open_page(f"{serve(browser_out)}/{path.name}", webgpu=webgpu)
    if webgpu:
        hardware_adapter(page.adapter())
    image = page.locator('canvas[data-quantem-scientific-output="show3d-image"]')
    plot = page.get_by_role("img", name="Histogram of intensity values with min and max clip handles")
    plot.wait_for()
    def centers():
        pixels = np.asarray(Image.open(BytesIO(image.screenshot())))
        h, w = pixels.shape[:2]
        return pixels[h // 2, [w // 4, 3 * w // 4], 0].astype(int)

    before = centers()
    plot.scroll_into_view_if_needed()
    rect = plot.bounding_box()
    y = rect["y"] + rect["height"] / 2
    page.mouse.move(rect["x"] + rect["width"] * .9, y)
    page.mouse.down()
    page.mouse.move(rect["x"] + rect["width"] * .6, y, steps=8)
    page.mouse.up()
    page.wait_for_timeout(250)
    after = centers()
    assert abs(before[0] - before[1]) <= 1 and abs(after[0] - after[1]) <= 1
    assert np.all(after > before + 20), (before, after)
    assert not page.errors
