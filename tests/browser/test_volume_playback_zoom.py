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
