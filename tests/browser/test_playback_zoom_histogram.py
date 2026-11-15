"""Exercise histogram dragging with the scientific panel zoomed and playing."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from test_show4dstem_offline import DP, counts
from test_showptycho_live import PHASE, SESSION

pytestmark = pytest.mark.browser


def drag_histogram_during_playback(page, image, play, pause, histogram):

    image = page.locator(image).first
    image.wait_for(timeout=60000)
    image.scroll_into_view_if_needed()

    def wheel_zoom():
        panel = image.bounding_box()
        page.mouse.move(
            panel["x"] + 0.4 * panel["width"], panel["y"] + 0.4 * panel["height"]
        )
        page.mouse.wheel(0, -240)
        page.wait_for_timeout(350)

    def image_patch():
        bounds = image.bounding_box()
        clip = {
            "x": bounds["x"] + 0.3 * bounds["width"],
            "y": bounds["y"] + 0.3 * bounds["height"],
            "width": 0.3 * bounds["width"],
            "height": 0.3 * bounds["height"],
        }
        return Image.open(BytesIO(page.screenshot(clip=clip))).convert("RGB").tobytes()

    # On a stationary frame, a pixel change isolates the effect of the wheel gesture.
    before_zoom = image_patch()
    wheel_zoom()
    assert image_patch() != before_zoom, (
        "wheel zoom left the stationary image unchanged"
    )
    page.locator(play).first.click()
    image.scroll_into_view_if_needed()
    wheel_zoom()
    hist = page.locator(histogram).first
    hist.scroll_into_view_if_needed(timeout=10000)
    box = hist.bounding_box()
    upper = hist.locator(".MuiSlider-thumb").last
    handle = upper.bounding_box() if upper.count() else None
    x = handle["x"] + handle["width"] / 2 if handle else box["x"] + box["width"] * 0.95
    y = box["y"] + box["height"] / 2

    def range_value():
        return (
            hist.locator("input").evaluate_all("xs=>xs.map(x=>x.value)")
            if handle
            else page.get_by_test_id("show1d-histogram-range-label").last.inner_text()
        )

    before = range_value()
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x - box["width"] * 0.22, y, steps=12)
    seen = set()
    for delay in (110, 180, 270, 320, 150, 420):
        page.wait_for_timeout(delay)
        seen.add(image_patch())
    held = range_value()
    page.mouse.up()
    if handle:
        page.wait_for_timeout(250)
        assert range_value() == held, "the clip handles jumped on release"
    assert held != before, (held, before)
    assert page.locator(pause).first.is_visible() and len(seen) > 1, (len(seen), pause)


@pytest.mark.parametrize(
    ("name", "webgpu"),
    [
        (name, gpu)
        for name in ["show2d", "pages", "show1d", "show3d"]
        for gpu in [False, True]
    ]
    + [("show4dstem", True)],
)
def test_zoom_and_adjust_contrast_during_playback(
    name, webgpu, serve, open_page, browser_out, hardware_adapter
):
    import quantem.widget as qw

    data = np.random.default_rng(3).random((12, 64, 64)).astype(np.float32)
    hist = '.MuiSlider-root:has(input[aria-label="Histogram intensity clip range"])'
    if name == "show2d":
        viewer = qw.Show2D([data, data.copy()], labels=["A", "B"])
        play = '[aria-label="Play frames for A"]'
        pause = '[aria-label="Pause frames for A"]'
        image = '[data-quantem-scientific-output="show2d-image-0"]'
    elif name == "pages":
        viewer = qw.Show2D(np.stack([data, data], axis=1), labels=["A", "B"])
        play = '[aria-label="Play pages"]'
        pause = '[aria-label="Pause page playback"]'
        image = '[data-quantem-scientific-output="show2d-image-0"]'
    elif name == "show4dstem":
        viewer = qw.Show4DSTEM(counts(4, 16, 16, seed=1))
        play = '[aria-label="Show4DSTEM play frames"]'
        pause = '[aria-label="Show4DSTEM pause frames"]'
        image = DP
    elif name == "show1d":
        viewer = qw.Show1D(np.arange(12))
        for i in range(12):
            viewer.snapshot(i, image=data[i])
        play = '[aria-label="Play"]'
        pause = '[aria-label="Pause playback"]'
        image = '[data-testid^="show1d-snapshot-panel-"] canvas'
        hist = '[data-testid="show1d-panel-histogram"] canvas'
    else:
        viewer = qw.Show3D(data, fps=8)
        play = '[aria-label="Play"]'
        pause = '[aria-label="Pause playback"]'
        image = '[data-quantem-scientific-output="show3d-image"]'
    path = browser_out / name / "combined.html"
    path.parent.mkdir(exist_ok=True)
    viewer.export_html(path)
    page = open_page(f"{serve(path.parent)}/{path.name}", webgpu=webgpu)
    if webgpu:
        hardware_adapter(page.adapter())
    drag_histogram_during_playback(page, image, play, pause, hist)
    assert not page.errors


def test_sweep(jupyter, open_page, browser_out, hardware_adapter):
    jupyter.write_notebook("sweep.ipynb", SESSION, "")
    page = jupyter.open(
        open_page, "sweep.ipynb", "sweep", webgpu=True, ready=PHASE, out=browser_out
    )
    hardware_adapter(page.adapter())
    drag_histogram_during_playback(
        page,
        PHASE,
        '[aria-label="Play sweep"]',
        '[aria-label="Pause sweep"]',
        'canvas[aria-label="Intensity histogram with linear counts"] + .MuiSlider-root',
    )
    # JupyterLab reports its idle debugger independently of the widget.
    assert [
        error for error in page.errors if error != "No active debugger session"
    ] == []
