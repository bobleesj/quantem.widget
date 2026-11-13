"""The 3D view of Show3DSlices does not queue frames faster than a slow GPU finishes them.

A full-quality frame marches up to 512 steps per device pixel. An integrated GPU can need hundreds of
milliseconds for it; frames queued on every pointer move then pile up until the driver's watchdog resets
the GPU. The page here runs on the hardware adapter with ``queue.onSubmittedWorkDone`` delayed, the signal
of a GPU that takes that long per frame. While the view rotates it must draw drafts (one canvas pixel per
CSS pixel), one frame in flight at a time, and return to full quality when the pointer rests; when even
one frame takes most of a second, it keeps the draft's pixel density and says so in the console.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

VOLUME = 'canvas[data-quantem-scientific-output="show3dslices-volume"]'
SLOW_GPU = """(() => {
  const delay = %d;
  window.__submits = 0;
  const done = GPUQueue.prototype.onSubmittedWorkDone;
  GPUQueue.prototype.onSubmittedWorkDone = function () { return done.call(this).then(() => new Promise(resolve => setTimeout(resolve, delay))); };
  const submit = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function (...args) { window.__submits++; return submit.apply(this, args); };
})();"""
STATE = f"""() => {{ const canvas = document.querySelector('{VOLUME}'); return {{ density: canvas.width / canvas.clientWidth, submits: window.__submits }}; }}"""


@pytest.mark.parametrize("frame_ms", [300, 900])
def test_3d_view_paces_itself_to_a_slow_gpu(frame_ms, chrome, serve, hardware_adapter, browser_out):
    from quantem.widget import Show3DSlices

    rows, cols = np.indices((64, 64))
    volume = np.stack([np.sin(rows / 5.0) * np.cos(cols / 7.0) + 0.1 * z for z in range(16)]).astype(np.float32)
    Show3DSlices(volume, title="slow").export_html(browser_out / "show3dslices_slow_gpu.html")
    context = chrome.new_context(viewport={"width": 1600, "height": 1050}, device_scale_factor=2)
    context.add_init_script(SLOW_GPU % frame_ms)
    page = context.new_page()
    console = []
    page.on("console", lambda message: console.append(message.text))
    page.goto(f"{serve(browser_out)}/show3dslices_slow_gpu.html")
    page.wait_for_selector(VOLUME, timeout=60_000)
    page.wait_for_timeout(6_000)
    hardware_adapter(page.evaluate("async () => { const a = await navigator.gpu.requestAdapter(); return {vendor: a.info.vendor, architecture: a.info.architecture}; }"))
    box = page.locator(VOLUME).bounding_box()
    centre = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

    def rotate(moves: int, seconds: float) -> dict:
        start = page.evaluate(STATE)
        page.mouse.move(*centre)
        page.mouse.down()
        for step in range(moves):
            page.mouse.move(centre[0] + (step % 40) * 4 - 80, centre[1] + (step % 17) * 3 - 24)
            page.wait_for_timeout(int(1000 * seconds / moves))
        during = page.evaluate(STATE)
        page.mouse.up()
        return {"density": during["density"], "submits": during["submits"] - start["submits"]}

    rotate(10, 0.5)  # the first measured full frame tells the view how slow this GPU is
    page.wait_for_timeout(3 * frame_ms + 1_000)
    moving = rotate(90, 3.0)
    # drafts while the view moves, and about one frame per GPU frame time instead of one per pointer move
    assert moving["density"] == 1, moving
    assert moving["submits"] <= 3_000 / frame_ms + 4, moving
    page.wait_for_timeout(3 * frame_ms + 1_500)
    rested = page.evaluate(STATE)
    page.screenshot(path=str(browser_out / f"show3dslices_slow_gpu_{frame_ms}.png"))
    said = [line for line in console if "full-quality 3D frame takes" in line]
    if frame_ms < 700:
        assert rested["density"] == 2 and not said, (rested, said)  # full quality once the pointer rests
    else:
        assert rested["density"] == 1 and len(said) == 1, (rested, said)  # capped, and said once
    context.close()
