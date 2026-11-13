"""Show3DSlices and Show3D keep working after the browser loses its GPU device.

A driver reset, a GPU process crash or too little GPU memory (most often on an integrated GPU) takes the
WebGPU device and every texture on it. The viewer must then say so, draw on the CPU with the same values
and still answer its controls, instead of freezing on a dead device. The test kills Chrome's GPU process
for real (CDP ``Browser.crashGpuProcess``) while each viewer is open on the hardware adapter.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser

PIXELS = """selector => [...document.querySelectorAll(selector)].map(canvas => canvas.toDataURL("image/png"))"""
# Focus the current-position thumb of the frame or slice slider (the middle of its three thumbs).
FOCUS_POSITION = """() => {
  const thumbs = [...document.querySelectorAll('input[type="range"]')].filter(input => /and current/i.test(input.getAttribute("aria-label") ?? ""));
  if (thumbs.length < 2) return null;
  thumbs[1].focus();
  return { value: Number(thumbs[1].value), max: Number(thumbs[1].max) };
}"""


def pixels(page, selector) -> list[np.ndarray]:
    """Every matching 2D canvas as an RGB array."""
    return [np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")).astype(int)
            for url in page.evaluate(PIXELS, selector)]


def same_share(first: np.ndarray, second: np.ndarray) -> float:
    """Share of pixels that are identical in two pictures of one size."""
    assert first.shape == second.shape, (first.shape, second.shape)
    return float((np.abs(first - second).max(axis=2) == 0).mean())


def volume() -> np.ndarray:
    rows, cols = np.indices((96, 96))
    return np.stack([np.sin(rows / (4.0 + z)) * np.cos(cols / 6.0) + 0.1 * z for z in range(12)]).astype(np.float32)


def crash_gpu(chrome) -> None:
    session = chrome.new_browser_cdp_session()
    session.send("Browser.crashGpuProcess")
    session.detach()


@pytest.mark.parametrize("viewer", ["show3dslices", "show3d"])
def test_viewer_survives_a_lost_gpu_device(viewer, chrome, serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show3D, Show3DSlices

    html = f"gpu_loss_{viewer}.html"
    (Show3DSlices if viewer == "show3dslices" else Show3D)(volume(), title="loss").export_html(browser_out / html)
    canvases = ('canvas[data-quantem-scientific-output^="show3dslices-slice-"]' if viewer == "show3dslices"
                else 'canvas[data-quantem-scientific-output="show3d-image"]')
    page = open_page(f"{serve(browser_out)}/{html}", webgpu=True)
    page.wait_for_selector(canvases, timeout=60_000)
    page.wait_for_timeout(4_000)
    hardware_adapter(page.adapter())
    assert page.locator("[data-render-path]").first.get_attribute("data-render-path") == "WebGPU"
    before = pixels(page, canvases)
    assert before and all(panel.max() > 0 for panel in before)

    crash_gpu(chrome)
    # the title chip names the path that draws now, and its tooltip says the device was lost
    page.wait_for_selector('[data-render-path="CPU"]', timeout=30_000)
    assert "lost" in page.locator('[data-render-path="CPU"]').first.get_attribute("title")
    page.wait_for_timeout(2_500)
    after = pixels(page, canvases)
    page.screenshot(path=str(browser_out / f"gpu_loss_{viewer}_after.png"))
    # The same picture, now drawn on the CPU. Show3DSlices matches to the pixel; Show3D's canvas path
    # rounds a few block edges of an enlarged frame differently from its WebGPU path.
    shares = [same_share(was, now) for was, now in zip(before, after, strict=True)]
    assert all(share >= (1.0 if viewer == "show3dslices" else 0.99) for share in shares), shares
    if viewer == "show3dslices":
        assert page.locator("[data-show3dslices-gpu-lost]").count() == 1, "the 3D view does not say the GPU stopped"
    else:
        assert page.get_by_text("GPU device was lost").count() >= 1, "the status line does not say the GPU stopped"

    # still alive: stepping the frame or slice slider draws another image
    position = page.evaluate(FOCUS_POSITION)
    assert position is not None, "no frame or slice slider"
    for _ in range(3):
        page.keyboard.press("ArrowRight" if position["value"] + 3 <= position["max"] else "ArrowLeft")
    page.wait_for_timeout(1_500)
    moved = pixels(page, canvases)
    page.screenshot(path=str(browser_out / f"gpu_loss_{viewer}_stepped.png"))
    assert moved[0].max() > 0 and same_share(after[0], moved[0]) < 0.9, "the viewer froze after the GPU was lost"
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
