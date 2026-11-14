"""Playback keeps running while a line is dragged over the image.

Show3D's Profile line and Show3DSlices' oblique cut line are moved by dragging on the image. Dragging the
line or one of its ends while the frames or slices play must not stop the playback: the frames keep
advancing and drawing during the drag, and the viewer is still playing after the release. When Show3DSlices
plays the cut itself (playback axis All), the cut stays where the hand holds it and moves on after release.
"""

import hashlib

import numpy as np
import pytest

pytestmark = pytest.mark.browser

POSITION = """label => {
  const thumbs = [...document.querySelectorAll('input[type="range"]')].filter(input => new RegExp(label, "i").test(input.getAttribute("aria-label") ?? ""));
  return thumbs.length ? Number(thumbs[thumbs.length > 1 ? 1 : 0].value) : null;
}"""
PLAYING = """() => [...document.querySelectorAll('button')].some(b => b.getAttribute('aria-label') === 'Pause playback' && b.offsetParent !== null)"""


def stack() -> np.ndarray:
    rows, cols = np.indices((96, 96))
    return np.stack([np.sin(rows / (4.0 + z * 0.4)) * np.cos(cols / 6.0) + 0.05 * z for z in range(40)]).astype(np.float32)


def region_hash(page, box):
    """Hash of the bottom fifth of the image, away from the dragged line."""
    clip = {"x": box["x"], "y": box["y"] + 0.78 * box["height"], "width": box["width"], "height": 0.18 * box["height"]}
    return hashlib.sha1(page.screenshot(clip=clip)).hexdigest()


def drag_and_watch(page, start, stop, label, box):
    """Drag from `start` to `stop` over about 2 s, holding still halfway; frame positions and images seen."""
    seen, images = [], []
    page.mouse.move(*start)
    page.mouse.down()
    for step in range(1, 41):
        t = step / 40
        page.mouse.move(start[0] + (stop[0] - start[0]) * t, start[1] + (stop[1] - start[1]) * t)
        page.wait_for_timeout(50)
        if step % 5 == 0:
            seen.append(page.evaluate(POSITION, label))
        if step == 20:  # hold the press still: the frames must keep drawing under it
            for _ in range(3):
                images.append(region_hash(page, box))
                page.wait_for_timeout(250)
    page.mouse.up()
    return seen, images


@pytest.mark.parametrize(("viewer", "grip"), [
    ("show3d", "line"), ("show3d", "endpoint"), ("show3dslices", "Top"), ("show3dslices", "All"),
])
def test_playback_continues_while_a_line_is_dragged(viewer, grip, serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show3D, Show3DSlices

    html = f"drag_playback_{viewer}.html"
    if viewer == "show3d":
        Show3D(stack(), title="drag", fps=20, verbose=False).export_html(browser_out / html)
        canvas, label = 'canvas[data-quantem-scientific-output="show3d-image"]', r"current (frame|slice) \("
    else:
        Show3DSlices(stack(), title="drag").export_html(browser_out / html)
        canvas, label = 'canvas[data-quantem-scientific-output="show3dslices-slice-0"]', "current top slice"
    page = open_page(f"{serve(browser_out)}/{html}", webgpu=True)
    page.wait_for_selector(canvas, timeout=60_000)
    page.wait_for_timeout(3_000)
    hardware_adapter(page.adapter())
    box = page.locator(canvas).first.bounding_box()
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
    if viewer == "show3d":
        page.locator('input[aria-label="Toggle line intensity profile tool"]').check()
        page.mouse.click(x + 0.25 * w, y + 0.5 * h)
        page.wait_for_timeout(150)
        page.mouse.click(x + 0.75 * w, y + 0.5 * h)
        page.wait_for_timeout(500)
        grab = (x + 0.5 * w, y + 0.5 * h) if grip == "line" else (x + 0.75 * w, y + 0.5 * h)
        drop = (grab[0], y + 0.3 * h)
        page.locator('button[aria-label="Play forward"]').first.click()
    else:
        if grip == "All":
            page.locator('[aria-label="Playback axis (Top, Side, or All)"]').locator("..").click()
            page.get_by_role("option", name="All").click()
            page.wait_for_timeout(300)
        # the oblique cut starts as a horizontal line through the middle row
        grab, drop = (x + 0.5 * w, y + 0.5 * h), (x + 0.5 * w, y + 0.3 * h)
        page.locator('button[aria-label="Play"]').first.click()
    page.wait_for_timeout(600)
    assert page.evaluate(PLAYING), "playback did not start"
    seen, images = drag_and_watch(page, grab, drop, label, box)
    held_cut = page.evaluate(POSITION, "oblique plane position") if viewer == "show3dslices" else None
    page.wait_for_timeout(600)
    after = [page.evaluate(POSITION, label)]
    page.wait_for_timeout(600)
    after.append(page.evaluate(POSITION, label))
    page.screenshot(path=str(browser_out / f"drag_playback_{viewer}_{grip}.png"))
    assert len(set(seen)) >= 4, f"frames stopped during the drag: {seen}"
    assert len(set(images)) == 3, "the image stopped drawing while the press was held"
    assert page.evaluate(PLAYING) and after[0] != after[1], f"playback stopped after the drag: {after}"
    if grip == "All":
        assert page.evaluate(POSITION, "oblique plane position") != held_cut, "the cut did not resume after release"
    assert page.errors == [], page.errors
