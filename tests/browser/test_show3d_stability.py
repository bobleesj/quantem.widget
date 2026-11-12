"""Show3D keeps what the user set while they work: no contrast jump when a histogram handle is grabbed, zoom kept
when a frame arrives, when another frame is shown and when a frame's sign is flipped; f flips and s stars the frame
on screen only. A streamed 2048 x 2048 viewer, as a screening session fills it.
"""

import io
import json

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser

IMAGE = 'canvas[data-quantem-scientific-output="show3d-image"]'
NOTEBOOK = """import json
import numpy as np
from quantem.widget import Show3D

rows, cols = np.indices((2048, 2048), dtype=np.float32)
lattice = np.sin(2 * np.pi * rows / 128) * np.sin(2 * np.pi * cols / 128)
# one lattice at six amplitudes and offsets: under Auto contrast every frame looks alike, their ranges differ widely
frames = [(lattice * (1 + 0.6 * index) + 2.0 * index).astype(np.float32) for index in range(6)]
viewer = Show3D.stream(title="t", cmap="gray", dim_label="Scan", save_state=True, show_scale_bar=False)
viewer"""
APPEND = """for index in range({first}, {stop}):
    viewer.append(frames[index], label=f"[{{index}}] a")
assert viewer._stream_wait_sent(60)"""
READ = """print(json.dumps({"frames": int(viewer.n_slices), "frame": int(viewer.slice_idx), "auto_contrast": bool(viewer.auto_contrast),
                  "flips": list(viewer.frame_flip_signs), "stars": list(viewer.bookmarked_frames), "limits": [viewer.image_vmin_pct, viewer.image_vmax_pct]}))"""
UPPER_HANDLE = """
() => {
  let box = document.querySelector('input[aria-label="Histogram intensity clip range"]');
  while (box && !box.classList.contains('MuiSlider-root')) box = box.parentElement;
  const thumbs = Array.from(box.querySelectorAll('.MuiSlider-thumb')).map(thumb => thumb.getBoundingClientRect());
  const right = thumbs.reduce((a, b) => (a.x > b.x ? a : b));
  return {x: right.x + right.width / 2, y: right.y + right.height / 2};
}
"""


def shot(page, path=None):
    """The image on screen as gray levels (rows, cols)."""
    png = page.screenshot(clip=page.locator(IMAGE).bounding_box())
    if path is not None:
        path.write_bytes(png)
    return np.asarray(Image.open(io.BytesIO(png)).convert("L")).astype(np.float64)


def period_px(image) -> float:
    """The lattice period along a row of the image on screen, from the strongest line of the row spectrum: a zoom
    of 2 doubles it."""
    row = image[int(np.argmax(image.std(axis=1)))]   # a row through the lattice's maxima, not along a node line
    profile = row - row.mean()
    spectrum = np.abs(np.fft.rfft(profile * np.hanning(profile.size), n=8 * profile.size))
    spectrum[:16] = 0   # the window's own slope
    return 8 * profile.size / float(np.argmax(spectrum))


def test_show3d_stream_keeps_contrast_and_zoom_while_the_user_works(jupyter, open_page, hardware_adapter, browser_out):
    name = "show3d_stability.ipynb"
    jupyter.write_notebook(name, NOTEBOOK)
    page = jupyter.open(open_page, name, "show3d-stability", webgpu=True, ready='text=Waiting for the first frame', out=browser_out)
    hardware_adapter(page.adapter())
    jupyter.run_in_kernel(name, APPEND.format(first=0, stop=4))
    page.wait_for_selector('[data-show3d-gpu-residency="ready"]', timeout=120_000)
    page.wait_for_timeout(5_000)
    assert page.render_paths() == ["WebGPU"]
    state = json.loads(jupyter.run_in_kernel(name, READ))
    assert state["frames"] == 4 and state["frame"] == 3 and state["auto_contrast"]
    page.evaluate("() => document.querySelector('input[aria-label=\"Histogram intensity clip range\"]').scrollIntoView({block: 'end'})")
    page.wait_for_timeout(1_000)

    # 1. grabbing a histogram handle under Auto contrast does not change the picture: the handles sit at what is shown
    rest = shot(page, browser_out / "show3d_stability_rest.png")
    handle = page.evaluate(UPPER_HANDLE)
    page.mouse.move(handle["x"], handle["y"])
    page.mouse.down()
    page.mouse.move(handle["x"] - 1, handle["y"])
    page.wait_for_timeout(400)
    grabbed = shot(page, browser_out / "show3d_stability_grabbed.png")
    page.mouse.up()
    page.wait_for_timeout(1_500)
    released = shot(page)
    grab_jump = abs(grabbed.mean() - rest.mean())
    assert grab_jump < 4 and abs(released.mean() - grabbed.mean()) < 2, (rest.mean(), grabbed.mean(), released.mean())
    manual = json.loads(jupyter.run_in_kernel(name, READ))
    assert not manual["auto_contrast"]   # the user's limits now

    # 2. zoom in on the image, then: a frame arrives, another frame is shown, the frame's sign is flipped
    box = page.locator(IMAGE).bounding_box()
    centre = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    before = period_px(shot(page))
    page.mouse.move(*centre)
    page.mouse.wheel(0, -400)
    page.wait_for_timeout(1_500)
    zoomed_image = shot(page, browser_out / "show3d_stability_zoomed.png")
    zoomed = period_px(zoomed_image)
    assert zoomed > 1.25 * before, (before, zoomed)   # one wheel notch zooms by 1.35
    page.locator(".show3d-root").focus()

    jupyter.run_in_kernel(name, APPEND.format(first=4, stop=6))
    page.wait_for_timeout(4_000)
    arrived = json.loads(jupyter.run_in_kernel(name, READ))
    assert arrived["frames"] == 6
    after_arrival = period_px(shot(page, browser_out / "show3d_stability_after_arrival.png"))
    assert after_arrival == pytest.approx(zoomed, rel=0.03), (zoomed, after_arrival)
    assert arrived["limits"] == manual["limits"] and not arrived["auto_contrast"]   # the user's limits survive new frames

    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(1_500)
    stepped = json.loads(jupyter.run_in_kernel(name, READ))
    assert stepped["frame"] == arrived["frame"] - 1
    after_step = period_px(shot(page))
    assert after_step == pytest.approx(zoomed, rel=0.03), (zoomed, after_step)
    assert stepped["limits"] == manual["limits"]

    # 3. f flips the sign of the frame on screen only, s stars it; neither moves the view
    unflipped = shot(page, browser_out / "show3d_stability_unflipped.png")
    page.keyboard.press("f")
    page.wait_for_timeout(1_500)
    flipped = shot(page, browser_out / "show3d_stability_flipped.png")
    marks = json.loads(jupyter.run_in_kernel(name, READ))
    assert marks["flips"] == [index == marks["frame"] for index in range(6)]
    assert period_px(flipped) == pytest.approx(zoomed, rel=0.03)
    # the negative of the same view: bright and dark swap about mid gray
    assert np.corrcoef(unflipped.ravel(), flipped.ravel())[0, 1] < -0.95, np.corrcoef(unflipped.ravel(), flipped.ravel())[0, 1]
    page.keyboard.press("s")
    page.wait_for_timeout(1_000)
    starred = json.loads(jupyter.run_in_kernel(name, READ))
    assert starred["stars"] == [starred["frame"]] and starred["flips"] == marks["flips"]
    assert np.abs(shot(page) - flipped).mean() < 2   # a star is a mark: the picture is as it was
    assert page.locator('[data-testid="show3d-thumbnail-flip-sign"]').count() == 1
    assert page.locator('[data-testid="show3d-thumbnail-star"]').count() == 1
    (browser_out / "show3d_stability.json").write_text(json.dumps({
        "grab_jump_gray_levels": grab_jump, "period_px": {"before": before, "zoomed": zoomed, "after_arrival": after_arrival, "after_step": after_step},
        "kernel": starred}, indent=1))
