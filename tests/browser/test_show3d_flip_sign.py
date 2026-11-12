"""Show3D flip_sign in a live kernel: the screen shows -value and no frame is sent again.

A ramp with one bright blob is not symmetric about its mid value, so its negative is a different
picture: the blob turns dark and the dark side of the ramp turns bright. Between the negated
limits every screen pixel is the mirror gray level (255 - level) of the unflipped picture, with
WebGPU on and off, toggled from Python and from the More menu. The numbers beside the image
(histogram limits, statistics, cursor value) are negated, and the kernel sends no binary buffer
while the sign flips. A stream keeps the flip across an appended frame.
"""

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser

IMAGE = 'canvas[data-quantem-scientific-output="show3d-image"]'
BLOB_ROW, BLOB_COL = 12, 22            # data pixel inside the bright blob
DARK_ROW, DARK_COL = 28, 2             # data pixel at the dark end of the ramp
STACK = """import numpy as np
from quantem.widget import Show3D

rows, cols = np.indices((64, 64))
frame = (cols / 63 * 2 - 0.5).astype(np.float32)      # ramp from -0.5 to 1.5
frame[10:15, 20:25] += 3.0                            # one bright blob
stack = np.stack([frame + 0.1 * index for index in range(3)]).astype(np.float32)
"""
NOTEBOOK = STACK + """viewer = Show3D(stack, cmap="gray", panel_width_px=256, auto_contrast=False, show_stats=True)
_buffers = []
_send = viewer._send
def _recorded(msg, buffers=None):
    _buffers.append(sum(memoryview(buffer).nbytes for buffer in buffers or []))
    return _send(msg, buffers)
viewer._send = _recorded
viewer"""
STREAM_NOTEBOOK = STACK + """viewer = Show3D.stream(stack[:2], cmap="gray", panel_width_px=256, auto_contrast=False)
viewer"""
READ = """import json
print(json.dumps({"flip_sign": viewer.flip_sign, "sent_bytes": sum(_buffers), "data_min": float(viewer._data.min()),
                  "data_max": float(viewer._data.max()), "vmin": viewer.data_min, "vmax": viewer.data_max}))"""
# Histogram limits (the two labels under the first histogram), the statistics row and nothing else.
READOUTS = """
() => {
  const slider = document.querySelector('input[aria-label="Histogram intensity clip range"]');
  let box = slider;
  while (box && !(box.nextElementSibling && box.nextElementSibling.children.length === 2 && !box.nextElementSibling.querySelector('canvas, input'))) box = box.parentElement;
  const stats = {};
  for (const p of document.querySelectorAll('p')) {
    const match = /^(Mean|Min|Max|Std) /.exec(p.textContent);
    if (match && !(match[1] in stats)) stats[match[1]] = parseFloat(p.querySelector('span').textContent);
  }
  return {limits: Array.from(box.nextElementSibling.children).map(label => parseFloat(label.textContent)), stats};
}
"""


# The histogram is a Canvas2D drawing, read in place: an element screenshot would scroll the notebook.
HISTOGRAM = """
() => {
  let box = document.querySelector('input[aria-label="Histogram intensity clip range"]');
  while (box && !box.querySelector('canvas')) box = box.parentElement;
  return box.querySelector('canvas').toDataURL('image/png');
}
"""


def histogram_ink(page, path) -> np.ndarray:
    """How much bar each column of the histogram canvas holds (the canvas is saved to ``path``)."""
    path.write_bytes(base64.b64decode(page.evaluate(HISTOGRAM).split(",", 1)[1]))
    gray = np.asarray(Image.open(path).convert("L")).astype(float)
    return (gray.max() - gray).sum(axis=0)


def screen_levels(page) -> np.ndarray:
    """Gray level of every screen pixel in the upper 256 x 160 CSS pixels of the image (clear of the scale bar)."""
    box = page.locator(IMAGE).bounding_box()
    assert round(box["width"]) == 256 and round(box["height"]) == 256   # 4 screen pixels per data pixel
    page.mouse.move(5, 5)                                               # no cursor readout over the image
    page.wait_for_timeout(700)
    clip = {"x": box["x"], "y": box["y"], "width": 256, "height": 160}
    rgb = np.asarray(Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("RGB")).astype(int)
    assert np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])   # gray colormap
    return rgb[..., 0]


def level_at(levels: np.ndarray, row: int, col: int) -> int:
    return int(levels[row * 4 + 2, col * 4 + 2])


def cursor_value(page, row: int, col: int) -> float:
    box = page.locator(IMAGE).bounding_box()
    page.mouse.move(box["x"] + col * 4 + 2, box["y"] + row * 4 + 2)
    page.wait_for_timeout(500)
    text = page.locator(".show3d-cursor-readout").inner_text()
    assert text.startswith(f"({row}, {col})"), text
    return float(text.split(")")[1])


def set_from_python(jupyter, page, notebook: str, code: str) -> None:
    jupyter.run_in_kernel(notebook, code)
    page.wait_for_timeout(1_500)


def toggle_in_more_menu(page) -> None:
    page.get_by_label("More tools").click()
    switch = page.get_by_label("Toggle sign flip")
    assert "Flip sign: show −value" in page.get_by_text("Flip Sign", exact=True).get_attribute("title")
    switch.click()
    page.keyboard.press("Escape")
    page.wait_for_timeout(1_500)


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_flip_sign_shows_the_exact_negative_without_sending_frames(webgpu, jupyter, open_page, hardware_adapter, browser_out):
    tag = "webgpu" if webgpu else "cpu"
    name = f"show3d_flip_sign_{tag}.ipynb"
    jupyter.write_notebook(name, NOTEBOOK)
    page = jupyter.open(open_page, name, f"show3d-flip-sign-{tag}", webgpu=webgpu,
                        ready=f'[data-show3d-gpu-residency="{"ready" if webgpu else "fallback"}"]', out=browser_out)
    adapter = page.adapter() if webgpu else None
    if webgpu:
        hardware_adapter(adapter)
    assert page.render_paths() == ["WebGPU" if webgpu else "CPU"]
    jupyter.run_in_kernel(name, "_buffers.clear()")     # from here on: display toggles only

    plain = screen_levels(page)
    page.screenshot(path=str(browser_out / f"show3d_flip_sign_{tag}_off.png"))
    plain_numbers = page.evaluate(READOUTS)
    plain_ink = histogram_ink(page, browser_out / f"show3d_flip_sign_{tag}_histogram_off.png")
    plain_blob, plain_dark = cursor_value(page, BLOB_ROW, BLOB_COL), cursor_value(page, DARK_ROW, DARK_COL)
    # stored limits are the stack range [-0.5, 3.46]; the blob is bright and the ramp's left end dark
    assert plain_numbers["limits"] == pytest.approx([-0.5, 3.46], abs=0.006)
    assert level_at(plain, BLOB_ROW, BLOB_COL) > 150 and level_at(plain, DARK_ROW, DARK_COL) < 20

    set_from_python(jupyter, page, name, "viewer.flip_sign = True")
    flipped = screen_levels(page)
    page.screenshot(path=str(browser_out / f"show3d_flip_sign_{tag}_on_from_python.png"))
    assert np.array_equal(flipped, 255 - plain)         # the exact negative, pixel for pixel
    assert level_at(flipped, BLOB_ROW, BLOB_COL) < 105 and level_at(flipped, DARK_ROW, DARK_COL) > 235
    numbers = page.evaluate(READOUTS)
    ink = histogram_ink(page, browser_out / f"show3d_flip_sign_{tag}_histogram_on.png")
    # The histogram is the stored one mirrored: the ramp's bars move from the left to the right. The
    # bars sit on a half-pixel grid, so compare where the ink lies, not single pixels.
    columns = np.arange(ink.size)
    centre, plain_centre = (ink * columns).sum() / ink.sum(), (plain_ink * columns).sum() / plain_ink.sum()
    assert plain_centre < 0.35 * ink.size and centre == pytest.approx(ink.size - 1 - plain_centre, abs=1.5)
    assert ink.sum() == pytest.approx(plain_ink.sum(), rel=0.02)
    stats, shown = plain_numbers["stats"], numbers["stats"]
    assert (shown["Mean"], shown["Min"], shown["Max"], shown["Std"]) == pytest.approx(
        (-stats["Mean"], -stats["Max"], -stats["Min"], stats["Std"]), abs=1e-9)
    assert cursor_value(page, BLOB_ROW, BLOB_COL) == pytest.approx(-plain_blob, abs=1e-9)
    assert cursor_value(page, DARK_ROW, DARK_COL) == pytest.approx(-plain_dark, abs=1e-9)
    assert plain_blob > 3 and plain_dark < 0

    # the More menu switch turns it off and on again, and Python follows
    toggle_in_more_menu(page)
    assert np.array_equal(screen_levels(page), plain)
    assert json.loads(jupyter.run_in_kernel(name, READ))["flip_sign"] is False
    toggle_in_more_menu(page)
    assert np.array_equal(screen_levels(page), flipped)
    page.screenshot(path=str(browser_out / f"show3d_flip_sign_{tag}_on_from_menu.png"))

    # auto contrast (percentile limits) and the colorbar: still the mirror of the unflipped picture
    page.get_by_label("Toggle colorbar overlay").click()
    set_from_python(jupyter, page, name, "viewer.auto_contrast = True")
    auto_flipped = screen_levels(page)
    page.screenshot(path=str(browser_out / f"show3d_flip_sign_{tag}_auto_on.png"))
    set_from_python(jupyter, page, name, "viewer.flip_sign = False")
    auto_plain = screen_levels(page)
    page.screenshot(path=str(browser_out / f"show3d_flip_sign_{tag}_auto_off.png"))
    bar = (slice(None), slice(0, 200))                  # left of the colorbar and its labels
    assert np.array_equal(auto_flipped[bar], 255 - auto_plain[bar]) and not np.array_equal(auto_plain, plain)

    held = json.loads(jupyter.run_in_kernel(name, READ))
    # the stored stack never changed and no binary buffer left the kernel during any toggle
    assert held["sent_bytes"] == 0 and held["flip_sign"] is False
    assert (held["data_min"], held["data_max"]) == pytest.approx((-0.5, 3.4619), abs=1e-4)
    (browser_out / f"show3d_flip_sign_{tag}.json").write_text(json.dumps({
        "adapter": adapter, "limits": plain_numbers["limits"], "limits_flipped": numbers["limits"],
        "stats": stats, "stats_flipped": shown, "blob_level": [level_at(plain, BLOB_ROW, BLOB_COL), level_at(flipped, BLOB_ROW, BLOB_COL)],
        "dark_level": [level_at(plain, DARK_ROW, DARK_COL), level_at(flipped, DARK_ROW, DARK_COL)], "kernel": held,
    }, indent=1))
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors


def test_show3d_stream_keeps_the_flip_across_an_appended_frame(jupyter, open_page, hardware_adapter, browser_out):
    name = "show3d_flip_sign_stream.ipynb"
    jupyter.write_notebook(name, STREAM_NOTEBOOK)
    page = jupyter.open(open_page, name, "show3d-flip-sign-stream", webgpu=True,
                        ready='[data-show3d-gpu-residency="ready"]', out=browser_out)
    hardware_adapter(page.adapter())
    plain = screen_levels(page)
    set_from_python(jupyter, page, name, "viewer.flip_sign = True")
    assert np.array_equal(screen_levels(page), 255 - plain)
    # a new frame arrives while flipped: it is drawn as its negative too
    set_from_python(jupyter, page, name, "viewer.append(stack[2] + 0.4)\nassert viewer._stream_wait_sent(10)")
    appended_flipped = screen_levels(page)
    page.screenshot(path=str(browser_out / "show3d_flip_sign_stream_on.png"))
    set_from_python(jupyter, page, name, "viewer.flip_sign = False")
    appended_plain = screen_levels(page)
    page.screenshot(path=str(browser_out / "show3d_flip_sign_stream_off.png"))
    assert np.array_equal(appended_flipped, 255 - appended_plain) and not np.array_equal(appended_plain, plain)
    assert level_at(appended_plain, BLOB_ROW, BLOB_COL) > level_at(appended_plain, DARK_ROW, DARK_COL)
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
