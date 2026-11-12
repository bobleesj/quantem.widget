"""Show3D histogram handles in a live kernel: the image follows the handle while it is held.

A trusted mouse press on the upper clip handle, 24 moves with the button down and no release.
While the button is still down the image on screen changes from move to move, the limit under
the histogram follows, every move is answered by a WebGPU draw of the image canvas, and nothing
goes to the kernel. Releasing commits the limits to the kernel once, with no binary buffer.
Checked for a streamed 2048 x 2048 viewer with the FFT inset, a plain stack, and flip_sign on.
"""

import io
import json
import statistics

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser

IMAGE = 'canvas[data-quantem-scientific-output="show3d-image"]'
MOVES = 24
FRAMES = """import numpy as np
from quantem.widget import Show3D

rows, cols = np.indices((2048, 2048), dtype=np.float32)
frames = [((rows + cols) / 4094 + 0.05 * index).astype(np.float32) for index in range(6)]   # smooth ramps: every limit shows
"""
RECORD = """
_changes, _buffers = [], []
viewer.observe(lambda change: _changes.append(change["name"]))
_send = viewer._send
def _recorded(msg, buffers=None):
    _buffers.append(sum(memoryview(buffer).nbytes for buffer in buffers or []))
    return _send(msg, buffers)
viewer._send = _recorded
viewer"""
STREAM = FRAMES + """viewer = Show3D.stream(title="t", cmap="gray", dim_label="Scan", save_state=True, show_fft=True, fft_layout="overlay", show_scale_bar=False{extra})
viewer.fft_overlay_position, viewer.fft_overlay_size = "bottom-left", 0.3""" + RECORD
APPEND = """for index, frame in enumerate(frames):
    viewer.append(frame, label=f"[{index}] a")
assert viewer._stream_wait_sent(60)"""
STACK = FRAMES + """viewer = Show3D(np.stack(frames), cmap="gray", show_scale_bar=False{extra})""" + RECORD
READ = """import json
print(json.dumps({"changes": _changes, "sent_bytes": sum(_buffers), "limits": [viewer.image_vmin_pct, viewer.image_vmax_pct],
                  "frames": int(viewer.n_slices), "auto_contrast": viewer.auto_contrast}))"""
# Page-side counters, installed by the test (the widget carries none): each WebGPU frame drawn to
# the image (its WebGPU canvas lies over the canvas that carries the label), each pointer move the page received, and each message the page sends to the kernel.
COUNTERS = """
() => {
  const log = window.__drag = {moves: [], draws: [], sent: 0, sentBytes: 0};
  const image = document.querySelector('canvas[data-quantem-scientific-output="show3d-image"]');
  const getCurrentTexture = GPUCanvasContext.prototype.getCurrentTexture;
  GPUCanvasContext.prototype.getCurrentTexture = function () {
    if (this.canvas.parentElement === image.parentElement) log.draws.push(performance.now());
    return getCurrentTexture.call(this);
  };
  const send = WebSocket.prototype.send;
  WebSocket.prototype.send = function (data) {
    log.sent += 1;
    log.sentBytes += typeof data === 'string' ? data.length : (data.byteLength ?? data.size ?? 0);
    return send.call(this, data);
  };
  document.addEventListener('pointermove', () => log.moves.push(performance.now()), true);
}
"""
LIMITS = """
() => {
  const slider = document.querySelector('input[aria-label="Histogram intensity clip range"]');
  let box = slider;
  while (box && !(box.nextElementSibling && box.nextElementSibling.children.length === 2 && !box.nextElementSibling.querySelector('canvas, input'))) box = box.parentElement;
  return Array.from(box.nextElementSibling.children).map(label => parseFloat(label.textContent));
}
"""
HANDLES = """
() => {
  let box = document.querySelector('input[aria-label="Histogram intensity clip range"]');
  while (box && !box.classList.contains('MuiSlider-root')) box = box.parentElement;
  const rail = box.getBoundingClientRect();
  return {rail: {x: rail.x, width: rail.width}, thumbs: Array.from(box.querySelectorAll('.MuiSlider-thumb')).map(thumb => {
    const rect = thumb.getBoundingClientRect();
    return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};
  })};
}
"""


def image_level(page, path=None) -> float:
    """Mean gray level of the upper right quarter of the image on screen (clear of the FFT inset)."""
    box = page.locator(IMAGE).bounding_box()
    clip = {"x": box["x"] + box["width"] / 2, "y": box["y"], "width": box["width"] / 2, "height": box["height"] / 2}
    png = page.screenshot(clip=clip)
    if path is not None:
        path.write_bytes(png)
    return float(np.asarray(Image.open(io.BytesIO(png)).convert("L")).mean())


CASES = {
    "stream_fft": (STREAM.format(extra=""), True, False),
    "stack": (STACK.format(extra=""), False, False),
    "stream_fft_flip_sign": (STREAM.format(extra=", flip_sign=True"), True, True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_show3d_histogram_handle_redraws_the_image_on_every_move_while_held(case, jupyter, open_page, hardware_adapter, browser_out):
    notebook, stream, flip_sign = CASES[case]
    name = f"show3d_histogram_drag_{case}.ipynb"
    jupyter.write_notebook(name, notebook)
    page = jupyter.open(open_page, name, f"show3d-histogram-drag-{case}", webgpu=True,
                        ready='text=Waiting for the first frame' if stream else '[data-show3d-gpu-residency="ready"]', out=browser_out)
    adapter = page.adapter()
    hardware_adapter(adapter)
    if stream:
        jupyter.run_in_kernel(name, APPEND)
        page.wait_for_selector('[data-show3d-gpu-residency="ready"]', timeout=120_000)
        page.wait_for_timeout(6_000)
    assert page.render_paths() == ["WebGPU"]
    assert json.loads(jupyter.run_in_kernel(name, READ))["frames"] == 6
    # the histogram sits under the image: bring both into the window
    page.evaluate("() => document.querySelector('input[aria-label=\"Histogram intensity clip range\"]').scrollIntoView({block: 'end'})")
    page.wait_for_timeout(1_000)
    jupyter.run_in_kernel(name, "_changes.clear(); _buffers.clear()")
    page.evaluate(COUNTERS)

    handles = page.evaluate(HANDLES)
    # The upper handle of the shown window; under flip_sign it holds the stored lower limit.
    start = max(handles["thumbs"], key=lambda thumb: thumb["x"])
    travel = 0.6 * handles["rail"]["width"]
    limits = [page.evaluate(LIMITS)]
    levels = [image_level(page, browser_out / f"show3d_histogram_drag_{case}_before.png")]
    page.mouse.move(start["x"], start["y"])
    page.mouse.down()
    page.wait_for_timeout(300)
    page.evaluate("() => { const log = window.__drag; log.moves.length = 0; log.draws.length = 0; log.sent = 0; log.sentBytes = 0; }")
    for step in range(1, MOVES + 1):
        page.mouse.move(start["x"] - travel * step / MOVES, start["y"])
        page.wait_for_timeout(50)
        if step % 4 == 0:   # the button is still down
            limits.append(page.evaluate(LIMITS))
            levels.append(image_level(page, browser_out / f"show3d_histogram_drag_{case}_held_{step:02d}.png"))
    page.wait_for_timeout(500)
    held = page.evaluate("() => window.__drag")
    held_kernel = json.loads(jupyter.run_in_kernel(name, READ))
    page.mouse.up()
    page.wait_for_timeout(2_000)
    limits.append(page.evaluate(LIMITS))
    levels.append(image_level(page, browser_out / f"show3d_histogram_drag_{case}_released.png"))
    released = page.evaluate("() => window.__drag")
    released_kernel = json.loads(jupyter.run_in_kernel(name, READ))

    moves, draws = held["moves"], held["draws"]
    # A move is answered when the image canvas is drawn before the next move arrives.
    waits = [next((draw - move for draw in draws if draw >= move), None) for move in moves]
    answered = [wait is not None and move + wait < (moves[index + 1] if index + 1 < len(moves) else move + 50)
                for index, (move, wait) in enumerate(zip(moves, waits))]
    summary = {
        "adapter": adapter, "moves": len(moves), "image_draws_while_held": len(draws),
        "answered_share": sum(answered) / max(1, len(moves)),
        "wait_ms_median": statistics.median(wait for wait in waits if wait is not None) if any(wait is not None for wait in waits) else None,
        "wait_ms_worst": max((wait for wait in waits if wait is not None), default=None),
        "levels": levels, "limits": limits,
        "page_messages_while_held": [held["sent"], held["sentBytes"]],
        "page_messages_at_release": [released["sent"] - held["sent"], released["sentBytes"] - held["sentBytes"]],
        "kernel_while_held": held_kernel, "kernel_after_release": released_kernel,
    }
    (browser_out / f"show3d_histogram_drag_{case}.json").write_text(json.dumps(summary, indent=1))

    held_levels, held_limits = levels[1:-1], limits[1:-1]
    assert len(moves) == MOVES
    # the picture changes between successive looks while the button is down, always the same way
    steps = np.diff([levels[0], *held_levels])
    assert (np.abs(steps) > 1).all() and (np.sign(steps) == np.sign(steps[0])).all(), summary
    # the limit under the dragged handle follows, the other stays
    assert (np.diff([limits[0][1], *(limit[1] for limit in held_limits)]) < 0).all(), summary
    assert all(limit[0] == limits[0][0] for limit in held_limits), summary
    # every move is answered by a draw of the image canvas within the frame
    assert summary["answered_share"] == 1 and summary["wait_ms_worst"] < 34, summary
    # nothing goes to the kernel while the handle is held; the release commits the limits, no frame bytes
    assert held["sent"] == 0 and held_kernel["changes"] == [] and held_kernel["sent_bytes"] == 0, summary
    # _property_lock is ipywidgets' own bookkeeping of a frontend update, not a trait of the viewer
    assert set(released_kernel["changes"]) - {"_property_lock"} <= {"image_vmin_pct", "image_vmax_pct", "auto_contrast"}, summary
    assert released_kernel["changes"] and released_kernel["sent_bytes"] == 0, summary
    assert levels[-1] == pytest.approx(held_levels[-1], abs=1) and limits[-1] == held_limits[-1], summary
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
