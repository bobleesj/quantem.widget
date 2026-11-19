"""Show4DSTEM in a live Jupyter kernel, driven like a user with WebGPU on and off.

A private JupyterLab server runs one notebook per mode on the public gold
4D-STEM dataset. The same clicks and drags run in both: three scan positions,
move and resize the detector, BF and ADF presets, a colormap change and a
contrast drag. After each step the diffraction, virtual-image and FFT panels
must draw identical pixels in both modes, the virtual image must equal the
masked detector sum computed in the kernel, and the badge must name the path.

A third notebook holds a 16 x 16 x 16 x 16 synthetic with large pixels: the
scan ROI is resized from its rim and must grow by the distance dragged, and
profile points clicked on pixel centers must be those pixels' indices (the
convention the line sampler reads) and be drawn on those centers. A click
that moves the scan ROI must reach the kernel as one reduction of the new region,
and a click that moves the detector, or its Reset, as one reduction of the image.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser

DP = '[data-quantem-scientific-output="show4dstem-diffraction-pattern"]'
VI = '[data-quantem-scientific-output="show4dstem-virtual-image"]'
FFT = '[data-quantem-scientific-output="show4dstem-fft"]'
REPORT = r"""
import json as _json, numpy as _np
_vi = viewer.virtual_image()
_mask = viewer._current_detector_mask().astype(_np.float64)
_points = [(0, 0), (viewer.shape_rows // 2, viewer.shape_cols // 3), (viewer.shape_rows - 1, viewer.shape_cols - 1)]
print(_json.dumps({
    "pos": [viewer.pos_row, viewer.pos_col],
    "roi": [viewer.roi_mode, viewer.roi_center_row, viewer.roi_center_col, viewer.roi_radius, viewer.roi_radius_inner],
    "vi_colormap": viewer.vi_colormap, "vi_vmin_pct": viewer.vi_vmin_pct,
    "vi_at": [float(_vi[row, col]) for row, col in _points],
    "masked_sum_at": [float((viewer.pattern(row, col).astype(_np.float64) * _mask).sum()) for row, col in _points],
}))
"""


@pytest.fixture(scope="module", autouse=True)
def notebooks(jupyter):
    source = ("from quantem.widget import Show4DSTEM\nfrom quantem.widget.datasets import show4dstem_gold\n\n"
              "viewer = Show4DSTEM(show4dstem_gold(size=\"medium\"), show_fft=True)\nviewer")
    synthetic = ("import numpy as np\nfrom quantem.widget import Show4DSTEM\n\n"
                 "rows, cols = np.mgrid[0:16, 0:16]\ndata = np.full((16, 16, 16, 16), 10, np.uint16)\n"
                 "data[:, :, 5, 9] = 1000\ndata += (rows * 16 + cols)[:, :, None, None].astype(np.uint16)\n"
                 "viewer = Show4DSTEM(data)\nviewer")
    for name, cell in (("show4dstem_webgpu", source), ("show4dstem_cpu", source), ("show4dstem_pixels", synthetic)):
        jupyter.write_notebook(f"{name}.ipynb", cell)


DETECTOR_NOTEBOOK = """
import numpy as np
from quantem.widget import Show4DSTEM

viewer = Show4DSTEM(np.ones((4, 4, 32, 32), dtype=np.uint16), device="cpu",
                    bright_field_center_px=(15.5, 15.5), bright_field_radius_px=6)
row, col = np.indices((4, 4))
viewer.attach("COM_row", row)
viewer.attach("COM_col", col)
viewer
"""


def test_com_hover_crosses_the_space_between_labels(jupyter, open_page, browser_out):
    notebook = "show4dstem_com_hover.ipynb"
    jupyter.write_notebook(notebook, DETECTOR_NOTEBOOK)
    page = jupyter.open(open_page, notebook, "com_hover", webgpu=False, ready=DP, out=browser_out)
    row = page.get_by_role("button", name="CoM_r", exact=True)
    col = page.get_by_role("button", name="CoM_c", exact=True)
    row.hover()
    page.get_by_title("CoM_r | 4×4", exact=True).wait_for()
    left, right = row.bounding_box(), col.bounding_box()
    gap_start = left["x"] + left["width"]
    gap = right["x"] - gap_start
    assert gap > 2
    for fraction, label in ((0.25, "CoM_r"), (0.75, "CoM_c")):
        page.mouse.move(gap_start + gap * fraction, left["y"] + left["height"] / 2)
        page.get_by_title(f"{label} | 4×4", exact=True).wait_for(timeout=5_000)
    page.mouse.move(5, 300)
    page.get_by_title("ROI | 4×4", exact=True).wait_for()
    assert jupyter.run_in_kernel(notebook, "print(viewer.vi_source)") == "roi"


def test_detector_preset_after_notebook_reload(jupyter, open_page, browser_out):
    """The refreshed view recovers its preset from the kernel and clears it for a custom annulus."""
    notebook = "show4dstem_detector_reload.ipynb"
    jupyter.write_notebook(notebook, DETECTOR_NOTEBOOK)
    page = jupyter.open(open_page, notebook, "detector_reload", webgpu=False, ready=DP, out=browser_out)
    adf = page.get_by_role("button", name="ADF", exact=True)
    adf.click()
    selected = page.get_by_role("button", name="ADF", exact=True, pressed=True)
    selected.wait_for()
    with page.expect_response(lambda response: response.request.method == "PUT" and notebook in response.url):
        page.keyboard.press("ControlOrMeta+s")
    page.reload()
    selected.wait_for(timeout=60_000)
    detector = jupyter.run_in_kernel(notebook, "import json; print(json.dumps([viewer.roi_mode, viewer.roi_radius]))")
    assert json.loads(detector) == ["annular", 12]
    jupyter.run_in_kernel(notebook, "viewer.roi_radius = 11")
    page.get_by_role("button", name="ADF", exact=True, pressed=False).wait_for()


def test_live_roi_paints_changed_rows_and_restores_the_retained_canvas(jupyter, open_page, hardware_adapter, browser_out):
    """New positions repaint one native strip while preserving the earlier measured pixels."""
    notebook = "show4dstem_live_rows.ipynb"
    jupyter.write_notebook(notebook, DETECTOR_NOTEBOOK)
    page = jupyter.open(open_page, notebook, "live_rows", webgpu=True, ready=DP, out=browser_out)
    hardware_adapter(page.adapter())
    jupyter.run_in_kernel(notebook, """
viewer.vi_colormap = "gray"
viewer.vi_vmin = 0
viewer.vi_vmax = 100
viewer.dp_colormap = "gray"
viewer.dp_scale_mode = "linear"
viewer.dp_vmin = 0
viewer.dp_vmax = 100
viewer.live_phase_preview = dict(scan="live_1", total=128, rows=128, cols=128,
                                 detRows=32, detCols=32, mode="Integrated intensity", upsample=1)
def positions(row, value):
    viewer.send(dict(type="live_image_positions", scan="live_1", generation=0,
                     start=128*row, count=128, total=128*128),
                buffers=[memoryview(np.full(128, value, np.float32))])
def pattern(value):
    viewer.live_pattern = np.full((32, 32), value, np.float32).tobytes()
positions(0, 25)
pattern(25)
""")
    pixel = """([selector, size, row, expected]) => {
      const canvas = document.querySelector(selector);
      return canvas?.height === size
        && Math.abs(canvas.getContext("2d").getImageData(size / 2, row, 1, 1).data[0] - expected) <= 1;
    }"""
    page.wait_for_function(pixel, arg=[VI, 128, 0, 63])
    page.wait_for_function(pixel, arg=[DP, 32, 16, 63])
    page.evaluate("""selector => {
      const paint = CanvasRenderingContext2D.prototype.putImageData;
      const canvas = document.querySelector(selector);
      window.liveDpPaints = [];
      CanvasRenderingContext2D.prototype.putImageData = function (...args) {
        if (this.canvas === canvas) window.liveDpPaints.push(args[0].data[0]);
        return paint.apply(this, args);
      };
    }""", DP)
    jupyter.run_in_kernel(notebook, "pattern(50)")
    page.wait_for_function(pixel, arg=[DP, 32, 16, 127])
    page.get_by_text("50.00", exact=True).first.wait_for(state="attached", timeout=2_000)
    assert page.evaluate("window.liveDpPaints.filter(value => value === 127).length") == 1
    jupyter.run_in_kernel(notebook, "pattern(25)")
    page.wait_for_function(pixel, arg=[DP, 32, 16, 63])
    page.evaluate("""(selector) => {
      const visible = document.querySelector(selector);
      const paint = CanvasRenderingContext2D.prototype.putImageData;
      window.liveStrips = [];
      CanvasRenderingContext2D.prototype.putImageData = function (...args) {
        if (this.canvas === visible) window.liveStrips.push(args.slice(1));
        return paint.apply(this, args);
      };
    }""", VI)
    jupyter.run_in_kernel(notebook, "positions(1, 75)")
    page.wait_for_function(pixel, arg=[VI, 128, 1, 191])
    assert page.evaluate(pixel, [VI, 128, 0, 63])
    assert [0, 0, 0, 1, 128, 1] in page.evaluate("window.liveStrips")

    # The first zoom needs the latest pattern, even when the unzoomed
    # fast path has never populated the staging canvas.
    page.evaluate("""(selector) => {
      window.zoomedPatternDraws = 0;
      const draw = CanvasRenderingContext2D.prototype.drawImage;
      CanvasRenderingContext2D.prototype.drawImage = function (...args) {
        if (this.canvas.matches(selector)) window.zoomedPatternDraws++;
        return draw.apply(this, args);
      };
    }""", DP)
    dp_box = page.locator(DP).bounding_box()
    page.mouse.move(dp_box["x"] + dp_box["width"] / 2, dp_box["y"] + dp_box["height"] / 2)
    page.mouse.wheel(0, -160)
    page.wait_for_function("window.zoomedPatternDraws > 0")
    assert page.evaluate(pixel, [DP, 32, 16, 63])
    jupyter.run_in_kernel(notebook, "pattern(75)")
    page.wait_for_function(pixel, arg=[DP, 32, 16, 191])

    jupyter.run_in_kernel(notebook, "viewer.live_phase_preview = {}")
    page.wait_for_function("""selector => {
      const canvas = document.querySelector(selector);
      return canvas?.height === 4 && canvas.getContext("webgpu") != null;
    }""", arg=VI)


def test_live_virtual_image_restores_the_matching_size_until_clear(jupyter, open_page, browser_out):
    notebook = "show4dstem_live_underlay.ipynb"
    jupyter.write_notebook(notebook, DETECTOR_NOTEBOOK)
    page = jupyter.open(open_page, notebook, "live_underlay", webgpu=False, ready=DP, out=browser_out)
    jupyter.run_in_kernel(notebook, """
viewer.vi_colormap = "gray"
viewer.vi_vmin, viewer.vi_vmax = 0, 100

def preview(scan, size, rows, value):
    viewer.live_phase_preview = dict(scan=scan, total=size, rows=size, cols=size, detRows=32,
                                     detCols=32, mode="Integrated intensity", upsample=1)
    viewer.send(dict(type="live_image_positions", scan=scan, generation=0,
                     start=0, count=rows*size, total=size*size),
                buffers=[memoryview(np.full(rows*size, value, np.float32))])

preview("series_1", 512, 512, 25)
""")
    pixel = """([selector, size, row, expected]) => {
      const canvas = document.querySelector(selector);
      return canvas?.height === size && canvas.width === size
        && Math.abs(canvas.getContext('2d').getImageData(size / 2, row, 1, 1).data[0] - expected) <= 1;
    }"""
    page.wait_for_function(pixel, arg=[VI, 512, 511, 63])
    jupyter.run_in_kernel(notebook, 'preview("live_1", 128, 128, 50)')
    page.wait_for_function(pixel, arg=[VI, 128, 127, 127])
    jupyter.run_in_kernel(notebook, 'preview("series_2", 512, 1, 75)')
    page.wait_for_function(pixel, arg=[VI, 512, 0, 191])
    assert page.evaluate(pixel, [VI, 512, 511, 63])

    jupyter.run_in_kernel(notebook, 'viewer.send({"type": "live_image_clear"}); preview("series_3", 512, 1, 50)')
    page.wait_for_function(pixel, arg=[VI, 512, 0, 127])
    assert page.evaluate(pixel, [VI, 512, 511, 0])
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors


def test_live_direct_restarts_with_a_smaller_scan_and_larger_detector(jupyter, open_page, browser_out):
    """An earlier retained image can outlive the shape traits as Live restarts."""
    notebook = "show4dstem_live_direct_resize.ipynb"
    jupyter.write_notebook(notebook, DETECTOR_NOTEBOOK)
    page = jupyter.open(open_page, notebook, "live_direct_resize", webgpu=False, ready=DP, out=browser_out)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    jupyter.run_in_kernel(notebook, """
def retained(size, value):
    with viewer.hold_sync():
        viewer.shape_rows = viewer.shape_cols = size
        viewer.virtual_image_bytes = np.full((size, size), value, np.float32).tobytes()

def direct(name, size, detector, value):
    viewer.live_phase_preview = dict(scan=name, rows=size, total=size, cols=size,
                                     detRows=detector, detCols=detector, mode="Direct", upsample=3)
    viewer.live_pattern = np.full((detector, detector), value, np.float32).tobytes()
    viewer.send(dict(type="live_image_rows", scan=name, image="Direct", first_row=0, rows=size,
                     cols=size, total=size, upsample=3),
                buffers=[memoryview(np.full((3 * size, 3 * size), value, np.float32))])

viewer.vi_colormap = "gray"
viewer.vi_scale_mode = "linear"
viewer.vi_vmin, viewer.vi_vmax = 0, 100
viewer.dp_colormap = "gray"
viewer.dp_scale_mode = "linear"
viewer.dp_vmin, viewer.dp_vmax = 0, 100
retained(512, 25)
viewer.live_phase_stream = True
viewer.vi_source = "Direct"
direct("live_1", 256, 96, 50)
""")
    pixel = """([selector, size, expected]) => {
      const canvas = document.querySelector(selector);
      return canvas?.width === size && canvas.height === size
        && Math.abs(canvas.getContext("2d").getImageData(size / 2, size / 2, 1, 1).data[0] - expected) <= 1;
    }"""
    page.wait_for_function(pixel, arg=[VI, 768, 127])
    jupyter.run_in_kernel(notebook, 'viewer.live_phase_preview = {**viewer.live_phase_preview, "frozen": True}')
    page.wait_for_function(pixel, arg=[VI, 768, 127])
    jupyter.run_in_kernel(notebook, """
with viewer.hold_sync():
    viewer.shape_rows = viewer.shape_cols = 128
    viewer.live_phase_preview = dict(scan="live_2", rows=128, total=128, cols=128,
                                     detRows=192, detCols=192, mode="Direct", upsample=3)
""")
    # Allow a browser render between the new descriptor and its first phase rows.
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    assert not errors
    assert page.locator(VI).count() == 1
    jupyter.run_in_kernel(notebook, 'direct("live_2", 128, 192, 75)')
    page.wait_for_function(pixel, arg=[VI, 384, 191])
    page.wait_for_function(pixel, arg=[DP, 192, 191])
    assert not errors

    # Switch back to binned Live while the retained DP is still native-sized.
    jupyter.run_in_kernel(notebook, """
with viewer.hold_sync():
    viewer.det_rows = viewer.det_cols = 192
    viewer.frame_bytes = np.full((192, 192), 75, np.float32).tobytes()
    viewer.live_phase_preview = {**viewer.live_phase_preview, "frozen": True}
""")
    page.wait_for_function(pixel, arg=[DP, 192, 191])
    jupyter.run_in_kernel(notebook, """
with viewer.hold_sync():
    viewer.det_rows = viewer.det_cols = 96
    viewer.live_phase_preview = dict(scan="live_3", rows=256, total=256, cols=256,
                                     detRows=96, detCols=96, mode="Direct", upsample=3)
""")
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    assert not errors
    page.wait_for_function(pixel, arg=[DP, 96, 191])
    jupyter.run_in_kernel(notebook, 'direct("live_3", 256, 96, 25)')
    page.wait_for_function(pixel, arg=[DP, 96, 63])
    page.wait_for_function(pixel, arg=[VI, 768, 63])
    assert not errors


def kernel_report(jupyter, notebook: str) -> dict:
    return json.loads(jupyter.run_in_kernel(notebook, REPORT).splitlines()[-1])


def detector_on_screen(page, out: Path):
    """Center and radius of the drawn detector, from its green outline (the view may be zoomed)."""
    locator = page.locator(DP).first
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    page.screenshot(path=str(out), clip=box)
    rgb = np.asarray(Image.open(out).convert("RGB"), dtype=np.int16)
    rows, cols = np.nonzero((rgb[..., 1] > rgb[..., 0] + 60) & (rgb[..., 1] > rgb[..., 2] + 60) & (rgb[..., 0] < 90))
    center_row, center_col = rows.mean(), cols.mean()
    return box["x"] + center_col, box["y"] + center_row, np.percentile(np.hypot(rows - center_row, cols - center_col), 90)


def test_complete_live_direct_source_labels_follow_the_drawn_phase(jupyter, open_page, browser_out):
    notebook = "show4dstem_complete_direct.ipynb"
    jupyter.write_notebook(notebook, DETECTOR_NOTEBOOK)
    page = jupyter.open(open_page, notebook, "complete_direct", webgpu=True, ready=DP, out=browser_out)
    jupyter.run_in_kernel(notebook, """
viewer.vi_colormap = "gray"
viewer.vi_vmin = 0
viewer.vi_vmax = 100
viewer.live_phase_stream = True
viewer.vi_source = "Direct"
def observe(scan, generation, frozen=False, waiting=False):
    viewer.live_phase_preview = dict(scan=scan, total=32, rows=32, cols=32, detRows=32, detCols=32,
                                     upsample=3, mode="Direct", completed_direct_generation=generation,
                                     frozen=frozen, waiting_for_direct_stripes=waiting)
    viewer.send(dict(type="live_image_positions", scan=scan, generation=0, start=0, count=32*32, total=32*32),
                buffers=[np.full(32*32,20,np.float32).tobytes()])
phase_for_ready = None
def complete(scan, generation, value):
    global phase_for_ready
    phase_for_ready = (scan, generation, value)
    viewer.send(dict(type="live_image_rows", scan=scan, image="Direct", total=32, cols=32, rows=32, first_row=0,
                     upsample=3, completed_direct_generation=generation),
                buffers=[np.full((96,96),value,np.float32).tobytes()])
def ready(widget, content, buffers):
    if content.get("type") == "show4dstem_frontend_ready" and phase_for_ready:
        if phase_for_ready[1] == viewer.live_phase_preview.get("completed_direct_generation"):
            complete(*phase_for_ready)
viewer.on_msg(ready)
observe("live_8", "session-a")
""")
    page.wait_for_function("(selector) => document.querySelector(selector)?.height === 32", arg=VI)
    page.get_by_text("ROI · Direct pending", exact=False).first.wait_for()
    jupyter.run_in_kernel(notebook, 'complete("live_7", "session-a", 25)')
    phase = """([selector, frame, value]) => {
      const label = document.querySelector('[aria-label="Direct phase source"]');
      const canvas = document.querySelector(selector);
      return label?.textContent === ` · frame ${frame}` && canvas?.height === 96
          && Math.abs(canvas.getContext("2d").getImageData(48, 48, 1, 1).data[0] - value) <= 1;
    }"""
    page.wait_for_function(phase, arg=[VI, 7, 63])
    jupyter.run_in_kernel(notebook, 'observe("live_9", "session-a"); complete("live_8", "session-a", 75)')
    page.wait_for_function(phase, arg=[VI, 8, 191])
    jupyter.run_in_kernel(notebook, 'observe("live_9", "session-a", frozen=True)')
    page.wait_for_function(phase, arg=[VI, 8, 191])
    with page.expect_response(lambda response: response.request.method == "PUT" and notebook in response.url):
        page.keyboard.press("ControlOrMeta+s")
    page.reload()
    page.wait_for_function(phase, arg=[VI, 8, 191], timeout=60_000)
    jupyter.run_in_kernel(notebook, 'observe("live_10", None, waiting=True); complete("live_9", "session-a", 50)')
    page.get_by_text("ROI · Direct pending", exact=False).first.wait_for()
    page.wait_for_function("(selector) => document.querySelector(selector)?.height === 32", arg=VI)
    jupyter.run_in_kernel(notebook, 'observe("live_10", None); complete("live_10", None, 75)')
    page.wait_for_function("(selector) => document.querySelector(selector)?.height === 96", arg=VI)
    assert page.locator('[aria-label="Direct phase source"]').count() == 0
    jupyter.run_in_kernel(notebook, 'observe("live_11", "session-b"); complete("live_9", "session-a", 50)')
    page.wait_for_function("(selector) => document.querySelector(selector)?.height === 32", arg=VI)
    assert page.locator('[aria-label="Direct phase source"]').count() == 0


def drag(page, start, end):
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, steps=5)
    page.mouse.move(*end, steps=5)
    page.mouse.up()


def drive(open_page, jupyter, mode: str, out: Path) -> dict:
    notebook = f"show4dstem_{mode}.ipynb"
    page = jupyter.open(open_page, notebook, mode, webgpu=mode == "webgpu", ready=DP, out=out)
    steps = {"adapter": page.adapter(), "paths": page.render_paths()}

    def record(name: str) -> None:
        # park the pointer off the panels: a click or drag leaves it over one, and the hover
        # readout drawn there (in one mode and not the other, by timing) is not data
        page.mouse.move(5, 300)
        page.wait_for_timeout(2500)
        shots = {}
        for key, selector in (("dp", DP), ("vi", VI), ("fft", FFT)):
            path = out / f"live_{mode}_{name}_{key}.png"
            page.locator(selector).first.screenshot(path=str(path))
            shots[key] = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
        page.screenshot(path=str(out / f"live_{mode}_{name}_page.png"))
        steps[name] = {"shots": shots, "kernel": kernel_report(jupyter, notebook)}

    record("initial")
    for index, (row_fraction, col_fraction) in enumerate([(0.25, 0.30), (0.62, 0.71), (0.80, 0.15)]):
        locator = page.locator(VI).first
        locator.scroll_into_view_if_needed()
        box = locator.bounding_box()
        page.mouse.click(box["x"] + col_fraction * box["width"], box["y"] + row_fraction * box["height"])
        record(f"scan_{index}")
    x, y, _ = detector_on_screen(page, out / f"live_{mode}_probe.png")
    drag(page, (x, y), (x + 40, y + 30))
    record("move_detector")
    x, y, radius = detector_on_screen(page, out / f"live_{mode}_probe.png")
    drag(page, (x + radius * 0.7071, y + radius * 0.7071), (x + radius * 1.1, y + radius * 1.1))
    record("resize_detector")
    page.get_by_text("BF", exact=True).first.click()
    record("preset_bf")
    page.get_by_text("ADF", exact=True).first.click()
    record("preset_adf")
    combos = page.locator('.jp-OutputArea-output [role="combobox"]')
    colormap_menus = [index for index in range(combos.count()) if combos.nth(index).inner_text().strip().lower() == "inferno"]
    combos.nth(colormap_menus[1]).click()
    page.locator('li[role="option"]', has_text="Viridis").first.click()
    record("colormap")
    left = page.locator(VI).first.bounding_box()["x"] - 2
    page.evaluate("""(left) => {
      const slider = Array.from(document.querySelectorAll('.jp-OutputArea-output .MuiSlider-root'))
        .find(s => s.getBoundingClientRect().left >= left && s.querySelectorAll('input[type=range]').length === 2);
      const input = slider.querySelectorAll('input[type=range]')[0];
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(input, '12');
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    }""", left)
    record("contrast")
    return steps


def test_show4dstem_live_kernel_same_pixels_and_numbers(jupyter, open_page, hardware_adapter, browser_out):
    gpu = drive(open_page, jupyter, "webgpu", browser_out)
    cpu = drive(open_page, jupyter, "cpu", browser_out)
    hardware_adapter(gpu.pop("adapter"))
    assert cpu.pop("adapter") is None
    assert gpu.pop("paths") == ["WebGPU"] and cpu.pop("paths") == ["CPU"]
    # Equal pixels in both modes prove nothing if both are blank: every panel
    # shows data, and each panel changes when the step that drives it runs.
    for mode, steps in (("webgpu", gpu), ("cpu", cpu)):
        for step, record in steps.items():
            for key, shot in record["shots"].items():
                pixels = shot.reshape(-1, 3)
                assert pixels.std(axis=0).max() > 1 and len(np.unique(pixels, axis=0)) >= 8, (mode, step, key, "blank")
        for step_a, step_b, key in (("scan_0", "scan_1", "dp"), ("preset_bf", "preset_adf", "vi"), ("preset_adf", "colormap", "vi")):
            assert np.abs(steps[step_a]["shots"][key] - steps[step_b]["shots"][key]).max() > 0, (mode, step_a, step_b, key)
    summary = {}
    for step, gpu_step in gpu.items():
        cpu_step = cpu[step]
        for kernel in (gpu_step["kernel"], cpu_step["kernel"]):
            np.testing.assert_allclose(kernel["vi_at"], kernel["masked_sum_at"], rtol=1e-6)
        summary[step] = {key: int(np.abs(gpu_step["shots"][key] - cpu_step["shots"][key]).max()) for key in ("dp", "vi", "fft")}
        # A drag can land a fraction of a pixel apart between the two runs;
        # pixels are only comparable when both kernels hold the same detector.
        if gpu_step["kernel"]["pos"] == cpu_step["kernel"]["pos"] and gpu_step["kernel"]["roi"] == cpu_step["kernel"]["roi"]:
            assert max(summary[step].values()) <= 1, (step, summary[step])
    (browser_out / "live_show4dstem.json").write_text(json.dumps(summary, indent=1))
    assert gpu["contrast"]["kernel"]["vi_colormap"] == "viridis"
    assert gpu["contrast"]["kernel"]["vi_vmin_pct"] == 12


def pixel_center(page, selector: str, row: float, col: float, size: int = 16) -> tuple[float, float]:
    """Screen position of the center of pixel (row, col) of an unzoomed square panel."""
    box = page.locator(selector).first.bounding_box()
    return box["x"] + (col + 0.5) * box["width"] / size, box["y"] + (row + 0.5) * box["height"] / size


def switch_profile(page, panel: int) -> None:
    """Turn on the Profile switch of the diffraction (0) or virtual-image (1) panel."""
    page.evaluate("""(panel) => {
      const labels = Array.from(document.querySelectorAll('.jp-OutputArea-output p, .jp-OutputArea-output span'))
        .filter(element => element.textContent === 'Profile' && element.children.length === 0);
      labels[panel].nextElementSibling.querySelector('input').click();
    }""", panel)


def test_scan_roi_resize_follows_the_grab_and_profiles_sit_on_pixel_centers(jupyter, open_page, hardware_adapter, browser_out):
    notebook = "show4dstem_pixels.ipynb"
    page = jupyter.open(open_page, notebook, "pixels", webgpu=True, ready=DP, out=browser_out)
    hardware_adapter(page.adapter())
    page.locator(VI).first.scroll_into_view_if_needed()
    # A radius-4 scan ROI at (7, 7), pressed on its rim band 3 px right of the center and dragged 3 px further out
    jupyter.run_in_kernel(notebook, "viewer.vi_roi_mode = 'circle'; viewer.vi_roi_radius = 4.0; "
                                     "viewer.vi_roi_center_row = 7.0; viewer.vi_roi_center_col = 7.0; viewer.vi_roi_center = [7.0, 7.0]")
    page.wait_for_timeout(2000)
    start, end = pixel_center(page, VI, 7, 10), pixel_center(page, VI, 7, 13)
    page.mouse.move(*start)
    page.mouse.down()
    radii = []
    for step in range(1, 7):
        page.mouse.move(start[0] + (end[0] - start[0]) * step / 6, start[1])
        page.wait_for_timeout(150)
        radii.append(float(jupyter.run_in_kernel(notebook, "print(viewer.vi_roi_radius)")))
    page.mouse.up()
    page.wait_for_timeout(1500)
    page.locator(VI).first.screenshot(path=str(browser_out / "live_pixels_scan_roi.png"))
    roi = json.loads(jupyter.run_in_kernel(notebook, "import json; print(json.dumps([viewer.vi_roi_radius, viewer.vi_roi_center_row, viewer.vi_roi_center_col]))"))
    # the edge follows the pointer from the press: never below the starting radius, 4 + 3 at the end, center kept
    assert min(radii) >= 4 and roi == [7.0, 7.0, 7.0], (radii, roi)
    jupyter.run_in_kernel(notebook, "viewer.vi_roi_mode = 'off'")
    page.wait_for_timeout(1500)
    # Diffraction profile clicked on the centers of pixels (5, 2) and (5, 13) stores those indices
    page.locator(DP).first.scroll_into_view_if_needed()
    switch_profile(page, 0)
    page.wait_for_timeout(800)
    for col in (2, 13):
        page.mouse.click(*pixel_center(page, DP, 5, col))
        page.wait_for_timeout(800)
    page.wait_for_timeout(1000)
    line = json.loads(jupyter.run_in_kernel(notebook, "import json; print(json.dumps(viewer.profile_line))"))
    page.locator(DP).first.screenshot(path=str(browser_out / "live_pixels_dp_profile.png"))
    assert np.allclose([[point["row"], point["col"]] for point in line], [[5, 2], [5, 13]], atol=0.05), line
    # Virtual-image profile clicked on the centers of (3, 3) and (12, 3) is drawn through column 3's center
    page.locator(VI).first.scroll_into_view_if_needed()
    switch_profile(page, 1)
    page.wait_for_timeout(800)
    for row in (3, 12):
        page.mouse.click(*pixel_center(page, VI, row, 3))
        page.wait_for_timeout(800)
    page.mouse.move(5, 300)
    page.wait_for_timeout(1000)
    path = browser_out / "live_pixels_vi_profile.png"
    page.locator(VI).first.screenshot(path=str(path))
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    # drawn like the diffraction profile, in the theme accent (#0066cc in the light JupyterLab theme)
    accent = np.abs(rgb - np.array([0, 102, 204])).max(axis=-1) <= 30
    line_cols = np.nonzero(accent)[1]
    assert line_cols.size, "no profile line drawn"
    assert abs(np.median(line_cols) / (rgb.shape[1] / 16) - 3.5) < 0.1, np.median(line_cols) / (rgb.shape[1] / 16)
    endpoints = [accent[int((row + 0.5) * rgb.shape[0] / 16), int(3.5 * rgb.shape[1] / 16)] for row in (3, 12)]
    assert endpoints == [True, True], endpoints
    # JupyterLab's debugger panel reports "No active debugger session" on its own
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors


def test_scan_roi_click_moves_the_region_with_one_kernel_reduction(jupyter, open_page, hardware_adapter, browser_out):
    notebook = "show4dstem_pixels.ipynb"
    page = jupyter.open(open_page, notebook, "pixels_click", webgpu=True, ready=DP, out=browser_out)
    hardware_adapter(page.adapter())
    page.locator(VI).first.scroll_into_view_if_needed()
    # every scan-region reduction is counted with the region it reduced
    jupyter.run_in_kernel(notebook, "_reductions = []\n_reduce = viewer._compute_vi_roi_dp\n"
                                     "def _counted_reduce():\n    _reductions.append([viewer.vi_roi_center_row, viewer.vi_roi_center_col])\n    _reduce()\n"
                                     "viewer._compute_vi_roi_dp = _counted_reduce\n"
                                     "viewer.vi_roi_mode = 'circle'; viewer.vi_roi_radius = 2.0; viewer.vi_roi_center = [3.0, 3.0]")
    page.wait_for_timeout(2000)
    jupyter.run_in_kernel(notebook, "_reductions.clear()")
    # a click outside the region moves its center there
    page.mouse.click(*pixel_center(page, VI, 11, 12))
    page.wait_for_timeout(3000)
    reductions = json.loads(jupyter.run_in_kernel(notebook, "import json; print(json.dumps(_reductions))"))
    # one reduction, of the new region: never a half-moved one (new row with the old column)
    assert reductions == [[11.0, 12.0]], reductions


def test_detector_click_and_reset_each_reach_the_kernel_as_one_reduction(jupyter, open_page, hardware_adapter, browser_out):
    notebook = "show4dstem_pixels.ipynb"
    page = jupyter.open(open_page, notebook, "pixels_detector", webgpu=True, ready=DP, out=browser_out)
    hardware_adapter(page.adapter())
    page.locator(DP).first.scroll_into_view_if_needed()
    # every virtual-image reduction is counted with the detector center it reduced
    jupyter.run_in_kernel(notebook, "_reduced = []\n_reduce_image = viewer._compute_virtual_image_from_roi\n"
                                     "def _counted_image():\n    _reduced.append([viewer.roi_center_row, viewer.roi_center_col])\n    _reduce_image()\n"
                                     "viewer._compute_virtual_image_from_roi = _counted_image")
    fitted = json.loads(jupyter.run_in_kernel(notebook, "import json; print(json.dumps([viewer.bright_field_center_row_px, viewer.bright_field_center_col_px]))"))
    # a click outside the detector moves it there: one message carrying the row, the column and the pair
    page.mouse.click(*pixel_center(page, DP, 12, 3))
    page.wait_for_timeout(3000)
    reduced = json.loads(jupyter.run_in_kernel(notebook, "import json; print(json.dumps(_reduced))"))
    assert len(reduced) == 1 and np.allclose(reduced[0], [12, 3], atol=0.6), reduced
    jupyter.run_in_kernel(notebook, "_reduced.clear()")
    # the diffraction panel's Reset returns the detector to the fitted disk in one message
    page.locator('.jp-OutputArea-output button:not([disabled])', has_text="Reset").first.click()
    page.wait_for_timeout(3000)
    reduced = json.loads(jupyter.run_in_kernel(notebook, "import json; print(json.dumps(_reduced))"))
    assert reduced == [fitted], (reduced, fitted)
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
