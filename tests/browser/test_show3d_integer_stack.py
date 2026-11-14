"""An integer stack in Show3D draws the pixels of its float32 copy, on the hardware adapter and on the CPU path.

Show3D keeps uint8, int8, uint16 and int16 stacks in their own type in the kernel, in the page and on the
GPU (one or two bytes per pixel instead of four) and converts only the frames it draws. Every value is
exact, so each frame must be pixel-identical to the same stack sent as float32, also after stepping to
another frame, with a moving average and with several panels.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

IMAGE = 'canvas[data-quantem-scientific-output="show3d-image"]'
HASH = f"""() => {{
  const canvas = document.querySelector('{IMAGE}');
  const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  let hash = 2166136261, lit = 0;
  for (let i = 0; i < data.length; i++) {{ hash = Math.imul(hash ^ data[i], 16777619) >>> 0; if (i % 4 < 3 && data[i] > 0) lit++; }}
  return {{ size: `${{canvas.width}}x${{canvas.height}}`, hash, lit }};
}}"""
# Records the size of every GPU buffer the page creates.
BUFFER_SIZES = """(() => {
  window.__bufferSizes = [];
  if (!window.GPUDevice) return;
  const create = GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer = function (desc) { window.__bufferSizes.push(desc.size); return create.call(this, desc); };
})();"""
FOCUS_FRAME = """() => {
  const thumbs = [...document.querySelectorAll('input[type="range"]')].filter(input => /and current frame/i.test(input.getAttribute("aria-label") ?? ""));
  thumbs[1].focus();
}"""


@pytest.mark.parametrize(("dtype", "avg_window", "panels"), [
    ("uint8", 1, 1), ("int8", 1, 1), ("uint16", 1, 1), ("int16", 1, 1), ("int16", 3, 1), ("uint8", 1, 2),
])
def test_integer_stack_draws_the_float32_pixels(dtype, avg_window, panels, serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show3D

    info = np.iinfo(dtype)
    frames_count = 8
    rows, cols = np.indices((70, 90))
    frames = np.stack([[np.sin(rows / (5.0 + z + panel)) * np.cos(cols / (7.0 + panel)) + 0.05 * z for z in range(frames_count)]
                       for panel in range(panels)])
    stack = (info.min + (frames - frames.min()) / np.ptp(frames) * (int(info.max) - int(info.min))).round().astype(dtype)
    stack = stack[0] if panels == 1 else list(stack)
    case = f"{dtype}_avg{avg_window}_panels{panels}"
    pages = {}
    as_float = [panel.astype(np.float32) for panel in stack] if panels > 1 else stack.astype(np.float32)
    for name, data in (("native", stack), ("float32", as_float)):
        widget = Show3D(data, title=f"{dtype} {name}", avg_window=avg_window, verbose=False)
        assert widget._stack_dtype == (dtype if name == "native" else "float32")
        widget.export_html(browser_out / f"show3d_{case}_{name}.html")
        pages[name] = f"show3d_{case}_{name}.html"
    drawn = {}
    for name, html in pages.items():
        for webgpu in (True, False):
            page = open_page(f"{serve(browser_out)}/{html}", webgpu=webgpu, init_script=BUFFER_SIZES)
            page.wait_for_selector(IMAGE, timeout=60_000)
            page.wait_for_timeout(4_000)
            if webgpu:
                hardware_adapter(page.adapter())
            first = page.evaluate(HASH)
            page.evaluate(FOCUS_FRAME)
            for _ in range(3):
                page.keyboard.press("ArrowRight")
            page.wait_for_timeout(1_500)
            later = page.evaluate(HASH)
            assert first["lit"] > 0 and later["lit"] > 0 and first != later, (first, later)
            assert page.errors == [], page.errors
            if webgpu and name == "native":
                # every resident frame sits on the GPU in its own type: one packed buffer per frame
                values = panels * 70 * 90
                packed = -(-values * np.dtype(dtype).itemsize // 4) * 4
                sizes = page.evaluate("window.__bufferSizes")
                assert sizes.count(packed) >= frames_count, (packed, sorted(set(sizes)))
            drawn[name, webgpu] = (first, later)
            page.screenshot(path=str(browser_out / f"show3d_{case}_{name}_{'webgpu' if webgpu else 'cpu'}.png"))
    assert drawn["native", True] == drawn["float32", True], drawn
    assert drawn["native", False] == drawn["float32", False], drawn
