"""Show1D snapshot panels in headed Chrome on the hardware adapter.

Stepping through snapshot groups whose images carry different labels mounts a
new contrast histogram for every group. The page counts the WebGPU buffers it
holds, and stepping through every group must not leave one behind. Zooming a
snapshot panel, or hovering the plot beside it, redraws the panel without
colouring its pixels again (no putImageData).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

COUNT_GPU_BUFFERS = """
window.__liveGpuBuffers = 0;
const createBuffer = GPUDevice.prototype.createBuffer;
GPUDevice.prototype.createBuffer = function (...args) {
  const buffer = createBuffer.apply(this, args);
  window.__liveGpuBuffers += 1;
  const destroy = buffer.destroy.bind(buffer);
  let destroyed = false;
  buffer.destroy = () => {
    if (!destroyed) window.__liveGpuBuffers -= 1;
    destroyed = true;
    destroy();
  };
  return buffer;
};
"""
COUNT_PUT_IMAGE_DATA = """
window.__putImageData = 0;
const putImageData = CanvasRenderingContext2D.prototype.putImageData;
CanvasRenderingContext2D.prototype.putImageData = function (...args) {
  window.__putImageData += 1;
  return putImageData.apply(this, args);
};
"""
GROUPS = 12


@pytest.fixture(scope="module")
def labelled_snapshots(browser_out) -> str:
    from quantem.widget import Show1D

    rng = np.random.default_rng(0)
    # Snapshots sit between the trace points, so hovering a point previews no group.
    widget = Show1D(np.stack([np.zeros(GROUPS), np.full(GROUPS, 0.1)]).astype(np.float32), labels=["loss", "error"])
    for group in range(GROUPS):
        widget.snapshot(group + 0.5, **{f"frame {group}": rng.random((96, 96)).astype(np.float32)})
    widget.export_html(browser_out / "show1d_snapshots.html")
    return "show1d_snapshots.html"


def test_stepping_snapshot_histograms_leaves_no_gpu_buffers(labelled_snapshots, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{labelled_snapshots}", webgpu=True)
    page.add_init_script(COUNT_GPU_BUFFERS)
    page.reload()
    page.wait_for_selector('[data-testid="show1d-panel-histogram"]', timeout=60_000)
    page.wait_for_timeout(3_000)
    hardware_adapter(page.adapter())
    before = page.evaluate("window.__liveGpuBuffers")
    previous = page.get_by_label("Previous snapshot")
    for _ in range(GROUPS - 1):
        previous.click()
        page.wait_for_timeout(250)
    page.wait_for_timeout(1_000)
    labels = page.locator('[data-testid="show1d-panel-histogram"] p').first.text_content()
    after = page.evaluate("window.__liveGpuBuffers")
    page.screenshot(path=str(browser_out / "show1d_histogram_steps.png"))
    assert labels == "frame 0", labels
    assert after == before, f"{after - before} WebGPU buffers left behind by {GROUPS - 1} histogram mounts"
    assert page.errors == [], page.errors


def test_plot_hover_and_zoom_redraw_snapshots_without_recolouring(labelled_snapshots, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{labelled_snapshots}", webgpu=True)
    page.add_init_script(COUNT_PUT_IMAGE_DATA)
    page.reload()
    page.wait_for_selector('[data-testid^="show1d-snapshot-panel-"] canvas', timeout=60_000)
    page.wait_for_timeout(3_000)
    hardware_adapter(page.adapter())
    start = page.evaluate("window.__putImageData")
    plot = page.locator("canvas[data-quantem-scientific-output='show1d-plot']").bounding_box()
    # Along the zero trace (plot frame top 16 px, bottom 46 px, y range padded by 4 %):
    # every move hovers a trace point, which re-renders the widget.
    trace_row = plot["y"] + 16 + (1 - 0.004 / 0.108) * (plot["height"] - 62)
    for step in range(12):
        page.mouse.move(plot["x"] + 70 + step * (plot["width"] - 80) / 12, trace_row)
        page.wait_for_timeout(80)
    hovered = page.evaluate("document.querySelector(\"canvas[data-quantem-scientific-output='show1d-plot']\").nextElementSibling.toDataURL()")
    tile = page.locator('[data-testid^="show1d-snapshot-panel-"] canvas').first
    before = tile.screenshot()
    box = tile.bounding_box()
    page.mouse.move(box["x"] + 0.4 * box["width"], box["y"] + 0.6 * box["height"])
    for _ in range(5):
        page.mouse.wheel(0, -100)
        page.wait_for_timeout(150)
    page.wait_for_timeout(500)
    zoomed = tile.screenshot()
    recoloured = page.evaluate("window.__putImageData") - start
    page.screenshot(path=str(browser_out / "show1d_hover_zoom.png"))
    assert len(hovered) > 2_000, "the plot hover marker was never drawn"
    assert zoomed != before, "the wheel did not zoom the snapshot panel"
    assert recoloured == 0, f"{recoloured} full-image recolours for 12 plot hover moves and 5 wheel steps"
    assert page.errors == [], page.errors
