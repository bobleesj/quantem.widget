"""Show1D snapshot panels in headed Chrome on the hardware adapter.

Stepping through snapshot groups whose images carry different labels mounts a
new contrast histogram for every group. The page counts the WebGPU buffers it
holds, and stepping through every group must not leave one behind.
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
GROUPS = 12


@pytest.fixture(scope="module")
def labelled_snapshots(browser_out) -> str:
    from quantem.widget import Show1D

    rng = np.random.default_rng(0)
    widget = Show1D(rng.random((2, GROUPS)).astype(np.float32), labels=["loss", "error"])
    for group in range(GROUPS):
        widget.snapshot(group, **{f"frame {group}": rng.random((96, 96)).astype(np.float32)})
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
