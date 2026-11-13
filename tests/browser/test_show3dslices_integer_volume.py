"""An integer volume draws the pixels of its float32 copy, on the hardware adapter and on the CPU path.

Show3DSlices keeps uint8, int8, uint16 and int16 volumes in their own type in the kernel, the page and
the GPU texture (one or two bytes per voxel instead of four). Every value is exact, so the slice panels
must be pixel-identical to the same volume sent as float32.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

HASHES = """() => [0, 1].map(axis => {
  const canvas = document.querySelector(`canvas[data-quantem-scientific-output="show3dslices-slice-${axis}"]`);
  const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  let hash = 2166136261, lit = 0;
  for (let i = 0; i < data.length; i++) { hash = Math.imul(hash ^ data[i], 16777619) >>> 0; if (i % 4 < 3 && data[i] > 0) lit++; }
  return { size: `${canvas.width}x${canvas.height}`, hash, lit };
})"""


@pytest.mark.parametrize("dtype", ["uint8", "int16"])
def test_integer_volume_draws_the_float32_pixels(dtype, chrome, serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show3DSlices

    info = np.iinfo(dtype)
    rows, cols = np.indices((70, 90))
    texture = np.sin(rows / 5.0) * np.cos(cols / 7.0)
    stack = np.stack([texture * (0.5 + 0.05 * z) + 0.02 * z for z in range(9)])
    # 70 x 90: rows of 90 and 180 bytes, the lengths the old upload padded to 256
    volume = (info.min + (stack - stack.min()) / np.ptp(stack) * (int(info.max) - int(info.min))).round().astype(dtype)
    pages = {}
    for name, data in (("native", volume), ("float32", volume.astype(np.float32))):
        widget = Show3DSlices(data, title=f"{dtype} {name}")
        assert widget.volume_dtype == (dtype if name == "native" else "float32")
        widget.export_html(browser_out / f"show3dslices_{dtype}_{name}.html")
        pages[name] = f"show3dslices_{dtype}_{name}.html"
    drawn = {}
    for name, html in pages.items():
        for webgpu in (True, False):
            page = open_page(f"{serve(browser_out)}/{html}", webgpu=webgpu)
            page.wait_for_selector('canvas[data-quantem-scientific-output="show3dslices-slice-1"]', timeout=60_000)
            page.wait_for_timeout(4_000)
            if webgpu:
                hardware_adapter(page.adapter())
            drawn[name, webgpu] = page.evaluate(HASHES)
            assert page.errors == [], page.errors
            assert all(panel["lit"] > 0 for panel in drawn[name, webgpu]), drawn[name, webgpu]
            page.screenshot(path=str(browser_out / f"show3dslices_{dtype}_{name}_{'webgpu' if webgpu else 'cpu'}.png"))
    reference = drawn["float32", True]
    assert drawn["native", True] == reference, drawn
    assert drawn["native", False] == drawn["float32", False], drawn
