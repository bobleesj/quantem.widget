"""Find drift in a live notebook, on the hardware WebGPU device and on the CPU path.

Six frames of a smooth random specimen move by a known (0.37, -0.23) px per frame,
shifted in Fourier space so the motion is exact and subpixel. With WebGPU the user
Ctrl-clicks frame 2 and Shift-clicks frame 6 in the thumbnail strip; without it the
strip is not drawn, so the notebook picks the same frames with `viewer.drift_frames`.
Then Find drift: the chart names the backend that ran, Python receives `viewer.drift`
with the picked frames and the velocity in nm/s, and the drift-corrected panel appears
beside the frames. WebGPU and the CPU path agree to a few thousandths of a pixel.
"""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.browser

SIZE, FRAMES = 256, 6
VELOCITY_PX = (0.37, -0.23)                 # (row, col) px per frame, the specimen's motion
PIXEL_NM, FRAME_S = 0.5, 2.0
PICKED = [1, 2, 3, 4, 5]                    # frames 2 to 6, as the strip numbers them

NOTEBOOK = f"""import numpy as np
from quantem.widget import Show3D

rows = np.fft.fftfreq({SIZE})[:, None]
cols = np.fft.fftfreq({SIZE})[None, :]
# a smooth specimen, as a HAADF frame is; a Fourier shift moves it exactly, also by a fraction of a pixel
spectrum = np.fft.fft2(np.random.default_rng(0).random(({SIZE}, {SIZE}))) * np.exp(-(rows**2 + cols**2) * 200)
stack = np.stack([np.fft.ifft2(spectrum * np.exp(-2j * np.pi * (rows * {VELOCITY_PX[0]} + cols * {VELOCITY_PX[1]}) * frame)).real
                  for frame in range({FRAMES})]).astype(np.float32)
viewer = Show3D(stack, sampling={PIXEL_NM}, units="nm", thumbnails=True, panel_width_px=320)
viewer.dim_sampling, viewer.dim_unit = {FRAME_S}, "s"
viewer"""

READ_DRIFT = """import json
print(json.dumps({"drift": viewer.drift, "panels": viewer.panel_titles, "n_panels": viewer.n_panels}))"""


def find_drift(jupyter, open_page, hardware_adapter, browser_out, webgpu: bool) -> dict:
    """Pick frames 2 to 6, click Find drift and return what Python then holds."""
    name = f"find_drift_{'webgpu' if webgpu else 'cpu'}.ipynb"
    # without WebGPU the thumbnail strip is not drawn: Python picks the frames
    picks = "" if webgpu else f"\nviewer.drift_frames = {PICKED}"
    jupyter.write_notebook(name, NOTEBOOK.removesuffix("\nviewer") + picks + "\nviewer")
    page = jupyter.open(open_page, name, name.removesuffix(".ipynb").replace("_", "-"), webgpu=webgpu,
                        ready=f'[data-show3d-gpu-residency="{"ready" if webgpu else "fallback"}"]', out=browser_out)
    adapter = page.adapter() if webgpu else None
    if webgpu:
        hardware_adapter(adapter)
        page.get_by_label("Frame 2 thumbnail").click(modifiers=["Control"])
        page.get_by_label("Frame 6 thumbnail").click(modifiers=["Shift"])
        for frame in range(1, FRAMES + 1):
            picked = page.get_by_label(f"Frame {frame} picked for Find drift")
            assert picked.count() == (1 if frame - 1 in PICKED else 0), f"frame {frame}"
    page.get_by_test_id("show3d-find-drift").click()
    chart = page.get_by_test_id("show3d-drift-chart")
    screenshot = browser_out / f"show3d_find_drift_{'webgpu' if webgpu else 'cpu'}.png"
    try:
        chart.wait_for(timeout=60_000)
        # the panel title carries the frame counter beside it ("drift corrected 4/6")
        page.get_by_text("drift corrected").first.wait_for(timeout=120_000)
    finally:
        page.wait_for_timeout(1_000)
        page.screenshot(path=str(screenshot))
    held = json.loads(jupyter.run_in_kernel(name, READ_DRIFT))
    held["chart"], held["adapter"] = chart.inner_text(), adapter
    # JupyterLab's debugger extension reports a missing session on its own.
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
    return held


def test_find_drift_in_a_notebook_on_webgpu_matches_the_cpu_path_and_the_known_drift(
        jupyter, open_page, hardware_adapter, browser_out):
    found = {webgpu: find_drift(jupyter, open_page, hardware_adapter, browser_out, webgpu) for webgpu in (True, False)}
    (browser_out / "show3d_find_drift.json").write_text(json.dumps(found, indent=2))
    for webgpu, held in found.items():
        drift = held["drift"]
        assert drift["backend"] == ("webgpu" if webgpu else "cpu"), drift["backend"]
        assert ("WebGPU" if webgpu else "CPU") in held["chart"], held["chart"]
        assert drift["frames"] == PICKED
        np.testing.assert_allclose(drift["velocity_px_per_frame"], VELOCITY_PX, atol=0.01)
        np.testing.assert_allclose(drift["velocity"], np.array(VELOCITY_PX) * PIXEL_NM / FRAME_S, atol=0.01 * PIXEL_NM / FRAME_S)
        assert drift["velocity_unit"] == "nm/s" and drift["frame_interval_s"] == FRAME_S
        assert min(drift["fit_r2"]) > 0.999, drift["fit_r2"]
        assert held["n_panels"] == 2 and held["panels"] == ["frames", "drift corrected"], held
    np.testing.assert_allclose(found[True]["drift"]["shift_px"], found[False]["drift"]["shift_px"], atol=0.005)
