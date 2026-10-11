"""The WebGPU display FFT equals NumPy's FFT on the hardware adapter, forward and inverse.

Every FFT panel, the frequency filter and Show3D's sub-pixel alignment run
``WebGPUFFT`` from ``js/display/fft.ts``. The module is bundled with esbuild
into a page that transforms non-square power-of-two images (row and column
passes of different lengths) and transforms the spectrum back; the spectrum
must match ``numpy.fft.fft2`` to float32 precision and the round trip must
return the image.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.browser

REPO = Path(__file__).resolve().parents[2]
SHAPES = [(32, 64), (128, 32), (64, 256)]
ENTRY = """
import { WebGPUFFT } from "./display/fft";
import { requireHardwareGPUDevice } from "./display/device";
window.transform = async (images) => {
  const fft = new WebGPUFFT(await requireHardwareGPUDevice("FFT parity"));
  const results = [];
  for (const { height, width, values } of images) {
    const forward = await fft.fft2D(Float32Array.from(values), new Float32Array(values.length), width, height);
    const inverse = await fft.fft2D(forward.real, forward.imag, width, height, true);
    results.push({ real: Array.from(forward.real), imag: Array.from(forward.imag), back: Array.from(inverse.real), backImag: Array.from(inverse.imag) });
  }
  return results;
};
"""


@pytest.fixture(scope="module")
def fft_page(browser_out) -> str:
    bundle = subprocess.run(
        ["npx", "esbuild", "--bundle", "--format=iife", "--loader=ts", f"--sourcefile={REPO / 'js' / 'entry.ts'}"],
        input=ENTRY, capture_output=True, text=True, cwd=REPO / "js", check=True,
    ).stdout
    (browser_out / "display_fft.html").write_text(f'<!doctype html><meta charset="utf-8"><script>{bundle}</script>')
    return "display_fft.html"


def test_webgpu_fft_matches_numpy_on_non_square_images(fft_page, serve, open_page, hardware_adapter, browser_out):
    rng = np.random.default_rng(0)
    images = [rng.standard_normal(shape).astype(np.float32) * 100 for shape in SHAPES]
    page = open_page(f"{serve(browser_out)}/{fft_page}", webgpu=True)
    hardware_adapter(page.adapter())
    payload = [{"height": image.shape[0], "width": image.shape[1], "values": image.ravel().tolist()} for image in images]
    results = page.evaluate("images => window.transform(images)", payload)
    errors = {}
    for image, result in zip(images, results):
        spectrum = np.fft.fft2(image.astype(np.float64))
        found = (np.array(result["real"]) + 1j * np.array(result["imag"])).reshape(image.shape)
        # float32 butterflies: error grows with log2(size) times the largest coefficient's ulp
        scale = np.abs(spectrum).max()
        errors[str(image.shape)] = (float(np.abs(found - spectrum).max() / scale),
                                    float(np.abs(np.array(result["back"]).reshape(image.shape) - image).max() / np.abs(image).max()),
                                    float(np.abs(np.array(result["backImag"])).max() / np.abs(image).max()))
    (browser_out / "display_fft.json").write_text(json.dumps(errors, indent=1))
    for forward, back, back_imag in errors.values():
        assert forward < 2e-6 and back < 2e-6 and back_imag < 2e-6, errors
    assert page.errors == [], page.errors
