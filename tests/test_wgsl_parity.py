"""WGSL/WebGPU compute parity: browser BF/DF/CoM/DPC kernels vs a numpy reference.

quantem.gpu owns the canonical WebGPU compute sources; quantem.widget syncs them
into js/engine before bundling. The Python torch path is covered by
test_dpc_virtual_parity.py; this is the missing leg - it proves the WGSL output
matches numpy on a deterministic fixture.

WGSL only runs on a real GPU in a browser, so this drives a headed Chrome over CDP and calls
the web app's `window.__wgslParity(scanCount, detRows, detCols)` hook. The fixture is a pure
index function (value = (s*31 + d*17) % 251) so numpy reproduces the exact bytes the JS builds.

Skips cleanly when google-chrome, websockets, or the built web dist are absent (CI without a GPU).
Run on the CUDA box: pytest tests/test_wgsl_parity.py -v
"""
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

websockets = pytest.importorskip("websockets")
import asyncio
import urllib.request

DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
MAC_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
CHROME = (
    shutil.which("google-chrome")
    or shutil.which("google-chrome-stable")
    or (str(MAC_CHROME) if MAC_CHROME.exists() else None)
)
NVIDIA_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"

pytestmark = [
    pytest.mark.skipif(CHROME is None, reason="google-chrome not installed"),
    pytest.mark.skipif(not (DIST / "index.html").exists(), reason="web app not built (run: cd web && npx vite build)"),
]

SCAN, DET_ROWS, DET_COLS = 64, 32, 32


def _numpy_reference():
    """The SAME deterministic stack + BF mask the JS hook builds, plus numpy BF sum, CoM, and DPC.

    value(s, d) = (s*31 + d*17) % 251 with d the row-major detector index; this is a pure
    function of indices so JS and numpy produce byte-identical input - any mismatch is a real
    kernel bug, not RNG drift.
    """
    det_size = DET_ROWS * DET_COLS
    s = np.arange(SCAN)[:, None]
    d = np.arange(det_size)[None, :]
    stack = ((s * 31 + d * 17) % 251).astype(np.uint8).reshape(SCAN, DET_ROWS, DET_COLS).astype(np.float64)
    cy, cx, radius = (DET_ROWS - 1) / 2, (DET_COLS - 1) / 2, min(DET_ROWS, DET_COLS) * 0.25
    rows = np.arange(DET_ROWS)[:, None]
    cols = np.arange(DET_COLS)[None, :]
    mask = ((rows - cy) ** 2 + (cols - cx) ** 2 <= radius * radius).astype(np.float64)
    intensity = stack * mask
    virtual = intensity.sum(axis=(1, 2))
    denom = intensity.sum(axis=(1, 2))
    com_y = (intensity * rows).sum(axis=(1, 2)) / denom
    com_x = (intensity * cols).sum(axis=(1, 2)) / denom
    dpc_y = com_y - com_y.mean()
    dpc_x = com_x - com_x.mean()
    return virtual, com_y, com_x, dpc_y, dpc_x


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _run_wgsl(cdp_port):
    """Connect to Chrome, call window.__wgslParity, return the parsed result dict."""
    tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json", timeout=5).read())
    page = next(t for t in tabs if t["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2**28) as ws:
        mid = 0

        async def cmd(method, params=None):
            nonlocal mid
            mid += 1
            await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == mid:
                    return msg

        await cmd("Runtime.enable")
        expr = (
            "(async () => {"
            f"const compute = await window.__wgslParity({SCAN},{DET_ROWS},{DET_COLS});"
            "const display = await window.__displayParity();"
            "return {...compute, display};"
            "})()"
        )
        res = await cmd("Runtime.evaluate", {"expression": expr, "awaitPromise": True, "returnByValue": True})
        return res.get("result", {}).get("result", {}).get("value")


@pytest.fixture(scope="module")
def wgsl_result():
    """Serve the built web app, launch headed Chrome on the real GPU, call the parity hook."""
    port = _free_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(DIST))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cdp_port = _free_port()
    profile = f"/tmp/cdp-wgsl-parity-{cdp_port}"
    import os
    env = dict(os.environ)
    if Path(NVIDIA_ICD).exists():
        env["VK_ICD_FILENAMES"] = NVIDIA_ICD  # force the real NVIDIA Vulkan device, never SwiftShader
    env.setdefault("DISPLAY", ":1")
    chrome_args = [
        CHROME,
        f"--remote-debugging-port={cdp_port}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--ignore-gpu-blocklist",
        "--enable-unsafe-webgpu",
        "--window-size=900,700",
    ]
    if sys.platform != "darwin":
        chrome_args.extend([
            "--enable-features=Vulkan",
            "--use-angle=vulkan",
            "--disable-gpu-sandbox",
        ])
    chrome_args.append(f"http://127.0.0.1:{port}/index.html")
    chrome = subprocess.Popen(
        chrome_args,
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):  # wait for CDP + page + the hook to mount
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json/version", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        time.sleep(4)
        result = None
        for _ in range(5):
            result = asyncio.new_event_loop().run_until_complete(_run_wgsl(cdp_port))
            if result and "error" not in result:
                break
            time.sleep(2)
        yield result
    finally:
        chrome.terminate()
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)


def test_wgsl_compute_available(wgsl_result):
    assert wgsl_result is not None, "no response from __wgslParity (Chrome/CDP failed)"
    if isinstance(wgsl_result, dict) and wgsl_result.get("error"):
        pytest.skip(f"WebGPU unavailable in this Chrome: {wgsl_result['error']}")
    assert wgsl_result["scanCount"] == SCAN


def test_wgsl_masked_sum_matches_numpy(wgsl_result):
    """Virtual image (BF mask sum) - integer sum, must be effectively exact in f32."""
    if not isinstance(wgsl_result, dict) or wgsl_result.get("error"):
        pytest.skip("WebGPU unavailable")
    virtual_ref, _, _, _, _ = _numpy_reference()
    virtual_wgsl = np.array(wgsl_result["virtual"], dtype=np.float64)
    np.testing.assert_allclose(virtual_wgsl, virtual_ref, rtol=1e-5, atol=1.0)


def test_wgsl_com_matches_numpy(wgsl_result):
    """CoM (intensity-weighted centroid in detector px) - f32 division, tight tolerance."""
    if not isinstance(wgsl_result, dict) or wgsl_result.get("error"):
        pytest.skip("WebGPU unavailable")
    _, com_y_ref, com_x_ref, _, _ = _numpy_reference()
    com_y = np.array(wgsl_result["comY"], dtype=np.float64)
    com_x = np.array(wgsl_result["comX"], dtype=np.float64)
    np.testing.assert_allclose(com_y, com_y_ref, atol=1e-3)
    np.testing.assert_allclose(com_x, com_x_ref, atol=1e-3)


def test_wgsl_dpc_matches_numpy(wgsl_result):
    """DPC row/col output: centered CoM components from the GPU reducer."""
    if not isinstance(wgsl_result, dict) or wgsl_result.get("error"):
        pytest.skip("WebGPU unavailable")
    _, _, _, dpc_y_ref, dpc_x_ref = _numpy_reference()
    dpc_y = np.array(wgsl_result["dpcY"], dtype=np.float64)
    dpc_x = np.array(wgsl_result["dpcX"], dtype=np.float64)
    np.testing.assert_allclose(dpc_y, dpc_y_ref, atol=1e-3)
    np.testing.assert_allclose(dpc_x, dpc_x_ref, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(wgsl_result["dpcMagnitude"], dtype=np.float64),
        np.hypot(dpc_y_ref, dpc_x_ref),
        atol=1e-3,
    )


def test_wgsl_display_histogram_colormap_and_fft_match_reference(wgsl_result):
    """Canonical WebGPU signed-log display and FFT match NumPy contracts."""
    if not isinstance(wgsl_result, dict) or wgsl_result.get("error"):
        pytest.skip("WebGPU unavailable")
    display = wgsl_result.get("display")
    assert isinstance(display, dict), "display parity hook did not return evidence"
    if display.get("error"):
        pytest.skip(display["error"])
    assert display["softwareAdapter"] is False
    assert display["adapter"]

    from quantem.gpu.display import colormap_lut
    from quantem.gpu.display.reference import colorize, histogram

    values = np.asarray(display["values"], dtype=np.float32)
    for scale in ("linear", "log"):
        reference_low, reference_high = (-7.0, 7.0)
        expected_histogram = histogram(
            values,
            reference_low,
            reference_high,
            scale,
        )
        nonzero = expected_histogram[expected_histogram > 0]
        expected_normalized = expected_histogram.astype(np.float64)
        expected_normalized /= max(1, int(nonzero.max()))
        np.testing.assert_array_equal(
            np.asarray(display[f"{scale}Histogram"]),
            expected_normalized,
        )
        expected_rgba = colorize(
            values,
            colormap_lut("gray"),
            reference_low,
            reference_high,
            scale,
        )
        np.testing.assert_array_equal(
            np.asarray(display[f"{scale}Rgba"], dtype=np.uint8).reshape(-1, 4),
            expected_rgba,
        )

    np.testing.assert_array_equal(
        np.asarray(display["namedViridisRgba"], dtype=np.uint8).reshape(-1, 4),
        colorize(values, colormap_lut("viridis"), -7, 7),
    )

    constant = np.asarray([-7, 0, 3, 7], dtype=np.float32)
    constant_histogram = histogram(constant, 3, 3)
    constant_histogram = constant_histogram / max(1, int(constant_histogram.max()))
    np.testing.assert_array_equal(
        np.asarray(display["constantHistogram"]),
        constant_histogram,
    )
    np.testing.assert_array_equal(
        np.asarray(display["constantRgba"], dtype=np.uint8).reshape(-1, 4),
        colorize(constant, colormap_lut("gray"), 3, 3),
    )

    nonfinite = np.asarray([np.nan, -np.inf, np.inf, -1, 0, 1], dtype=np.float32)
    nonfinite_histogram = histogram(nonfinite, -1, 1)
    nonfinite_histogram = nonfinite_histogram / max(1, int(nonfinite_histogram.max()))
    np.testing.assert_array_equal(
        np.asarray(display["nonfiniteHistogram"]),
        nonfinite_histogram,
    )
    np.testing.assert_array_equal(
        np.asarray(display["nonfiniteRgba"], dtype=np.uint8).reshape(-1, 4),
        colorize(nonfinite, colormap_lut("gray"), -1, 1),
    )

    limit = np.float32(1e20)
    extreme = np.asarray([-limit, 0, limit, -limit, 0, limit], dtype=np.float32)
    extreme_histogram = histogram(extreme, -limit, limit)
    extreme_histogram = extreme_histogram / max(1, int(extreme_histogram.max()))
    np.testing.assert_array_equal(
        np.asarray(display["extremeHistogram"]),
        extreme_histogram,
    )
    np.testing.assert_array_equal(
        np.asarray(display["extremeRgba"], dtype=np.uint8).reshape(-1, 4),
        colorize(extreme, colormap_lut("gray"), -limit, limit),
    )

    expected_fft = np.fft.fft2(
        np.asarray(display["fftInput"], dtype=np.float32).reshape(4, 4)
    )
    actual_fft = (
        np.asarray(display["fftReal"], dtype=np.float32)
        + 1j * np.asarray(display["fftImag"], dtype=np.float32)
    ).reshape(4, 4)
    np.testing.assert_allclose(actual_fft, expected_fft, rtol=2e-6, atol=2e-5)


def test_wgsl_display_and_frequency_filters_match_numpy_scipy(wgsl_result):
    """Odd/nonsquare WebGPU filters preserve the frozen scientific formulas."""
    if not isinstance(wgsl_result, dict) or wgsl_result.get("error"):
        pytest.skip("WebGPU unavailable")
    display = wgsl_result.get("display")
    if not isinstance(display, dict) or display.get("error"):
        pytest.skip("WebGPU display unavailable")

    scipy_ndimage = pytest.importorskip("scipy.ndimage")
    image = np.asarray(display["filterInput"], dtype=np.float32).reshape(3, 5)
    expected_gaussian = scipy_ndimage.gaussian_filter(
        image,
        sigma=1.25,
        mode="reflect",
        truncate=4.0,
    )
    np.testing.assert_allclose(
        np.asarray(display["gaussianOdd"], dtype=np.float32).reshape(3, 5),
        expected_gaussian,
        rtol=2e-5,
        atol=2e-5,
    )

    padded_rows, padded_cols = 4, 8
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    padded[:3, :5] = image
    spectrum = np.fft.fft2(padded)
    rows = np.arange(padded_rows)
    cols = np.arange(padded_cols)
    fy = np.minimum(rows, padded_rows - rows) / max(1, padded_rows / 2)
    fx = np.minimum(cols, padded_cols - cols) / max(1, padded_cols / 2)
    radius = np.minimum(1, np.hypot(fy[:, None], fx[None, :]))
    mask = 1 / (1 + np.exp(-(radius - 0.2) / 0.035))
    expected_frequency = np.fft.ifft2(spectrum * mask).real[:3, :5]
    np.testing.assert_allclose(
        np.asarray(display["frequencyOdd"], dtype=np.float32).reshape(3, 5),
        expected_frequency,
        rtol=2e-5,
        atol=2e-5,
    )


def test_wgsl_geometry_matches_numpy_reference(wgsl_result):
    """ROI masking, width-averaged line sampling, and FFT peak refinement."""
    if not isinstance(wgsl_result, dict) or wgsl_result.get("error"):
        pytest.skip("WebGPU unavailable")
    display = wgsl_result.get("display")
    if not isinstance(display, dict) or display.get("error"):
        pytest.skip("WebGPU display unavailable")

    image = np.asarray(display["filterInput"], dtype=np.float32).reshape(3, 5)
    crop = display["cropOdd"]
    assert (crop["height"], crop["width"]) == (3, 4)
    expected_crop = np.zeros((3, 4), dtype=np.float32)
    for row in range(3):
        for col in range(4):
            if (row - 1) ** 2 + (col - 2) ** 2 <= 1.5**2:
                expected_crop[row, col] = image[row, col]
    np.testing.assert_array_equal(
        np.asarray(crop["values"], dtype=np.float32).reshape(3, 4),
        expected_crop,
    )

    def bilinear(row, col):
        base_row = np.floor(row).astype(int)
        base_col = np.floor(col).astype(int)
        rf = row - base_row
        cf = col - base_col
        top = np.clip(base_row, 0, 2)
        bottom = np.clip(base_row + 1, 0, 2)
        left = np.clip(base_col, 0, 4)
        right = np.clip(base_col + 1, 0, 4)
        return (
            image[top, left] * (1 - cf) * (1 - rf)
            + image[top, right] * cf * (1 - rf)
            + image[bottom, left] * (1 - cf) * rf
            + image[bottom, right] * cf * rf
        )

    row0, col0, row1, col1 = -0.25, 0.5, 2.25, 4.5
    length = np.hypot(col1 - col0, row1 - row0)
    count = max(2, int(np.ceil(length)))
    perpendicular_row = -(col1 - col0) / length
    perpendicular_col = (row1 - row0) / length
    expected_line = []
    for index in range(count):
        fraction = index / (count - 1)
        value = 0.0
        for offset in (-1, 0, 1):
            value += bilinear(
                row0 + offset * perpendicular_row + fraction * (row1 - row0),
                col0 + offset * perpendicular_col + fraction * (col1 - col0),
            )
        expected_line.append(value / 3)
    np.testing.assert_allclose(display["lineOdd"], expected_line, atol=2e-5)
    np.testing.assert_allclose(
        [display["peak"]["row"], display["peak"]["col"]],
        [1.25, 2.25],
        atol=2e-6,
    )


def test_wgsl_quantization_and_rotation_match_qgpu_reference(wgsl_result):
    """Direct uint8+range and fixed-shape rotation run on the hardware adapter."""
    if not isinstance(wgsl_result, dict) or wgsl_result.get("error"):
        pytest.skip("WebGPU unavailable")
    display = wgsl_result.get("display")
    if not isinstance(display, dict) or display.get("error"):
        pytest.skip("WebGPU display unavailable")
    assert display["softwareAdapter"] is False

    from quantem.gpu.display.geometry import rotate_stack_inplane
    from quantem.gpu.display.reference import dequantize_uint8

    quantized = np.asarray(display["quantizedInput"], dtype=np.uint8)
    np.testing.assert_allclose(
        np.asarray(display["dequantized"], dtype=np.float32),
        dequantize_uint8(quantized, -7.5, 12.5),
        rtol=0,
        atol=2e-6,
    )

    image_u8 = np.asarray(display["quantizedImage"], dtype=np.uint8).reshape(3, 5)
    image = dequantize_uint8(image_u8, -3.25, 6.75)

    def bilinear(row, col):
        base_row, base_col = int(np.floor(row)), int(np.floor(col))
        row_fraction, column_fraction = row - base_row, col - base_col
        top, bottom = np.clip([base_row, base_row + 1], 0, 2)
        left, right = np.clip([base_col, base_col + 1], 0, 4)
        return (
            image[top, left] * (1 - column_fraction) * (1 - row_fraction)
            + image[top, right] * column_fraction * (1 - row_fraction)
            + image[bottom, left] * (1 - column_fraction) * row_fraction
            + image[bottom, right] * column_fraction * row_fraction
        )

    row0, col0, row1, col1 = -0.25, 0.5, 2.25, 4.5
    count = max(2, int(np.ceil(np.hypot(col1 - col0, row1 - row0))))
    length = np.hypot(col1 - col0, row1 - row0)
    perpendicular_row = -(col1 - col0) / length
    perpendicular_col = (row1 - row0) / length
    expected_line = []
    for index in range(count):
        fraction = index / (count - 1)
        expected_line.append(sum(
            bilinear(
                row0 + offset * perpendicular_row + fraction * (row1 - row0),
                col0 + offset * perpendicular_col + fraction * (col1 - col0),
            )
            for offset in (-1, 0, 1)
        ) / 3)
    np.testing.assert_allclose(display["quantizedLine"], expected_line, atol=3e-5)

    rotation_input = np.asarray(display["rotationInput"], dtype=np.float32).reshape(1, 3, 5)
    expected_rotation = rotate_stack_inplane(rotation_input, 30)
    np.testing.assert_allclose(
        np.asarray(display["rotatedOdd"], dtype=np.float32).reshape(1, 3, 5),
        expected_rotation,
        rtol=0,
        atol=3e-5,
    )
