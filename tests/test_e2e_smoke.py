"""
Playwright smoke tests — verify each widget renders in a real browser.

All widgets are loaded in a single notebook to avoid repeated
JupyterLab page loads. Total runtime ~1 min.

Run with:
    python -m pytest tests/test_e2e_smoke.py -v

Requires: playwright, jupyterlab
"""

import pytest

from conftest import TESTS_DIR, _run_notebook_and_wait, _write_notebook

NOTEBOOK_PATH = TESTS_DIR / "_smoke_all_widgets.ipynb"


@pytest.fixture(scope="module")
def smoke_page(browser_context):
    """Single notebook with all widgets, opened once for the module."""
    _write_notebook(NOTEBOOK_PATH, [
        {"source": [
            "import numpy as np\n",
            "from quantem.widget import Clicker, Show2D, Show3D, Show3DVolume, Show4DSTEM\n",
        ]},
        {"source": [
            "Clicker(np.random.rand(64, 64).astype(np.float32))\n",
        ]},
        {"source": [
            "Show2D(np.random.rand(64, 64).astype(np.float32))\n",
        ]},
        {"source": [
            "Show3D(np.random.rand(10, 64, 64).astype(np.float32))\n",
        ]},
        {"source": [
            "Show3DVolume(np.random.rand(32, 32, 32).astype(np.float32))\n",
        ]},
        {"source": [
            "Show4DSTEM(np.random.rand(8, 8, 32, 32).astype(np.float32))\n",
        ]},
    ])
    page = browser_context.new_page()
    _run_notebook_and_wait(page, NOTEBOOK_PATH)
    yield page
    page.close()
    NOTEBOOK_PATH.unlink(missing_ok=True)


@pytest.mark.parametrize("css_class", [
    "clicker-root",
    "show2d-root",
    "show3d-root",
    "show3dvolume-root",
    "show4dstem-root",
])
def test_widget_root_exists(smoke_page, css_class):
    root = smoke_page.locator(f".{css_class}")
    assert root.count() >= 1, f"Widget .{css_class} not found on page"


@pytest.mark.parametrize("css_class", [
    "clicker-root",
    "show2d-root",
    "show3d-root",
    "show3dvolume-root",
    "show4dstem-root",
])
def test_canvas_rendered(smoke_page, css_class):
    canvas = smoke_page.locator(f".{css_class} canvas")
    assert canvas.count() >= 1, f"No canvas in .{css_class}"
    box = canvas.first.bounding_box()
    assert box is not None, f"Canvas in .{css_class} has no bounding box"
    assert box["width"] > 0 and box["height"] > 0, f"Canvas in .{css_class} has zero dimensions"
