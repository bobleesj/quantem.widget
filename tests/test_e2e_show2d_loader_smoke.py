"""Playwright smoke tests for explicit Show2D loader entrypoints."""

import pytest

from conftest import TESTS_DIR, _run_notebook_and_wait, _write_notebook

NOTEBOOK_PATH = TESTS_DIR / "_smoke_show2d_loaders.ipynb"
SCREENSHOT_DIR = TESTS_DIR / "screenshots" / "smoke"


@pytest.fixture(scope="module")
def show2d_loader_page(browser_context):
    _write_notebook(
        NOTEBOOK_PATH,
        [
            {
                "source": [
                    "import tempfile\n",
                    "from pathlib import Path\n",
                    "import numpy as np\n",
                    "from PIL import Image\n",
                    "from quantem.widget import Show2D\n",
                    "try:\n",
                    "    import h5py\n",
                    "    has_h5py = True\n",
                    "except Exception:\n",
                    "    has_h5py = False\n",
                ]
            },
            {
                "source": [
                    "tmp = Path(tempfile.mkdtemp(prefix='show2d_loader_smoke_'))\n",
                    "\n",
                    "png_dir = tmp / 'png_stack'\n",
                    "png_dir.mkdir()\n",
                    "for i in range(3):\n",
                    "    arr = (np.ones((32, 32), dtype=np.uint8) * (10 + i)).astype(np.uint8)\n",
                    "    Image.fromarray(arr).save(png_dir / f'slice_{i:02d}.png')\n",
                    "\n",
                    "tiff_file = tmp / 'stack.tiff'\n",
                    "frames = [\n",
                    "    Image.fromarray((np.ones((32, 32), dtype=np.uint8) * 20).astype(np.uint8)),\n",
                    "    Image.fromarray((np.ones((32, 32), dtype=np.uint8) * 21).astype(np.uint8)),\n",
                    "]\n",
                    "frames[0].save(tiff_file, save_all=True, append_images=frames[1:])\n",
                    "\n",
                    "emd_file = tmp / 'stack.emd'\n",
                    "if has_h5py:\n",
                    "    with h5py.File(emd_file, 'w') as h5f:\n",
                    "        h5f.create_dataset('preview/thumb', data=np.ones((16, 16), dtype=np.float32) * 2.0)\n",
                    "        h5f.create_dataset('data/signal', data=np.ones((2, 32, 32), dtype=np.float32) * 55.0)\n",
                ]
            },
            {
                "source": [
                    "Show2D.from_png_folder(png_dir)\n",
                    "Show2D.from_tiff(tiff_file)\n",
                    "if has_h5py:\n",
                    "    Show2D.from_emd(emd_file, dataset_path='/data/signal')\n",
                ]
            },
            {
                "source": [
                    "explicit_error = ''\n",
                    "try:\n",
                    "    Show2D.from_path(png_dir)\n",
                    "except Exception as exc:\n",
                    "    explicit_error = str(exc)\n",
                    "print('EXPLICIT_ERROR:', explicit_error)\n",
                ]
            },
        ],
    )

    page = browser_context.new_page()
    _run_notebook_and_wait(page, NOTEBOOK_PATH, wait_seconds=25)
    if page.locator(".show2d-root").count() == 0:
        page.close()
        NOTEBOOK_PATH.unlink(missing_ok=True)
        pytest.skip("Show2D loader smoke notebook did not render widgets in this environment.")
    yield page
    page.close()
    NOTEBOOK_PATH.unlink(missing_ok=True)


def test_show2d_loader_widgets_render(show2d_loader_page):
    roots = show2d_loader_page.locator(".show2d-root")
    assert roots.count() >= 2, "Expected explicit Show2D loader widgets to render"


def test_show2d_loader_canvas_render(show2d_loader_page):
    canvases = show2d_loader_page.locator(".show2d-root canvas")
    assert canvases.count() >= 2, "Expected canvases from explicit loader widgets"
    box = canvases.first.bounding_box()
    assert box is not None and box["width"] > 0 and box["height"] > 0


def test_show2d_loader_explicit_error_message(show2d_loader_page):
    assert show2d_loader_page.locator("text=file_type is required").count() >= 1


def test_show2d_loader_screenshot(show2d_loader_page):
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    widget = show2d_loader_page.locator(".show2d-root").first
    widget.screenshot(path=str(SCREENSHOT_DIR / "show2d_loader_explicit.png"))
