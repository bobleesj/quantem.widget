"""
E2E Screenshot Capture for Show2D Widget

Captures screenshots simulating user interactions:
- Click to select different images
- Toggle FFT on/off
- Verify blue selection highlight moves correctly

Usage:
    python tests/capture_show2d.py

Screenshots are saved to tests/screenshots/show2d/
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Configuration
JUPYTER_PORT = 8899
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "show2d"
NOTEBOOK_PATH = Path(__file__).parent / "_test_show2d_features.ipynb"


def create_test_notebook():
    """Generate test notebook with a gallery widget for interaction testing."""
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "from quantem.widget import Show2D\n",
                    "\n",
                    "np.random.seed(42)\n",
                    "y, x = np.mgrid[0:128, 0:128]\n",
                    "r = np.sqrt((x - 64)**2 + (y - 64)**2)\n",
                    "img1 = (np.sin(r / 5) * np.exp(-r / 40) + np.random.rand(128, 128) * 0.3).astype(np.float32)\n",
                    "img2 = (np.cos(r / 8) * np.exp(-r / 50) + np.random.rand(128, 128) * 0.3).astype(np.float32)\n",
                    "img3 = (np.sin(r / 3) * np.exp(-r / 30) + np.random.rand(128, 128) * 0.3).astype(np.float32)\n",
                    "print('Data ready')",
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Gallery for interaction testing\n",
                    "Show2D([img1, img2, img3], labels=['Image A', 'Image B', 'Image C'], ncols=3)",
                ]
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created test notebook: {NOTEBOOK_PATH}")


def cleanup_test_notebook():
    if NOTEBOOK_PATH.exists():
        NOTEBOOK_PATH.unlink()
        print(f"Cleaned up: {NOTEBOOK_PATH}")


def start_jupyter():
    print("Starting JupyterLab...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "jupyter", "lab",
            f"--port={JUPYTER_PORT}", "--no-browser",
            "--NotebookApp.token=''", "--NotebookApp.password=''",
            "--ServerApp.disable_check_xsrf=True",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent.parent,
    )
    import socket
    print("Waiting for JupyterLab to start...")
    for _ in range(30):
        time.sleep(1)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', JUPYTER_PORT))
            sock.close()
            if result == 0:
                print("JupyterLab is ready!")
                time.sleep(2)
                return proc
        except Exception:
            pass
    raise RuntimeError("JupyterLab failed to start within 30 seconds")


def stop_jupyter(proc):
    print("Stopping JupyterLab...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def save(widget, name, out_dir):
    widget.screenshot(path=str(out_dir / f"{name}.png"))
    print(f"  Saved: {name}.png")


def get_border_info(page, widget):
    """Get border widths of all image and FFT containers via JS."""
    return page.evaluate("""(root) => {
        // Find all divs that directly contain a canvas (these are image/FFT containers)
        const allCanvases = root.querySelectorAll('canvas');
        const containerBorders = [];
        const seen = new Set();

        for (const canvas of allCanvases) {
            // Walk up to find the container div with a border
            let el = canvas.parentElement;
            while (el && el !== root) {
                const style = getComputedStyle(el);
                const bw = style.borderTopWidth;
                // Look for the element that has a visible border (1px or 2px)
                if (bw && bw !== '0px' && !seen.has(el)) {
                    seen.add(el);
                    containerBorders.push({
                        tag: el.tagName,
                        borderWidth: bw,
                        borderColor: style.borderTopColor,
                        hasCanvas: el.querySelector('canvas') !== null,
                        canvasCount: el.querySelectorAll('canvas').length,
                        text: el.textContent?.substring(0, 30) || ''
                    });
                    break;
                }
                el = el.parentElement;
            }
        }

        return containerBorders;
    }""", widget.element_handle())


def check_borders(page, widget, step_name, expected_selected, fft_on):
    """Print border state for debugging."""
    info = get_border_info(page, widget)
    print(f"  [{step_name}] Found {len(info)} bordered containers:")
    for i, item in enumerate(info):
        bw = item['borderWidth']
        thick = "THICK" if bw == '2px' else "thin"
        print(f"    [{i}] {thick} ({bw}) canvases={item['canvasCount']}")


def run_interaction_tests(page):
    """Simulate user behavior: click, resize, toggle FFT, verify dark mode."""
    out_dir = SCREENSHOT_DIR / "interactions"
    out_dir.mkdir(parents=True, exist_ok=True)

    widget = page.locator(".show2d-root").first
    widget.scroll_into_view_if_needed()
    time.sleep(1)

    # Step 1: Initial state (light mode)
    save(widget, "01_light_initial", out_dir)

    # Step 2: Click image 2
    canvases = widget.locator("canvas")
    print(f"  Found {canvases.count()} canvases")
    print("  Clicking image 2...")
    canvases.nth(2).click()
    time.sleep(0.5)
    save(widget, "02_light_click_img2", out_dir)
    check_borders(page, widget, "Click img2", expected_selected=1, fft_on=False)

    # Step 3: Resize gallery image (drag resize handle)
    print("  Resizing gallery images...")
    # Find the resize handle (bottom-right triangle gradient)
    resize_handles = widget.locator('[style*="nwse-resize"]')
    if resize_handles.count() == 0:
        resize_handles = widget.locator("div").filter(has_text="").locator("xpath=//div[contains(@style,'nwse-resize') or contains(@class,'nwse')]")
    handle_count = resize_handles.count()
    print(f"  Found {handle_count} resize handles")
    if handle_count > 0:
        handle = resize_handles.first
        bbox = handle.bounding_box()
        if bbox:
            page.mouse.move(bbox["x"] + 8, bbox["y"] + 8)
            page.mouse.down()
            page.mouse.move(bbox["x"] + 108, bbox["y"] + 108, steps=5)
            page.mouse.up()
            time.sleep(0.5)
    save(widget, "03_light_after_resize", out_dir)

    # Step 4: Toggle FFT on
    print("  Toggling FFT on...")
    fft_switch = widget.locator(".MuiSwitch-root").first
    fft_switch.click()
    time.sleep(2)
    save(widget, "04_light_fft_on", out_dir)
    check_borders(page, widget, "FFT on", expected_selected=1, fft_on=True)

    # Step 5: Click image 3 FFT card
    print("  Clicking image 3 FFT...")
    img3_label = widget.locator("text=Image C")
    if img3_label.count() > 0:
        bbox = img3_label.bounding_box()
        if bbox:
            page.mouse.click(bbox["x"] + bbox["width"] / 2, bbox["y"] + 80)
            time.sleep(0.5)
    save(widget, "05_light_fft_click_img3", out_dir)

    # Step 6: Switch to dark mode
    print("  Switching to dark mode...")
    page.evaluate("""() => {
        document.body.dataset.jpThemeLight = 'false';
        document.body.dataset.jpThemeName = 'JupyterLab Dark';
        document.body.classList.remove('jp-theme-light');
        document.body.classList.add('jp-theme-dark');
    }""")
    time.sleep(1.5)
    save(widget, "06_dark_fft_on", out_dir)

    # Step 7: Toggle FFT off in dark mode
    print("  Toggle FFT off (dark mode)...")
    fft_switch = widget.locator(".MuiSwitch-root").first
    fft_switch.click()
    time.sleep(1)
    save(widget, "07_dark_fft_off", out_dir)

    # Step 8: Click image 1 in dark mode
    print("  Click image 1 (dark mode)...")
    canvases = widget.locator("canvas")
    canvases.nth(0).click()
    time.sleep(0.5)
    save(widget, "08_dark_click_img1", out_dir)

    print(f"\n  Interaction screenshots saved to: {out_dir}")


def main():
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    jupyter_proc = None
    try:
        create_test_notebook()
        jupyter_proc = start_jupyter()
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1400, "height": 900})
            page = context.new_page()
            notebook_rel_path = NOTEBOOK_PATH.relative_to(Path(__file__).parent.parent)
            url = f"http://localhost:{JUPYTER_PORT}/lab/tree/{notebook_rel_path}"
            print(f"Opening: {url}")
            page.goto(url, timeout=60000)
            print("Waiting for JupyterLab to load...")
            time.sleep(10)
            try:
                page.keyboard.press("Escape")
                time.sleep(0.5)
            except Exception:
                pass
            # Run all cells
            page.keyboard.press("Meta+Shift+C")
            time.sleep(0.3)
            page.keyboard.type("Run All Cells")
            time.sleep(0.3)
            page.keyboard.press("Enter")
            print("Waiting for cells to execute...")
            time.sleep(15)

            # Run interaction tests
            run_interaction_tests(page)

            print(f"\nAll screenshots saved to: {SCREENSHOT_DIR}")
            browser.close()
    finally:
        if jupyter_proc:
            stop_jupyter(jupyter_proc)
        cleanup_test_notebook()


if __name__ == "__main__":
    main()
