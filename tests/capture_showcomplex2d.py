"""
E2E Screenshot Capture for ShowComplex2D Widget

Captures screenshots to verify:
- Canvas renders without black padding
- Controls (Scale, Color, Mode, Auto) are functional
- FFT side panel renders correctly
- Display modes switch properly

Usage:
    python tests/capture_showcomplex2d.py

Screenshots saved to tests/screenshots/showcomplex2d/
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

JUPYTER_PORT = 8899
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "showcomplex2d"
NOTEBOOK_PATH = Path(__file__).parent / "_test_showcomplex2d.ipynb"


def create_test_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"\n',
                    "import numpy as np\n",
                    "from quantem.widget import ShowComplex2D\n",
                    "\n",
                    "np.random.seed(42)\n",
                    "y, x = np.mgrid[0:128, 0:128]\n",
                    "r = np.sqrt((x - 64)**2 + (y - 64)**2)\n",
                    "phase = r / 10.0\n",
                    "amp = np.exp(-r / 30.0)\n",
                    "obj = (amp * np.exp(1j * phase)).astype(np.complex64)\n",
                    "print('Data ready:', obj.shape, obj.dtype)",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Default amplitude view\n",
                    "ShowComplex2D(obj, title='Test Complex', pixel_size=0.5)",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent.parent,
    )
    import socket

    print("Waiting for JupyterLab to start...")
    for _ in range(30):
        time.sleep(1)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", JUPYTER_PORT))
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


def dismiss_dialog(page):
    """Dismiss JupyterLab server connection error dialog if present."""
    try:
        close_btn = page.locator("button.jp-Dialog-button:has-text('Close')")
        if close_btn.is_visible(timeout=500):
            close_btn.click()
            time.sleep(0.3)
    except Exception:
        pass


def run_tests(page):
    out_dir = SCREENSHOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    dismiss_dialog(page)

    widget = page.locator(".showcomplex-root").first
    widget.scroll_into_view_if_needed()
    time.sleep(1)

    # 1. Initial state
    save(widget, "01_initial_amplitude", out_dir)

    # 2. Check canvas doesn't have black padding (measure widths)
    canvas_box = widget.locator("canvas").first.bounding_box()
    container_box = widget.locator("canvas").first.locator("..").bounding_box()
    if canvas_box and container_box:
        print(f"  Canvas: {canvas_box['width']:.0f}x{canvas_box['height']:.0f}")
        print(f"  Container: {container_box['width']:.0f}x{container_box['height']:.0f}")
        padding = container_box["width"] - canvas_box["width"]
        print(f"  Right padding: {padding:.0f}px {'OK' if padding < 5 else 'BUG - black area!'}")

    # Helper to click MUI menu items (avoids matching notebook code cells)
    def click_menu_item(text, timeout=3000):
        item = page.locator(f'.MuiMenuItem-root:has-text("{text}")').first
        item.wait_for(state="visible", timeout=timeout)
        item.click()

    # 3. Try changing Mode dropdown to Phase
    dismiss_dialog(page)
    print("  Switching to Phase mode...")
    mode_selects = widget.locator(".MuiSelect-select")
    mode_count = mode_selects.count()
    print(f"  Found {mode_count} Select components")
    try:
        if mode_count >= 3:
            mode_selects.nth(2).click()
            time.sleep(0.5)
            click_menu_item("Phase")
            time.sleep(1)
            save(widget, "02_phase_mode", out_dir)
    except Exception as e:
        print(f"  WARNING: Phase mode failed: {e}")

    # 4. Switch back to amplitude, then change colormap to Viridis
    dismiss_dialog(page)
    print("  Switching colormap to Viridis...")
    try:
        mode_selects = widget.locator(".MuiSelect-select")
        if mode_selects.count() >= 3:
            mode_selects.nth(2).click()
            time.sleep(0.3)
            click_menu_item("Amplitude")
            time.sleep(0.5)

        mode_selects = widget.locator(".MuiSelect-select")
        if mode_selects.count() >= 2:
            mode_selects.nth(1).click()
            time.sleep(0.5)
            click_menu_item("Viridis")
            time.sleep(1)
            save(widget, "03_viridis_colormap", out_dir)
    except Exception as e:
        print(f"  WARNING: Viridis colormap failed: {e}")

    # 5. Try toggling Log scale
    dismiss_dialog(page)
    print("  Switching to Log scale...")
    try:
        mode_selects = widget.locator(".MuiSelect-select")
        if mode_selects.count() >= 1:
            mode_selects.nth(0).click()
            time.sleep(0.5)
            click_menu_item("Log")
            time.sleep(1)
            save(widget, "04_log_scale", out_dir)
    except Exception as e:
        print(f"  WARNING: Log scale failed: {e}")

    # 6. Toggle FFT
    dismiss_dialog(page)
    print("  Toggling FFT...")
    try:
        fft_switch = widget.locator(".MuiSwitch-root").first
        fft_switch.click()
        time.sleep(3)
        save(widget, "05_fft_on", out_dir)
    except Exception as e:
        print(f"  WARNING: FFT toggle failed: {e}")

    # 7. Switch to HSV mode
    dismiss_dialog(page)
    print("  Switching to HSV mode...")
    try:
        mode_selects = widget.locator(".MuiSelect-select")
        if mode_selects.count() >= 3:
            mode_selects.nth(2).click()
            time.sleep(0.3)
            click_menu_item("HSV")
            time.sleep(1)
            save(widget, "06_hsv_mode", out_dir)
    except Exception as e:
        print(f"  WARNING: HSV mode failed: {e}")

    print(f"\n  All screenshots saved to: {out_dir}")


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

            # Capture JS console messages
            js_errors = []
            def on_console(msg):
                if msg.type in ("error", "warning"):
                    print(f"  [JS {msg.type}] {msg.text}")
            def on_pageerror(err):
                js_errors.append(str(err))
                print(f"  [JS ERROR] {err}")
                # Try to get the stack trace
                err_str = str(err)
                if hasattr(err, 'stack'):
                    print(f"  [JS STACK] {err.stack}")
            page.on("console", on_console)
            page.on("pageerror", on_pageerror)
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
            # Click on the notebook area to ensure focus
            notebook_area = page.locator(".jp-Notebook").first
            if notebook_area.is_visible():
                notebook_area.click()
                time.sleep(0.5)
            else:
                # Fallback: click center of page
                page.mouse.click(700, 400)
                time.sleep(0.5)

            # Run all cells via Run menu (most reliable method)
            page.click('text=Run')
            time.sleep(0.3)
            page.click('text=Run All Cells')
            print("Waiting for cells to execute...")
            time.sleep(20)

            # Debug: save full page screenshot
            page.screenshot(path=str(SCREENSHOT_DIR / "00_full_page.png"))
            print(f"  Saved: 00_full_page.png")

            # Debug: check what widget-related elements exist
            for selector in [".showcomplex-root", ".jp-RenderedWidget", ".jp-OutputArea-output", "canvas", ".widget-output"]:
                count = page.locator(selector).count()
                print(f"  Elements matching '{selector}': {count}")

            # Debug: check for JS errors
            if js_errors:
                print(f"  JS ERRORS: {len(js_errors)}")
                for err in js_errors:
                    print(f"    {err}")

            run_tests(page)

            print(f"\nAll screenshots saved to: {SCREENSHOT_DIR}")
            browser.close()
    finally:
        if jupyter_proc:
            stop_jupyter(jupyter_proc)
        cleanup_test_notebook()


if __name__ == "__main__":
    main()
