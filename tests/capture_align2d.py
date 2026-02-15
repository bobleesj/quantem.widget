"""
E2E Screenshot Capture for Align2D Widget

Captures screenshots of Align2D in JupyterLab with both dark and light themes.
Uses Playwright to automate the browser.

Test data: hexagonal crystal lattice (spacing=12) with:
  - Image A: crystal + slight noise
  - Image B: shifted by (7.3, -4.6) and rotated by 1.5 degrees (via scipy.ndimage)

Usage:
    python tests/capture_align2d.py

Screenshots are saved to tests/screenshots/align2d/
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Configuration
JUPYTER_PORT = 8899
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "align2d"
NOTEBOOK_PATH = Path(__file__).parent / "_test_align2d_features.ipynb"

FEATURE_NAMES = ["default", "viridis_cmap"]


def create_test_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "from scipy.ndimage import shift, rotate\n",
                    "from quantem.widget import Align2D\n",
                    "\n",
                    "rng = np.random.default_rng(42)\n",
                    "size = 256\n",
                    "y, x = np.mgrid[:size, :size]\n",
                    "\n",
                    "# Hexagonal crystal lattice (spacing=12)\n",
                    "spacing = 12\n",
                    "a1 = np.array([spacing, 0.0])\n",
                    "a2 = np.array([spacing * 0.5, spacing * np.sqrt(3) / 2])\n",
                    "lattice = np.zeros((size, size), dtype=np.float32)\n",
                    "sigma = 1.8\n",
                    "for i in range(-size // spacing - 2, size // spacing + 2):\n",
                    "    for j in range(-size // spacing - 2, size // spacing + 2):\n",
                    "        cx = i * a1[0] + j * a2[0]\n",
                    "        cy = i * a1[1] + j * a2[1]\n",
                    "        if -20 < cx < size + 20 and -20 < cy < size + 20:\n",
                    "            lattice += np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))\n",
                    "\n",
                    "# Image A: crystal lattice + slight noise\n",
                    "img_a = lattice + rng.normal(0, 0.05, (size, size)).astype(np.float32)\n",
                    "img_a = img_a.astype(np.float32)\n",
                    "\n",
                    "# Image B: shifted by (7.3, -4.6) and rotated by 1.5 degrees\n",
                    "img_b = shift(lattice, shift=(-4.6, 7.3), order=3, mode='constant', cval=0.0)\n",
                    "img_b = rotate(img_b, angle=1.5, reshape=False, order=3, mode='constant', cval=0.0)\n",
                    "img_b = img_b + rng.normal(0, 0.05, (size, size)).astype(np.float32)\n",
                    "img_b = img_b.astype(np.float32)\n",
                    "\n",
                    "print(f'Images ready: A={img_a.shape}, B={img_b.shape}')\n",
                    "print(f'Known shift: dx=7.3, dy=-4.6, rotation=1.5 deg')\n",
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Default (auto-alignment should recover the known shift/rotation)\n",
                    "Align2D(img_a, img_b, title='Hex Lattice Alignment', label_a='Reference', label_b='Shifted+Rotated')\n",
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Viridis colormap\n",
                    "Align2D(img_a, img_b, title='Viridis', cmap='viridis', label_a='Reference', label_b='Shifted+Rotated')\n",
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
            sys.executable,
            "-m",
            "jupyter",
            "lab",
            f"--port={JUPYTER_PORT}",
            "--no-browser",
            "--NotebookApp.token=''",
            "--NotebookApp.password=''",
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
            result = sock.connect_ex(('localhost', JUPYTER_PORT))
            sock.close()
            if result == 0:
                print("JupyterLab is ready!")
                time.sleep(2)
                return proc
        except:
            pass
    raise RuntimeError("JupyterLab failed to start within 30 seconds")


def stop_jupyter(proc):
    print("Stopping JupyterLab...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except:
        proc.kill()


def set_theme(page, theme: str):
    print(f"  Setting {theme} theme...")
    if theme == "dark":
        page.evaluate("""() => {
            document.body.dataset.jpThemeLight = 'false';
            document.body.dataset.jpThemeName = 'JupyterLab Dark';
            document.body.classList.remove('jp-theme-light');
            document.body.classList.add('jp-theme-dark');
            localStorage.setItem('@jupyterlab/apputils-extension:themes', JSON.stringify({theme: 'JupyterLab Dark'}));
        }""")
    else:
        page.evaluate("""() => {
            document.body.dataset.jpThemeLight = 'true';
            document.body.dataset.jpThemeName = 'JupyterLab Light';
            document.body.classList.remove('jp-theme-dark');
            document.body.classList.add('jp-theme-light');
            localStorage.setItem('@jupyterlab/apputils-extension:themes', JSON.stringify({theme: 'JupyterLab Light'}));
        }""")
    time.sleep(1)


def capture_widgets(page, theme: str):
    print(f"Capturing {theme} theme screenshots...")
    theme_dir = SCREENSHOT_DIR / theme
    theme_dir.mkdir(parents=True, exist_ok=True)

    page.keyboard.press("Meta+Home")
    time.sleep(0.5)
    page.screenshot(path=str(theme_dir / "full_page.png"), full_page=True)
    print(f"  Saved: full_page.png")

    widgets = page.locator(".align2d-root")
    widget_count = widgets.count()
    print(f"  Found {widget_count} widgets")

    for i in range(min(widget_count, len(FEATURE_NAMES))):
        try:
            widget = widgets.nth(i)
            widget.scroll_into_view_if_needed()
            time.sleep(0.5)
            filename = f"{FEATURE_NAMES[i]}.png"
            widget.screenshot(path=str(theme_dir / filename))
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Warning: Could not capture {FEATURE_NAMES[i]}: {e}")


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
            except:
                pass

            set_theme(page, "light")

            page.keyboard.press("Meta+Shift+C")
            time.sleep(0.3)
            page.keyboard.type("Run All Cells")
            time.sleep(0.3)
            page.keyboard.press("Enter")
            print("  Waiting for cells to execute...")
            time.sleep(30)

            capture_widgets(page, "light")

            set_theme(page, "dark")
            time.sleep(2)
            capture_widgets(page, "dark")

            print(f"\nScreenshots saved to: {SCREENSHOT_DIR}")
            browser.close()

    finally:
        if jupyter_proc:
            stop_jupyter(jupyter_proc)
        cleanup_test_notebook()


if __name__ == "__main__":
    main()
