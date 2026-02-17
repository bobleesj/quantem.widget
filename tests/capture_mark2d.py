"""
E2E Screenshot Capture for Mark2D Widget

Captures screenshots of Mark2D in JupyterLab with both dark and light themes.
Uses Playwright to automate the browser.

Usage:
    python tests/capture_mark2d.py

Screenshots are saved to tests/screenshots/mark2d/
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Configuration
JUPYTER_PORT = 8899
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "mark2d"
NOTEBOOK_PATH = Path(__file__).parent / "_test_mark2d_features.ipynb"

# Feature names for captured widgets
FEATURE_NAMES = ["default", "preloaded_points", "viridis_log", "gallery", "diffraction_snap"]


def create_test_notebook():
    """Generate test notebook with specific feature configurations."""
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "from quantem.widget import Mark2D\n",
                    "\n",
                    "np.random.seed(42)\n",
                    "# HAADF-STEM-like atomic columns on hexagonal lattice\n",
                    "image = np.zeros((128, 128), dtype=np.float32)\n",
                    "a = 12  # lattice parameter in pixels\n",
                    "for i in range(-10, 15):\n",
                    "    for j in range(-10, 15):\n",
                    "        cx = 64 + int(i * a + j * a * 0.5)\n",
                    "        cy = 64 + int(j * a * 0.866)\n",
                    "        if 0 <= cx < 128 and 0 <= cy < 128:\n",
                    "            y, x = np.ogrid[max(0,cy-4):min(128,cy+5), max(0,cx-4):min(128,cx+5)]\n",
                    "            r2 = (x - cx)**2 + (y - cy)**2\n",
                    "            image[max(0,cy-4):min(128,cy+5), max(0,cx-4):min(128,cx+5)] += np.exp(-r2 / 3.0)\n",
                    "image += np.random.normal(0, 0.02, image.shape).astype(np.float32)\n",
                    "print(f'Image ready: {image.shape}')",
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 1. Default View\n", "Mark2D(image, scale=2.0)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Pre-loaded points with ROI and pixel calibration\n",
                    "pts = [(30, 30), (42, 30), (54, 30), (36, 41), (48, 41), (60, 41)]\n",
                    "w = Mark2D(image, scale=2.0, points=pts, pixel_size_angstrom=1.5)\n",
                    "w.add_roi(64, 64, mode='circle', radius=20, color='#00ff00')\n",
                    "w.add_roi(40, 80, mode='square', radius=15, color='#ff9800')\n",
                    "w"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Viridis colormap + log scale\n",
                    "Mark2D(image, scale=2.0, colormap='viridis', log_scale=True,\n",
                    "        marker_border=0, marker_opacity=0.7, label_color='yellow')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. Gallery mode\n",
                    "imgs = [image, np.log1p(np.maximum(image, 0)) * 5,\n",
                    "        np.flipud(image)]\n",
                    "Mark2D(imgs, scale=1.5, ncols=3, labels=['Original', 'Enhanced', 'Flipped'])"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. Diffraction pattern with snap-to-peak\n",
                    "diff = np.random.normal(0.02, 0.005, (128, 128))\n",
                    "y, x = np.mgrid[:128, :128]\n",
                    "a = 20\n",
                    "g1 = np.array([a, 0.0])\n",
                    "g2 = np.array([a * 0.5, a * np.sqrt(3) / 2])\n",
                    "for i in range(-5, 6):\n",
                    "    for j in range(-5, 6):\n",
                    "        sx = 64 + i * g1[0] + j * g2[0]\n",
                    "        sy = 64 + i * g1[1] + j * g2[1]\n",
                    "        if 0 <= sx < 128 and 0 <= sy < 128:\n",
                    "            dist = np.sqrt((sx - 64)**2 + (sy - 64)**2)\n",
                    "            intensity = np.exp(-dist**2 / (2 * (3*a)**2))\n",
                    "            if i == 0 and j == 0: intensity = 1.0\n",
                    "            diff += intensity * np.exp(-((x-sx)**2 + (y-sy)**2) / (2*0.8**2))\n",
                    "diff = np.clip(diff, 0, None).astype(np.float32)\n",
                    "Mark2D(diff, scale=2.0, snap_enabled=True, snap_radius=8,\n",
                    "        colormap='viridis', log_scale=True, dot_size=8, max_points=10)"
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
    """Remove generated test notebook."""
    if NOTEBOOK_PATH.exists():
        NOTEBOOK_PATH.unlink()
        print(f"Cleaned up: {NOTEBOOK_PATH}")


def start_jupyter():
    """Start JupyterLab server in the background."""
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
        except:
            pass
    raise RuntimeError("JupyterLab failed to start within 30 seconds")


def stop_jupyter(proc):
    """Stop JupyterLab server."""
    print("Stopping JupyterLab...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except:
        proc.kill()


def set_theme(page, theme: str):
    """Set JupyterLab theme via JavaScript."""
    print(f"  Setting {theme} theme...")
    if theme == "dark":
        page.evaluate("""() => {
            document.body.dataset.jpThemeLight = 'false';
            document.body.dataset.jpThemeName = 'JupyterLab Dark';
            document.body.classList.remove('jp-theme-light');
            document.body.classList.add('jp-theme-dark');
        }""")
    else:
        page.evaluate("""() => {
            document.body.dataset.jpThemeLight = 'true';
            document.body.dataset.jpThemeName = 'JupyterLab Light';
            document.body.classList.remove('jp-theme-dark');
            document.body.classList.add('jp-theme-light');
        }""")
    time.sleep(1)


def capture_widgets(page, theme: str):
    """Capture screenshots of widgets."""
    print(f"Capturing {theme} theme screenshots...")
    theme_dir = SCREENSHOT_DIR / theme
    theme_dir.mkdir(parents=True, exist_ok=True)

    page.keyboard.press("Meta+Home")
    time.sleep(0.5)
    page.screenshot(path=str(theme_dir / "full_page.png"), full_page=True)
    print(f"  Saved: full_page.png")

    widgets = page.locator(".mark2d-root")
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
    """Main entry point."""
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
            time.sleep(20)
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
