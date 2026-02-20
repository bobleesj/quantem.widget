"""
E2E Screenshot Capture for Show4DSTEM Widget

Captures screenshots of Show4DSTEM in JupyterLab with both dark and light themes.
Uses Playwright to automate the browser.

Usage:
    python tests/capture_show4dstem.py

Screenshots are saved to tests/screenshots/show4dstem/
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Configuration
JUPYTER_PORT = 8899
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "show4dstem"
NOTEBOOK_PATH = Path(__file__).parent / "_test_show4dstem_features.ipynb"

# Feature names for captured widgets
FEATURE_NAMES = ["default", "circle_roi", "annular_roi", "log_mode"]


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
                    "from quantem.widget import Show4DSTEM\n",
                    "\n",
                    "np.random.seed(42)\n",
                    "data = np.random.rand(8, 8, 32, 32).astype(np.float32)\n",
                    "# Add bright field disk\n",
                    "y, x = np.mgrid[0:32, 0:32]\n",
                    "r = np.sqrt((x - 16)**2 + (y - 16)**2)\n",
                    "bf_disk = np.exp(-r**2 / 20)\n",
                    "data += bf_disk[np.newaxis, np.newaxis, :, :] * 5\n",
                    "print(f'Data ready: {data.shape}')",
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 1. Default View\n", "Show4DSTEM(data)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Circle ROI\n",
                    "w = Show4DSTEM(data)\n",
                    "w.roi_circle(5)\n",
                    "w",
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Annular ROI\n",
                    "w = Show4DSTEM(data)\n",
                    "w.roi_annular(3, 10)\n",
                    "w",
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. Log Scale\n",
                    "w = Show4DSTEM(data)\n",
                    "w.dp_scale_mode = 'log'\n",
                    "w",
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

    widgets = page.locator(".show4dstem-root")
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
