"""
E2E Screenshot Capture for Show3DVolume Widget

Captures screenshots of Show3DVolume in JupyterLab with both dark and light themes.
Uses Playwright to automate the browser.

Usage:
    python tests/capture_show3dvolume.py

Screenshots are saved to tests/screenshots/show3dvolume/
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Configuration
JUPYTER_PORT = 8899
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "show3dvolume"
NOTEBOOK_PATH = Path(__file__).parent / "_test_show3dvolume_features.ipynb"

# Feature names for captured widgets
FEATURE_NAMES = ["default", "viridis_cmap", "log_scale", "auto_contrast", "gray_cmap"]


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
                    "from quantem.widget import Show3DVolume\n",
                    "\n",
                    "# Create synthetic 3D volume: nested spheres\n",
                    "N = 64\n",
                    "coords = np.linspace(-1, 1, N)\n",
                    "zz, yy, xx = np.meshgrid(coords, coords, coords, indexing='ij')\n",
                    "r = np.sqrt(xx**2 + yy**2 + zz**2)\n",
                    "spheres = np.zeros((N, N, N), dtype=np.float32)\n",
                    "spheres[r < 0.9] = 0.3\n",
                    "spheres[r < 0.6] = 0.6\n",
                    "spheres[r < 0.3] = 1.0\n",
                    "print(f'Volume ready: {spheres.shape}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 1. Default View\n", "Show3DVolume(spheres, title='Default View')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 2. Viridis Colormap\n", "Show3DVolume(spheres, title='Viridis Colormap', cmap='viridis')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 3. Log Scale\n", "Show3DVolume(spheres, title='Log Scale', log_scale=True)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. Auto Contrast with noisy volume\n",
                    "np.random.seed(42)\n",
                    "noisy = np.random.randn(64, 64, 64).astype(np.float32) * 0.1\n",
                    "noisy[20:44, 20:44, 20:44] = 1.0\n",
                    "Show3DVolume(noisy, title='Auto Contrast', auto_contrast=True)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 5. Gray Colormap\n", "Show3DVolume(spheres, title='Gray Colormap', cmap='gray')"]
            }
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
    # Wait for server to start
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
    """Capture screenshots of widgets."""
    print(f"Capturing {theme} theme screenshots...")
    theme_dir = SCREENSHOT_DIR / theme
    theme_dir.mkdir(parents=True, exist_ok=True)
    # Capture full page
    page.keyboard.press("Meta+Home")
    time.sleep(0.5)
    page.screenshot(path=str(theme_dir / "full_page.png"), full_page=True)
    print(f"  Saved: full_page.png")
    # Capture individual widgets
    widgets = page.locator(".show3dvolume-root")
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
        # Generate test notebook
        create_test_notebook()
        jupyter_proc = start_jupyter()
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1400, "height": 900})
            page = context.new_page()
            # Navigate to generated notebook
            notebook_rel_path = NOTEBOOK_PATH.relative_to(Path(__file__).parent.parent)
            url = f"http://localhost:{JUPYTER_PORT}/lab/tree/{notebook_rel_path}"
            print(f"Opening: {url}")
            page.goto(url, timeout=60000)
            print("Waiting for JupyterLab to load...")
            time.sleep(10)
            # Dismiss dialogs
            try:
                page.keyboard.press("Escape")
                time.sleep(0.5)
            except:
                pass
            # Light theme
            set_theme(page, "light")
            # Run all cells
            page.keyboard.press("Meta+Shift+C")
            time.sleep(0.3)
            page.keyboard.type("Run All Cells")
            time.sleep(0.3)
            page.keyboard.press("Enter")
            print("  Waiting for cells to execute...")
            time.sleep(30)
            capture_widgets(page, "light")
            # Dark theme
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
