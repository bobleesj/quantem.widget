"""
E2E Screenshot Capture for Show3D Widget

Captures screenshots of Show3D in JupyterLab with both dark and light themes.
Uses Playwright to automate the browser.

Usage:
    python tests/capture_show3d.py

Screenshots are saved to tests/screenshots/show3d/
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Configuration
JUPYTER_PORT = 8899
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "show3d"
NOTEBOOK_PATH = Path(__file__).parent / "_test_show3d_features.ipynb"

# Feature names for captured widgets
FEATURE_NAMES = ["default", "fft_enabled", "viridis_cmap", "log_scale", "roi_active"]

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
                    "import torch\n",
                    "from quantem.widget import Show3D\n",
                    "from quantem.core.datastructures.dataset3d import Dataset3d\n",
                    "\n",
                    "def make_synthetic_stack(size=128, n_frames=10, seed=42):\n",
                    "    torch.manual_seed(seed)\n",
                    "    y = torch.arange(size, dtype=torch.float32).view(-1, 1).expand(size, size)\n",
                    "    x = torch.arange(size, dtype=torch.float32).view(1, -1).expand(size, size)\n",
                    "    center = size // 2\n",
                    "    r = torch.sqrt((x - center)**2 + (y - center)**2)\n",
                    "    base = torch.sin(r / 8) * torch.exp(-r / (size / 3))\n",
                    "    n_atoms = 30\n",
                    "    atom_x = torch.randint(size // 4, 3 * size // 4, (n_atoms,)).float()\n",
                    "    atom_y = torch.randint(size // 4, 3 * size // 4, (n_atoms,)).float()\n",
                    "    atom_contrib = torch.exp(-((x.unsqueeze(0) - atom_x.view(-1, 1, 1))**2 + (y.unsqueeze(0) - atom_y.view(-1, 1, 1))**2) / 15).sum(dim=0) * 2\n",
                    "    base = base + atom_contrib\n",
                    "    frame_idx = torch.arange(n_frames, dtype=torch.float32)\n",
                    "    sigmas = 1 + torch.abs(frame_idx - n_frames // 2) / max(1, n_frames // 8)\n",
                    "    freq_y = torch.fft.fftfreq(size).view(-1, 1)\n",
                    "    freq_x = torch.fft.fftfreq(size).view(1, -1)\n",
                    "    freq_sq = freq_y**2 + freq_x**2\n",
                    "    gaussian_filters = torch.exp(-2 * (torch.pi**2) * (sigmas.view(-1, 1, 1)**2) * freq_sq)\n",
                    "    base_fft = torch.fft.fft2(base)\n",
                    "    blurred_fft = base_fft.unsqueeze(0) * gaussian_filters\n",
                    "    stack = torch.fft.ifft2(blurred_fft).real\n",
                    "    stack = stack + torch.randn(n_frames, size, size) * 0.05\n",
                    "    return stack.numpy()\n",
                    "\n",
                    "dataset = Dataset3d.from_array(\n",
                    "    make_synthetic_stack(),\n",
                    "    name='Test Data',\n",
                    "    sampling=[1, 1.5, 1.5],\n",
                    "    units=['index', 'nm', 'nm'],\n",
                    ")\n",
                    "print(f'Dataset ready: {dataset.shape}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 1. Default View\n", "Show3D(dataset, title='Default View')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 2. FFT Enabled\n", "Show3D(dataset, title='FFT Enabled', show_fft=True)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 3. Viridis Colormap\n", "Show3D(dataset, title='Viridis Colormap', cmap='viridis')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 4. Log Scale\n", "Show3D(dataset, title='Log Scale', log_scale=True)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. ROI Active\n",
                    "w = Show3D(dataset, title='ROI Active')\n",
                    "w.roi_active = True\n",
                    "w.roi_x = 64\n",
                    "w.roi_y = 64\n",
                    "w.roi_radius = 20\n",
                    "w"
                ]
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
    widgets = page.locator(".show3d-root")
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
