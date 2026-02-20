"""
E2E Screenshot Capture for Bin widget.

Captures Bin in JupyterLab with light/dark theme screenshots and
basic interactive control changes.

Usage:
    python tests/capture_bin.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

JUPYTER_PORT = 8899
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "bin"
NOTEBOOK_PATH = Path(__file__).parent / "_test_bin_features.ipynb"


def create_test_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "from quantem.widget import Bin\n",
                    "\n",
                    "np.random.seed(42)\n",
                    "data = np.random.rand(8, 8, 32, 32).astype(np.float32)\n",
                    "Bin(data, pixel_size=2.4, k_pixel_size=0.48, device='cpu')\n",
                ],
            }
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
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2))


def cleanup_test_notebook() -> None:
    NOTEBOOK_PATH.unlink(missing_ok=True)


def start_jupyter():
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

    for _ in range(30):
        time.sleep(1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", JUPYTER_PORT))
        sock.close()
        if result == 0:
            time.sleep(2)
            return proc
    raise RuntimeError("JupyterLab failed to start within 30 seconds")


def stop_jupyter(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def set_theme(page, dark: bool) -> None:
    if dark:
        page.evaluate(
            """() => {
                document.body.dataset.jpThemeLight = 'false';
                document.body.dataset.jpThemeName = 'JupyterLab Dark';
                document.body.classList.remove('jp-theme-light');
                document.body.classList.add('jp-theme-dark');
            }"""
        )
    else:
        page.evaluate(
            """() => {
                document.body.dataset.jpThemeLight = 'true';
                document.body.dataset.jpThemeName = 'JupyterLab Light';
                document.body.classList.remove('jp-theme-dark');
                document.body.classList.add('jp-theme-light');
            }"""
        )
    time.sleep(1.5)


def click_menu_item(page, text: str) -> None:
    item = page.locator(f'.MuiMenuItem-root:has-text("{text}")').first
    item.wait_for(state="visible", timeout=4000)
    item.click()


def change_dropdown(widget, page, nth: int, value: str, wait: float = 0.8) -> None:
    page.keyboard.press("Escape")
    time.sleep(0.2)
    widget.locator(".MuiSelect-select").nth(nth).click()
    time.sleep(0.4)
    click_menu_item(page, value)
    time.sleep(wait)


def capture(page) -> None:
    light_dir = SCREENSHOT_DIR / "light"
    dark_dir = SCREENSHOT_DIR / "dark"
    light_dir.mkdir(parents=True, exist_ok=True)
    dark_dir.mkdir(parents=True, exist_ok=True)

    widget = page.locator(".bin-root").first
    widget.wait_for(state="visible", timeout=15000)
    widget.scroll_into_view_if_needed()
    time.sleep(1)

    widget.screenshot(path=str(light_dir / "default.png"))

    change_dropdown(widget, page, 0, "2")
    change_dropdown(widget, page, 1, "2")
    change_dropdown(widget, page, 2, "2")
    change_dropdown(widget, page, 3, "2")
    change_dropdown(widget, page, 4, "sum")
    change_dropdown(widget, page, 5, "pad")
    change_dropdown(widget, page, 6, "magma")
    change_dropdown(widget, page, 7, "log")
    widget.screenshot(path=str(light_dir / "binned_sum_pad_magma_log.png"))

    set_theme(page, dark=True)
    widget.screenshot(path=str(dark_dir / "dark_mode.png"))


def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    jupyter_proc = None
    try:
        create_test_notebook()
        jupyter_proc = start_jupyter()
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1500, "height": 1000})
            page = context.new_page()
            rel_path = NOTEBOOK_PATH.relative_to(Path(__file__).parent.parent)
            page.goto(f"http://localhost:{JUPYTER_PORT}/lab/tree/{rel_path}", timeout=60000)
            page.wait_for_timeout(2000)
            page.keyboard.press("Shift+Enter")
            page.wait_for_timeout(6000)
            capture(page)
            context.close()
            browser.close()
    finally:
        if jupyter_proc:
            stop_jupyter(jupyter_proc)
        cleanup_test_notebook()


if __name__ == "__main__":
    main()
