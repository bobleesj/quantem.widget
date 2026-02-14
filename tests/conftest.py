"""Shared fixtures for Playwright smoke tests."""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

JUPYTER_PORT = 8898
TESTS_DIR = Path(__file__).parent
PROJECT_DIR = TESTS_DIR.parent


@pytest.fixture(scope="session")
def jupyter_server():
    """Start a JupyterLab server for the entire test session."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "jupyter", "lab",
            f"--port={JUPYTER_PORT}", "--no-browser",
            "--NotebookApp.token=''", "--NotebookApp.password=''",
            "--ServerApp.disable_check_xsrf=True",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=PROJECT_DIR,
    )
    # Wait for server
    for _ in range(30):
        time.sleep(1)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", JUPYTER_PORT))
            sock.close()
            if result == 0:
                time.sleep(2)
                break
        except OSError:
            pass
    else:
        proc.kill()
        raise RuntimeError("JupyterLab failed to start within 30 seconds")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def browser_context(jupyter_server):
    """Provide a Playwright browser context for the session."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        yield context
        browser.close()


def _write_notebook(path: Path, cells: list[dict]) -> Path:
    """Write a Jupyter notebook with the given code cells."""
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell["source"],
            }
            for cell in cells
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
    with open(path, "w") as f:
        json.dump(notebook, f, indent=2)
    return path


def _run_notebook_and_wait(page, notebook_path: Path, wait_seconds: int = 20):
    """Open a notebook in JupyterLab, run all cells, and wait for output."""
    rel_path = notebook_path.relative_to(PROJECT_DIR)
    url = f"http://localhost:{JUPYTER_PORT}/lab/tree/{rel_path}"
    page.goto(url, timeout=60000)
    time.sleep(8)

    # Dismiss any dialogs
    try:
        page.keyboard.press("Escape")
        time.sleep(0.5)
    except Exception:
        pass

    # Run All Cells via command palette
    page.keyboard.press("Meta+Shift+C")
    time.sleep(0.3)
    page.keyboard.type("Run All Cells")
    time.sleep(0.3)
    page.keyboard.press("Enter")
    time.sleep(wait_seconds)
