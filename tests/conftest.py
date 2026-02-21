"""Shared fixtures for Playwright smoke tests."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

JUPYTER_PORT = 8898
TESTS_DIR = Path(__file__).parent
PROJECT_DIR = TESTS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

# Ensure tests import this checkout (src layout), not an installed package.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Keep matplotlib cache on a writable path to avoid expensive first-import churn.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "quantem-widget-mplconfig"))

@pytest.fixture(scope="session")
def jupyter_server():
    """Start a JupyterLab server for the entire test session."""
    def _port_open(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("localhost", port)) == 0

    def _pick_port(start_port: int) -> int:
        for offset in range(16):
            candidate = start_port + offset
            if not _port_open(candidate):
                return candidate
        return start_port

    def _wait_for_server(proc: subprocess.Popen, port: int, timeout_s: int = 90) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"JupyterLab exited early with code {proc.returncode}")
            if _port_open(port):
                return
            time.sleep(1.0)
        raise RuntimeError(f"JupyterLab failed to start within {timeout_s} seconds")

    def _tail_log(path: Path, max_lines: int = 80) -> str:
        if not path.exists():
            return ""
        lines = path.read_text(errors="ignore").splitlines()
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[-max_lines:])

    def _stop(proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    global JUPYTER_PORT
    base_port = int(os.environ.get("QUANTEM_TEST_JUPYTER_PORT", JUPYTER_PORT))
    JUPYTER_PORT = _pick_port(base_port)

    proc = None
    log_path = Path("/tmp") / f"quantem-widget-jupyter-{int(time.time())}.log"
    log_handle = None
    startup_errors: list[str] = []

    for attempt in (1, 2):
        if log_handle is not None and not log_handle.closed:
            log_handle.close()
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "jupyter",
                "lab",
                f"--port={JUPYTER_PORT}",
                "--ServerApp.port_retries=0",
                "--no-browser",
                "--NotebookApp.token=''",
                "--NotebookApp.password=''",
                "--ServerApp.disable_check_xsrf=True",
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_DIR,
        )
        try:
            _wait_for_server(proc, JUPYTER_PORT, timeout_s=90)
            time.sleep(2)
            break
        except RuntimeError as exc:
            log_handle.flush()
            startup_errors.append(f"attempt {attempt} on port {JUPYTER_PORT}: {exc}\n{_tail_log(log_path)}")
            _stop(proc)
            proc = None
            JUPYTER_PORT = _pick_port(JUPYTER_PORT + 1)
    else:
        if log_handle is not None and not log_handle.closed:
            log_handle.close()
        details = "\n\n".join(startup_errors)
        raise RuntimeError(f"JupyterLab failed to start after 2 attempts.\n{details}")

    yield proc

    _stop(proc)
    if log_handle is not None and not log_handle.closed:
        log_handle.close()

@pytest.fixture(scope="session")
def browser_context(jupyter_server):
    """Provide a Playwright browser context for the session."""
    from playwright.sync_api import sync_playwright

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
