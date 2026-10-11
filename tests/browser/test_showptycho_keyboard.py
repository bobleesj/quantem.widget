"""ShowPtycho keyboard shortcuts act only while the widget has focus, in a live JupyterLab.

A notebook holds an SSB session in ShowPtycho and an empty code cell. Text
typed into the cell, with the letters and keys the widget binds (P, R, S,
Space, arrows), must land in the cell editor untouched while the widget sits
on the page. After a click on the widget the same keys must reach it: P pins
the current phase and Space starts and stops the aberration sweep, and the
cell text stays as it was.
"""

import pytest

pytestmark = pytest.mark.browser

PHASE = '[data-quantem-scientific-output="Phase"]'
PINS = "[data-showptycho-pin-id]"
SESSION = """import numpy as np
from quantem.gpu import SSB
from quantem.widget import ShowPtycho

rows, cols = np.mgrid[0:32, 0:32]
disk = (np.hypot(rows - 16, cols - 16) < 6).astype(np.float32)
scan = np.random.default_rng(0).normal(size=(24, 24, 1, 1)).astype(np.float32)
data = disk * (100 + scan) * (1 + 0.02 * np.cos(rows / 3.0))
viewer = ShowPtycho(SSB(data, voltage_kV=300, semiangle_mrad=20, scan_sampling_A=0.5, det_sampling=20 / 6))
viewer"""


def test_shortcuts_need_widget_focus_and_leave_cell_typing_alone(jupyter, open_page, hardware_adapter, browser_out):
    pytest.importorskip("quantem.gpu")
    jupyter.write_notebook("showptycho_keys.ipynb", SESSION, "")
    page = jupyter.open(open_page, "showptycho_keys.ipynb", "keys", webgpu=True, ready=PHASE, out=browser_out)
    hardware_adapter(page.adapter())
    editor = page.locator(".jp-Cell .cm-content").nth(1)
    editor.click()
    page.keyboard.type("pp rs")
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("ArrowLeft")
    page.keyboard.type("x")
    page.wait_for_timeout(500)
    page.screenshot(path=str(browser_out / "showptycho_keys_typed.png"))
    assert editor.inner_text() == "pp xrs"
    assert page.locator(PINS).count() == 0
    assert page.get_by_label("Play sweep").count() == 1
    page.locator(PHASE).first.click()
    page.keyboard.press("p")
    page.wait_for_timeout(300)
    page.keyboard.press("p")
    page.wait_for_timeout(300)
    assert page.locator(PINS).count() == 2
    page.keyboard.press(" ")
    page.wait_for_timeout(300)
    assert page.get_by_label("Pause sweep").count() == 1
    page.keyboard.press(" ")
    page.wait_for_timeout(300)
    assert page.get_by_label("Play sweep").count() == 1
    page.screenshot(path=str(browser_out / "showptycho_keys_widget.png"))
    assert editor.inner_text() == "pp xrs"
    # JupyterLab's debugger panel reports "No active debugger session" on its own
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
