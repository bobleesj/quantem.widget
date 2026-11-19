"""Multiple recordings stay interactive while their detector is moved."""

import json

import pytest

pytestmark = pytest.mark.browser

DP = '[data-quantem-scientific-output="show4dstem-diffraction-pattern"]'
COMPARISON = '[data-quantem-scientific-output="show4dstem-compare-0"]'
SOURCE = """
import torch
from quantem.widget import Show4DSTEM

scan_t = torch.arange(16 * 16).reshape(1, 16, 16, 1, 1)
row_t = torch.arange(32).reshape(1, 1, 1, 32, 1)
col_t = torch.arange(32).reshape(1, 1, 1, 1, 32)
frame_t = torch.arange(3).reshape(3, 1, 1, 1, 1)
data_t = ((scan_t * 5 + row_t * 3 + col_t * 7 + frame_t * 11) % 97 + 1).float()
viewer = Show4DSTEM(
    data_t, device="cpu", offline=False, precompute_virtual_images=False,
    bright_field_center_px=(15.5, 15.5), bright_field_radius_px=5,
    view_mode="multiple", compare_max_panels=3, compare_group_mode="all",
)
viewer
"""
REPORT = """
import hashlib, json
mask_t = torch.as_tensor(viewer._current_detector_mask(), dtype=torch.float64)
expected_t = (data_t[viewer.compare_panel_indices].double() * mask_t).sum((-2, -1)).float()
expected_t /= max(1, int(torch.count_nonzero(mask_t)))
actual_t = torch.frombuffer(bytearray(viewer.compare_virtual_image_bytes), dtype=torch.float32)
print(json.dumps({
    "center": list(viewer.roi_center),
    "pos": [viewer.pos_row, viewer.pos_col],
    "region": list(viewer.vi_roi_center),
    "panels": len(viewer.compare_panel_indices),
    "equal": torch.equal(actual_t, expected_t.flatten()),
    "sha256": hashlib.sha256(viewer.compare_virtual_image_bytes).hexdigest(),
}))
"""


def _position(page, selector, row, col, size):
    box = page.locator(selector).first.bounding_box()
    return (box["x"] + (col + 0.5) * box["width"] / size,
            box["y"] + (row + 0.5) * box["height"] / size)


def _wait_state(page, jupyter, notebook, predicate):
    for _ in range(30):
        state = json.loads(jupyter.run_in_kernel(notebook, REPORT).splitlines()[-1])
        if predicate(state):
            return state
        page.wait_for_timeout(100)
    pytest.fail(f"Comparison did not update: {state}")


def test_an_old_live_direct_preview_does_not_label_retained_comparison_phases(jupyter, open_page, browser_out):
    notebook = "show4dstem_compare_direct_title.ipynb"
    source = SOURCE.removesuffix("viewer\n") + """
for index in range(3):
    viewer.attach("Direct", torch.arange(48 * 48).reshape(48, 48).float() + index, frame=index)
viewer.vi_source = "Direct"
viewer.live_phase_stream = True
viewer.live_phase_preview = dict(scan="live_19", total=16, rows=16, cols=16, detRows=32, detCols=32,
                                 mode="Direct", upsample=3, completed_direct_generation="old", frozen=True)
viewer
"""
    jupyter.write_notebook(notebook, source)
    page = jupyter.open(open_page, notebook, "compare_direct_title", webgpu=False, ready=COMPARISON, out=browser_out)
    page.locator('[title^="Multiple Direct ptycho"]').first.wait_for(timeout=10_000)
    assert page.get_by_text("ROI · Direct pending", exact=False).count() == 0
    assert page.get_by_label("Direct phase source").count() == 0


def test_comparison_detector_drag_and_scan_region_switch(
    jupyter, open_page, hardware_adapter, browser_out,
):
    notebook = "show4dstem_compare_pointer.ipynb"
    jupyter.write_notebook(notebook, SOURCE)
    page = jupyter.open(open_page, notebook, "compare_pointer", webgpu=True,
                        ready=COMPARISON, out=browser_out)
    adapter = page.adapter()
    hardware_adapter(adapter)
    before = _wait_state(page, jupyter, notebook, lambda state: state["equal"])
    before_png = page.locator(COMPARISON).first.screenshot(
        path=str(browser_out / "compare_pointer_before.png"))
    page.locator(DP).first.scroll_into_view_if_needed()
    page.mouse.move(*_position(page, DP, 15.5, 15.5, 32))
    page.mouse.down()
    try:
        page.mouse.move(*_position(page, DP, 9, 22, 32), steps=12)
        held = _wait_state(page, jupyter, notebook, lambda state: (
            state["center"] != before["center"] and state["equal"]
            and state["sha256"] != before["sha256"]
        ))
        assert held["panels"] == 3
        held_png = page.locator(COMPARISON).first.screenshot(
            path=str(browser_out / "compare_pointer_held.png"))
        assert held_png != before_png
    finally:
        page.mouse.up()

    jupyter.run_in_kernel(notebook,
        "viewer.vi_roi_mode = 'circle'; viewer.vi_roi_radius = 2; "
        "viewer.vi_roi_center = [3.0, 3.0]")
    page.wait_for_timeout(300)
    page.locator(COMPARISON).first.scroll_into_view_if_needed()
    page.mouse.click(*_position(page, COMPARISON, 11, 10, 16))
    region = _wait_state(page, jupyter, notebook, lambda state: (
        abs(state["region"][0] - 11) < 0.05
        and abs(state["region"][1] - 10) < 0.05
    ))
    assert region["pos"] == held["pos"]
    jupyter.run_in_kernel(notebook, "viewer.vi_roi_mode = 'off'")
    page.wait_for_timeout(300)
    page.mouse.click(*_position(page, COMPARISON, 6, 4, 16))
    point = _wait_state(page, jupyter, notebook, lambda state: state["pos"] == [6, 4])
    assert page.adapter() == adapter
    (browser_out / "compare_pointer.json").write_text(json.dumps({
        "adapter": adapter, "before": before, "held": held,
        "region": region, "point": point,
    }, indent=2))
    assert [error for error in page.errors if "debugger session" not in error] == []


def test_wheel_zoom_after_stream_tiles_are_replaced(jupyter, open_page, hardware_adapter, browser_out):
    notebook = "show4dstem_rolling_zoom.ipynb"
    jupyter.write_notebook(notebook, SOURCE)
    page = jupyter.open(open_page, notebook, "rolling_zoom", webgpu=True, ready=COMPARISON, out=browser_out)
    hardware_adapter(page.adapter())
    tile = page.get_by_role("button", name="Show4DSTEM multiple panel 1", exact=True)
    original = tile.element_handle()
    jupyter.run_in_kernel(notebook, "viewer._stream_source_ids = ['new-a', 'new-b', 'new-c']")
    page.wait_for_function("tile => !tile.isConnected", arg=original)
    tile.scroll_into_view_if_needed()
    box = tile.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    before = tile.screenshot(path=str(browser_out / "rolling_zoom_before.png"))
    page.mouse.wheel(0, -120)
    page.wait_for_function("tile => Number(tile.dataset.comparisonZoom) > 1", arg=tile.element_handle(), timeout=4000)
    assert tile.screenshot(path=str(browser_out / "rolling_zoom_after.png")) != before
