"""ShowSearch: candidates planned, trained and landed reach the saved traits on flush, each array once and native; a
pick marks its stage's tiles; a view's hello flushes only news."""

import numpy as np

from quantem.widget import ShowSearch


def frames(view):
    """The flushed arrays back as the browser decodes them, {(name, kind): array}."""
    values, out, offset = np.frombuffer(view.images, np.float32), {}, 0
    for entry in view.image_index:
        size = int(np.prod(entry["shape"]))
        out[(entry["name"], entry["kind"])] = values[offset:offset + size].reshape(entry["shape"])
        offset += size
    return out


def searched():
    rng = np.random.default_rng(0)
    view = ShowSearch("AutoPtycho · 64 × 64 region", pixel_size_A=0.25, code='AutoPtychography.load("search").reconstruction("{name}")')
    sent = []
    view.send = lambda content, buffers=None: sent.append((content, buffers or []))
    for index, (thickness, focus) in enumerate([(20, -5), (20, 0), (40, 0)]):
        view.candidate(f"pass1-00{index}", stage="geometry", thickness_nm=thickness, focus_nm=focus, slices=12, modes=6, epochs=4)
    arrays = {}
    for index in range(3):
        name = f"pass1-00{index}"
        for epoch in range(5):
            view.epoch(name, epoch, 0.4 - 0.05 * epoch - 0.001 * index)
        arrays[name] = (rng.random((16, 18)), rng.random((20, 20)), rng.random((12, 32)))
        view.land(name, 0.2 - 0.001 * index, projection=arrays[name][0], probe=arrays[name][1], section=arrays[name][2])
    view.candidate("pass1-1002", stage="geometry", thickness_nm=40, focus_nm=0, slices=12, modes=6, epochs=2, parent="pass1-002")
    view.pick("geometry", "pass1-002", held_out=0.198, paired_SE=0.0004, ties=["pass1-001"])
    return view, sent, arrays


def test_a_search_saves_every_landed_candidates_arrays_native_and_its_curves():
    view, sent, arrays = searched()
    assert view.images == b"" and view.curves == {}  # nothing synced until flush: each array went once as a message
    assert [content["kind"] for content, _ in sent].count("arrays") == 3
    assert sum(len(buffers) for _, buffers in sent) == 9
    view.flush()
    saved = frames(view)
    for name, (projection, probe, section) in arrays.items():
        np.testing.assert_array_equal(saved[(name, "projection")], projection.astype(np.float32))
        np.testing.assert_array_equal(saved[(name, "probe")], probe.astype(np.float32))
        np.testing.assert_array_equal(saved[(name, "section")], section.astype(np.float32))
    assert view.curves["pass1-002"][-1] == [4, 0.4 - 0.2 - 0.002]
    assert [row["state"] for row in view.candidates] == ["done", "done", "done", "queued"]
    assert view.candidates[3]["parent"] == "pass1-002"


def test_a_pick_marks_its_stage_and_the_view_says_it_in_five_decimals():
    view, _, _ = searched()
    assert [row["mark"] for row in view.candidates] == ["", "tie", "best", ""]
    assert view.best == "pass1-002" and view.picks[0]["paired_SE"] == 0.0004
    assert repr(view) == "ShowSearch('AutoPtycho · 64 × 64 region', 3/4 candidates, best pass1-002 held-out 0.19800)"


def test_a_finishing_stage_draws_on_the_strip_and_a_new_thickness_as_a_grid_row():
    view, _, _ = searched()
    view.stage("vibration path", verdict="kept", held_out=0.17914, start=0.17953, control=0.17949, projection=np.zeros((16, 18)))
    view.stage("slab 450 Å", verdict="refused", held_out=0.17920, start=0.17914, control=0.17910, thickness_nm=45, focus_nm=0)
    view.stage("vibration path", verdict="kept", held_out=0.17913, start=0.17953, control=0.17949)  # read again: replaced, not added
    assert [row["name"] for row in view.stages] == ["slab 450 Å", "vibration path"]
    assert view.stages[0]["thickness_nm"] == 45.0 and view.stages[1]["thickness_nm"] is None
    view.flush()
    assert ("stage:vibration path", "projection") in frames(view)


def test_a_view_that_opens_gets_a_flush_only_when_something_is_new():
    view, _, _ = searched()
    view._on_message(view, {"kind": "hello"}, [])
    flushed = view.images
    assert flushed != b"" and not view._changed
    view.images = b""
    view._on_message(view, {"kind": "hello"}, [])
    assert view.images == b""
