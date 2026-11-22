"""ShowReconstruction: a run model holds its own run and saves what keep says; the dashboard and compare reference the
run models and copy no array; frames of another shape stay in their stack; a view's hello flushes only news."""

import numpy as np
import pytest

from quantem.widget import CompareReconstruction, ShowReconstruction
from quantem.widget.showreconstruction import KEEP


def frames(stack_bytes, shapes):
    """A flushed stack back as its frames, as the browser decodes it."""
    values, out, offset = np.frombuffer(stack_bytes, np.float32), [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        out.append(values[offset:offset + size].reshape(shape))
        offset += size
    return out


def run_model(start=0, epochs=8, every=4, modes=3, field=(12, 10)):
    """One run as myptycho draws it: its start, a projection every epoch, a probe and depth row at the draws, the volume at the end."""
    rng = np.random.default_rng(start)
    run = ShowReconstruction("test", pixel_size_A=0.2, slice_A=10.0)
    run.stage("run", {"probe.modes": (2, modes)}, start=start)
    run.add({"epoch": start, "held_out": 0.2, "start": True}, projection=rng.random(field))
    for epoch in range(1, epochs + 1):
        drawn = epoch == 1 or epoch % every == 0 or epoch == epochs
        record = {"epoch": start + epoch, "train": 1.0 / epoch, "held_out": 0.2 - 0.001 * epoch if drawn else None}
        if not drawn:
            run.add(record, projection=rng.random(field))
            continue
        run.add(record, projection=rng.random(field), slices=rng.random((4, *field)) if epoch == epochs else None, depth=rng.random(4),
                probe=rng.random((modes, 6, 6)), probe_shares=[0.6, 0.3, 0.1][:modes])
    run.flush()
    return run


def test_a_run_saves_every_projection_the_last_probe_its_strongest_mode_at_every_draw_and_its_last_volume():
    run = run_model()
    assert run.keep == KEEP
    assert run.snapshot_epochs == list(range(9)) and run.projection_shapes == [[12, 10]] * 9  # its start and every epoch
    assert run.probe_epochs == [8] and run.probe_shapes == [[3, 6, 6]] and run.probe_shares == [[0.6, 0.3, 0.1]]
    assert run.strongest_epochs == [1, 4] and run.strongest_shapes == [[6, 6]] * 2  # the last draw's strongest mode is its saved probe's first: not stored twice
    assert len(run._kept["strongest"]) == 3 and np.array_equal(run.last("strongest")[1], run.last("probes")[1][0])
    assert run.volumes == b"" and run.last("volumes")[0] == 8 and run.last("volumes")[1].shape == (4, 12, 10)  # the kernel keeps the volume; a saved state its ends
    assert run.depth_epochs == [1, 4, 8] and len(run.depth_rows[0]) == 4
    assert [row["probe_shares"] for row in run.records if "probe_shares" in row] == [[0.6, 0.3, 0.1]] * 3  # the shares panel reads them from the records
    assert run.stages == [{"name": "run", "start": 0, "changes": "probe.modes 2 → 3", "changed": [{"key": "probe.modes", "before": "2", "after": "3"}]}]


def test_an_earlier_run_saves_its_projections_at_its_draws():
    run = run_model()
    whole = run.saved_bytes()
    run.keep = {"projections": "draws", "probes": "last", "volumes": "none"}
    run.flush()
    assert run.snapshot_epochs == [0, 1, 4, 8] and run.volume_epochs == [] and run.volumes == b""
    assert run.strongest_epochs == [1, 4] and run.probe_epochs == [8]
    assert run.saved_bytes() == whole - 5 * 12 * 10 * 4
    assert len(run._kept["projections"]) == 9  # the kernel keeps every epoch


def test_frames_of_another_shape_stay_in_their_stack():
    """MyPtycho.changes can set probe.modes part way through a call: the probe's frames change shape and every
    draw stays, where one shape a stack threw the earlier draws away."""
    run = ShowReconstruction()
    run.stage("modes", start=0)
    run.add({"epoch": 1}, projection=np.ones((5, 5)), probe=np.ones((2, 4, 4)), probe_shares=[0.7, 0.3])
    run.add({"epoch": 2}, projection=np.ones((5, 6)), probe=np.ones((3, 4, 4)), probe_shares=[0.6, 0.3, 0.1])
    run.keep["probes"] = "every"
    run.flush()
    assert run.probe_epochs == [1, 2] and run.probe_shapes == [[2, 4, 4], [3, 4, 4]]
    assert run.projection_shapes == [[5, 5], [5, 6]] and len(run.projections) == 4 * (25 + 30)


def test_the_dashboard_references_its_runs_and_holds_no_array():
    first, second = run_model(0), run_model(8)
    board = ShowReconstruction("test", view="dashboard", runs=[first, second])
    state = board.get_state()
    assert state["runs"] == [f"IPY_MODEL_{first.model_id}", f"IPY_MODEL_{second.model_id}"]
    assert all(state[name] == b"" for name in ("projections", "volumes", "probes", "strongest"))
    assert "dashboard of 2 runs" in repr(board)


def test_compare_draws_a_run_from_its_models_ends_and_keeps_three_slices_of_its_own():
    """A run with a model is drawn from its saved after state (set_ends), referenced; a run given as a volume keeps
    three of its slices, the cut along the line and its projection, not the volume."""
    run = run_model()
    volume = np.random.default_rng(2).random((4, 12, 10)).astype(np.float32)
    run.set_ends("after", [({"kind": "slice", "slice": index, "slices": 4, "depth_A": [10.0 * index, 10.0 * index + 10]}, volume[index]) for index in (1, 2, 3)])
    own = {"name": "before", "volume": np.random.default_rng(1).random((8, 12, 10)), "probe": np.ones((2, 6, 6)) + 0j, "shares": [0.5, 0.5], "slice_A": 10.0, "pixel_A": 0.2}
    compare = CompareReconstruction([own, {"name": "after", "source": run, "slice_A": 10.0, "pixel_A": 0.2}], profile_line=[6, 0, 6, 9])
    assert [meta["source"] for meta in compare.runs] == [-1, 0] and compare.sources == [run] and compare.runs[1]["items"] == []
    kinds = [(item["kind"], item.get("slice")) for item in compare.runs[0]["items"]]
    assert kinds == [("slice", 2), ("slice", 4), ("slice", 6), ("cut", None), ("projection", None), ("probe", None)]
    cut = [item for item in compare.runs[0]["items"] if item["kind"] == "cut"][0]
    assert cut["shape"] == [8, 19] and cut["length_A"] == pytest.approx(9 * 0.2)  # two points a pixel along 9 px, every slice
    assert len(compare.arrays) == 4 * (3 * 120 + 8 * 19 + 120 + 2 * 2 * 36)


def test_a_hello_flushes_only_news():
    run = run_model()
    flushed = []
    run.flush = lambda: flushed.append(True)
    run._on_message(run, {"kind": "hello"}, [])
    assert flushed == []  # nothing since run_model's flush
    run.add({"epoch": 9}, projection=np.ones((12, 10)))
    run._on_message(run, {"kind": "hello"}, [])
    assert flushed == [True]


def test_a_complex_probe_is_kept_with_its_phase_and_its_aperture():
    """A complex probe is drawn in complex-wave colors: its frame keeps the real and imaginary parts, (modes, 2, rows,
    cols); its strongest mode (2, rows, cols); the aperture is its own stack, saved as the probe is."""
    run = ShowReconstruction()
    run.stage("complex", start=0)
    modes = (np.arange(2 * 4 * 5).reshape(2, 4, 5) + 1j * np.ones((2, 4, 5))).astype(np.complex64)
    run.add({"epoch": 1}, probe=modes, probe_shares=[0.8, 0.2], aperture=np.ones((6, 6)))
    run.flush()
    assert run.probe_shapes == [[2, 2, 4, 5]] and run.last("strongest")[1].shape == (2, 4, 5) and run.aperture_shapes == [[6, 6]]  # the strongest mode is read from the saved probe
    frame = frames(run.probes, run.probe_shapes)[0]
    np.testing.assert_array_equal(frame[:, 0], modes.real)
    np.testing.assert_array_equal(frame[:, 1], modes.imag)


def test_a_view_has_a_size_and_refuses_another():
    import pytest

    assert ShowReconstruction().size == "medium" and ShowReconstruction(size="large").size == "large"
    with pytest.raises(ValueError, match="size='huge'"):
        ShowReconstruction(size="huge")


def test_an_earlier_run_saves_its_middle_slice_of_each_state():
    """keep["ends"] = "middle" (an earlier run): each state keeps the slice nearest the slab's middle, its cut and its
    probe; the kernel keeps all three slices."""
    run = ShowReconstruction()
    slab = np.random.default_rng(3).random((8, 5, 5))
    for which in ("before", "after"):
        run.set_ends(which, [({"kind": "slice", "slice": index, "slices": 8}, slab[index]) for index in (2, 4, 6)] + [({"kind": "cut", "slices": 8}, slab[:, 2])])
    run.keep["ends"] = "middle"
    run.flush()
    assert [(item["which"], item["kind"], item.get("slice")) for item in run.ends_items] == [("before", "slice", 4), ("before", "cut", None), ("after", "slice", 4), ("after", "cut", None)]
    assert len(run.ends) == 4 * 2 * (25 + 40) and len(run._ends) == 8


def test_a_model_carries_the_bundle_only_once_a_cell_shows_it():
    """A run model the dashboard references but no cell shows stores a one-line module, not the 0.6 MB bundle; shown,
    it takes the bundle."""
    run, board = run_model(), None
    board = ShowReconstruction(view="dashboard", runs=[run])
    assert len(run._esm) < 100 and len(board._esm) < 100
    board._repr_mimebundle_()
    assert len(board._esm) > 100_000 and len(run._esm) < 100


def test_show_direct_holds_only_what_it_draws():
    """ShowDirect: the virtual images at scan resolution, the region's phase window, the region and the readout."""
    from quantem.widget import ShowDirect

    rng = np.random.default_rng(4)
    view = ShowDirect(rng.random((16, 20)), rng.random((16, 20)), rng.random((24, 24)), scan_sampling_A=0.5, upsample=2, region=(4, 8, 6, 10),
                      phase_region=(2, 14, 4, 16), readout={"region": {"tilt_mrad": [-9.9, 3.1], "defocus_nm": 3.6}})
    assert view.image_shapes == [[16, 20], [16, 20], [24, 24]] and len(view.images) == 4 * (2 * 320 + 576)
    assert view.region == [4, 8, 6, 10] and view.phase_region == [2, 14, 4, 16] and view.kind == "direct"


def test_the_region_box_stays_square_on_whole_scan_positions():
    """square_region: rounded to whole positions, square on the shorter side, an even side of at least 2, inside the scan;
    an editable ShowDirect starts from it, and with no region from the middle eighth of the scan."""
    from quantem.widget import ShowDirect
    from quantem.widget.showreconstruction import square_region

    assert square_region((100.4, 165, 200, 260), (512, 512)) == (100, 160, 200, 260)
    assert square_region((500, 540, -5, 30), (512, 512)) == (478, 512, 0, 34)
    assert square_region((10, 11, 10, 11), (512, 512)) == (10, 12, 10, 12)
    view = ShowDirect(np.zeros((64, 64)), np.zeros((64, 64)), scan_sampling_A=0.5, region=(10, 25, 10, 30), editable=True)
    assert view.region == [10, 24, 10, 24] and view.image_shapes == [[64, 64], [64, 64]]
    assert ShowDirect(np.zeros((64, 64)), np.zeros((64, 64)), scan_sampling_A=0.5, editable=True).region == [28, 36, 28, 36]


def test_compare_takes_a_run_of_unknown_thickness_and_says_its_band():
    """A run with no slab thickness (slice_A None) is compared per slice, not refused; the lattice band reaches the view."""
    volume = np.random.default_rng(5).random((4, 12, 10))
    compare = CompareReconstruction([{"name": "a", "volume": volume, "slice_A": None, "pixel_A": 0.2}, {"name": "b", "volume": volume, "slice_A": 10.0, "pixel_A": 0.2}],
                                    band_per_A=(0.24, 1.9))
    assert compare.runs[0]["slice_A"] == 0.0 and compare.runs[0]["items"][0]["depth_A"] is None and compare.band_per_A == [0.24, 1.9]


def test_a_rebuilt_bundle_does_not_reach_a_run_model_no_cell_shows():
    """The views carry their code as text, not as a watched file: anywidget's live reload would write the whole bundle
    into every run model when a builder rebuilds, and the saved notebook would carry it once a run (21 Nov 2026)."""
    from quantem.widget.showreconstruction import STUB

    assert isinstance(ShowReconstruction._esm, str) or isinstance(ShowReconstruction().__class__._esm, str)
    assert ShowReconstruction()._esm == STUB


def test_the_status_view_reads_each_runs_settings_and_what_its_stage_changed():
    """view="status": the runs referenced, each with the settings it started with (text, as a person set them) and its
    stage's changes, old against new: what the browser lays out as one row a setting, one column a run."""
    from quantem.widget import ShowReconstruction

    first, second = ShowReconstruction("run"), ShowReconstruction("run")
    first.stage("search", {})
    second.stage("10 modes", {"probe.modes": (8, 10)})
    second.settings = {"probe.modes": "10", "object.slices": "27"}
    status = ShowReconstruction("status", view="status", runs=[first, second])
    assert status.view == "status" and len(status.runs) == 2
    assert second.stages[0]["changed"] == [{"key": "probe.modes", "before": "8", "after": "10"}]
    assert second.settings["object.slices"] == "27" and first.settings == {}
