"""ShowReconstruction: the live views of an iterative reconstruction, one model a run and a dashboard of all of them.

A reconstruction is staged in runs ("learned", "10 modes", ...). Each run owns one model (``view="run"``): the cell
that ran it shows that run only, while it trains and in the saved notebook. It holds the run's records, its projected
phase at every epoch (its start included), the probe's modes at the draws and its last volume. The dashboard
(``view="dashboard"``) holds no arrays: it references the run models and draws them as one story on one axis of total
epochs, each run a band and a tile in its stage strip. The browser draws every panel at the epoch the stepper picks,
with no kernel round trip.

The widget only displays what it is given: the code that trains reads its model between epochs and calls ``add``,
so drawing never enters a training step. Arrays stay native (no binning). What a saved notebook keeps of a run is
``keep``: by default every projection, the probe's modes at the last draw (its strongest mode at every draw) and the
last volume; the code that stages runs may save less of the earlier ones and says so in one line.
"""

from __future__ import annotations

import pathlib

import anywidget
import ipywidgets
import numpy as np
import traitlets

from quantem.widget.utils.array import to_numpy

# What the kernel keeps of a run for stepping back through it, in bytes. Every checkpoint is kept until its stack
# passes the budget; then every other one across the stack goes (the newest always stays), said once per stack.
BUDGETS = {"projections": 400e6, "volumes": 600e6, "probes": 200e6, "strongest": 50e6, "apertures": 50e6}
STACKS = ("projections", "volumes", "probes", "strongest", "apertures")
SIZES = ("small", "medium", "large")  # a view's drawn size: small the compact strip, medium the default, large the images dominant


def _probe_frame(probe) -> np.ndarray:
    """A probe's modes as the browser draws them: complex modes (modes, rows, cols) as (modes, 2, rows, cols) float32,
    the real and imaginary parts (hue the phase, brightness the amplitude); a real array (an intensity) as it is."""
    values = to_numpy(probe)
    if np.iscomplexobj(values):
        values = np.stack([values.real, values.imag], axis=-3)
    return np.ascontiguousarray(values, dtype=np.float32)
KEEP = {"projections": "every", "probes": "last", "volumes": "none", "ends": "all"}  # a run's saved state: its depth is its ends (set_ends), the kernel keeps the volume


BUNDLE = pathlib.Path(__file__).parent / "static" / "showreconstruction.js"
STUB = "export default { render() {} };"  # the module of a run model no cell shows: the views that reference it read its state, not its code


def _float32(array) -> np.ndarray:
    """A C-contiguous float32 copy of any array (torch on any device, numpy): what the browser decodes."""
    return np.ascontiguousarray(to_numpy(array), dtype=np.float32)


def _plain(value) -> str:
    if value is None:
        return "-"
    return f"{value:.5f}" if abs(value) >= 1e-5 or value == 0 else f"{value:.8f}"


class ShowReconstruction(anywidget.AnyWidget):
    """One run's live view, or the dashboard of several runs.

    Run view (``view="run"``): a header (held-out at the run's start and now), the projected phase, its spectrum and
    its change since the run started, the probe's modes and its strongest mode at each draw, a third panel chosen by
    what the run changed (``panel``: "path" the scan's vibration path and line offsets, "shares" each mode's power,
    "depth" the cut through the slab, "tilt", or "" the run's objective), and an epoch filmstrip.

    Dashboard (``view="dashboard"``, ``runs`` the run models): the story of every run, a stage strip with one tile a
    run, the object at any epoch (slice, projection, line profile, depth cut, lattice power), the probe, and the tilt,
    tilt map and scan panels when the runs hold them.

    Parameters
    ----------
    title : str
        What the reconstruction is, in plain words (specimen, scans).
    pixel_size_A : float
        Object pixel, for the scale bar and the profile's axis.
    slice_A : float or None
        Slice thickness, for the depth axis.
    bars : dict
        The pass bars of the probe's health, {name: (value, "below" or "above")}, e.g.
        {"copies_percent": (0.55, "below"), "r99_A": (3.26, "below")}.
    view : {"run", "dashboard"}
    runs : list of ShowReconstruction
        The dashboard's run models, in order; referenced, never copied.

    Examples
    --------
    >>> run = ShowReconstruction("scans 9 + 10", pixel_size_A=0.178, slice_A=14.7)
    >>> run.stage("learned", start=0)
    >>> run.add({"epoch": 4, "train": 0.21, "held_out": 0.1512}, projection=phase.sum(0), slices=phase)
    >>> ShowReconstruction(view="dashboard", runs=[run])
    """

    _esm = BUNDLE.read_text()  # the code as text, not a watched file: anywidget's live reload would put the whole bundle back into every run model when a builder rebuilds

    view = traitlets.Unicode("run").tag(sync=True)  # "run": one run's model; "dashboard": the runs referenced
    runs = traitlets.List(traitlets.Instance(ipywidgets.Widget)).tag(sync=True, **ipywidgets.widget_serialization)
    parent = traitlets.Instance(ipywidgets.Widget, allow_none=True).tag(sync=True, **ipywidgets.widget_serialization)  # the run model this run continued from: its end is this run's before
    ends = traitlets.Bytes(b"").tag(sync=True)  # the run's before (when it has no parent model) and after states: slices and cuts, float32 back to back
    ends_items = traitlets.List(traitlets.Dict()).tag(sync=True)  # one an array of ends: {which: "before" or "after", kind: "slice", "cut" or "probe", shape, slice, slices, depth_A, length_A, shares}
    shown = traitlets.Unicode("projection").tag(sync=True)  # the run view's object: "projection" or "slice" (slice_index into the kept slices)
    title = traitlets.Unicode("").tag(sync=True)
    pixel_size_A = traitlets.Float(1.0).tag(sync=True)
    slice_A = traitlets.Float(0.0).tag(sync=True)
    bars = traitlets.Dict().tag(sync=True)
    status = traitlets.Dict().tag(sync=True)  # the header: seconds an epoch, time left, GPU memory
    records = traitlets.List(traitlets.Dict()).tag(sync=True)  # one an epoch: epoch (total) and every number the panels plot; written on flush
    earlier = traitlets.List(traitlets.Dict()).tag(sync=True)  # the runs before this one, epoch, train and held_out only: the faint curve
    stages = traitlets.List(traitlets.Dict()).tag(sync=True)  # {name, start, changes, changed}: the run's band (one in a run model)
    panel = traitlets.Unicode("").tag(sync=True)  # the run view's third panel: "path", "shares", "depth", "tilt" or "" (the objective)
    views = traitlets.Int(1).tag(sync=True)  # scans fitted together: 1 a single scan, 2 a joint 0/90 pair, ...
    views_label = traitlets.Unicode("").tag(sync=True)  # how the joint scans relate, e.g. "0/90"; empty: "0/90" for two
    # Each stack: the frames back to back (float32), one shape and one epoch a frame.
    projections = traitlets.Bytes(b"").tag(sync=True)  # the projected phase (rows, cols) at each epoch kept
    projection_shapes = traitlets.List(traitlets.List(traitlets.Int())).tag(sync=True)
    snapshot_epochs = traitlets.List(traitlets.Int()).tag(sync=True)
    volumes = traitlets.Bytes(b"").tag(sync=True)  # the volume (slices, rows, cols) at each kept epoch
    volume_shapes = traitlets.List(traitlets.List(traitlets.Int())).tag(sync=True)
    volume_epochs = traitlets.List(traitlets.Int()).tag(sync=True)
    probes = traitlets.Bytes(b"").tag(sync=True)  # the probe's modes (modes, rows, cols), intensity at focus
    probe_shapes = traitlets.List(traitlets.List(traitlets.Int())).tag(sync=True)
    probe_epochs = traitlets.List(traitlets.Int()).tag(sync=True)
    probe_shares = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)  # each kept probe's mode power shares
    strongest = traitlets.Bytes(b"").tag(sync=True)  # the strongest mode (rows, cols) at every draw: the probe's story
    strongest_shapes = traitlets.List(traitlets.List(traitlets.Int())).tag(sync=True)
    strongest_epochs = traitlets.List(traitlets.Int()).tag(sync=True)
    apertures = traitlets.Bytes(b"").tag(sync=True)  # the probe's Fourier-space amplitude (rows, cols) at each kept draw: the aperture
    aperture_shapes = traitlets.List(traitlets.List(traitlets.Int())).tag(sync=True)
    aperture_epochs = traitlets.List(traitlets.Int()).tag(sync=True)
    size = traitlets.Unicode("medium").tag(sync=True)  # "small", "medium" or "large": how big the view draws; the browser's S M L writes it back
    tiles = traitlets.Unicode("").tag(sync=True)  # what a stage tile holds beside the projection: a comma list of "fft", "slice", "probe" ("image": none; "": the view's default, FFT and probe in the report, the probe in the dashboard); the browser's chips write it back
    settings = traitlets.Dict().tag(sync=True)  # the settings this run started with, flattened to "part.setting": text, as the status view lists the ones no stage changed
    phase_range = traitlets.List(traitlets.Float(), default_value=[0.0, 1.0]).tag(sync=True)  # one linear range, every slice
    profile_line = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)  # row0, col0, row1, col1 in pixels; the browser writes a dragged line back
    profile_line_given = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)  # the line Python gave: what "reset line" restores
    peaks = traitlets.List(traitlets.List(traitlets.Float()), default_value=[]).tag(sync=True)  # [row_px, col_px, heavy 1 or 0] per atomic column
    depth_rows = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)  # lattice power per slice at each draw
    depth_epochs = traitlets.List(traitlets.Int()).tag(sync=True)
    path = traitlets.Bytes(b"").tag(sync=True)  # (lines, knots) float32: the vibration path on its axis, Å
    path_shape = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    path_step_ms = traitlets.Float(0.0).tag(sync=True)
    line_offsets_A = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)  # [row, col] Å per scan line
    tilt_map = traitlets.Bytes(b"").tag(sync=True)  # (grid, grid, 2) float32 mrad: the local tilt across the field
    tilt_map_shape = traitlets.List(traitlets.Int(), default_value=[0, 0, 0]).tag(sync=True)
    slider_epoch = traitlets.Int(-1).tag(sync=True)  # -1 follows the newest epoch
    slice_index = traitlets.Int(-1).tag(sync=True)  # -1 the middle slice
    smooth = traitlets.Bool(False).tag(sync=True)  # bilinear display of the images, as Show2D's Smooth; off shows the pixels as they are
    band_per_A = traitlets.List(traitlets.Float()).tag(sync=True)  # (low, high) 1/Å: the band the lattice power reads (depth_rows), said on its axis
    reference_tilt_mrad = traitlets.List(traitlets.Float()).tag(sync=True)  # the direct fit's tilt (row, col), the tilt's dashed reference; empty when unknown

    def __init__(self, title="", *, pixel_size_A=1.0, slice_A=None, bars=None, views=1, views_label="", view="run", runs=(), size="medium", **kwargs):
        if size not in SIZES:
            raise ValueError(f"size={size!r}: one of {', '.join(SIZES)}")
        super().__init__(title=title, pixel_size_A=float(pixel_size_A), slice_A=float(slice_A or 0.0), bars=dict(bars or {}),
                         views=int(views), views_label=str(views_label), view=str(view), runs=list(runs), size=size, **kwargs)
        self._records: list[dict] = []
        self._kept = {name: [] for name in STACKS}  # (epoch, array, extra) per checkpoint, oldest first
        self._draws: set[int] = set()  # the epochs a probe was drawn at
        self._thinned = set()
        self._depth: list[tuple[int, list[float]]] = []
        self._ends: list[tuple[dict, np.ndarray]] = []
        self._changed = False  # something added since the last flush
        self._esm = STUB  # a model carries the 0.6 MB bundle only once a cell shows it (_repr_mimebundle_): a notebook stores it once a view, not once a run
        self.keep = dict(KEEP)  # what the saved state holds: projections "every" or "draws", probes "every" or "last", volumes "last" or "none"
        self.on_msg(self._on_message)

    def _on_message(self, widget, content, buffers):
        """A view that opens asks for the run's history: it may have missed the checkpoints sent before it existed.
        Nothing new since the last flush: nothing sent."""
        if isinstance(content, dict) and content.get("kind") == "hello" and self._changed:
            self.flush()

    def _keep(self, name, epoch, array, extra=None):
        """Add one checkpoint to a stack and thin it under its byte budget: past the budget every other entry across
        the whole stack is dropped (the newest always kept), so early and late epochs stay evenly spaced; one line
        says so the first time. Volumes keep the newest only. Returns the kept epochs."""
        kept = self._kept[name]
        if name == "volumes":
            kept.clear()
        kept[:] = [entry for entry in kept if entry[0] != epoch]
        kept.append((epoch, array, extra))
        while sum(entry[1].nbytes for entry in kept) > BUDGETS[name] and len(kept) > 2:
            kept[:] = kept[-1::-2][::-1]  # every other one counted back from the newest
            if name not in self._thinned:
                print(f"ShowReconstruction: the {name} passed {BUDGETS[name] / 1e6:.0f} MB, every other one is dropped (the numbers keep every epoch)")
                self._thinned.add(name)
        return [entry[0] for entry in kept]

    def _saved(self, name):
        """The entries of a stack the saved state holds, under keep."""
        kept = self._kept[name]
        rule = self.keep.get(name, "every")
        if name == "projections" and rule == "draws":
            return [entry for index, entry in enumerate(kept) if index == 0 or entry[0] in self._draws or entry is kept[-1]]
        if name == "strongest":  # the strongest mode at a draw whose probe is saved whole is that probe's first mode: the browser reads it there, not stored twice
            probes = {entry[0] for entry in self._saved("probes")}
            return [entry for entry in kept if entry[0] not in probes]
        if name == "apertures":
            rule = self.keep.get("probes", "every")
        if name in ("probes", "apertures") and rule == "last" or name == "volumes" and rule == "last":
            return kept[-1:]
        if rule == "none":
            return []
        return kept

    def flush(self):
        """Write the run's history into the synced traits, as keep says: what a saved notebook, an exported page
        and a newly opened view read. During a run each add sends only its own checkpoint, so this runs rarely: when
        a run ends, when what is kept changes, on export_html, and when a view opens after something new."""
        with self.hold_sync():
            self.records = list(self._records)
            for name, epochs_trait, shapes_trait in (("projections", "snapshot_epochs", "projection_shapes"), ("volumes", "volume_epochs", "volume_shapes"),
                                                     ("probes", "probe_epochs", "probe_shapes"), ("strongest", "strongest_epochs", "strongest_shapes"),
                                                     ("apertures", "aperture_epochs", "aperture_shapes")):
                saved = self._saved(name)
                setattr(self, name, b"".join(entry[1].tobytes() for entry in saved))
                setattr(self, shapes_trait, [list(entry[1].shape) for entry in saved])
                setattr(self, epochs_trait, [entry[0] for entry in saved])
            self.probe_shares = [entry[2] for entry in self._saved("probes")]
            self.depth_rows = [row for _, row in self._depth]
            self._write_ends()
            self.depth_epochs = [epoch for epoch, _ in self._depth]
        self._changed = False

    def saved_bytes(self):
        """The bytes of arrays the saved state holds (before the notebook's base64)."""
        return sum(entry[1].nbytes for name in STACKS for entry in self._saved(name)) + sum(array.nbytes for _, array in self._saved_ends())

    def set_ends(self, which, items):
        """The run's before or after state as the run view compares them: items, (meta, array) pairs, meta a dict with
        kind ("slice" with slice, slices and depth_A; "cut" with length_A; "probe" with shares). A run with a parent
        model reads its before from the parent's after, so only a first run sets its own before."""
        kept = [(meta, array) for meta, array in self._ends if meta["which"] != which]
        arrays = [_probe_frame(array) if meta["kind"] == "probe" else _float32(array) for meta, array in items]
        kept += [({**meta, "which": which, "shape": list(array.shape)}, array) for (meta, _), array in zip(items, arrays)]
        self._ends = kept
        self._write_ends()

    def _saved_ends(self):
        """The ends the saved state holds, under keep["ends"]: "all", or "middle" (an earlier run: the slice nearest the
        slab's middle of each state, its cut, projection and probe)."""
        if self.keep.get("ends", "all") != "middle":
            return self._ends
        out = []
        for which in ("before", "after"):
            state = [(meta, array) for meta, array in self._ends if meta["which"] == which]
            slices = [(meta, array) for meta, array in state if meta["kind"] == "slice"]
            middle = min(slices, key=lambda entry: abs(entry[0]["slice"] + 0.5 - entry[0]["slices"] / 2), default=None)
            out += [(meta, array) for meta, array in state if meta["kind"] != "slice" or (middle is not None and meta is middle[0])]
        return out

    def _write_ends(self):
        saved = self._saved_ends()
        self.ends_items = [meta for meta, _ in saved]
        self.ends = b"".join(array.tobytes() for _, array in saved)

    def last(self, name):
        """The newest checkpoint of a stack, (epoch, array, extra), or None."""
        kept = self._kept[name]
        return kept[-1] if kept else None

    def stage(self, name, changes="", *, start=None):
        """The run's band from epoch start (the epoch after the last record when None): name as the run is called,
        changes the settings set since the last run, as text ("probe.modes 4 → 10") or as {key: (before, after)},
        which the hover shows old against new."""
        if start is None:
            start = int(self._records[-1]["epoch"]) if self._records else 0
        entry = {"name": str(name), "start": int(start), "changes": str(changes), "changed": []}
        if isinstance(changes, dict):
            entry["changed"] = [{"key": str(key), "before": str(before), "after": str(after)} for key, (before, after) in changes.items()]
            entry["changes"] = ", ".join(f"{item['key']} {item['before']} → {item['after']}" for item in entry["changed"])
        self.stages = [*self.stages, entry]

    def add(self, record, *, projection=None, slices=None, phase_range=None, profile_line=None, depth=None, probe=None,
            probe_shares=None, aperture=None, path=None, path_step_ms=None, line_offsets_A=None, tilt_map=None, status=None, peaks=None):
        """One epoch: record, the numbers at this epoch ({"epoch": total epoch, "train", "held_out", "copies_percent",
        "r99_A", "far_power_percent", "tilt_mrad": [[row, col] per scan], ...}), and the arrays the panels show, each
        optional. peaks: the atomic columns found on the field, rows of (row_px, col_px, heavy), drawn as rings that
        the line's ends snap to. probe: the modes (modes, rows, cols), complex (drawn hue phase, brightness amplitude) or an
        intensity; aperture: its Fourier-space amplitude. Every projection, volume (slices) and probe is a checkpoint the
        epoch stepper can return to; a probe's power shares also go into the record ("probe_shares"). Records may
        carry "held_out_per_view" (a list, one held-out a scan) for a joint fit. A line the user dragged stays until
        "reset line"; a new profile_line replaces it only while the user has not moved it.

        The record and the new checkpoints go to the browser as one message carrying only this epoch's arrays, so an
        add costs the same at epoch 200 as at epoch 1; the stacks reach the synced traits on flush."""
        record = {name: (value.tolist() if isinstance(value, np.ndarray) else value) for name, value in dict(record).items()}
        epoch = int(record["epoch"])
        shares = [float(value) for value in (probe_shares if probe_shares is not None else [])]
        if probe is not None:
            record["probe_shares"] = shares
        self._records = [row for row in self._records if int(row["epoch"]) != epoch] + [record]
        message, buffers = {"kind": "checkpoint", "record": record}, []
        for name, array, extra in (("projections", projection, None), ("volumes", slices, None), ("probes", probe, shares), ("apertures", aperture, None)):
            if array is None:
                continue
            frame = _probe_frame(array) if name == "probes" else _float32(array)
            message[name] = {"shape": list(frame.shape), "epochs": self._keep(name, epoch, frame, extra), "extra": extra}
            buffers.append(memoryview(frame).cast("B"))
            if name == "probes":
                self._draws.add(epoch)
                message["strongest"] = {"shape": list(frame.shape[1:]), "epochs": self._keep("strongest", epoch, np.ascontiguousarray(frame[0]))}
        with self.hold_sync():
            if phase_range is not None:
                self.phase_range = [float(phase_range[0]), float(phase_range[1])]
            if profile_line is not None:
                given = [float(value) for value in profile_line]
                if not self.profile_line or self.profile_line == self.profile_line_given:
                    self.profile_line = given
                self.profile_line_given = given
            if peaks is not None:
                self.peaks = [[float(row), float(col), float(bool(heavy))] for row, col, heavy in np.asarray(to_numpy(peaks), dtype=float).reshape(-1, 3)]
            if depth is not None:
                self._depth = [entry for entry in self._depth if entry[0] != epoch] + [(epoch, [float(value) for value in _float32(depth).ravel()])]
                message["depth"] = self._depth[-1][1]
            if path is not None:
                lines = _float32(path)
                self.path, self.path_shape = lines.tobytes(), list(lines.shape)
                if path_step_ms is not None:
                    self.path_step_ms = float(path_step_ms)
            if line_offsets_A is not None:
                self.line_offsets_A = [[float(value) for value in row] for row in np.asarray(to_numpy(line_offsets_A), dtype=float).reshape(-1, 2)]
            if tilt_map is not None:
                grid = _float32(tilt_map)
                self.tilt_map, self.tilt_map_shape = grid.tobytes(), list(grid.shape)
            if status is not None:
                self.status = dict(status)
        self._changed = True
        self.send(message, buffers)

    def _repr_mimebundle_(self, **kwargs):
        """Shown in a cell: the model takes its bundle first."""
        self.shown_code()
        return super()._repr_mimebundle_(**kwargs)

    def shown_code(self):
        if self._esm == STUB:
            self._esm = BUNDLE.read_text()

    def export_html(self, path):
        """The view as one standalone page that opens offline (a dashboard with every run it references)."""
        from quantem.widget.export import write_widget_html

        self.shown_code()
        for run in [self, *self.runs]:
            if isinstance(run, ShowReconstruction) and run._changed:
                run.flush()
        return write_widget_html(path, self, self.title or "ShowReconstruction")

    def __repr__(self):
        if self.view == "dashboard":
            names = ", ".join(stage["name"] for run in self.runs for stage in run.stages)
            return f"ShowReconstruction({self.title!r}, dashboard of {len(self.runs)} runs: {names})"
        records = [row for row in self._records if row.get("held_out") is not None]
        name = self.stages[0]["name"] if self.stages else "no run"
        start, now = (records[0]["held_out"], records[-1]["held_out"]) if records else (None, None)
        last = self._records[-1]["epoch"] if self._records else 0
        return f"ShowReconstruction({name!r}, epoch {last}, held-out {_plain(start)} → {_plain(now)})"


def _cut_along(volume, line, pixel_A):
    """The volume along a line through every slice, (slices, points), two points a pixel (bilinear, display only, as
    the browser's profile reads it), and the line's length in Å."""
    slices, rows, cols = volume.shape
    r0, c0, r1, c1 = (float(value) for value in line)
    length = float(np.hypot(r1 - r0, c1 - c0))
    t = np.linspace(0, 1, max(2, round(length * 2)) + 1)
    row, col = r0 + t * (r1 - r0), c0 + t * (c1 - c0)
    top, left = np.clip(np.floor(row), 0, rows - 2).astype(int), np.clip(np.floor(col), 0, cols - 2).astype(int)
    down, across = row - top, col - left
    cut = (volume[:, top, left] * (1 - down) * (1 - across) + volume[:, top, left + 1] * (1 - down) * across
           + volume[:, top + 1, left] * down * (1 - across) + volume[:, top + 1, left + 1] * down * across)
    return np.ascontiguousarray(cut, dtype=np.float32), length * pixel_A


def _kept_slices(count):
    """The slices a saved state keeps: a quarter, half and three quarters through the slab."""
    return sorted({count // 4, count // 2, (3 * count) // 4})


class CompareReconstruction(anywidget.AnyWidget):
    """Two to four reconstructions side by side, one column each: each run's slice at a shared depth (from the slices
    a saved state keeps, a quarter, half and three quarters through its slab: its own slicing, no resampling), its
    spectrum, its depth cut along the line, its probe (complex modes, their shares and the aperture); below, the
    lattice power per slice against depth for every run, and B - A on the projected phase when there are exactly two.

    Contrast: phase per slice grows with slice thickness, so each run is shown from its own floor (1st percentile)
    over one shared span per Å of slice thickness (the largest run's 1st to 99.8th percentile span per Å).

    Parameters
    ----------
    runs : list of dict
        One a run, in column order (the first is the reference): name, slice_A, pixel_A, and either "source", a run
        model (ShowReconstruction) whose saved after state (set_ends) and last projection and probe the column
        draws, referenced so a notebook stores them once, or volume (slices, rows, cols) phase in rad with optionally
        probe (modes, rows, cols, complex), shares, aperture and power (lattice power per slice). Optionally record
        (held_out, copies_percent, r99_A, ...), changed ({key, before, after} against the first run) and epochs.
    title : str
        What is compared, in plain words.
    peaks : array-like or None
        The atomic columns, rows of (row_px, col_px, heavy), drawn as rings.
    profile_line : sequence or None
        row0, col0, row1, col1 in pixels, the line the depth cut follows; default a horizontal line through the middle.

    Examples
    --------
    >>> CompareReconstruction([{"name": "learned", "volume": a, "slice_A": 14.7, "pixel_A": 0.178},
    ...                        {"name": "10 modes", "source": run_model, "slice_A": 9.6, "pixel_A": 0.178}])
    """

    _esm = BUNDLE.read_text()  # the code as text, not a watched file: anywidget's live reload would put the whole bundle back into every run model when a builder rebuilds

    kind = traitlets.Unicode("compare").tag(sync=True)  # which view the shared bundle draws
    title = traitlets.Unicode("").tag(sync=True)
    runs = traitlets.List(traitlets.Dict()).tag(sync=True)  # per run: name, source, items (its own arrays), slice_A, pixel_A, low, high, record, changed, epochs, power
    sources = traitlets.List(traitlets.Instance(ipywidgets.Widget)).tag(sync=True, **ipywidgets.widget_serialization)  # run models whose arrays a run uses
    arrays = traitlets.Bytes(b"").tag(sync=True)  # the runs without a model: their kept slices, cut, projection, probe and aperture, float32 back to back in column order
    peaks = traitlets.List(traitlets.List(traitlets.Float()), default_value=[]).tag(sync=True)
    profile_line = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    profile_line_given = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    smooth = traitlets.Bool(False).tag(sync=True)
    size = traitlets.Unicode("medium").tag(sync=True)  # "small", "medium" or "large"; the browser's S M L writes it back
    band_per_A = traitlets.List(traitlets.Float()).tag(sync=True)  # (low, high) 1/Å: the band the runs' lattice power reads, said on the plot

    def __init__(self, runs, *, title="", peaks=None, profile_line=None, size="medium", band_per_A=None, **kwargs):
        sources, arrays, meta, kept_of = [], [], [], []
        for run in runs:
            source = run.get("source")
            if source is not None:  # the run model's saved after state: three slices and the cut
                if source._changed:
                    source.flush()
                if source not in sources:
                    sources.append(source)
                kept_of.append([array for item, array in source._ends if item["which"] == "after" and item["kind"] == "slice"])
                continue
            volume = _float32(run["volume"])
            rows, cols = volume.shape[1:]
            line = [float(value) for value in profile_line] if profile_line is not None else [rows / 2, 0.0, rows / 2, cols - 1.0]
            count = len(volume)
            slice_A = float(run["slice_A"]) if run.get("slice_A") else None
            items = [({"kind": "slice", "slice": index, "slices": count, "depth_A": None if slice_A is None else [index * slice_A, (index + 1) * slice_A]}, volume[index]) for index in _kept_slices(count)]
            cut, length_A = _cut_along(volume, line, float(run["pixel_A"]))
            items += [({"kind": "cut", "slices": count, "length_A": length_A}, cut), ({"kind": "projection"}, volume.sum(0))]
            if run.get("probe") is not None:
                items.append(({"kind": "probe", "shares": [float(value) for value in run.get("shares") or []]}, _probe_frame(run["probe"])))
            if run.get("aperture") is not None:
                items.append(({"kind": "aperture"}, _float32(run["aperture"])))
            arrays.append(items)
            kept_of.append([array for item, array in items if item["kind"] == "slice"])
        floors = [float(np.percentile(np.stack(kept), 1)) for kept in kept_of]
        slice_of = [float(run["slice_A"]) if run.get("slice_A") else 1.0 for run in runs]  # a run of unknown thickness: per slice (slice_A 0 tells the browser)
        span_per_A = max((float(np.percentile(np.stack(kept), 99.8)) - floor) / slice for kept, floor, slice in zip(kept_of, floors, slice_of))
        own = iter(arrays)
        for run, floor, slice_A in zip(runs, floors, slice_of):
            items = [] if run.get("source") is not None else next(own)
            meta.append({
                "name": str(run["name"]), "source": sources.index(run["source"]) if run.get("source") is not None else -1,
                "items": [{**item, "shape": list(np.shape(array))} for item, array in items],
                "slice_A": float(run["slice_A"]) if run.get("slice_A") else 0.0, "pixel_A": float(run["pixel_A"]),
                "low": floor, "high": floor + span_per_A * slice_A,
                "power": [float(value) for value in run["power"]] if run.get("power") is not None else None,
                "record": {name: (value.tolist() if isinstance(value, np.ndarray) else value) for name, value in dict(run.get("record") or {}).items()},
                "changed": [{"key": str(item["key"]), "before": str(item["before"]), "after": str(item["after"])} for item in (run.get("changed") or [])],
                "epochs": run.get("epochs"),
            })
        shape = kept_of[0][0].shape
        line = [float(value) for value in profile_line] if profile_line is not None else [shape[0] / 2, 0.0, shape[0] / 2, shape[1] - 1.0]
        found = [[float(row), float(col), float(bool(heavy))] for row, col, heavy in np.asarray(to_numpy(peaks), dtype=float).reshape(-1, 3)] if peaks is not None else []
        super().__init__(title=str(title), runs=meta, sources=sources, arrays=b"".join(np.ascontiguousarray(array, dtype=np.float32).tobytes() for items in arrays for _, array in items),
                         peaks=found, profile_line=line, profile_line_given=line, size=size, band_per_A=[float(value) for value in (band_per_A or [])], **kwargs)

    def export_html(self, path):
        """The comparison as one standalone page that opens offline."""
        from quantem.widget.export import write_widget_html

        return write_widget_html(path, self, self.title or "CompareReconstruction")

    def __repr__(self):
        return f"CompareReconstruction({self.title!r}, {', '.join(run['name'] for run in self.runs)})"


def square_region(region, scan_shape):
    """A scan region as the draggable box keeps it: whole scan positions, square (the shorter side), an even side of at
    least 2, inside the scan; (row start, row stop, col start, col stop)."""
    row_start, row_stop, col_start, col_stop = (int(round(float(value))) for value in region)
    rows, cols = int(scan_shape[0]), int(scan_shape[1])
    side = max(2, min(row_stop - row_start, col_stop - col_start, rows, cols) // 2 * 2)
    row_start = min(max(0, row_start), rows - side)
    col_start = min(max(0, col_start), cols - side)
    return (row_start, row_start + side, col_start, col_start + side)


class ShowDirect(anywidget.AnyWidget):
    """Direct ptychography of a scan at a glance, in the reconstruction views' language: the whole scan's virtual bright
    field (annular dark field on a toggle) with the region to reconstruct drawn on it, the region's direct phase with the
    scored positions boxed, its spectrum (the dashed ring at 50 pm), and the region's tilt and defocus beside the whole
    scan's. It holds only what it draws: the virtual images at scan resolution and the region's phase window.

    Parameters
    ----------
    bright_field, dark_field : array (scan rows, scan cols)
        The whole scan's virtual bright- and annular-dark-field images.
    phase : array (rows, cols) or None
        Direct ptychography's phase of phase_region, at upsample pixels a scan step; None draws the whole scan only
        (choosing the region: editable).
    scan_sampling_A : float
        The scan step, Å.
    region : (row start, row stop, col start, col stop)
        The scan positions scored, drawn on every image.
    phase_region : (row start, row stop, col start, col stop) or None
        The scan positions the phase covers; None the whole scan.
    readout : dict
        {"region": {"tilt_mrad": [row, col], "defocus_nm": value}, "whole scan": {...}}: shown as a plain line.
    origin : (row, col)
        The scan position the virtual images start at, when they are a crop (a fitted region's view crops them to the
        positions its phase covers); region and phase_region stay in whole-scan positions.
    editable : bool
        The region's box can be dragged (moved, or resized from its corner), kept square on whole scan positions
        (square_region); the browser writes it to region, which a caller observes.
    """

    _esm = BUNDLE.read_text()

    kind = traitlets.Unicode("direct").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    scan_sampling_A = traitlets.Float(1.0).tag(sync=True)
    upsample = traitlets.Int(1).tag(sync=True)
    images = traitlets.Bytes(b"").tag(sync=True)  # bright field, dark field, phase, float32 back to back
    image_shapes = traitlets.List(traitlets.List(traitlets.Int())).tag(sync=True)  # their shapes, in that order
    region = traitlets.List(traitlets.Int()).tag(sync=True)
    phase_region = traitlets.List(traitlets.Int()).tag(sync=True)
    readout = traitlets.Dict().tag(sync=True)
    dark = traitlets.Bool(False).tag(sync=True)  # the annular dark field in place of the bright field
    editable = traitlets.Bool(False).tag(sync=True)  # the region's box can be dragged; the browser writes region back
    origin = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)  # the scan position the virtual images start at: (0, 0) the whole scan, else a crop about the region
    size = traitlets.Unicode("medium").tag(sync=True)  # "small", "medium" or "large"; the browser's S M L writes it back

    def __init__(self, bright_field, dark_field, phase=None, *, scan_sampling_A, upsample=1, region=None, phase_region=None, readout=None, title="", size="medium",
                 editable=False, origin=(0, 0), **kwargs):
        arrays = [_float32(bright_field), _float32(dark_field)] + ([] if phase is None else [_float32(phase)])
        rows, cols = arrays[0].shape
        if region is None:  # the middle eighth of the scan, a start to drag from
            side = max(2, min(rows, cols) // 8 // 2 * 2)
            region = (rows // 2 - side // 2, rows // 2 - side // 2 + side, cols // 2 - side // 2, cols // 2 - side // 2 + side)
        if editable:
            region = square_region(region, (rows, cols))
        super().__init__(title=str(title), scan_sampling_A=float(scan_sampling_A), upsample=int(upsample), images=b"".join(array.tobytes() for array in arrays),
                         image_shapes=[list(array.shape) for array in arrays], region=[int(value) for value in region],
                         phase_region=[int(value) for value in (phase_region or (0, rows, 0, cols))], readout=dict(readout or {}), size=str(size),
                         editable=bool(editable), origin=[int(value) for value in origin], **kwargs)

    def export_html(self, path):
        """The view as one standalone page that opens offline."""
        from quantem.widget.export import write_widget_html

        return write_widget_html(path, self, self.title or "ShowDirect")

    def __repr__(self):
        return f"ShowDirect({self.title!r}, region {tuple(self.region)})"
