"""ShowSearch: a parameter search as one live view, the candidates as thumbnails that carry their held-out.

A search trains many short candidates (AutoPtycho: slab thickness × focus, then slice counts, then a refocus) and
keeps the one whose held-out is lowest. A log line an epoch per candidate says none of that; this view does: the
geometry candidates as a grid on thickness × focus (★ the pick, ▲ within its noise, a dashed row for a thickness the
search added), every later candidate and finishing stage as a strip under it, each candidate's held-out against epoch
as a curve (convergence), and the hovered or chosen candidate's projection, spectrum, strongest probe mode and depth
cut. Hover looks, a click chooses (``selected``, and the line of code that reopens it), leaving goes back.

The widget only displays what it is given: the search calls ``candidate`` when it plans one, ``epoch`` per epoch,
``land`` with its arrays when it completes, ``stage`` per finishing stage. Arrays stay native (no binning); each
reaches the browser once as a message, and the synced traits (what a saved notebook keeps) are written on ``flush``.
"""

from __future__ import annotations

import pathlib

import anywidget
import numpy as np
import traitlets

from quantem.widget.utils.array import to_numpy

KINDS = ("projection", "probe", "section")  # the arrays a candidate may carry: phase (rows, cols), probe intensity (rows, cols), depth cut (slices, samples)


def _float32(array) -> np.ndarray:
    return np.ascontiguousarray(to_numpy(array), dtype=np.float32)


def _plain(value, digits=5) -> str:
    """At most five decimals, never scientific notation."""
    return "-" if value is None else f"{value:.{digits}f}"


TOP = 5  # the lowest held-out candidates the view draws as cards: a saved notebook keeps their probe and depth cut
TILE_PX = 150  # the largest a candidate's grid tile is drawn (at large): what a saved notebook keeps of a candidate not shown large


def tile_size(image) -> np.ndarray:
    """An image binned by block means to at most TILE_PX a side, the factor the same on both axes (true proportions):
    a display copy of a candidate's projection for the saved notebook."""
    factor = max(1, -(-max(image.shape[-2:]) // TILE_PX))
    if factor == 1:
        return image
    rows, cols = (image.shape[-2] // factor) * factor, (image.shape[-1] // factor) * factor
    binned = image[..., :rows, :cols].reshape(*image.shape[:-2], rows // factor, factor, cols // factor, factor).mean(axis=(-3, -1))
    return np.ascontiguousarray(binned, dtype=np.float32)


class ShowSearch(anywidget.AnyWidget):
    """A search's candidates, their convergence and the chosen one's pictures, live and saved.

    Parameters
    ----------
    title : str
        What is searched, in plain words ("AutoPtycho · 64 × 64 region").
    pixel_size_A : float
        Object pixel of the projections, for the scale bar and the spectrum.
    code : str
        The line that reopens a candidate, with ``{name}`` where its name goes, shown when one is clicked.
    rules : list of str
        How the search decides, one rule a line: behind the header's i, not printed under the cell.

    Examples
    --------
    >>> view = ShowSearch("AutoPtycho · 64 × 64 region", pixel_size_A=0.247)
    >>> view.candidate("pass1-000", stage="geometry", thickness_nm=20, focus_nm=-5, slices=12, modes=6, epochs=4)
    >>> view.epoch("pass1-000", 1, 0.20312)
    >>> view.land("pass1-000", 0.20240, projection=phase, probe=intensity, section=cut)
    """

    _esm = (pathlib.Path(__file__).parent / "static" / "showsearch.js").read_text()  # text, not a watched file: a rebuild never reaches a running kernel's saved state

    title = traitlets.Unicode("").tag(sync=True)
    pixel_size_A = traitlets.Float(1.0).tag(sync=True)
    code = traitlets.Unicode("").tag(sync=True)
    rules = traitlets.List(traitlets.Unicode()).tag(sync=True)  # how the search decides, behind the header's i
    status = traitlets.Dict().tag(sync=True)  # the header: stage, done, total, elapsed_s, left_s, device, memory_GB, note
    candidates = traitlets.List(traitlets.Dict()).tag(sync=True)  # one a candidate, in plan order: name, stage, thickness_nm, focus_nm, slices, modes, epochs, state, held_out, parent, mark, added
    curves = traitlets.Dict().tag(sync=True)  # name: [[epoch, held_out], ...]
    stages = traitlets.List(traitlets.Dict()).tag(sync=True)  # the finishing stages: name, verdict, held_out, start, control, flag
    picks = traitlets.List(traitlets.Dict()).tag(sync=True)  # one a decision: stage, name, held_out, paired_SE, flags
    best = traitlets.Unicode("").tag(sync=True)  # the candidate or stage the search keeps now
    selected = traitlets.Unicode("").tag(sync=True)  # the one clicked; the browser writes it
    images = traitlets.Bytes(b"").tag(sync=True)  # every kept array, float32 back to back
    image_index = traitlets.List(traitlets.Dict()).tag(sync=True)  # name, kind, shape, in the order of images
    size = traitlets.Unicode("medium").tag(sync=True)  # "small", "medium" or "large": how big the view draws; the browser's S M L writes it back

    def __init__(self, title="", *, pixel_size_A=1.0, code="", rules=(), size="medium", **kwargs):
        super().__init__(title=str(title), pixel_size_A=float(pixel_size_A), code=str(code), rules=[str(rule) for rule in rules], size=str(size), **kwargs)
        self._arrays: dict[tuple[str, str], np.ndarray] = {}
        self._curves: dict[str, list] = {}
        self._changed = False
        self.on_msg(self._on_message)

    def _on_message(self, widget, content, buffers):
        """A view that opens asks for what it missed: the arrays sent before it existed."""
        if isinstance(content, dict) and content.get("kind") == "hello" and self._changed:
            self.flush()

    def flush(self):
        """Write the arrays and curves into the synced traits: what a saved notebook and a newly opened view read. The
        saved state keeps every picture whole only for what the view shows large: the best, each pass's pick, the
        finishing stages and the five lowest held-out (the view's top-five cards: projection, probe, depth cut); every
        other candidate keeps its projection at the size its grid tile is drawn at most (block means, TILE_PX) and no
        probe or depth cut, as the view's "i" says. The kernel keeps every array whole."""
        entries = self._saved()
        with self.hold_sync():
            self.images = b"".join(array.tobytes() for _, array in entries)
            self.image_index = [{"name": name, "kind": kind, "shape": list(array.shape)} for (name, kind), array in entries]
            self.curves = {name: list(points) for name, points in self._curves.items()}
        self._changed = False

    def _saved(self):
        """((name, kind), array) the saved state holds, under flush's rule."""
        landed = sorted((row for row in self.candidates if row.get("held_out") is not None), key=lambda row: row["held_out"])
        large = {self.best, *(pick["name"] for pick in self.picks), *(row["name"] for row in landed[:TOP])}
        out = []
        for (name, kind), array in self._arrays.items():
            if name in large or name.startswith("stage:"):
                out.append(((name, kind), array))
            elif kind == "projection":
                out.append(((name, kind), tile_size(array)))
        return out

    def _row(self, name):
        return next((index for index, row in enumerate(self.candidates) if row["name"] == name), None)

    def candidate(self, name, *, stage, thickness_nm, focus_nm, slices, modes, epochs, parent=None, state="queued", added=False):
        """Plan one candidate (a gray tile until it starts). A candidate that continues another (parent, same stage)
        shares its tile: its epochs extend the parent's curve."""
        row = {"name": str(name), "stage": str(stage), "thickness_nm": float(thickness_nm), "focus_nm": float(focus_nm),
               "slices": int(slices), "modes": int(modes), "epochs": int(epochs), "state": str(state), "held_out": None,
               "parent": None if parent is None else str(parent), "mark": "", "added": bool(added)}
        index = self._row(name)
        rows = list(self.candidates)
        if index is None:
            rows.append(row)
        else:
            rows[index] = {**rows[index], **{key: value for key, value in row.items() if key not in ("held_out", "mark")}}
        self.candidates = rows

    def update(self, name, **fields):
        """Change a candidate's fields (state "running", "done", "failed"; mark "best", "tie" or ""; held_out)."""
        index = self._row(name)
        if index is None:
            return
        rows = list(self.candidates)
        rows[index] = {**rows[index], **fields}
        self.candidates = rows

    def epoch(self, name, epoch, held_out):
        """One epoch's held-out of a running candidate: a point on its curve, sent alone (the curves sync on flush)."""
        point = [int(epoch), float(held_out)]
        points = self._curves.setdefault(str(name), [])
        points[:] = [each for each in points if each[0] != point[0]] + [point]
        index = self._row(name)
        if index is not None and self.candidates[index]["state"] != "running":
            self.update(name, state="running")
        self._changed = True
        self.send({"kind": "epoch", "name": str(name), "point": point})

    def curve(self, name, points):
        """A candidate's whole curve at once, [[epoch, held_out], ...]: a saved search read back."""
        self._curves[str(name)] = [[int(epoch), float(value)] for epoch, value in points]
        self._changed = True

    def land(self, name, held_out, *, projection=None, probe=None, section=None, state="done"):
        """A candidate completed: its held-out on the tile and its pictures (projected phase, strongest probe mode,
        complex or an intensity, depth cut), each sent once as one message."""
        self.update(name, state=state, held_out=None if held_out is None else float(held_out))
        self._send_arrays(name, projection=projection, probe=probe, section=section)

    def _send_arrays(self, name, **arrays):
        message, buffers = {"kind": "arrays", "entries": []}, []
        for kind, array in arrays.items():
            if array is None:
                continue
            values = to_numpy(array)
            if np.iscomplexobj(values):  # a complex probe mode as (2, rows, cols), the real and imaginary parts: drawn hue phase, brightness amplitude
                values = np.stack([values.real, values.imag])
            frame = _float32(values)
            self._arrays[(str(name), kind)] = frame
            message["entries"].append({"name": str(name), "kind": kind, "shape": list(frame.shape)})
            buffers.append(memoryview(frame).cast("B"))
        if buffers:
            self._changed = True
            self.send(message, buffers)

    def stage(self, name, *, verdict, held_out, start, control=None, flag="", thickness_nm=None, focus_nm=None, projection=None, probe=None, section=None):
        """A finishing stage on the strip: its verdict ("kept", "refused", "running"), held-out against its start and
        its plain control. A stage that sets a new thickness also draws as a dashed row of the grid."""
        entry = {"name": str(name), "verdict": str(verdict), "held_out": None if held_out is None else float(held_out),
                 "start": None if start is None else float(start), "control": None if control is None else float(control),
                 "flag": str(flag), "thickness_nm": None if thickness_nm is None else float(thickness_nm),
                 "focus_nm": None if focus_nm is None else float(focus_nm)}
        rows = [row for row in self.stages if row["name"] != entry["name"]]
        self.stages = [*rows, entry]
        self._send_arrays(f"stage:{name}", projection=projection, probe=probe, section=section)

    def pick(self, stage, name, *, held_out, paired_SE=None, flags=(), ties=()):
        """A decision: the candidate kept after a stage, ★ on its tile and ▲ on those within its noise (ties)."""
        entry = {"stage": str(stage), "name": str(name), "held_out": float(held_out),
                 "paired_SE": None if paired_SE is None else float(paired_SE), "flags": [str(flag) for flag in flags]}
        self.picks = [*[row for row in self.picks if row["stage"] != entry["stage"]], entry]
        rows = []
        for row in self.candidates:
            if row["stage"] == stage:
                row = {**row, "mark": "best" if row["name"] == name else "tie" if row["name"] in ties else ""}
            rows.append(row)
        self.candidates = rows
        self.best = str(name)

    def export_html(self, path):
        """The view as one standalone page that opens offline."""
        from quantem.widget.export import write_widget_html

        if self._changed:
            self.flush()
        return write_widget_html(path, self, self.title or "ShowSearch")

    def __repr__(self):
        done = [row for row in self.candidates if row["held_out"] is not None]
        best = min(done, key=lambda row: row["held_out"]) if done else None
        tail = f", best {best['name']} held-out {_plain(best['held_out'])}" if best else ""
        return f"ShowSearch({self.title!r}, {len(done)}/{len(self.candidates)} candidates{tail})"
