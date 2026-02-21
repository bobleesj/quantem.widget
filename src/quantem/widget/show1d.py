"""
show1d: Interactive 1D data viewer for spectra, profiles, and time series.

The 1D counterpart to Show2D. Displays single or multiple 1D traces with
interactive zoom/pan, cursor readout, and calibrated axes. Common uses:
line profiles, ROI time series, optimization loss curves, EELS/EDX spectra,
convergence plots, and any 1D signal.
"""

import json
import pathlib
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload

_DEFAULT_COLORS = [
    "#4fc3f7",  # light blue
    "#81c784",  # green
    "#ffb74d",  # orange
    "#ce93d8",  # purple
    "#ef5350",  # red
    "#ffd54f",  # yellow
    "#90a4ae",  # blue-gray
    "#a1887f",  # brown
]


class Show1D(anywidget.AnyWidget):
    """
    Interactive 1D data viewer for spectra, profiles, and time series.

    Display one or more 1D traces with interactive zoom, pan, cursor readout,
    and calibrated axes. Supports log scale, grid lines, legend, and
    publication-quality export.

    Parameters
    ----------
    data : array_like
        1D array for a single trace, 2D array (n_traces, n_points) for
        multiple traces, or a list of 1D arrays.
    x : array_like, optional
        Shared X-axis values. Must have the same length as data points.
        If not provided, uses integer indices [0, 1, 2, ...].
    labels : list of str, optional
        Labels for each trace. Used in legend and stats bar.
    colors : list of str, optional
        Hex color strings for each trace. If not provided, uses a default
        8-color palette.
    title : str, optional
        Title displayed above the plot.
    x_label : str, optional
        Label for the X axis (e.g. "Energy", "Distance", "Epoch").
    y_label : str, optional
        Label for the Y axis (e.g. "Counts", "Intensity", "Loss").
    x_unit : str, optional
        Unit for the X axis (e.g. "eV", "nm", "").
    y_unit : str, optional
        Unit for the Y axis.
    log_scale : bool, default False
        Use logarithmic Y axis.
    show_stats : bool, default True
        Show statistics bar (mean, min, max, std per trace).
    show_legend : bool, default True
        Show legend when multiple traces are displayed.
    show_grid : bool, default True
        Show grid lines on the plot.
    show_controls : bool, default True
        Show control row below the plot.
    line_width : float, default 1.5
        Line width for traces.
    state : dict or str or Path, optional
        Restore display settings from a dict or a JSON file path.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show1D
    >>>
    >>> # Single trace
    >>> Show1D(np.sin(np.linspace(0, 10, 200)))
    >>>
    >>> # Multiple traces with labels
    >>> x = np.linspace(0, 10, 200)
    >>> Show1D([np.sin(x), np.cos(x)], x=x, labels=["sin", "cos"])
    >>>
    >>> # Optimization loss curve
    >>> Show1D(losses, x_label="Epoch", y_label="Loss", log_scale=True)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show1d.js"
    _css = pathlib.Path(__file__).parent / "static" / "show1d.css"

    # =========================================================================
    # Core State
    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    y_bytes = traitlets.Bytes(b"").tag(sync=True)
    x_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_traces = traitlets.Int(1).tag(sync=True)
    n_points = traitlets.Int(0).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    colors = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    title = traitlets.Unicode("").tag(sync=True)
    x_label = traitlets.Unicode("").tag(sync=True)
    y_label = traitlets.Unicode("").tag(sync=True)
    x_unit = traitlets.Unicode("").tag(sync=True)
    y_unit = traitlets.Unicode("").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_legend = traitlets.Bool(True).tag(sync=True)
    show_grid = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    line_width = traitlets.Float(1.5).tag(sync=True)
    focused_trace = traitlets.Int(-1).tag(sync=True)

    # =========================================================================
    # Statistics (per-trace)
    # =========================================================================
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)

    def __init__(
        self,
        data,
        x=None,
        labels=None,
        colors=None,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        x_unit: str = "",
        y_unit: str = "",
        log_scale: bool = False,
        show_stats: bool = True,
        show_legend: bool = True,
        show_grid: bool = True,
        show_controls: bool = True,
        line_width: float = 1.5,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Dataset duck typing
        _extracted_title = ""
        if hasattr(data, "array") and hasattr(data, "name"):
            if data.name:
                _extracted_title = data.name
            data = data.array

        # Normalize data to 2D (n_traces, n_points)
        if isinstance(data, list):
            arrays = [to_numpy(d).astype(np.float32).ravel() for d in data]
            n_pts = len(arrays[0])
            for i, a in enumerate(arrays):
                if len(a) != n_pts:
                    raise ValueError(
                        f"All traces must have the same length. "
                        f"Trace 0 has {n_pts} points, trace {i} has {len(a)}."
                    )
            data_2d = np.stack(arrays)
        else:
            arr = to_numpy(data).astype(np.float32)
            if arr.ndim == 1:
                data_2d = arr[np.newaxis, :]
            elif arr.ndim == 2:
                data_2d = arr
            else:
                raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D.")

        self._data = data_2d
        self.n_traces = int(data_2d.shape[0])
        self.n_points = int(data_2d.shape[1])

        # X axis
        self._x = None
        if x is not None:
            x_arr = to_numpy(x).astype(np.float32).ravel()
            if len(x_arr) != self.n_points:
                raise ValueError(
                    f"x has {len(x_arr)} points but data has {self.n_points} points."
                )
            self._x = x_arr
            self.x_bytes = x_arr.tobytes()

        # Labels
        if labels is not None:
            self.labels = [str(l) for l in labels]
        else:
            if self.n_traces == 1:
                self.labels = ["Trace"]
            else:
                self.labels = [f"Trace {i + 1}" for i in range(self.n_traces)]

        # Colors
        if colors is not None:
            self.colors = [str(c) for c in colors]
        else:
            self.colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(self.n_traces)]

        # Display options
        self.title = title or _extracted_title
        self.x_label = x_label
        self.y_label = y_label
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.log_scale = log_scale
        self.show_stats = show_stats
        self.show_legend = show_legend
        self.show_grid = show_grid
        self.show_controls = show_controls
        self.line_width = line_width

        # Compute stats and send data
        self._compute_stats()
        self.y_bytes = self._data.tobytes()

        # Restore state
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def set_data(self, data, x=None, labels=None) -> Self:
        """Replace displayed data. Preserves display settings.

        Parameters
        ----------
        data : array_like
            1D, 2D (n_traces, n_points), or list of 1D arrays.
        x : array_like, optional
            New X-axis values.
        labels : list of str, optional
            New trace labels. If not provided, generates defaults.
        """
        if hasattr(data, "array") and hasattr(data, "name"):
            data = data.array

        if isinstance(data, list):
            arrays = [to_numpy(d).astype(np.float32).ravel() for d in data]
            n_pts = len(arrays[0])
            for i, a in enumerate(arrays):
                if len(a) != n_pts:
                    raise ValueError(
                        f"All traces must have the same length. "
                        f"Trace 0 has {n_pts} points, trace {i} has {len(a)}."
                    )
            data_2d = np.stack(arrays)
        else:
            arr = to_numpy(data).astype(np.float32)
            if arr.ndim == 1:
                data_2d = arr[np.newaxis, :]
            elif arr.ndim == 2:
                data_2d = arr
            else:
                raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D.")

        self._data = data_2d
        self.n_traces = int(data_2d.shape[0])
        self.n_points = int(data_2d.shape[1])

        if x is not None:
            x_arr = to_numpy(x).astype(np.float32).ravel()
            if len(x_arr) != self.n_points:
                raise ValueError(
                    f"x has {len(x_arr)} points but data has {self.n_points} points."
                )
            self._x = x_arr
            self.x_bytes = x_arr.tobytes()
        else:
            self._x = None
            self.x_bytes = b""

        if labels is not None:
            self.labels = [str(l) for l in labels]
        else:
            if self.n_traces == 1:
                self.labels = ["Trace"]
            else:
                self.labels = [f"Trace {i + 1}" for i in range(self.n_traces)]

        # Assign default colors if count changed
        self.colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(self.n_traces)]

        self._compute_stats()
        self.y_bytes = self._data.tobytes()
        return self

    def add_trace(self, y, label=None, color=None) -> Self:
        """Append a trace.

        Parameters
        ----------
        y : array_like
            1D array with the same number of points as existing traces.
        label : str, optional
            Trace label.
        color : str, optional
            Hex color string.
        """
        arr = to_numpy(y).astype(np.float32).ravel()
        if self.n_points > 0 and len(arr) != self.n_points:
            raise ValueError(
                f"New trace has {len(arr)} points but existing traces have {self.n_points}."
            )

        if self._data.size == 0:
            self._data = arr[np.newaxis, :]
            self.n_points = int(len(arr))
        else:
            self._data = np.vstack([self._data, arr[np.newaxis, :]])

        self.n_traces = int(self._data.shape[0])

        lbl = label if label is not None else f"Trace {self.n_traces}"
        self.labels = list(self.labels) + [lbl]

        clr = color if color is not None else _DEFAULT_COLORS[(self.n_traces - 1) % len(_DEFAULT_COLORS)]
        self.colors = list(self.colors) + [clr]

        self._compute_stats()
        self.y_bytes = self._data.tobytes()
        return self

    def remove_trace(self, index: int) -> Self:
        """Remove a trace by index.

        Parameters
        ----------
        index : int
            Zero-based trace index.
        """
        if index < 0 or index >= self.n_traces:
            raise IndexError(f"Trace index {index} out of range [0, {self.n_traces}).")
        self._data = np.delete(self._data, index, axis=0)
        self.n_traces = int(self._data.shape[0])
        lbls = list(self.labels)
        lbls.pop(index)
        self.labels = lbls
        clrs = list(self.colors)
        clrs.pop(index)
        self.colors = clrs
        self._compute_stats()
        self.y_bytes = self._data.tobytes()
        return self

    def clear(self) -> Self:
        """Remove all traces."""
        self._data = np.empty((0, 0), dtype=np.float32)
        self.n_traces = 0
        self.n_points = 0
        self.labels = []
        self.colors = []
        self.stats_mean = []
        self.stats_min = []
        self.stats_max = []
        self.stats_std = []
        self.y_bytes = b""
        return self

    def _compute_stats(self):
        means, mins, maxs, stds = [], [], [], []
        for i in range(self.n_traces):
            trace = self._data[i]
            means.append(float(np.mean(trace)))
            mins.append(float(np.min(trace)))
            maxs.append(float(np.max(trace)))
            stds.append(float(np.std(trace)))
        self.stats_mean = means
        self.stats_min = mins
        self.stats_max = maxs
        self.stats_std = stds

    # =========================================================================
    # State Persistence
    # =========================================================================
    def state_dict(self):
        return {
            "title": self.title,
            "labels": self.labels,
            "colors": self.colors,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "log_scale": self.log_scale,
            "show_stats": self.show_stats,
            "show_legend": self.show_legend,
            "show_grid": self.show_grid,
            "show_controls": self.show_controls,
            "line_width": self.line_width,
            "focused_trace": self.focused_trace,
        }

    def save(self, path: str):
        save_state_file(path, "Show1D", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        lines = [self.title or "Show1D", "═" * 32]
        lines.append(f"Traces:   {self.n_traces} × {self.n_points} points")
        if self.labels:
            lines.append(f"Labels:   {', '.join(self.labels)}")
        if self._x is not None:
            lines.append(f"X range:  {float(self._x[0]):.4g} – {float(self._x[-1]):.4g}")
        if self.x_label or self.x_unit:
            x_desc = self.x_label
            if self.x_unit:
                x_desc += f" ({self.x_unit})" if x_desc else self.x_unit
            lines.append(f"X axis:   {x_desc}")
        if self.y_label or self.y_unit:
            y_desc = self.y_label
            if self.y_unit:
                y_desc += f" ({self.y_unit})" if y_desc else self.y_unit
            lines.append(f"Y axis:   {y_desc}")
        for i in range(self.n_traces):
            if i < len(self.stats_mean):
                lines.append(
                    f"  [{i}] {self.labels[i] if i < len(self.labels) else ''}: "
                    f"mean={self.stats_mean[i]:.4g}  min={self.stats_min[i]:.4g}  "
                    f"max={self.stats_max[i]:.4g}  std={self.stats_std[i]:.4g}"
                )
        scale = "log" if self.log_scale else "linear"
        display = f"{scale}"
        if self.show_grid:
            display += " | grid"
        lines.append(f"Display:  {display}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        if self.n_traces == 1:
            return f"Show1D({self.n_points} points)"
        return f"Show1D({self.n_traces} traces × {self.n_points} points)"
