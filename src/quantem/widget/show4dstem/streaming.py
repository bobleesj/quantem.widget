"""Internal polling lifecycle for a viewer borrowing live acquisition data."""

import re
import threading
import time
import traceback
from contextlib import nullcontext
from typing import TYPE_CHECKING

from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.adapters.gpu import AcquisitionSeries
from quantem.widget.show4dstem.session import SessionSeries

if TYPE_CHECKING:
    from quantem.widget.show4dstem.widget import Show4DSTEM


# Seconds after the user last touched the viewer before the probe follows the newest pattern again.
STREAM_FOLLOW_PAUSE_S = 1.5
# Seconds before a failed live update is tried again (the receiver still runs; the next scan must show).
STREAM_RETRY_S = 1.0


def short_scan_name(name) -> str:
    """A scan's name as the viewer prints it: its number ("41" for "series_41", "41_rot90" for "series_41_rot90");
    any other name unchanged. The same rule as the page's shortScanName (js/show4dstem/viewLabels.ts)."""
    return re.sub(r"^series_(\d+)", r"\1", str(name))


def live_frame_number(name) -> str | None:
    """The number of a Live frame (``live_<n>``: one at a time, each replacing the last, never kept), else None."""
    match = re.fullmatch(r"live_(\d+)", str(name))
    return match.group(1) if match else None


def _shape_of(source) -> tuple:
    """A source's (scan rows, scan cols, detector rows, detector cols): from ``shape``, else its counts."""
    shape = getattr(source, "shape", None)
    if shape is None:
        shape = getattr(getattr(source, "counts", None), "shape", ())
    return tuple(int(side) for side in tuple(shape)[-4:])


def stopped_title(error) -> str:
    """The live view's title when the stream stops: one short line, never a traceback (a 2048 x 2048 pair that ran
    the GPU out of memory put the whole CUDA traceback in the title).

    Examples
    --------
    >>> stopped_title(RuntimeError("replay failed: Traceback (most recent call last): ... OutOfMemoryError: ..."))
    'Live acquisition stopped: the GPU ran out of memory'
    """
    text = str(error).replace("\\x0a", "\n")
    if "OutOfMemoryError" in text or "out of memory" in text.lower():
        return "Live acquisition stopped: the GPU ran out of memory"
    first = next((line.strip() for line in text.splitlines() if line.strip()), type(error).__name__)
    first = first.split("Traceback (most recent call last)")[0].strip(" :")
    return f"Live acquisition stopped: {first[:160]}"


def _size_label(shape, count: int) -> str:
    """A size group's name: scan rows x columns x detector (one number when square), and how many scans."""
    rows, cols, det_rows, det_cols = shape
    detector = f"{det_rows}" if det_rows == det_cols else f"{det_rows}×{det_cols}"
    return f"{rows}×{cols}×{detector} · {count} scan{'s' if count != 1 else ''}"


class StreamFollower:
    """Follow one recording without taking ownership of its acquisitions."""

    def __init__(self, viewer: "Show4DSTEM", recording, sources: list, *,
                 labels: list | None, refresh_s: float, options: dict):
        self.viewer = viewer
        self.recording = recording
        self.partial = hasattr(recording, "acquisitions")
        self.lock = getattr(recording, "lock", nullcontext())
        self.shown_sources = sources[:1]
        self.previous = None
        self.initialized = bool(self.shown_sources)
        self.last_refresh = 0.0
        self.labels = labels
        self.refresh_s = refresh_s
        self.options = options
        self.title = options.get("title") or "Live acquisition"
        # a live folder shows each scan as it lands; a live recording keeps the scan the user selected
        self.select_newest = getattr(recording, "select_newest", False)
        # scans of several sizes: one size in view, the others one click away (the size tabs)
        self.size_shapes, self.shown_shape, self.newest_source = [], None, None

    def snapshot(self) -> list:
        return list(self.recording.acquisitions if self.partial else self.recording.datasets)

    def of_shown_size(self, sources: list) -> list:
        """The scans of the size in view, and the size tabs when the recording holds several sizes.

        One viewer draws one scan and detector shape. A scan that begins brings its size into view; a size tab
        brings back the scans of another size, which stay in GPU memory all along."""
        viewer = self.viewer
        shapes = [_shape_of(source) for source in sources]
        if sources and (id(sources[-1]) != self.newest_source or self.shown_shape not in shapes):
            self.shown_shape = shapes[-1]
        self.newest_source = id(sources[-1]) if sources else None     # its identity only: no scan is kept by this
        self.size_shapes = list(dict.fromkeys(reversed(shapes)))
        several = len(self.size_shapes) > 1
        labels = [_size_label(shape, shapes.count(shape)) for shape in self.size_shapes] if several else []
        index = self.size_shapes.index(self.shown_shape) if several else 0
        if list(viewer.size_groups) != labels:
            viewer.size_groups = labels
        if viewer.size_group != index:
            viewer.size_group = index
        return [source for source, shape in zip(sources, shapes) if shape == self.shown_shape]

    def choose_size(self, change) -> None:
        """A size tab was clicked: the scans of that size come into view at the next refresh."""
        index = int(change["new"])
        if 0 <= index < len(self.size_shapes) and self.size_shapes[index] != self.shown_shape:
            self.shown_shape = self.size_shapes[index]
            self.previous = None

    def start(self) -> None:
        """Publish independently of blocking cells; register only this viewer's hooks."""
        from IPython import get_ipython

        viewer = self.viewer
        viewer.observe(self.choose_size, names=["size_group"])
        viewer._stream_update = self.update
        viewer._stream_recording = self.recording
        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            def before_cell(*_args, **_kwargs):
                viewer._stream_cell_running = True

            def after_cell(*_args, **_kwargs):
                viewer._stream_cell_running = False

            viewer._stream_cell_running = shell.kernel.get_parent().get("header", {}).get("msg_type") == "execute_request"
            shell.events.register("pre_run_cell", before_cell)
            shell.events.register("post_run_cell", after_cell)
            viewer._stream_cell_hooks = (shell, before_cell, after_cell)
        viewer._stream_thread = threading.Thread(target=self.follow, name="Show4DSTEM-stream", daemon=True)
        viewer._stream_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Join before releasing any sessions, and detach borrowed recording references."""
        viewer = self.viewer
        viewer._stream_stop.set()
        thread = viewer._stream_thread
        if thread is not None and thread.is_alive():
            if thread is not threading.current_thread():
                thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("Show4DSTEM is still refreshing. Wait for the query to finish, then call close() again.")
        hooks = viewer._stream_cell_hooks
        if hooks is not None:
            shell, before, after = hooks
            shell.events.unregister("pre_run_cell", before)
            shell.events.unregister("post_run_cell", after)
            viewer._stream_cell_hooks = None
        viewer.unobserve(self.choose_size, names=["size_group"])
        with viewer._stream_lock:
            viewer._stream_update = None
            viewer._stream_recording = None
            self.shown_sources = []
            self.recording = None
            _release_error_frames(viewer.stream_error)

    def update_state(self, force: bool = False) -> bool:
        """Adopt retained sources and refresh only when counts or topology change."""
        viewer = self.viewer
        if viewer._stream_stop.is_set():
            return False
        try:
            if self.recording.error is not None and not force:
                raise self.recording.error
            # Sample completion first: if it changes during this update,
            # the next update must still publish the final counts.
            complete = not self.recording.receiving
            now = time.monotonic()
            # Full scan images share Jupyter's ordered output channel with
            # the small DP replies. Keep room for direct manipulation;
            # detector gestures still recompute immediately, and the final
            # acquisition refresh is never deferred.
            if (not force and not complete and not viewer._stream_deferred_state
                    and now - viewer._last_interaction_at < 0.15
                    and now - self.last_refresh < 0.1):
                return True
            with self.lock, viewer._defer_stream_observers():
                sources = self.snapshot()
                if not sources:
                    self.of_shown_size(sources)         # no scans: no size tabs
                if sources:
                    # One viewer holds one scan and detector shape: a scan of another size replaces the others and
                    # the scans of its size are shown (2026-11-16: a 256 x 256 scan after 512 x 512 ones stopped the
                    # live page: 'Live sessions must share scan/detector shape')
                    kept = self.of_shown_size(sources)
                    if self.shown_sources and _shape_of(self.shown_sources[-1]) != self.shown_shape:
                        viewer._replace_stream_sources(self.shown_sources, [])
                        self.shown_sources, self.previous, self.initialized = [], None, False
                    sources = kept
                shown = len(self.shown_sources)
                if not sources:
                    if self.shown_sources:
                        viewer._replace_stream_sources(self.shown_sources, [])
                        self.last_refresh = time.monotonic()
                    self.shown_sources, self.previous = [], None
                    if not self.partial:
                        # the next completed scan may have another shape: it sets the geometry again
                        self.initialized = False
                    return not complete
                if not self.initialized and self.partial and not sources[0].received:
                    return not complete
                if viewer._data is None:
                    with viewer.hold_sync(), viewer.hold_trait_notifications():
                        viewer._replace_stream_sources([], sources, labels=self.labels)
                        if not self.initialized:
                            viewer._initialize_stream_geometry(sources[0], self.options)
                            self.initialized = True
                    self.previous = None
                elif sources[:shown] == self.shown_sources:
                    if len(sources) > shown:
                        viewer.append(sources[shown:], self.labels[shown:len(sources)] if self.labels else None)
                        self.previous = None
                elif sources:
                    viewer._replace_stream_sources(self.shown_sources, sources)
                    self.previous = None
                arrived = not self.shown_sources or sources[-1] is not self.shown_sources[-1]
                self.shown_sources = sources
                if (self.select_newest and arrived and viewer.stream_follow
                        and time.monotonic() - viewer._last_interaction_at >= STREAM_FOLLOW_PAUSE_S):
                    with viewer.hold_sync(), viewer.hold_trait_notifications():
                        viewer.frame_idx = len(sources) - 1
                        viewer._stream_selected_source = viewer._stream_source_ids[viewer.frame_idx]
                if arrived:
                    viewer._adopt_shown_calibration()      # a new scan may have another step (11-16)
                shown = len(sources)
                if self.partial and isinstance(viewer._data, SessionSeries):
                    # Live sessions remain open as leases until eviction. Their
                    # completed datasets can already use batched GPU queries;
                    # there is no need to wait for continuous acquisition to end.
                    completed = {dataset.metadata.get("acquisition_id"): dataset
                                 for dataset in self.recording.datasets
                                 if gpu_adapter.is_acquisition(dataset) and dataset.metadata.get("acquisition_id")}
                    with viewer._compare_lock:
                        viewer._stream_completed_sources = {
                            idx: completed[source.identifier] for idx, source in enumerate(sources)
                            if getattr(source, "identifier", None) in completed
                        }
            progress = tuple(source.received for source in sources[:shown]) if self.partial else ()
            if self.partial and complete and isinstance(viewer._data, SessionSeries) and len(self.recording.datasets) == shown and all(
                    gpu_adapter.is_acquisition(dataset) for dataset in self.recording.datasets):
                # Borrow the same ANS storage through the regular joint
                # query path once every acquisition is immutable.
                completed = AcquisitionSeries(self.recording.datasets)
                with viewer._compare_lock, viewer._session_lock:
                    viewer._close_sessions()
                    viewer._data = completed
                self.previous = None
            if force or progress != self.previous:
                viewer.refresh()
                if self.partial:
                    current = sources[shown - 1]
                    self.follow_probe(current)
                    scan = current.metadata.get("series", shown)
                    # One short line that never wraps beside a control panel; the counts are in the stats bar.
                    done = " ✓" if current.received == current.total else ""
                    counts = f"{current.received:,}/{current.total:,}{done}"
                    frame = live_frame_number(scan)
                    if frame is not None:
                        # a Live frame replaces the one before and is never kept: no count of kept scans
                        viewer.title = f"Live · frame {frame} · {counts}"
                    else:
                        kept = sum(live_frame_number(source.metadata.get("series", "")) is None
                                   for source in sources[:shown])
                        viewer.title = f"Live · {short_scan_name(scan)} · {counts} · {kept} kept"
                else:
                    viewer.title = self.options.get("title") or getattr(self.recording, "title", None) or self.title
                self.previous = progress
                self.last_refresh = time.monotonic()
            if viewer.stream_error is not None and self.recording.error is None:
                viewer.stream_error = None              # recovered: the update after a failed one went through
            return not complete
        except Exception as error:
            previous = viewer.stream_error
            if previous is not None and previous is not error:
                _release_error_frames(previous)     # a retried failure: the earlier one must not hold scans
            viewer.stream_error = error
            viewer.title = stopped_title(error)
            if force:
                raise
            return False

    def follow_probe(self, current) -> None:
        """Point the diffraction pattern at the newest arrived pattern of the scan being received, unless the
        user interacted within STREAM_FOLLOW_PAUSE_S: their probe stays until they let go."""
        viewer = self.viewer
        if not viewer.stream_follow or not current.received or current.received >= current.total:
            return
        if time.monotonic() - viewer._last_interaction_at < STREAM_FOLLOW_PAUSE_S:
            return
        row, col = divmod(int(current.received) - 1, int(viewer.shape_cols))
        if (row, col) != (viewer.pos_row, viewer.pos_col) and row < viewer.shape_rows:
            with viewer.hold_trait_notifications():
                viewer.pos_row, viewer.pos_col = row, col

    def update(self, force: bool = False) -> bool:
        """Coalesce display traffic without deferring acquisition retirement."""
        viewer = self.viewer
        with viewer._stream_lock:
            if viewer._stream_stop.is_set():
                return False
            if force:
                self.previous = None
            # Jupyter carries images and interactive replies on the same
            # ordered channel. Bound unacknowledged acquisition images to
            # the credit window. A busy user cell cannot dispatch ACKs.
            now = time.monotonic()
            absent = viewer._stream_display_negotiated and viewer._stream_display_view is None
            outstanding = viewer._stream_display_seq - viewer._stream_display_acknowledged
            waiting = outstanding >= viewer._stream_display_credits and not (
                viewer._stream_cell_running and now - viewer._stream_display_sent_at >= 2.0)
            # Also bound the initial cell before its frontend-ready message
            # can dispatch (for example, display(viewer); recording.wait()).
            busy = viewer._stream_cell_running and self.last_refresh and now - self.last_refresh < 2.0
            if (not force and viewer._stream_display_pending and now - max(
                    viewer._stream_display_sent_at, viewer._stream_display_probe_at) >= 2.0):
                # A final update may occupy only part of the window. A
                # surviving view must still recover a hard-closed owner.
                viewer._stream_display_probe_at = now
                viewer._stream_display_probe_seq = viewer._stream_display_pending
                viewer.send({"type": "show4dstem_stream_probe", "version": 1,
                             "seq": viewer._stream_display_probe_seq})
            if not force and (absent or waiting or busy) and self.recording.error is None:
                # Retention is independent of display delivery. A closed
                # browser must not keep retired acquisitions in GPU memory.
                try:
                    with self.lock:
                        sources = self.of_shown_size(self.snapshot())
                        if any(source not in sources for source in self.shown_sources):
                            if viewer._stream_deferred_snapshot is None:
                                viewer._stream_deferred_snapshot = viewer.get_state(key=viewer.keys)
                            with viewer.hold_sync(), viewer._defer_stream_observers():
                                viewer._replace_stream_sources(self.shown_sources, sources, labels=self.labels)
                                viewer._stream_deferred_state.update(viewer._states_to_send)
                                viewer._states_to_send.clear()
                            self.shown_sources, self.previous = sources, None
                except Exception as error:
                    viewer.stream_error = error
                    viewer.title = stopped_title(error)
                    return False
                return True
            before = self.last_refresh
            deferred = bool(viewer._stream_deferred_state)
            old_images = viewer.compare_virtual_image_bytes
            old_revision = viewer.compare_image_revision
            published = viewer._stream_deferred_snapshot
            old_ids = (published or {}).get("_stream_source_ids", viewer._stream_source_ids)
            old_indices = (published or {}).get("compare_panel_indices", viewer.compare_panel_indices)
            old_panel_ids = [old_ids[index] for index in old_indices] if old_ids else []
            old_topology = (tuple(map(id, self.shown_sources)), tuple(viewer.compare_panel_indices),
                            viewer.shape_rows, viewer.shape_cols)
            delta = None
            with viewer.hold_sync():
                receiving = self.update_state(force)
                viewer._states_to_send.update(viewer._stream_deferred_state)
                viewer._stream_deferred_state.clear()
                viewer._stream_deferred_snapshot = None
                topology = (tuple(map(id, self.shown_sources)), tuple(viewer.compare_panel_indices),
                            viewer.shape_rows, viewer.shape_cols)
                if (not force and not deferred and viewer._stream_display_delta
                        and old_topology == topology):
                    delta = viewer._stream_compare_delta(old_images, old_revision)
                elif (not force and viewer._stream_display_views
                      and viewer._stream_display_views.keys() <= viewer._stream_layout_views
                      and old_topology[2:] == topology[2:]):
                    viewer._stream_compare_layout(old_images, old_revision, old_panel_ids)
            if delta is not None:
                message, buffers = delta
                viewer.send(message, buffers=buffers)
            if deferred or self.last_refresh != before:
                viewer._stream_display_barrier()
            return viewer.stream_error is None and (receiving or bool(viewer._stream_display_pending))

    def retrying(self) -> bool:
        """After a failed update: try again while the receiver still runs. A method, not a local in ``follow``: an
        error's frames lead back to that loop's frame, and a local there would keep the recording after close."""
        recording = self.recording
        return (recording is not None and recording.error is None and bool(recording.receiving)
                and self.viewer.stream_error is not None)

    def follow(self) -> None:
        """Keep background failures visible; yield between expensive refreshes.

        A failed update (a scan released from GPU memory while it was drawn, a display hiccup) shows in the title
        and is tried again after ``STREAM_RETRY_S``: the follower ends only when the receiver has stopped (its
        ``error`` is set or it is no longer receiving) or the viewer closes. 11-16 at the microscope: one failure
        ended it, and the live page's left panel never showed another scan."""
        viewer = self.viewer
        while not viewer._stream_stop.is_set():
            started = time.monotonic()
            try:
                alive = self.update()
            except Exception as error:
                viewer.stream_error = error
                viewer.title = stopped_title(error)
                alive = False
            if not alive:
                if not self.retrying():
                    break                               # the receiver stopped, or the acquisition ended
                viewer.title = f"Live: retrying after {type(viewer.stream_error).__name__}: {viewer.stream_error}"[:160]
                viewer._stream_stop.wait(STREAM_RETRY_S)
                continue
            viewer._stream_stop.wait(max(0.005, started + self.refresh_s - time.monotonic()))


def _release_error_frames(error: BaseException | None) -> None:
    """Keep failure identities and stack locations without retaining scan locals."""
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        # Called only after this viewer's worker and explicit refresh have ended.
        # clear_frames also leaves any still-executing receiver frame untouched.
        traceback.clear_frames(error.__traceback__)
        pending.extend((error.__cause__, error.__context__))
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
