"""Public Show4DSTEM factory.

``quantem.widget.Show4DSTEM`` is one user-facing API. Acquisitions that
``quantem.gpu.io.load`` returns stay in their encoded GPU storage and open as
bounded live views; arrays, tensors and Dataset5dstem series open in the base
viewer. Keeping that routing here leaves the package initializer a plain
export list.
"""

import pathlib
import warnings

import numpy as np

from quantem.gpu import io as gpu_io
from quantem.gpu.io import Dataset4dstemGPU
from quantem.widget.show4dstem import Show4DSTEM as _Show4DSTEMBase
from quantem.widget.show4dstem_bounded import acquisition_name, show_bounded, show_series


def Show4DSTEM(data, **kwargs):
    """Open a 4D-STEM viewer over ``io.load`` output or an array, on any backend.

    Examples::

        from quantem.gpu.io import load
        from quantem.widget import Show4DSTEM

        Show4DSTEM(load("a_master.h5"))                    # one acquisition
        Show4DSTEM(load(["a_master.h5", "b_master.h5"]))   # dataset comparison

    An acquisition from ``io.load`` (CUDA or MPS) is never expanded: virtual
    images are summed on its encoded storage and diffraction patterns come from
    bounded reads. Pass ``scan_region=(row_start, row_stop, col_start, col_stop)``
    to view part of one scan. A list of acquisitions opens as a comparison
    labelled by source file. Arrays, tensors and Dataset5dstem series open in the
    base viewer, and the browser WebGPU export decodes inside the page.
    """
    if isinstance(data, Dataset4dstemGPU):
        return show_bounded([data], **kwargs)
    if isinstance(data, (list, tuple)) and any(isinstance(item, Dataset4dstemGPU) for item in data):
        return show_bounded(list(data), **kwargs)
    return _Show4DSTEMBase(data, **kwargs)


def _master_file_contract(master) -> dict:
    """Read the raw shape and dtype needed to validate a watched master."""
    report = gpu_io.inspect(str(master))
    required = {
        "scan_shape": report.scan_shape,
        "detector_shape": report.detector_shape,
        "n_frames": report.actual_frames,
        "dtype": report.dtype,
    }
    missing = [name for name, value in required.items() if value is None]
    if not report.ready or missing:
        problems: list[str] = []
        if not report.ready:
            problems.append(report.reason or "the source is not ready")
        if missing:
            problems.append("inspection did not provide " + ", ".join(missing))
        action = report.action or "Verify that the master and external data are complete."
        raise ValueError(
            f"Cannot open 4D-STEM master {str(master)!r}: {'; '.join(problems)}. {action}"
        )
    return {
        "scan_shape": report.scan_shape,
        "detector_shape": report.detector_shape,
        "n_frames": report.actual_frames,
        "dtype": np.dtype(report.dtype).str,
    }


def _largest_compatible_master_group(masters: list, *, verbose: bool) -> list:
    """Keep the largest group of masters sharing scan and detector geometry.

    A comparison needs one geometry; a folder that also holds test scans of
    another size would otherwise fail to open at all.
    """
    if len(masters) <= 1:
        return masters
    groups: dict[tuple, list] = {}
    for master in masters:
        try:
            report = gpu_io.inspect(str(master))
            key = (report.scan_shape, report.detector_shape, report.actual_frames)
        except (OSError, ValueError, KeyError):
            key = (None, None, None)
        groups.setdefault(key, []).append(master)
    if len(groups) <= 1:
        return masters
    key, compatible = max(groups.items(), key=lambda item: len(item[1]))
    if verbose:
        skipped = len(masters) - len(compatible)
        warnings.warn(
            "Show4DSTEM.from_folder found mixed 4D-STEM shapes in the folder; "
            f"using the largest compatible group ({len(compatible)}/{len(masters)}) "
            f"with scan_shape={key[0]}, detector_shape={key[1]}. "
            f"Skipped {skipped} master file{'s' if skipped != 1 else ''}. "
            "Use scan_size= or a narrower pattern= to select a different group.",
            RuntimeWarning,
            stacklevel=3,
        )
    return compatible


def from_folder(
    folder,
    *,
    pattern: str = "*_master.h5",
    recursive: bool = True,
    scan_size: int | None = None,
    max_masters: int | None = None,
    min_masters: int | None = None,
    ready_only: bool = True,
    backend: str = "auto",
    device: int | None = None,
    view_mode: str = "multiple",
    columns: int | None = None,
    page_size: int | None = None,
    watch: bool = True,
    watch_interval: float = 2.0,
    verbose: bool = False,
    **viewer_kwargs,
):
    """Open every ready ``*_master.h5`` in a folder as one live comparison viewer.

    Each acquisition is loaded with :func:`quantem.gpu.io.load` into encoded
    storage on one CUDA device or the Apple GPU, at full detector resolution;
    a 512 x 512 x 192 x 192 uint16 Arina scan (18 GiB dense) occupies about
    0.1 GiB, so the whole folder stays resident without binning or paging.
    Masters that are not completely written are skipped (``ready_only``), and
    when the folder mixes geometries the largest group sharing one scan and
    detector shape is shown. The viewer opens once the first master is loaded;
    the others join in the background (``wait_for_folder()`` blocks until they
    have), and ``free()`` or ``close()`` releases them.

    ``columns`` sets the grid width and ``page_size`` the number of datasets
    per page. With ``watch=True`` (default) a master that completes while the
    viewer is open is appended once its headers are unchanged on two polls,
    every ``watch_interval`` seconds. Set ``watch=False`` for a fixed snapshot.

    Examples
    --------
    >>> viewer = Show4DSTEM.from_folder("/data/session", scan_size=512)  # doctest: +SKIP
    """
    folder_path = pathlib.Path(folder).expanduser().resolve()
    scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
    masters = list(gpu_io.discover(
        str(folder_path), pattern=pattern, recursive=recursive,
        scan_shape=scan_shape, verbose=False,
    ))
    if ready_only:
        ready = [master for master in masters if gpu_io.inspect(master).ready]
        if verbose and len(ready) < len(masters):
            names = ", ".join(
                acquisition_name(master) for master in masters if master not in ready
            )
            warnings.warn(
                f"Show4DSTEM.from_folder skipped incomplete master files: {names}.",
                RuntimeWarning,
                stacklevel=2,
            )
        masters = ready
    if not masters:
        state = "ready " if ready_only else ""
        raise ValueError(
            f"No {state}{pattern!r} files found in {folder_path}. "
            "Wait for linked data files to finish writing, or pass "
            "ready_only=False if you know the masters are complete."
        )
    masters = _largest_compatible_master_group(masters, verbose=verbose)
    if min_masters is not None and len(masters) < min_masters:
        raise ValueError(
            f"Show4DSTEM.from_folder requires at least {min_masters} "
            f"compatible master(s), but found {len(masters)}."
        )
    if max_masters is not None:
        masters = masters[:max_masters]
    expected = _master_file_contract(masters[0])

    def validate_master(master) -> None:
        """Reject a watched master whose geometry cannot join the comparison."""
        contract = _master_file_contract(master)
        # dtype is not compared: io.load narrows uint32 counts that fit into
        # uint16, and the views read float32 either way.
        mismatches = [
            name for name in ("scan_shape", "detector_shape", "n_frames")
            if contract[name] != expected[name]
        ]
        if mismatches:
            observed = ", ".join(f"{name}={contract[name]!r}" for name in mismatches)
            wanted = ", ".join(f"{name}={expected[name]!r}" for name in mismatches)
            raise ValueError(
                f"Incompatible 4D-STEM master {acquisition_name(master)!r}: "
                f"{observed}; expected {wanted}. Use scan_size= or a narrower "
                "pattern= for a uniform folder."
            )

    def load_master(master) -> Dataset4dstemGPU:
        return gpu_io.load(master, backend=backend, device=device, verbose=False)

    viewer = show_series(
        [load_master(masters[0])],
        view_mode=view_mode,
        compare_cols=3 if columns is None else int(columns),
        compare_max_panels=12 if page_size is None else int(page_size),
        verbose=verbose,
        **viewer_kwargs,
    )
    viewer._attach_folder_source(
        folder=folder_path,
        pattern=pattern,
        recursive=recursive,
        scan_shape=scan_shape,
        ready_only=ready_only,
        known_masters=masters,
        load_master=load_master,
        validate_master=validate_master,
    )
    viewer._fill_folder(masters[1:])
    if watch:
        viewer.watch_folder(interval=watch_interval)
    return viewer


Show4DSTEM.from_folder = from_folder


__all__ = ["Show4DSTEM", "from_folder"]
