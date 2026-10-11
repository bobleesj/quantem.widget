"""Tutorial datasets: small real microscopy data downloaded once from Hugging Face.

Every loader takes a friendly ``size`` name and returns a calibrated quantem
dataset (or a folder of files), so the tutorials never ship data in git: a
quantem core ``Dataset2d``/``Dataset3d`` for images and stacks, and for the
4D-STEM scan a ``quantem.gpu.io.Dataset4dstemGPU`` on a GPU or a quantem core
``Dataset4dstem`` elsewhere (the widget's stand-in where quantem core cannot
install).
"""

import json
import os
from pathlib import Path

import numpy as np

from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.adapters.core import as_array, make_dataset
from quantem.widget.show4dstem.reader import read_4dstem

TUTORIAL_DATA_REPO_ID = "bobleesj/quantem-data"
TUTORIAL_DATA_ROOT = "widget-tutorials"
TUTORIAL_SIZES = ("small", "medium", "large", "full")
_SHOW2D_STRIDE_BY_SIZE = {
    "small": 8,
    "medium": 4,
    "large": 2,
    "full": 1,
}
_SHOW3D_PARAMS_BY_SIZE = {
    "small": {"n_frames": 32, "stride": 8, "crop_size": 256},
    "medium": {"n_frames": 48, "stride": 4, "crop_size": 384},
    "large": {"n_frames": 64, "stride": 2, "crop_size": 512},
    "full": {"n_frames": 64, "stride": 1, "crop_size": 512},
}
_SHOW4DSTEM_SCAN_STRIDE_BY_SIZE = {
    "small": 4,
    "medium": 2,
    "large": 1,
    "full": 1,
}


def _normalize_tutorial_size(size: str) -> str:
    """Lowercase tutorial size name; anything outside ``TUTORIAL_SIZES`` is a ValueError naming the valid ones."""
    value = str(size).strip().lower()
    if value not in TUTORIAL_SIZES:
        valid = ", ".join(TUTORIAL_SIZES)
        raise ValueError(f"size must be one of {valid}; got {size!r}")
    return value


def _download_widget_tutorial_folder(
    viewer: str,
    name: str,
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Download one ``widget-tutorials/<viewer>/<name>/<size>`` folder and return its local path.

    Only that folder is fetched (``allow_patterns``), so a small tutorial
    never pulls the full-size payloads stored next to it.
    """
    size = _normalize_tutorial_size(size)
    path_in_repo = f"{TUTORIAL_DATA_ROOT}/{viewer}/{name}/{size}"
    kwargs = {
        "repo_id": TUTORIAL_DATA_REPO_ID,
        "repo_type": "dataset",
        "allow_patterns": [f"{path_in_repo}/*"],
        "force_download": bool(force_download),
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision is not None:
        kwargs["revision"] = str(revision)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download  # noqa: PLC0415  (imported on first download: importing the loaders stays light)

    folder = Path(snapshot_download(**kwargs)) / path_in_repo
    if not folder.is_dir():
        raise FileNotFoundError(
            f"downloaded tutorial folder is missing: {folder}. "
            f"Expected Hugging Face path {TUTORIAL_DATA_REPO_ID}/{path_in_repo}"
        )
    return folder


def show1d_ducky(
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """Download the real ducky joint-time ptychography Show1D tutorial run.

    The returned folder contains ``show1d_monitor.jsonl`` and snapshot ``.npy``
    files. Open it with :meth:`quantem.widget.Show1D.from_monitor_file`.

    Parameters
    ----------
    cache_dir
        Optional Hugging Face cache directory.
    revision
        Optional Hugging Face dataset revision.
    force_download
        If ``True``, ask Hugging Face Hub to refresh the cached files.
    verbose
        If ``True``, print a short dataset summary.

    Returns
    -------
    Path
        Local folder containing the Show1D monitor run.
    """

    folder = _download_widget_tutorial_folder(
        "show1d",
        "ducky",
        size="small",
        cache_dir=cache_dir,
        revision=revision,
        force_download=force_download,
    )
    monitor = folder / "show1d_monitor.jsonl"
    if not monitor.is_file():
        raise FileNotFoundError(f"Show1D ducky tutorial monitor is missing: {monitor}")
    if verbose:
        events = sum(1 for line in monitor.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"Show1D ducky tutorial run: {folder}")
        print(f"Monitor events: {events}")
    return folder


def show2d_gold(
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
):
    """Load the gold HAADF Show2D tutorial image by friendly size name.

    The source image is downloaded once from
    ``widget-tutorials/shared/gold-haadf/full``; every size but ``"full"``
    keeps every Nth pixel of it (stride 8, 4, 2) and says so in one line.
    The same source is reused by :func:`show3d_gold`.
    """

    size = _normalize_tutorial_size(size)
    folder = _download_widget_tutorial_folder(
        "shared", "gold-haadf", size="full", cache_dir=cache_dir, revision=revision, force_download=force_download,
    )
    metadata = json.loads((folder / "meta.json").read_text())
    full_image = np.load(folder / "data.npy", mmap_mode="r")
    stride = _SHOW2D_STRIDE_BY_SIZE[size]
    image = np.asarray(full_image[::stride, ::stride], dtype=np.float32)
    sampling = tuple(float(value) * stride for value in metadata["sampling"])
    units = tuple(metadata["units"])
    if stride > 1:
        print(f"show2d_gold(size={size!r}) keeps every {stride}th pixel of the {full_image.shape[0]}x{full_image.shape[1]} "
              f"image ({image.shape[0]}x{image.shape[1]}); size='full' loads native pixels.")
    if verbose:
        print(f"Source: {metadata['name']} from Hugging Face")
        print(f"Full image: {full_image.shape[0]} x {full_image.shape[1]} {full_image.dtype}")
        print(f"Preview: {image.shape[0]} x {image.shape[1]}, pixel size {sampling[0]:.4f} {units[0]}")
    return make_dataset(image, sampling=sampling, units=units, name="Gold HAADF preview")


def show3d_gold(
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
):
    """Load the gold HAADF Show3D tutorial stack by friendly size name.

    The stack is built from moving crops of the shared gold HAADF tutorial
    source instead of storing a second copy of the image for Show3D. Every
    size but ``"full"`` crops from every Nth pixel and says so in one line.
    """

    size = _normalize_tutorial_size(size)
    folder = _download_widget_tutorial_folder(
        "shared", "gold-haadf", size="full", cache_dir=cache_dir, revision=revision, force_download=force_download,
    )
    n_frames, stride, crop_size = (_SHOW3D_PARAMS_BY_SIZE[size][key] for key in ("n_frames", "stride", "crop_size"))
    metadata = json.loads((folder / "meta.json").read_text())
    full_image = np.load(folder / "data.npy", mmap_mode="r")
    preview = np.asarray(full_image[::stride, ::stride], dtype=np.float32)
    # the crop window walks from the top right to the bottom left, one frame per step
    rows = np.linspace(0, preview.shape[0] - crop_size, n_frames)
    cols = np.linspace(preview.shape[1] - crop_size, 0, n_frames)
    stack = np.empty((n_frames, crop_size, crop_size), dtype=np.float32)
    for idx, (row, col) in enumerate(zip(rows, cols)):
        row_start, col_start = int(round(float(row))), int(round(float(col)))
        stack[idx] = preview[row_start : row_start + crop_size, col_start : col_start + crop_size]
    pixel_sampling = float(metadata["sampling"][0]) * stride
    units = tuple(metadata["units"])
    if stride > 1:
        print(f"show3d_gold(size={size!r}) crops from every {stride}th pixel of the {full_image.shape[0]}x{full_image.shape[1]} "
              "image; size='full' crops native pixels.")
    if verbose:
        print(f"Source: {metadata['name']} from Hugging Face")
        print(f"Full image: {full_image.shape[0]} x {full_image.shape[1]} {full_image.dtype}")
        print(f"Stack: {stack.shape}, stride {stride}, pixel size {pixel_sampling:.4f} {units[0]}")
    return make_dataset(
        stack,
        sampling=(1.0, pixel_sampling, pixel_sampling),
        units=("frame", units[0], units[1]),
        name="Gold HAADF moving-crop stack",
    )


def show4dstem_gold(
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
):
    """Load the gold 4D-STEM Show4DSTEM tutorial scan by friendly size name.

    The source scan is stored once under
    ``widget-tutorials/show4dstem/gold-128-bin8/full``; ``"small"`` and
    ``"medium"`` keep every 4th or 2nd scan position and say so in one line.
    The file is read as every 4D-STEM file is (``read_4dstem``): a
    ``quantem.gpu.io.Dataset4dstemGPU`` that stays on the GPU when quantem.gpu
    and a GPU are present, else a quantem core ``Dataset4dstem``.
    """

    size = _normalize_tutorial_size(size)
    folder = _download_widget_tutorial_folder(
        "show4dstem", "gold-128-bin8", size="full", cache_dir=cache_dir, revision=revision, force_download=force_download,
    )
    scan_stride = _SHOW4DSTEM_SCAN_STRIDE_BY_SIZE[size]
    metadata = json.loads((folder / "meta.json").read_text())
    source = folder / "data.npy"
    loaded = read_4dstem(source, verbose=verbose)
    name = "Gold 4D-STEM bin8 preview"
    sampling = [float(value) * scan_stride if axis < 2 else float(value) for axis, value in enumerate(metadata["sampling"])]
    units = [str(unit) for unit in metadata["units"]]
    if gpu_adapter.is_acquisition(loaded):
        with loaded:
            preview_t = loaded[::scan_stride, ::scan_stride].contiguous()
        dataset = gpu_adapter.acquisition_from_tensor(
            preview_t, {"name": name, "sampling": sampling, "units": units, "source_path": str(source)})
    else:
        stack = np.ascontiguousarray(as_array(loaded)[::scan_stride, ::scan_stride])
        dataset = make_dataset(stack, sampling=sampling, units=units, name=name)
    if scan_stride > 1:
        print(f"show4dstem_gold(size={size!r}) keeps every {scan_stride}th scan position of the "
              f"{loaded.shape[0]}x{loaded.shape[1]} scan; size='large' or 'full' loads every position.")
    if verbose:
        print(f"Source: {metadata['name']} from Hugging Face")
        print(f"Full stack: {tuple(loaded.shape)} {loaded.dtype}")
        print(f"Preview: {tuple(dataset.shape)}, scan stride {scan_stride}")
        print(f"Sampling: {[float(value) for value in dataset.sampling]} {[str(unit) for unit in dataset.units]}")
        print(f"Processing: {metadata.get('processing', 'none')}")
    return dataset


def gold_session(
    *, size: str = "small", cache_dir: str | Path | None = None,
    revision: str | None = None, force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """Download the public gold HAADF session for image-reader tutorials.

    Parameters
    ----------
    size
        Tutorial size, normally ``"small"``.
    cache_dir, revision, force_download
        Download cache, dataset revision and refresh controls.
    verbose
        Print the number of downloaded EMD files.

    Returns
    -------
    Path
        Cached folder containing the public EMD images.

    Examples
    --------
    >>> folder = gold_session()
    >>> images = sorted(folder.glob("*.emd"))
    """
    size = _normalize_tutorial_size(size)
    folder = _download_widget_tutorial_folder(
        "showfolder", "gold-haadf-session", size=size, cache_dir=cache_dir,
        revision=revision, force_download=force_download,
    )
    if verbose:
        print(f"Gold image session: {len(list(folder.glob('*.emd')))} EMD files")
    return folder


def showdiffraction_fe3o4(
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
):
    """Load the real Fe3O4 nanoparticle SAED pattern used by the ShowDiffraction tutorial.

    The pattern is downloaded from
    ``widget-tutorials/showdiffraction/fe3o4-saed``. It is uncalibrated by
    design; the tutorial calibrates it against the Fe3O4 phase.

    Parameters
    ----------
    cache_dir
        Optional Hugging Face cache directory.
    revision
        Optional Hugging Face dataset revision.
    force_download
        If ``True``, ask Hugging Face Hub to refresh the cached files.
    verbose
        If ``True``, print a short dataset summary.

    Returns
    -------
    Dataset2d
        The 512 by 512 float32 diffraction pattern (``.array``), a quantem core
        ``Dataset2d`` like every 2D tutorial loader returns.
    """

    folder = _download_widget_tutorial_folder(
        "showdiffraction",
        "fe3o4-saed",
        size="small",
        cache_dir=cache_dir,
        revision=revision,
        force_download=force_download,
    )
    source = folder / "data.npy"
    if not source.is_file():
        raise FileNotFoundError(f"tutorial pattern is missing: {source}")
    pattern = np.asarray(np.load(source), dtype=np.float32)
    if verbose:
        print(f"Fe3O4 SAED tutorial pattern: {source}")
        print(f"Pattern: {pattern.shape[0]} x {pattern.shape[1]} {pattern.dtype}")
    return make_dataset(pattern, name="Fe3O4 SAED")


__all__ = [
    "gold_session",
    "show1d_ducky",
    "show2d_gold",
    "show3d_gold",
    "show4dstem_gold",
    "showdiffraction_fe3o4",
]
