"""GIF and MP4 export for image stacks and widget-rendered frames.

Array stacks go to :mod:`quantem.gpu.movie`, which chooses the CUDA, Metal or
CPU encoder and applies contrast and grid layout. Frames a widget has already
rendered as PIL images (Show3D exports) need only the container, so they are
written here without passing through that pipeline.
"""

from pathlib import Path

from PIL import Image

from quantem.gpu import movie as gpu_movie
from quantem.gpu.movie import MovieData
from quantem.widget.render import gif as gif_utils


def _is_rendered_frames(data: MovieData) -> bool:
    """True for a non-empty list of PIL frames rendered by a widget."""
    return (
        isinstance(data, (list, tuple))
        and len(data) > 0
        and all(isinstance(frame, Image.Image) for frame in data)
    )


def save_gif(data: MovieData, path: str | Path, *, fps: float = 12.0, **kwargs) -> Path:
    """Write ``data`` as a looping GIF at ``fps`` frames per second.

    Rendered PIL frames are written as given; array stacks accept the layout and
    contrast options of :func:`quantem.gpu.movie.save_gif`.
    """
    if _is_rendered_frames(data) and not kwargs:
        return gif_utils.write_gif(list(data), path, fps=fps)
    return gpu_movie.save_gif(data, path, fps=fps, **kwargs)


def save_mp4(
    data: MovieData,
    path: str | Path,
    *,
    fps: float = 12.0,
    crf: int = 18,
    **kwargs,
) -> Path:
    """Write ``data`` as an H.264 MP4 at ``fps`` frames per second.

    ``crf`` is the H.264 quality factor (lower is higher quality). Rendered PIL
    frames are written with ffmpeg as given; array stacks accept the backend,
    layout and contrast options of :func:`quantem.gpu.movie.save_mp4`.
    """
    if _is_rendered_frames(data) and not kwargs:
        return gif_utils.write_mp4(list(data), path, fps=fps, crf=crf)
    return gpu_movie.save_mp4(data, path, fps=fps, crf=crf, **kwargs)


def save_movie(data: MovieData, path: str | Path, *, format: str | None = None, **kwargs) -> Path:
    """Write a GIF or MP4 chosen by ``format`` or by the suffix of ``path``."""
    suffix = (format or Path(path).suffix.lstrip(".")).lower()
    if suffix == "gif":
        return save_gif(data, path, **kwargs)
    if suffix == "mp4":
        return save_mp4(data, path, **kwargs)
    return gpu_movie.save_movie(data, path, format=format, **kwargs)


__all__ = ["MovieData", "save_gif", "save_movie", "save_mp4"]
