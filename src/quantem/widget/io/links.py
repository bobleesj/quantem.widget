"""Files placed beside a viewer that read another file's bytes, on every OS.

A symlink is the first choice. Windows refuses one without the symlink privilege (Developer Mode or an
administrator; WinError 1314), so there a hardlink follows (same volume, no extra space), and a copy last.
"""

import os
import pathlib
import shutil


def place_link(link: pathlib.Path, target: pathlib.Path, *, relative: bool = False) -> str:
    """Make ``link`` read ``target``: ``"symlink"``, else ``"hardlink"``, else ``"copy"`` (which is returned).

    ``relative`` writes a symlink relative to the link's folder. ``link`` must not exist; see ``replaceable``.
    """
    link, target = pathlib.Path(link), pathlib.Path(target)
    try:
        link.symlink_to(pathlib.Path(os.path.relpath(target, link.parent)) if relative else target)
        return "symlink"
    except OSError:
        pass
    try:
        os.link(target, link)
        return "hardlink"
    except OSError:
        pass
    print(f"{link.name}: copied from {target} ({target.stat().st_size / 2**30:.2f} GiB): this OS allows neither "
          "a symlink nor a hardlink here (another volume, or no symlink privilege on Windows).")
    shutil.copy2(target, link)
    return "copy"


def replaceable(link: pathlib.Path, target: pathlib.Path | None = None) -> bool:
    """Whether ``link`` is a link this module could have placed, safe to remove before placing it again.

    A symlink always is. A hardlink or copy is only when ``target`` is given and ``link`` is that file (the same
    file, or a copy with its size and modification time); any other real file is not.
    """
    link = pathlib.Path(link)
    if link.is_symlink():
        return True
    if target is None or not link.exists():
        return False
    target = pathlib.Path(target)
    if os.path.samefile(link, target):
        return True
    ours, theirs = link.stat(), target.stat()
    return ours.st_size == theirs.st_size and ours.st_mtime_ns == theirs.st_mtime_ns
