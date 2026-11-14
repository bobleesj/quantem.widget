"""quantem.widget.io.links: a file beside a viewer that reads another, where symlinks are refused too."""

import os
import pathlib

import pytest

from quantem.widget.io.links import place_link, replaceable


def _refuse_symlinks(monkeypatch):
    def refused(self, target, target_is_directory=False):
        raise OSError(1314, "A required privilege is not held by the client")  # Windows, no symlink privilege

    monkeypatch.setattr(pathlib.Path, "symlink_to", refused)


def test_a_link_reads_its_target_whatever_the_os_allows(tmp_path):
    target = tmp_path / "scan_master.h5"
    target.write_bytes(b"counts")
    (tmp_path / "viewer").mkdir()
    link = tmp_path / "viewer" / "scan_master.h5"
    kind = place_link(link, target, relative=True)
    assert kind in ("symlink", "hardlink") and link.samefile(target)
    if kind == "symlink":
        assert not pathlib.Path(os.readlink(link)).is_absolute()
    assert replaceable(link, target)


def test_without_the_symlink_privilege_a_hardlink_is_placed(tmp_path, monkeypatch):
    _refuse_symlinks(monkeypatch)
    target = tmp_path / "a.qem"
    target.write_bytes(b"counts")
    link = tmp_path / "link.qem"
    assert place_link(link, target, relative=True) == "hardlink"
    assert link.samefile(target) and not link.is_symlink()
    assert replaceable(link, target)


def test_without_symlinks_or_hardlinks_the_file_is_copied_and_still_replaceable(tmp_path, monkeypatch, capsys):
    _refuse_symlinks(monkeypatch)

    def refused(src, dst):
        raise OSError(17, "The system cannot move the file to a different disk drive")

    monkeypatch.setattr(os, "link", refused)
    target = tmp_path / "a.qem"
    target.write_bytes(b"counts")
    link = tmp_path / "link.qem"
    assert place_link(link, target) == "copy"
    assert link.read_bytes() == b"counts" and not link.samefile(target)
    assert "copied" in capsys.readouterr().out
    assert replaceable(link, target)  # the same size and time: the copy this module made


def test_a_real_file_is_never_replaceable(tmp_path):
    target = tmp_path / "a.qem"
    target.write_bytes(b"counts")
    other = tmp_path / "b.qem"
    other.write_bytes(b"someone else's file")
    assert not replaceable(other, target)
    assert not replaceable(other)
    assert not replaceable(tmp_path / "missing.qem", target)


def test_a_symlink_is_replaceable_even_to_another_target(tmp_path):
    target = tmp_path / "a.qem"
    target.write_bytes(b"counts")
    link = tmp_path / "link.qem"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"this OS refuses symlinks here: {error}")
    assert replaceable(link) and replaceable(link, tmp_path / "elsewhere.qem")
