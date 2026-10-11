"""Download only the selected tutorial family at its recorded revision."""

import sys
from types import ModuleType

import pytest
import huggingface_hub

from quantem.widget.io import hub


def test_qualified_download_preserves_revision_and_file_metadata(tmp_path, monkeypatch):
    files = ["haadf/gold.npy", "haadf/gold.json", "4dstem/gold/data.npy", "widget/ducky/show1d_monitor.jsonl", "widget/ducky/snapshots/reference.npy", "denoise/README.md", "denoise/empty/README.md"]
    calls = []
    monkeypatch.setattr(hub, "_shared_download", lambda *args, **kwargs: None)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda **kwargs: calls.append(("list", kwargs)) or files)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: calls.append(("download", kwargs)) or str(tmp_path))
    assert hub.download("haadf/gold", revision="recorded", verbose=False) == tmp_path / "haadf/gold.npy"
    assert calls[-1][1]["allow_patterns"] == ["haadf/gold.npy", "haadf/gold.json"]
    assert all(options["revision"] == "recorded" for _, options in calls)
    with pytest.raises(ValueError, match="qualified"):
        hub.download("gold", verbose=False)
    with pytest.raises(FileNotFoundError):
        hub.download("denoise/empty", verbose=False)
    assert hub.download("widget/ducky", revision="recorded", verbose=False) == tmp_path / "widget/ducky"
    assert calls[-1][1]["allow_patterns"] == "widget/ducky/*"
    assert hub.download("4dstem/gold/data.npy", verbose=False) == tmp_path / "4dstem/gold"


def test_download_delegates_all_options_to_compatible_shared_api(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "_shared_download", lambda name, **kwargs: calls.append((name, kwargs)) or tmp_path)
    assert hub.download("widget/ducky", revision="recorded", cache_dir=tmp_path, force_download=True, verbose=False) == tmp_path
    assert calls[0][0] == "widget/ducky"
    assert calls[0][1]["revision"] == "recorded"
    assert calls[0][1]["cache_dir"] == tmp_path
    assert calls[0][1]["force_download"] is True


def test_old_optional_shared_package_uses_widget_compatibility_resolver(monkeypatch):
    module = ModuleType("quantem.data.huggingface")

    def old_download(name, *, repo=None, out=None, verbose=True):
        raise AssertionError("Old resolver must not receive new revision options")

    module.download = old_download
    monkeypatch.setitem(sys.modules, "quantem.data.huggingface", module)
    assert hub._shared_download("widget/ducky", revision="recorded") is None


def test_existing_session_name_downloads_canonical_folder(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "_shared_download", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files",
        lambda **kwargs: ["widget/gold_session/gold.emd"],
    )
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda **kwargs: calls.append(kwargs) or str(tmp_path),
    )
    assert hub.download("gold_haadf_session", verbose=False) == tmp_path / "widget/gold_session"
    assert calls[0]["allow_patterns"] == "widget/gold_session/*"
