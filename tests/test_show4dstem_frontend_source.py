from __future__ import annotations

import pathlib


def test_show4dstem_preset_clicks_sync_without_comm_guard() -> None:
    """C1: BF/ABF/ADF clicks must sync in JupyterLab models without ``comm``."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    source = (repo_root / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )
    save_start = source.index("const saveChangesIfLiveComm")
    save_end = source.index("const publishVirtualImageBytes", save_start)
    save_block = source[save_start:save_end]
    preset_start = source.index("const requestViPreset")
    preset_end = source.index("const setViSource", preset_start)
    preset_block = source[preset_start:preset_end]

    assert "liveModel.comm" not in save_block
    assert 'typeof liveModel.save_changes !== "function"' in save_block
    assert 'model.set("_preset_request", preset);' in preset_block
    assert "saveChangesIfLiveComm();" in preset_block
