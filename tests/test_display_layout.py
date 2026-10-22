"""Plain layout keywords preserve images and the existing saved widget state."""

import numpy as np
import pytest

from quantem.widget import Show2D, Show3D


@pytest.mark.parametrize("viewer_class, legacy", [(Show2D, "ncols"), (Show3D, "max_cols")])
def test_old_and_new_gallery_calls_share_layout_and_image_state(viewer_class, legacy):
    images = np.random.default_rng(8).random((5, 3, 24, 32)).astype(np.float32)
    data = list(images[:, 0] if viewer_class is Show2D else images)
    with pytest.warns(DeprecationWarning, match="use columns"):
        old = viewer_class(data, **{legacy: 2}, size="medium", verbose=False)
    current = viewer_class(data, columns=2, size="medium", verbose=False)
    assert current.columns == old.columns == 2
    assert current.get_state()[legacy] == old.get_state()[legacy]
    np.testing.assert_array_equal(current._data, old._data)
    assert current.panel_width_px == old.panel_width_px == 450
    if viewer_class is Show2D:
        assert current._static_png_b64() == old._static_png_b64()
    current.columns = 3
    assert current.get_state()[legacy] == 3
    np.testing.assert_array_equal(current._data, old._data)


@pytest.mark.parametrize("viewer_class", [Show2D, Show3D])
def test_rows_reflow_each_page_without_counting_other_pages(viewer_class):
    images = np.random.default_rng(9).random((2, 5, 3, 24, 32)).astype(np.float32)
    data = images[:, :, 0] if viewer_class is Show2D else images
    viewer = viewer_class(data, rows=2, size="small", verbose=False)
    assert viewer.columns == 3
    assert viewer.n_pages == 2
    viewer.page_idx = 1
    assert viewer.columns == 3
    assert viewer.panel_width_px == 300


@pytest.mark.parametrize("viewer_class, legacy", [(Show2D, "ncols"), (Show3D, "max_cols")])
def test_conflicting_layout_requests_explain_how_to_keep_all_panels(viewer_class, legacy):
    images = np.ones((5, 3, 12, 16), dtype=np.float32)
    data = list(images[:, 0] if viewer_class is Show2D else images)
    with pytest.raises(ValueError, match="not both"):
        viewer_class(data, columns=2, **{legacy: 2}, verbose=False)
    with pytest.raises(ValueError, match="cannot hold 5 panels"):
        viewer_class(data, rows=2, columns=2, verbose=False)
    for value in (0, True, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            viewer_class(data, columns=value, verbose=False)


def test_legacy_single_row_and_saved_state_still_export(tmp_path):
    stacks = [np.ones((3, 12, 16), dtype=np.float32)] * 3
    with pytest.warns(DeprecationWarning, match="use columns"):
        viewer = Show3D(stacks, max_cols=0, verbose=False)
    assert viewer.columns == 3
    viewer.max_cols = 0  # Existing saved state uses zero for one row.
    viewer.export_html(tmp_path / "single-row.html")
    assert viewer.columns == 3
