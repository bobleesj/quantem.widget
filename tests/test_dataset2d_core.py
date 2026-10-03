import numpy as np


def test_read_image_returns_core_dataset2d(tmp_path):
    from quantem.core.datastructures import Dataset2d
    from quantem.widget.io.image import read_image

    path = tmp_path / "survey.npy"
    np.save(path, np.arange(4, dtype=np.float32).reshape(2, 2))

    ds = read_image(path)

    assert isinstance(ds, Dataset2d)
    assert ds.name == "survey"
    np.testing.assert_array_equal(ds.array, np.array([[0, 1], [2, 3]], dtype=np.float32))


def test_native_torch_scan_selections_keep_widget_pixels_and_calibration():
    import torch

    from quantem.core.datastructures import Dataset4dstem
    from quantem.widget import Show2D, Show3D, Show4DSTEM, ShowDiffraction

    counts_t = torch.arange(3 * 4 * 6 * 7, dtype=torch.float32).reshape(3, 4, 6, 7)
    data = Dataset4dstem.from_tensor(
        counts_t,
        sampling=(0.4, 0.6, 0.02, 0.03),
        units=["nm", "nm", "1/angstrom", "1/angstrom"],
        name="calibrated scan",
    )
    pattern = data[1, 2]
    image = Show2D(pattern, verbose=False)
    gallery = Show2D([pattern, data[2, 3]], verbose=False)
    stack = Show3D(data[1], verbose=False)
    diffraction = ShowDiffraction(pattern, verbose=False)
    acquisition = Show4DSTEM(data, precompute_virtual_images=False, verbose=False)
    try:
        np.testing.assert_array_equal(image._data[0], counts_t[1, 2].numpy())
        np.testing.assert_array_equal(gallery._data[1], counts_t[2, 3].numpy())
        assert image.pixel_size == 0.03
        assert diffraction.k_pixel_size == 0.02
        assert acquisition.pixel_size == 0.6
        assert acquisition.k_pixel_size == 0.03
        assert stack.n_slices == 4
        image.set_image(data[2, 1])
        np.testing.assert_array_equal(image._data[0], counts_t[2, 1].numpy())
        stack.set_image(data[0])
        np.testing.assert_array_equal(stack._data, counts_t[0].numpy())
    finally:
        for widget in (image, gallery, stack, diffraction, acquisition):
            widget.close()
