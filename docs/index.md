# quantem.widget

Interactive Jupyter widgets for electron microscopy visualization.
Works with NumPy, CuPy, and PyTorch arrays.

## install

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem-widget
```

## widgets

| widget | description |
|--------|-------------|
| [Show2D](examples/show2d/show2d_simple) | 2D image viewer with gallery, FFT, histogram |
| [Show3D](examples/show3d/show3d_simple) | 3D stack viewer with playback, ROI, FFT, export |
| [Show3DVolume](examples/show3dvolume/show3dvolume_simple) | orthogonal slice viewer (XY, XZ, YZ) |
| [Show4DSTEM](examples/show4dstem/show4dstem_simple) | 4D-STEM diffraction pattern viewer with virtual imaging |
| [Show4D](examples/show4d/show4d_simple) | general 4D data viewer with dual navigation/signal panels |
| [Align2D](examples/align2d/align2d_simple) | image alignment overlay with phase correlation |
| [Mark2D](examples/mark2d/mark2d_simple) | interactive point picker for 2D images |

```{toctree}
:maxdepth: 2
:hidden:

examples/index
api/index
changelog
```
