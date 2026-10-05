# Image and acquisition I/O

Use `quantem.gpu.io` for 4D-STEM acquisitions and `quantem.widget` image readers
for survey images. The [QuantEM.GPU I/O guide](https://github.com/bobleesj/quantem.gpu/blob/main/docs/api/io.md) owns the detector
storage, supported formats, metadata, backend, and exactness contracts.
For a walkthrough of `uint8`/`uint16`, memory estimates and the image readers,
start with {doc}`IO/GPU <../tutorials/io_gpu>`; GPU selection and cleanup are in
{doc}`Memory management <../tutorials/memory_management>`.

## Open an acquisition and inspect a pattern

```python
from quantem.gpu import detector, io
from quantem.widget import Show2D, Show4DSTEM

data = io.load("scan_master.h5")
Show4DSTEM(data)
```

`data` is a `quantem.gpu.io.Dataset4dstemGPU`, ANS encoded on the CUDA device or
the Apple GPU at native detector sampling and count dtype: a 512 x 512 x 192 x 192
uint16 scan, 18 GiB as a dense array, occupies about 0.1 to 2 GiB. Keep it open
while viewers or calculations use its encoded buffers. Array-style selection
decodes only the selected region into a Torch tensor on the acquisition's GPU;
it does not unpack the complete acquisition:

```python
Show2D(data[10, 12])
```

| Expression | Selection |
| --- | --- |
| `data[10, 12]` | One diffraction pattern |
| `data[8:12, 10:16]` | A rectangular scan patch |
| `data[10, 12, 64:128, 64:128]` | A detector crop at one position |
| `data[:, :, 95, 100]` | One detector pixel across the scan |
| `data.metadata` | Scientific calibration and source metadata |

```python
Show2D([detector.bf(data), detector.adf(data)], labels=["BF", "ADF"])
io.save("scan.qem", data)
```

Call `data.close()` after the last use. Scripted jobs can use
`with io.load(path) as data:` for automatic cleanup. A list of paths loads as a
list of encoded acquisitions, one per file; `Show4DSTEM(io.load(paths))` opens
them as a comparison grid labelled by file, and `Show4DSTEM(series[1])` opens
one of them. `io.load(path, backend="cuda", device=1)` selects a CUDA device; a
Mac loads onto MPS. See [load](load.md) for the function reference and the GPU
guide for multi-device storage policies.

## Read a scan region

Reconstruction, denoise and ROI workflows read the rectangular scan region they
need. `read` decodes that region into a Torch tensor on the acquisition's GPU:

```python
with io.load("scan_master.h5") as data:
    patch_t = data.read(scan_region=(160, 293, 234, 367))  # row_start, row_stop, col_start, col_stop
print(patch_t.shape)  # torch.Size([133, 133, 192, 192])
```

Stops are exclusive, and `detector_region=` bounds the detector pixels the same
way. For a drift-corrected time series, derive `scan_region` from the shared
specimen ROI, the frame shift and a small scan halo, then sample the final ROI
from the local patch. The detector counts remain raw; drift stays as
scan-position metadata.

## Discover acquisitions

```python
from quantem.gpu import io
from quantem.widget import Show4DSTEM

files = io.discover("/data/session")
Show4DSTEM(io.load(files[0]))
```

`io.discover(path, scan_shape=(512, 512))` keeps only matching acquisitions when
a folder mixes scan sizes. `io.inspect(path)` reads readiness and calibration
headers without decoding measurements. Use `Show4DSTEM.from_folder(path)` for a
live acquisition viewer.

For named sample data, see [Tutorial datasets](datasets.md). The [IO/GPU
notebook](../tutorials/io_gpu.ipynb) demonstrates a real Gold image session.

## Read images and image stacks

```python
from quantem.widget import Show2D, Show3D, read_image, read_images, read_image_stack

image = read_image("overview.tif")
Show2D(image)
```

```python
images = read_images("survey_images", workers=8)
Show2D(images)
```

```python
stack = read_image_stack("frames", file_type="tif", workers=8)
Show3D(stack)
```

These format readers run on the host. Calibrated image inputs retain their
sampling for viewer scale bars. This path is for individual images and image
sequences, not compressed 4D-STEM detector acquisitions.

## Share data and finished figures

Use [Contribute tutorial data](../tutorials/contribute_data.md) for shared
example datasets and [HTML and file export](../tutorials/widget_export.ipynb)
for interactive figures. Acquisition export belongs to `io.save`, described
in the GPU I/O guide; it is separate from exporting a viewer.

## Image function reference

```{eval-rst}
.. autofunction:: quantem.widget.read_image
.. autofunction:: quantem.widget.read_images
.. autofunction:: quantem.widget.read_image_stack
```
