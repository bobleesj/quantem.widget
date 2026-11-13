# Set up a Windows PC with an NVIDIA card

Browse 4D-STEM scans on the GPU with `Show4DSTEM`, from source checkouts of quantem.gpu and quantem.widget.
First used on a detector PC with an RTX A400 (4 GB).

## Before you start

- [Miniforge](https://conda-forge.org/download/)
- NVIDIA driver 580 or newer (`nvidia-smi` shows the version)

Nothing else: the environment brings Python, Git, Node.js, JupyterLab and PyTorch with CUDA.

## Install

Open a **Miniforge Prompt**, go to an empty folder, and paste:

```bash
curl -LO https://raw.githubusercontent.com/bobleesj/quantem.widget/arina-pc/environments/install.bat && install.bat
```

It prints five numbered steps and takes several minutes. JupyterLab opens `browse.ipynb` when it is done.
Run `install.bat` again at any time to repair or update the installation: finished steps are skipped.

## Browse a scan

In `browse.ipynb`, set `SCAN` to your file and run the cells:

```python
SCAN = Path(r"D:\data\scan_master.h5")  # an Arina master (its data files beside it) or a .qem file
```

```python
data = io.load(SCAN)
Show4DSTEM(data)
```

The scan stays lossless and compressed on the GPU. Nothing is binned or cropped. `io.load` prints the size it takes.

## Every later session

```bash
conda activate arina-pc && jupyter lab
```

## Get new features

```bash
conda activate arina-pc && widget update
```

Then restart the notebook kernel.

## If something goes wrong

| What you see | What to do |
|---|---|
| Any step of the install stops | Read its last lines, fix that, run `install.bat` again |
| `No module named ...` | The wrong Python is active: open a new Miniforge Prompt, `conda activate arina-pc`, run `install.bat` again |
| `conda` is not recognized | Use the Miniforge Prompt, not the plain Command Prompt |
| `out of memory` when loading | The scan is too large for the card once compressed, or another notebook holds the card: shut down other kernels and try again |
| The viewer area stays empty | Run `widget build`, restart the kernel, reload the browser tab |
| `widget update` says a checkout could not be fast-forwarded | That folder has local changes; nothing was changed. Keep or undo them, then run it again |

## What fits on a 4 GB card

A 512 x 512 scan on a 192 x 192 detector (19.3 GB of counts) took 2.6 GB on the card and 3.2 GB with the viewer
open, measured with 3.7 GB free. A scan with more counts compresses less and may not fit.
