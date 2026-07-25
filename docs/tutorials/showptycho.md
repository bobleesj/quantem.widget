# ShowPtycho in Jupyter

`ShowPtycho` is an interactive **SSB** (single-sideband) aberration explorer for
4D-STEM data. You tune defocus (C10), astigmatism (C12 / phi12), and scan-detector
rotation and watch the reconstructed phase and its FFT update live in the notebook.

SSB is a *direct* (non-iterative) phase retrieval: fast and interactive, but
lower quality than iterative multislice ptychography. Use ShowPtycho for quick
aberration tuning and review, not as a substitute for a full iterative
reconstruction.

To export a standalone HTML viewer you can open without a kernel, see
[Export and run ShowPtycho](showptycho_export.md).

## The one rule: always optimize before you view

```python
import cupy as cp
from quantem.gpu.io import load
from quantem.gpu.ssb.reconstruction import SSB
from quantem.widget import ShowPtycho

# 1. Load the native detector — do NOT bin (see "No detector binning" below).
data = load("scan_master.h5", dtype=None).data        # dtype=None keeps native uint16

# 2. Build the SSB reconstruction with your microscope calibration.
ssb = SSB(
    data,
    semiangle=30.0,          # convergence semiangle, mrad
    scan_sampling=0.264,     # real-space scan step, Angstrom
    voltage_kV=300.0,
    rotation_angle_deg=158.9,   # scan-detector rotation (run find_rotation if unknown)
)

# 3. Solve the aberrations.  THIS STEP IS REQUIRED.
ssb.optimize(n_trials=200)   # Optuna TPE global search for C10/C12/phi12 (~1-2 s)
ssb.refine()                 # Nelder-Mead exact minimum (~1 s)

# 4. Open the interactive widget — it starts at the solved optimum.
ShowPtycho(ssb)
```

### Do NOT skip step 3

```python
# WRONG — this NEVER optimizes.  It uses whatever aberrations you pass verbatim,
# so the phase and FFT are junk unless your numbers were already perfect.
ShowPtycho(data, semiangle=30.0, scan_sampling=0.264,
           aberrations={"C10": 78.0, "C12": 17.0, "phi12": 0.5})
```

`ShowPtycho(data, aberrations=...)` is a convenience constructor that trusts the
aberrations you hand it. It does not run Optuna or Nelder-Mead. If you want the
solver to find the aberrations, build an `SSB`, call `optimize()` then `refine()`,
and pass the `ssb` object: `ShowPtycho(ssb)`.

You can confirm the solve ran: the stats bar shows a non-null `loss`, and the
`Optuna trials + nmead` panel at the bottom is populated.

## No detector binning

Build the reconstruction at the **native detector size** (`det_bin=1`, the
default). Native (e.g. 192x192) is what resolves light columns such as oxygen in
a perovskite; binning throws that away. Binning also breaks the HTML export (the
browser cannot bin), so keep the whole workflow un-binned.

## Region-specific refit (crop)

A smaller crop often converges more physically than the full field of view: a
single global aberration and rotation hold better over a small region, so a crop
can resolve oxygen the full FOV cannot.

Two ways to crop:

- **Interactively.** Construct the widget with the raw master path so the `Crop`
  action appears next to `Export`/`Reset`. Enable `Crop`, drag a rectangle on the
  phase, then `Refit SSB` — the widget reloads only that scan region from the
  HDF5 source, runs 200 optimization trials plus refinement, and replaces the
  phase/FFT and calibration.

- **In code.** Load only the region, then optimize as usual:

  ```python
  data = load("scan_master.h5", dtype=None,
              scan_region=(128, 384, 128, 384)).data   # 256x256 center crop
  ssb = SSB(data, semiangle=30.0, scan_sampling=0.264,
            voltage_kV=300.0, rotation_angle_deg=158.9)
  ssb.optimize(n_trials=200); ssb.refine()
  ShowPtycho(ssb)
  ```

  256x256 is a good crop size: small enough for region-specific aberrations, big
  enough that the phase is not blocky. 128x128 works but displays coarse.

## Checklist

1. `dtype=None` on load — native uint16, no lossy `uint8` clip.
2. Native detector, `det_bin=1` — do not bin.
3. `ssb.optimize(n_trials=200)` then `ssb.refine()` — the solve is not optional.
4. Pass the `ssb` object to `ShowPtycho`, not `data` + hand-typed aberrations.
5. Confirm: stats bar `loss` is non-null and the trials panel is populated.
