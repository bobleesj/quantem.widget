# 2026-10-05: Show4DSTEM on encoded acquisitions

## Question

quantem.gpu's restructure makes `io.load` return only encoded acquisitions
(`Dataset4dstemGPU`, ANS-encoded on CUDA or MPS) and removes detector binning,
dtype casts, dense and packed residency, multi-GPU loading, source112, count-ANS
and the CUDA resident owner. Which widget features keep working on the encoded
data, how fast are they, and which features lose their backend?

## Setup

- quantem.gpu: `gpu-refactor` worktree (5265b312 at the start, 0d955476 at the end).
- quantem.widget: branch `gpu-refactor-callers` from `main` f9cbccfc.
- CUDA: cudahost GPU 0, RTX PRO 6000 Blackwell, shared with other jobs (load
  average about 20 during most runs, so timings are upper bounds).
- MPS: maca (Apple M5).
- Data: Arina 512 x 512 x 192 x 192 uint16 masters (one BTO zone series of 41
  ready masters; a MAPED tilt folder and an SSB acquisition on maca).
- Browser: headed Chrome on the NVIDIA Vulkan adapter (`adapter.info` reported
  `nvidia` / `blackwell`), private JupyterLab with its own settings.

## Results

| Measurement | CUDA (cudahost) | MPS (maca) |
|---|---|---|
| `io.load(master)`, warm disk | 1.0 s | 2.3 to 2.8 s |
| Encoded size (18 GiB dense) | 1.98 GiB; 0.09 GiB for a blank scan | 1.93 GiB |
| `Show4DSTEM(load(master))` construction | 1.4 s | 6.4 s |
| Detector ROI change | 2 ms | 4 ms |
| Diffraction pattern move | 0.8 ms | not timed |
| Virtual image vs `detector.prepare(acq).masked_sum` | identical | identical |
| Diffraction pattern vs `acq.read(...)` | identical | identical |
| Comparison of 8 masters: construction / ROI change | 6.9 s / 22 ms | not run |
| `from_folder`, 12 masters: first viewer / all loaded | 3.6 s / 22.5 s (268 s under heavy contention) | 4 masters: 8.7 s / 14.8 s |
| Compute SSB, 512 x 512, 3 trials | 1.2 s, phase identical to `SSB(acq).reconstruct(fitted)` | 2.1 to 3.4 s |
| DPC maps vs rotated `dpc.center_of_mass(acq)` | equal (atol 1e-5) | equal (atol 1e-5) |
| `io.save("a.qem", acq)` then browser QEM viewer decode | 2.0 GiB file; Chrome WebGPU decode 15.9 s; DP stats equal to the live kernel | not run |

Construction time is dominated by the mean diffraction pattern used to find
the bright-field disk: quantem.gpu's `BoundedDetectorCompute.mean_dp` reads
every scan position (1.5 s on CUDA) although the encoded session computes the
same pattern natively in 0.07 s.

## Conclusions

- `Show4DSTEM(load(path))`, `Show4DSTEM(load([a, b]))`, `from_folder`,
  `ShowFolder.open_show4dstem`, Compute SSB, DPC and ShowPtycho work on encoded
  acquisitions on CUDA and MPS, at full detector resolution, with exact virtual
  images and diffraction patterns.
- A folder of masters fits on one GPU encoded (6 masters: 6.3 GB for 108 GiB
  dense), so the lazy dense pages, LRU paging, preview cache, progressive page
  streaming and detector binning of the old folder viewer were removed rather
  than rebuilt.
- The browser export reads `.qem` files written by `io.save`.

## Rejected ideas

- Keep the lazy `Dataset5dstem` folder pages by expanding each master with
  `acq.read()`: 18 GiB per master instead of about 2 GiB, the expansion the
  encoded API exists to avoid.
- Bin detectors in the widget for live views: a lossy reduction with no
  backend and contrary to the no-surprise-reduction rule; `--bin` remains only
  for offline HTML exports, where the whole array is embedded.
- Put encoded views inside `Dataset5dstem` pages: paging moves tensors between
  devices and host memory, which encoded storage cannot do.
- Load the whole folder before showing the viewer: 22 s warm and several
  minutes cold for 12 masters, so the viewer opens after the first master and
  loads the rest in the background.

## Open points

- The virtual-image canvas renders black in headed-Chrome screenshots on the
  Xvfb display at `main` and on this branch alike; its statistics and histogram
  update, and the diffraction canvas renders.
- ShowPtycho's drag preview calls `close()` on the SSB preview context; the MPS
  context has no `close()`, so changing `drag_bf` (and a crop refit) fails on
  MPS. This predates the restructure.
