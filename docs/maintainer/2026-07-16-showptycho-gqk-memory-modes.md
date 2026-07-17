# 2026-07-16 — ShowPtycho WebGPU resident-memory experiment: Hermitian half-plane and snorm16 block quantization

## Question

The ShowPtycho WebGPU folder viewer keeps the per-BF-pixel `G(q,k)` reducers
resident on the GPU for the whole review session (that residency is what makes
aberration-slider drags ~15 ms instead of a full HDF5 re-decode). Resident cost
is `active_BF_pixels x scan_pixels x 8 bytes` (complex64). On a 512x512 scan at
full BF that projects to tens of GB — workstation-only. Question: **can the
resident footprint shrink without losing precision or drag speed**, so
collaborators can open these folders on ordinary laptops?

## Setup

- Dataset: a real experimental 4D-STEM acquisition
  (512x512 scan, 192x192 Arina, 19.3 GB raw, max pixel 17 counts).
  Calibration: fresh SSB fit, 200 Optuna trials + Nelder-Mead refine
  (`Show4DSTEM.compute_ssb`, rotation seeded from a prior batch screen fit).
- Export: `quantem showptycho <master> --calibration <fit.json> --html`
  (WebGPU folder, compressed HDF5 + `bf_columns.u8` companion, 5.9 GB on disk).
- Viewer host: phil (M5 MacBook, 24 GB unified, Chrome, adapter `apple/metal-3`
  — real hardware confirmed via `adapter.info`, not SwiftShader).
- Harness: folder served by `scripts/serve_sidecar_range.py` (plain
  `python -m http.server` FAILS — no HTTP Range support, viewer dies with
  "Failed to fetch"). Page driven over CDP; each mode selected with the new
  `?gqk=full|herm|herm16` URL parameter; the engine stashes
  `globalThis.__quantemSsbLast = {gqkMode, residentGqkBytes, gpuMs, phase, ...}`
  after every reconstruct, and the harness pulls the raw `Float32Array` phase
  out over CDP for offline numpy comparison. Same aberrations, same BF set
  (preview 0.30 => 3941 requested, 408 active aperture pixels) in all runs.

## Raw numbers

| G(q,k) mode | resident VRAM | reconstruct (gpuMs) | max abs dphase vs full | rms dphase | phase corr |
|---|---|---|---|---|---|
| `full` (complex64 full plane, prior behavior) | 0.856 GB | 18.2 ms | — | — | — |
| `herm` (complex64 Hermitian half-plane) | 0.429 GB (2.0x) | 16.4 ms | **0.0 (bit-exact)** | 0.0 | 1.0 |
| `herm16` (half-plane + snorm16, one f32 scale per BF px) | 0.215 GB (4.0x) | 12.6 ms | 1.204e-4 rad | 2.60e-5 rad | 0.9999954 |

Phase image span was 0.0748 rad, so `herm16` worst-case error is 0.16 % of
span (rms 0.03 %) — far below shot noise on 17-count data.

Projection for this dataset at **full BF (13137 px, ~1360 active est.)**:
full ~2.9 GB, herm ~1.4 GB, herm16 ~0.7 GB. Upper-bound projection if every
selected BF pixel were aperture-active: 27.6 / 13.9 / 6.9 GB. Per-BF-pixel
cost by scan size: 2 MB at 512^2, 0.5 MB at 256^2, 0.125 MB at 128^2 (full
mode; divide by 2 or 4 for herm/herm16).

## Why herm is exactly lossless — and faster

The stored `G(q,k)` is the scan-space FFT of each BF pixel's intensity trace,
and intensities are real, so `G(-q,k) = conj(G(q,k))`. Storing the
`n x (n/2+1)` half-plane and mirror-conjugating on fetch is algebra, not
approximation. Measured **bit-exact** (max diff literally 0.0): the radix-2
FFT's rounding errors are themselves conjugate-symmetric for real input, so
the discarded half was a bitwise mirror all along. It is *faster* because the
per-drag reduce is bandwidth-bound (re-reads all of G every slider move) and
now reads half the bytes. Strictly better on every axis => **`herm` is the new
default**; `?gqk=full` restores the old layout.

## Implementation (js/showptycho/webgpu-ssb.ts)

- `GqkMode` = `full | herm | herm16`, resolved from `?gqk=` URL param or
  `globalThis.__QUANTEM_SHOWPTYCHO_GQK_MODE__`; default `herm`.
- `makeSsbShader(n, mode)` templates a `fetch_g(local_bf, bf_global, row, x)`
  WGSL helper: direct read for `x <= n/2`, mirror `(r,c) = ((n-row) % n, n-x)`
  + conjugate otherwise; `herm16` additionally
  `unpack2x16snorm(word) * gqkScale[bf_global]`.
- Build path unchanged (full-plane chunks, gather + in-place FFT), then a new
  GPU post-pass `transformGqkChunks` runs `scaleMax` (per-BF-pixel max |G| over
  the half-plane, workgroup tree reduce) and `compact` (copy or
  `pack2x16snorm(clamp(v/scale))`) per chunk, destroying each full chunk
  immediately — build peak only briefly exceeds the old peak by one compacted
  chunk; the *resident* session footprint is what shrinks.
- `__quantemSsbLast` debug hook on every reconstruct for harnesses.

## Rejected / deferred ideas

- **float16 storage**: 2x, but real mantissa loss on FFT accumulations
  (10-bit mantissa vs values spanning ~4.5e6 dynamic range). Rejected —
  strictly worse than herm16 which spends its 16 bits after per-pixel scaling.
- **Band-limit crop in q**: exact only when the scan oversamples the 2-alpha
  double-overlap disk. This dataset (0.5 A sampling, 30 mrad, 300 kV) is
  in-band across the whole q-plane — zero win here, dataset-dependent in
  general. Not implemented.
- **Store raw counts, rebuild G per drag**: unbounded memory win but turns
  every slider move into a full FFT rebuild — kills the 15 ms interactivity.
  Only sensible as a future explicit "final full-BF render" button.
- **Aberration-basis factorization of the k-sum**: the gamma weight is
  nonlinear in the coefficients (`e^{i chi}`), no exact low-rank split. Rejected.
- **Streaming the initial build** (bounded peak, not just bounded resident):
  requires either re-decoding all HDF5 chunks per BF chunk (N_chunks x slower
  load) or a compact real-u16 time-domain gather buffer. Deferred; noted as
  the remaining lever for laptop-friendly *first load*.

## Gotchas recorded

- `python -m http.server` cannot serve ShowPtycho folders (no Range support);
  use `scripts/serve_sidecar_range.py --dir <folder> --port <p>`.
- `collectActiveBfIndices` drops zero-aperture-weight BF pixels: the
  "3941/13137 BF" UI label overstates the resident set (408 active here).
  Memory projections must use *active* BF counts.
- `FULL_STACK_GPU_BUDGET_BYTES` (4.5 GB) in `webgpu-ssb.ts` is still dead code
  — the VRAM clamp for the BF slider remains unimplemented. With `herm` default
  the pressure is halved but a 512^2 full-BF drag can still device-lost a small
  GPU. Follow-up: cap effective BF by `budget / (storedPlane x bytesPer)`.

## Backend coverage checklist (for follow-up agents)

Status as of 2026-07-16. The math is backend-independent: every SSB backend
builds `G(q,k)` as the scan-space FFT of real intensity traces, so the
Hermitian identity `G(-q,k) = conj(G(q,k))` holds everywhere, and the
per-BF-pixel snorm16/int16 block quantization transfers directly.

| Optimization | WebGPU (`quantem.widget/js/showptycho/webgpu-ssb.ts`) | CUDA (`quantem.gpu/src/quantem/gpu/ssb/engine.py`) | MPS (`quantem.gpu/src/quantem/gpu/ssb/mps.py`) |
|---|---|---|---|
| Hermitian half-plane G(q,k) (2x, bit-exact, faster) | **DONE — default** (`?gqk=full` opt-out) | TODO — `self.G_qk` is full-plane complex64; also the streaming `result_buffer`/staging buffers (batch x bf x scan^2 x c64) would halve | TODO — `mx.complex64` full plane; gamma kernels at mps.py:430-440 already compute conj explicitly, mirror fetch slots in there |
| snorm16/int16 block-quantized G (4x, ~1e-4 rad error) | **DONE — opt-in** `?gqk=herm16` | TODO — cupy int16 pairs + per-BF f32 scale; dequant inside the variance/correction kernels | TODO — mx int16 + scale; check MLX gather perf before committing |
| VRAM budget clamp on BF count | TODO — `FULL_STACK_GPU_BUDGET_BYTES` still dead code | n/a (96 GB workstation assumption baked in; revisit for L40S) | TODO — unified memory, clamp matters most on 8-16 GB Macs |
| Streamed initial build (bounded peak, not just resident) | TODO — needs real-u16 time-domain gather or per-chunk re-decode | n/a today | TODO |

Verification recipe for a port (what was used here): compute the same
reconstruction with the optimization off and on (same aberrations, same BF
set), assert `max|dphase|` is 0.0 for Hermitian and < ~1e-3 of the phase span
for int16; then compare per-drag wall time — Hermitian must not be slower
(it reads half the bytes; if it is slower, the mirror fetch broke coalescing).
Raw parity harness for the WebGPU case: CDP + `globalThis.__quantemSsbLast`
(this doc's Setup section).

## Scan-size sweep (added same day)

Full kernel sweep across all supported scan sizes, three modes each, on phil
(`apple/metal-3`). 128/256/1024 are synthetic Arina-style masters written with
`quantem.gpu.io.save` (uint16, 48x48 detector, disk + gradient-shift phase
object, semiangle 8 mrad, det sampling 1 mrad/px); 512 is the real
experimental row from the table above. Preview BF (0.30); activeBf ~60 for
the synthetic sets, 408 for real 512. gpuMs is the first reconstruct (launch-
dominated at small BF counts; the 512-real row is the bandwidth-relevant one).

| scan | mode | resident | gpuMs | max abs dphase vs full | rms | phase span |
|---|---|---|---|---|---|---|
| 128 | full | 8.0 MB | 4.7 | — | — | 1.002 rad |
| 128 | herm | 4.1 MB | 4.5 | **0.0** | 0.0 | |
| 128 | herm16 | 2.0 MB | 4.8 | 7.0e-4 (0.07 % span) | 1.5e-4 | |
| 256 | full | 31.5 MB | 8.5 | — | — | 0.813 rad |
| 256 | herm | 15.9 MB | 5.8 | **0.0** | 0.0 | |
| 256 | herm16 | 7.9 MB | 5.3 | 1.3e-3 (0.16 % span) | 3.1e-4 | |
| 512 (real) | full | 856 MB | 18.2 | — | — | 0.075 rad |
| 512 (real) | herm | 429 MB | 16.4 | **0.0** | 0.0 | |
| 512 (real) | herm16 | 215 MB | 12.6 | 1.2e-4 (0.16 % span) | 2.6e-5 | |
| 1024 | full | 495 MB | 13.1 | — | — | 0.459 rad |
| 1024 | herm | 248 MB | 12.5 | **0.0** | 0.0 | |
| 1024 | herm16 | 124 MB | 11.6 | 6.2e-3 (1.35 % span) | 1.3e-3 | |

Conclusions:

- **herm is bit-exact at every supported size** (128/256/512/1024, synthetic
  and real data) and never slower. Safe as the unconditional default.
- **herm16 error grows with scan size**: 0.07 % of span at 128 up to 1.35 %
  at 1024. Cause: one snorm16 scale per BF pixel spans the whole q-plane, and
  the dynamic range inside G(q,k) (DC-dominated peak vs weak high-q tail)
  widens with n, so a single per-pixel scale under-resolves the tail.
  Recommendation: herm16 is comfortably below shot noise up to 512; at 1024
  treat it as a preview mode, or implement **per-q-row block scales**
  (n scales per BF pixel instead of 1, +0.4 % memory) to pull the error back
  down — noted as the follow-up for whoever extends the quantization.
- Repro: masters under `/home/owner/ssd/tmp/claude-1000/ssb_sweep/`, harness
  `sweep_run.py` in the session scratchpad, per-mode viewer selection via
  `?gqk=`.

## Lessons learned (process, not just numbers)

1. **Ask the symmetry question before the hardware question.** The 2x win here
   did not come from a faster kernel — it came from noticing the input is real,
   so half the stored spectrum was a mathematical mirror. Physics/math
   equivalences (Hermitian symmetry, band limits, separability, known output
   realness) reduce the PROBLEM; occupancy and coalescing only speed up
   whatever problem is left. Always ask first: what symmetry, invariance, or
   physical constraint makes part of this data or compute redundant?
2. **Bandwidth-bound loops convert memory wins into speed wins for free.** The
   per-drag reduce re-reads all of G(q,k) every slider move, so halving the
   bytes halved the traffic — herm was faster, not merely smaller. When a loop
   is bandwidth-bound, compression IS optimization.
3. **Quantize after per-block scaling, in the domain with bounded dynamic
   range.** snorm16 works because each BF pixel gets its own scale. The same
   16 bits as raw float16 would have failed (mantissa loss across ~1e6 dynamic
   range). And the residual error law is set by the dynamic range INSIDE each
   block - which grows with scan size - hence the 1024^2 degradation and the
   per-q-row-scale follow-up.
4. **Bit-exactness is testable and worth demanding.** Expected ~1e-7 rounding
   differences from the mirror fetch; measured literally 0.0 because radix-2
   FFT rounding is itself conjugate-symmetric for real input. A tolerance-free
   `array_equal` assertion is a far stronger regression net than atol=1e-6.
5. **Measure the active set, not the labeled set.** UI said 3941 BF pixels;
   only 408 carried nonzero aperture weight. Memory projections from labels
   were 10x off. Instrument the engine (resident-bytes counter, `__quantemSsbLast`
   hook) instead of computing footprints from UI numbers.
6. **A parity harness is one page of code.** URL-param mode switch + a
   globalThis hook holding the raw Float32Array + CDP pull + numpy compare.
   Built once, it validated the default flip, the sweep, and will validate the
   CUDA/MPS ports.
7. **Verify the adapter before believing any GPU number** (`adapter.info` must
   not be SwiftShader), and serve folder exports with a Range-capable server -
   two silent failure modes that produce plausible-looking nonsense.
