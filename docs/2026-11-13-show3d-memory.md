# Show3DSlices and Show3D memory on a real stack (13 Nov 2026)

## Question

How much memory do Show3DSlices and Show3D hold for one stack, in the kernel, in the page and on the GPU,
and how much of it can go without changing any value or pixel?

## Setup

- Data: a real drift series, 24 frames of 2048 x 2048, stored as uint8 (100 MB on disk and in NumPy).
- `Show3DSlices(data)` and `Show3D(data)` in a live JupyterLab notebook, one widget per fresh kernel and fresh
  headless Chrome, 1512 x 982 at device pixel ratio 2, WebGPU on the hardware adapter (`apple metal-3`).
- Kernel: unique set size after the widget is shown, minus the size after `import quantem.widget` and
  `np.load`. Page: ArrayBuffer bytes from `Runtime.getHeapUsage` after a forced collection. GPU: bytes of
  live WebGPU buffers and textures, summed by wrapping `createBuffer`, `createTexture` and `destroy`.
  Process peaks: resident size of Chrome's renderer and GPU processes, sampled every 50 ms.
- Chrome is driven over its debugging port without the Network domain: a driver that mirrors websocket
  frames cannot carry a comm message of several hundred MB.
- Pixels: an FNV hash of every 2D canvas of the widget, before and after.

## Numbers

| | Kernel | Page arrays | GPU buffers and textures | Renderer peak | GPU process peak |
|---|---|---|---|---|---|
| Show3DSlices before | 908 MB | 432 MB | 504 MB | 2983 MB | 790 MB |
| Show3DSlices after | 3 MB | 130 MB | 202 MB | 1657 MB | 391 MB |
| Show3D before | 907 MB | 446 MB | 403 MB | 2746 MB | 581 MB |
| Show3D after | 129 MB | 295 MB | 117 MB | 1585 MB | 292 MB |

A page with no widget holds 8 MB of arrays; its renderer process sits at about 765 MB and its GPU process
at about 140 MB. All 7 Show3DSlices canvases and all 3 Show3D canvases hash the same before and after.

## What changed

- Show3DSlices widened every volume to float32 (4 bytes per voxel) in the kernel, copied it again into the
  synced bytes, held it as float32 in the page and as an `r32float` texture, plus an 8-bit 3D texture for the
  volume view. It now keeps uint8, int8, uint16, int16 and float32 in their own type end to end: the synced
  bytes are a view of the caller's array, the page views them as the matching typed array, and the slice
  shader reads an integer texture (`r8uint`, `r16sint`, ...) with `textureLoad`, so every value is exact.
- Both volume uploads padded each row to 256 bytes through a second full copy. `queue.writeTexture` has no
  such rule (it is a rule of buffer-to-texture copies), so the volume's own memory is uploaded, a few slices
  per call.
- Show3D widened every stack to float32 in the kernel and copied it into the synced bytes with `tobytes()`.
  It now keeps uint8, int8, uint16 and int16 stacks in their own type and syncs a view of the array. The page
  holds those bytes and converts one frame at a time to float32 (exact), keeping at most 128 MB of converted
  frames; a float32 stack needs no such cache, its frames are views. Differences, ranges and percentiles in
  the kernel are taken one frame at a time in float32, so nothing wraps and no second stack is built. In a
  bare kernel the widget adds 32 MB (46 MB at its peak); the 129 MB above includes the notebook's display.
- Show3D also held every resident frame on the GPU as a float32 storage buffer (16 MB per 2048 x 2048
  frame). An integer frame now sits on the GPU packed in its own type (4 MB for uint8). Its colormap, range,
  histogram and thumbnail passes read it through a float32 view that one compute pass fills when the frame
  is drawn (exact, as every 8- and 16-bit value is a float32); only the frame on screen and the one before
  keep that view, and frames drawn together (panels) keep theirs while they are drawn. A moving average
  converts its window straight into its scratch buffer, and the thumbnail strip converts each frame into
  one reused buffer, so neither keeps a view per frame. On this stack: 24 packed frames (96 MB) and one
  view (16 MB).
- The device asks for the adapter's `maxTextureArrayLayers` and `maxTextureDimension3D`, so a volume deeper
  than 256 slices or wider than 2048 voxels stays in a texture.

## Does it survive a GPU that stops, or a slow one?

The report that started this: Show3DSlices froze on a Windows laptop with an integrated GPU. No such
machine was at hand, so the three ways a weak GPU fails were each produced on purpose, on the hardware
adapter (`apple metal-3`), as browser tests:

- **The GPU device is lost.** `tests/browser/test_gpu_loss_recovery.py` kills Chrome's GPU process (CDP
  `Browser.crashGpuProcess`) with each viewer open. Before the fix both viewers went blank: the device is
  gone and Chrome also clears every 2D canvas. Now the title chip reads CPU, the viewer says the GPU
  stopped, Show3DSlices draws the same pixels (100% identical) and Show3D the same frame (over 99%
  identical; its canvas path rounds a few block edges of an enlarged frame differently), and the frame
  or slice slider still draws new images.
- **Frames take too long.** `tests/browser/test_show3dslices_slow_gpu.py` delays
  `queue.onSubmittedWorkDone` by 300 ms and by 900 ms, the signal of a GPU that slow. Over 90 pointer moves
  in 3 s the 3D view submits at most one frame per GPU frame time (about 10 and 4 frames, not 90), at one
  canvas pixel per CSS pixel; at rest it returns to full quality (300 ms) or stays reduced and says so (900 ms).
- **No GPU memory for the volume.** `js/display/device.test.ts`: an allocation that reports out-of-memory
  moves the page to the CPU paths with the reason.

Stress drive on the real stack, live notebook, per viewer: 600 pointer moves rotating the view, 80 wheel
ticks, 160 slider steps in about 22 s.

| | GPU device lost | Frames per second | Longest gap between frames | GPU bytes growth | Page arrays after |
|---|---|---|---|---|---|
| Show3DSlices | no | 59.4 | 83 ms | 17 MB | 130 MB |
| Show3D | no | 59.7 to 59.8 (3 runs) | 33 to 83 ms | 17 MB (the previous frame's float32 view) | 295 MB |

Not shown: a run on a Windows integrated GPU itself. The driver watchdog, the shared-memory limit and
Chrome's D3D backend are only on that machine; the tests above cover what the viewers do when those fail.

## Conclusion

Show3DSlices holds about a quarter of the page and GPU memory for 8-bit data and no kernel copy; its
renderer peak fell by 1.3 GB. Show3D holds no kernel copy, a third less in the page and a little over a
quarter on the GPU (117 MB instead of 403 MB); its renderer peak fell by 1.2 GB. An HTML export without binning now ships an integer stack in its own type
too (it shipped float32 before). The saving is for integer data: a float32 stack keeps 4 bytes per value everywhere and
only loses the kernel copies.
