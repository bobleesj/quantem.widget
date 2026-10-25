# ShowGPU

Monitor NVIDIA VRAM, utilization, temperature and power alongside system RAM
while a notebook runs. The readings describe the **notebook server**, which
may be a different machine from the browser.

```python
from quantem.widget import ShowGPU

monitor = ShowGPU(users=True, size="medium")
monitor
```

Place this cell above acquisition or reconstruction. The sampler runs in a
background thread and updates once per second. It reads the NVIDIA driver
through `quantem.gpu.device.status()` without creating a CUDA context,
allocating VRAM, synchronizing work, or releasing caches. GPU rows include
every process using that physical device. Driver indices are not CUDA logical
indices; any `CUDA_VISIBLE_DEVICES` mask is shown separately.

`users=True` adds each compute user's reported VRAM and process count.
Click a user row or focus it and press Enter to expand the process list below
it: process ID, executable name and reported VRAM. Expanded rows stay open as
samples update. Only executable names are read, not command-line arguments.
`users=False` hides that breakdown and skips the process query. User totals
exclude graphics processes and driver allocations, so they need not sum to
device usage. Inaccessible process ownership is labeled **Unknown user**;
unsupported memory readings remain **unavailable**.

`size` accepts `"small"`, `"medium"`, `"large"` and `"xlarge"`. For this
text-based monitor, these select 11, 13, 15 and 18 CSS-pixel text respectively,
with matching controls and responsive cards. Choose **S / M / L / XL** or
toggle **Users** directly in the widget without rerunning the cell. The same
settings are mutable Python attributes: `monitor.size = "large"` and
`monitor.users = False`.

`ShowGPU(interval=2.0)` reduces sampling frequency. Pause/Resume controls
automatic updates; Refresh requests one sample even while paused. Browser
commands may wait for a busy kernel. A native call holding Python's GIL can
also delay samples; **No recent update** labels stale readings. Saved output
does not count as live until new samples arrive.

When finished, use `monitor.close()`. Before rerunning the creation cell, close
the previous monitor to stop its sampler. System RAM remains visible without
NVIDIA telemetry. Apple unified GPU memory is not reported as dedicated VRAM.
Use matching development versions of quantem.gpu and quantem.widget when
trying this API before release.

Memory is shown in GiB (2³⁰ bytes). NVIDIA's driver reserve can make reported
used plus free smaller than total. RAM used means total minus available;
available includes reclaimable memory. The kernel's resident host RAM is
shown separately and is not its VRAM allocation. Recent VRAM traces retain
at most 120 samples per device in the browser.

## Interactive controls

| Control | Effect | Expected observation |
| --- | --- | --- |
| Pause / Resume | `running` | Sampling pauses or resumes |
| Refresh | one sample | Sample counter and timestamp advance |
| Users | `users` | Compute-user VRAM breakdown shows or hides |
| User row | expand / collapse | Process IDs, names and VRAM appear underneath |
| S / M / L / XL | `size` | Text, controls and card layout resize |

```{eval-rst}
.. autoclass:: quantem.widget.showgpu.ShowGPU
   :members: refresh, close
```
