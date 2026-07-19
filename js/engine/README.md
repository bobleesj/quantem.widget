# WebGPU Engine Copy

The reusable WebGPU browser-compute sources in this directory are synced from
`quantem.gpu.webgpu` before `npm run build`.

Edit the canonical sources in `quantem.gpu/src/quantem/gpu/webgpu/`, then run:

```bash
npm run sync:webgpu
```

`quantem.widget` owns the UI, bundling, and exported HTML runtime. The shared
kernel math and browser compute engine source belong in `quantem.gpu`,
including Show4DSTEM WebGPU IO/reductions and ShowPtycho WebGPU SSB.
