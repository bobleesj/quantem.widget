import {WebGPUDirectBackend} from "../.generated/engine/direct_ptycho/webgpu/backend";
import type {RansResidentSet} from "../.generated/engine/detector/webgpu/rans";

export async function standalonePhase(folder: string, resident: RansResidentSet, progress: (s: string) => void) {
  const base = new URL(folder, document.baseURI);
  if (base.origin !== location.origin) throw Error("Serve the calibrated Direct Ptycho folder beside the viewer.");
  const json = async (path: string) => {
    const response = await fetch(new URL(path, base));
    if (!response.ok) throw Error("Missing Direct Ptycho calibration: " + path);
    return response.json();
  };
  const [full, crop, manifest] = await Promise.all([
    json("full.json"), json("crop.json"), json("source.json"),
  ]);
  const shape = manifest.scan_shape;
  if (crop.g_shape[1] !== 128 || crop.g_shape[2] !== 128 || shape.some((v: number) => v < 128))
    throw Error("Provide calibration for the central 128 × 128 scan region.");
  const region = [Math.floor((shape[0]-128)/2), Math.floor((shape[1]-128)/2), 128, 128];
  const solver = new WebGPUDirectBackend(JSON.stringify(crop), {resident, json: JSON.stringify({
    ...manifest, region, num_bright_field_pixels: crop.num_bright_field_pixels,
  })});
  let fit;
  const start = performance.now();
  let prepMs = 0;
  try {
    progress("Preparing the central 128 × 128 fit");
    const preparation = solver.modelAgreement(0,0,0,{sample:{depthSpreadA:100}}).then(() => {
      prepMs = performance.now()-start;
    });
    const policy = import(/* webpackIgnore: true */ new URL("load.mjs", base).href);
    // The browser fit worker can initialize while WebGPU transforms the crop.
    const [, result] = await Promise.all([preparation, policy.then(p => p.fit(solver, crop, progress, preparation))]);
    fit = result;
  } finally { solver.close(); }
  progress("Reconstructing the full-field phase");
  const engine = new WebGPUDirectBackend(JSON.stringify(full), {resident, json: JSON.stringify(manifest)});
  let percent = -1;
  engine.setProgressHandler(event => {
    if (event.stage !== "fetch" || !event.total) return;
    const next = Math.min(95, 5 * Math.floor(20 * (event.current ?? 0) / event.total));
    if (next !== percent) { percent = next; progress(`Full-field phase · ${percent}%`); }
  });
  try {
    const [c10,c12,phi,opts] = fit.args;
    const phase = await engine.reconstructPhase(c10,c12,phi,{...opts, upsample:"auto", phaseDamp:0.3});
    return {phase, loadMs:resident.readyMs, prepMs, trialsMs:fit.trials_s*1000, refinementMs:fit.refinement_s*1000, totalMs:performance.now()-start, splitRms:fit.split_rms};
  } finally { engine.close(); }
}
