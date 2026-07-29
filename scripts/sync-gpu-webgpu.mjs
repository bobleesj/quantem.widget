// Sync canonical WebGPU browser-compute sources from quantem.gpu into the
// widget frontend tree before bundling. Browsers need TypeScript/WGSL bundled
// into the anywidget JS artifact, but quantem.gpu owns the reusable kernel
// source.

import { spawnSync } from "child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

export function syncGpuWebgpuSources({ targetDir = "js/.generated/engine" } = {}) {
  const outputDir = path.isAbsolute(targetDir) ? targetDir : path.join(repoRoot, targetDir);
  const python = process.env.PYTHON || "python";
  const code = `
import json
from quantem.gpu import webgpu
print(json.dumps({name: webgpu.source_text(name) for name in webgpu.source_names()}))
`;
  const runExport = (env = process.env) => spawnSync(python, ["-c", code], {
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
    env,
  });

  let result = runExport();
  if (result.status !== 0) {
    const home = process.env.HOME || "";
    const srcDirs = [
      process.env.QUANTEM_GPU_SRC,
      path.resolve(repoRoot, "../quantem.gpu/src"),
      path.resolve(repoRoot, "../../quantem.gpu/src"),
      home ? path.resolve(home, "repos/quantem.gpu/src") : "",
      home ? path.resolve(home, "quantem.gpu/src") : "",
    ].filter((srcDir) => srcDir && existsSync(srcDir));
    if (srcDirs.length) {
      const pythonPath = [
        ...srcDirs,
        process.env.PYTHONPATH || "",
      ].filter(Boolean).join(path.delimiter);
      result = runExport({ ...process.env, PYTHONPATH: pythonPath });
    }
  }
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim();
    throw new Error(
      "Unable to sync WebGPU sources from quantem.gpu. Install quantem.gpu in " +
      "the active Python environment, set PYTHON explicitly, or set " +
      `QUANTEM_GPU_SRC to the quantem.gpu/src directory. ${detail}`
    );
  }

  const sources = JSON.parse(result.stdout);
  mkdirSync(outputDir, { recursive: true });
  for (const legacyName of [
    "bslz4.ts",
    "compute.ts",
    "device.ts",
    "fft-shader.ts",
    "h5reader.ts",
    "lazy.ts",
    "local-h5.ts",
    "showptycho-ssb.ts",
  ]) {
    const legacyPath = path.join(outputDir, legacyName);
    if (existsSync(legacyPath)) rmSync(legacyPath);
  }
  let changed = 0;
  let unchanged = 0;
  for (const [name, text] of Object.entries(sources)) {
    const dest = path.join(outputDir, name);
    mkdirSync(path.dirname(dest), { recursive: true });
    const current = existsSync(dest) ? readFileSync(dest, "utf8") : null;
    if (current === text) {
      unchanged += 1;
      continue;
    }
    writeFileSync(dest, text, "utf8");
    changed += 1;
  }
  console.log(
    `synced quantem.gpu.webgpu -> ${targetDir} (${changed} updated, ${unchanged} unchanged)`
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  syncGpuWebgpuSources();
}
