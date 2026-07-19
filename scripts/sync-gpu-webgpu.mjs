// Sync canonical WebGPU browser-compute sources from quantem.gpu into the
// widget frontend tree before bundling. Browsers need TypeScript/WGSL bundled
// into the anywidget JS artifact, but quantem.gpu owns the reusable kernel
// source.

import { spawnSync } from "child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import path from "path";

export function syncGpuWebgpuSources({ targetDir = "js/engine" } = {}) {
  const python = process.env.PYTHON || "python3";
  const code = `
import json
from quantem.gpu import webgpu
print(json.dumps({name: webgpu.source_text(name) for name in webgpu.source_names()}))
`;
  const result = spawnSync(python, ["-c", code], {
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
  });
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim();
    throw new Error(
      "Unable to sync WebGPU sources from quantem.gpu. Install quantem.gpu in " +
      `the active Python environment or set PYTHON explicitly. ${detail}`
    );
  }

  const sources = JSON.parse(result.stdout);
  mkdirSync(targetDir, { recursive: true });
  let changed = 0;
  let unchanged = 0;
  for (const [name, text] of Object.entries(sources)) {
    const dest = path.join(targetDir, name);
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
