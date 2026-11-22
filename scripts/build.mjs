// Bundle each widget as a self-contained ESM file.
// anywidget loads bundles via Blob URL; relative imports break in that context.
// esbuild flattens everything into one file per widget.
//
//   npm run build                     all bundles
//   npm run build -- show2d show3d    only the named bundles
//   npm run build:web                 only the web-page bundles (WEB_BUNDLES below)
//
// Every build first syncs js/.generated/engine from quantem.gpu (QUANTEM_GPU_SRC
// or an installed quantem.gpu; CI checks out the ref in gpu-ref.txt): show4dstem
// and showptycho bundle its WebGPU science kernels, and every bundle that draws
// through js/display validates borrowed count images with its contract.

import { build, context } from "esbuild";
import { copyFileSync, mkdirSync, readdirSync, renameSync, rmSync, writeFileSync } from "fs";
import { syncGpuWebgpuSources } from "./sync-gpu-webgpu.mjs";

const watch = process.argv.includes("--watch");
const baseOpts = {
  bundle: true,
  format: "esm",
  jsx: "automatic",
  target: "es2022",
  define: { "process.env.NODE_ENV": '"production"' },
  minify: true,
  sourcemap: false,
  legalComments: "none",
};
const widgets = [
  "plot2d",
  "show1d",
  "show2d",
  "show3d",
  "show3dslices",
  "show4dstem",
  "showpdf",
  "showptycho",
  "showgpu",
  "showreconstruction",
  "showsearch",
  "scancontrol",
  "diffractionsim",
];
// Framework-free bundles for web pages, built only by `npm run build:web`: one self-contained ESM
// file each, with `export default { render }`, written into the quantem.diffsim package (PyPI),
// which serves them to web pages and Dash apps. The MyST `anywidget` directive loads the same file.
const WEB_BUNDLES = [
  {
    entry: "js/diffractionsim-web/index.ts",
    outfile: "packages/quantem.diffsim/src/quantem/diffsim/static/diffraction-sim.js",
  },
];
const webOnly = process.argv.includes("--web-only");
const requested = process.argv.slice(2).filter((arg) => !arg.startsWith("--"));
const unknown = requested.filter((name) => !widgets.includes(name));
if (unknown.length) throw new Error(`unknown bundle(s): ${unknown.join(", ")}`);
const selected = requested.length ? widgets.filter((name) => requested.includes(name)) : widgets;

if (webOnly) {
  for (const web of WEB_BUNDLES) {
    const start = Date.now();
    await build({ ...baseOpts, entryPoints: [web.entry], outfile: web.outfile });
    console.log(`built ${web.outfile} (${Date.now() - start}ms)`);
  }
  process.exit(0);
}

try {
  syncGpuWebgpuSources();
} catch (error) {
  console.error(`\nNOT built: ${selected.join(", ")}\n${error.message}`);
  process.exit(1);
}

// Every file in static/ is replaced atomically: written to a temporary name beside it, then renamed into place, so a
// notebook running while a builder rebuilds always reads a whole bundle and never finds it missing (a delete-then-write
// build broke a running notebook's widgets on 21 Nov 2026).
mkdirSync("src/quantem/widget/static", { recursive: true });
function replaceAtomically(path, contents) {
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, contents);
  renameSync(temporary, path);
}
// The wheel names the quantem.gpu sources its bundles carry; quantem.widget.profile() prints it.
copyFileSync("js/.generated/engine/MANIFEST.json", "src/quantem/widget/static/engine-manifest.json.tmp");
renameSync("src/quantem/widget/static/engine-manifest.json.tmp", "src/quantem/widget/static/engine-manifest.json");

for (const name of selected) {
  const opts = {
    ...baseOpts,
    entryPoints: [`js/${name}/index.tsx`],
    outfile: `src/quantem/widget/static/${name}.js`,
  };
  if (watch) {
    const watcher = await context(opts);
    await watcher.watch();
    console.log(`watching ${name}...`);
  } else {
    const start = Date.now();
    const result = await build({ ...opts, write: false });
    for (const file of result.outputFiles) replaceAtomically(file.path, file.contents);
    console.log(`built ${name}.js (${Date.now() - start}ms)`);
  }
}

// A full build then removes the bundles no widget builds any more, after the new ones are in place: retired bundles
// cannot linger, and no current one is ever absent.
if (!requested.length && !watch) {
  const kept = new Set([...selected.map((name) => `${name}.js`), "engine-manifest.json"]);
  for (const file of readdirSync("src/quantem/widget/static")) {
    if (file.endsWith(".js") && !kept.has(file)) rmSync(`src/quantem/widget/static/${file}`, { force: true });
  }
}

if (!watch) console.log("done.");
