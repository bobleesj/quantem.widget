/**
 * The vendored quantem.gpu WebGPU SSB engine, spoken to in the names of the vocabulary.
 *
 * ShowPtycho's traits, messages and files use one name per concept (`scan_detector_rotation_deg`,
 * `detector_sampling_mrad`, `C10_nm` / `C12_nm` / `phi12_deg`, `C21_nm` / `phi21_deg`,
 * `num_bright_field_pixels`, `sample_tilt_mrad`, `depth_spread_nm`). The engine under
 * `js/.generated/engine` is copied from quantem.gpu and keeps that package's names, so every
 * engine name appears in this file only, beside the name it stands for. This is the one file to
 * change when the engine takes the same vocabulary; `src/quantem/widget/adapters/gpu.py` is its
 * Python counterpart.
 *
 * engine name -> vocabulary name
 *   num_bf, numBf                        -> num_bright_field_pixels
 *   kx_bf, ky_bf                         -> bright_field_k_row, bright_field_k_col (the engine's x is the scan row axis)
 *   qx_1d, qy_1d                         -> scan_q_row, scan_q_col
 *   semiangle_rad                        -> semiangle_mrad
 *   angular_sampling_rad                 -> detector_sampling_mrad
 *   rotation_angle_deg, rotationDeg      -> scan_detector_rotation_deg
 *   reconstruct(c10, c12, phi12) in Angstrom and radians -> C10_nm, C12_nm, phi12_deg
 *   higherOrder <name>_mag (Angstrom), <name>_angle (degrees) -> <name>_nm, phi<order>_deg
 *   sample tiltRowMrad, tiltColMrad, thickness (Angstrom) -> sample_tilt_mrad [row, col], depth_spread_nm
 *   upsample                             -> upsampling_factor
 *   bfCount                              -> the bright-field pixels of this reconstruction
 *   scanShape, bytesPerBf, bitsPerValue  -> scan_shape, bytes_per_bright_field_pixel, bits_per_value
 */

import type { WebGPUSSBBackend } from "../.generated/engine/ssb/webgpu/backend";
import type { WebGPUReconstructionOptions } from "../.generated/engine/ssb/webgpu/protocol";

/** The layout of `snapshots/cal.json` and of the calibration embedded in an exported page. */
export const BROWSER_CALIBRATION_SCHEMA = "quantem.ssb-browser-calibration/2";
/** The layout of `snapshots/snapshots.json`, written by the exported page. */
export const SNAPSHOTS_SCHEMA = "quantem.ssb-snapshots/1";

// The engine evaluates chi = (pi / lambda[A]) alpha^2 (C10 + ...), so it takes Angstrom where the vocabulary says nm.
const ANGSTROM_PER_NM = 10;

/** Thick-sample model of one reconstruction: sample tilt (scan frame) and depth spread; a depth spread of 0 is standard SSB. */
export interface SampleValues { tiltRowMrad: number; tiltColMrad: number; depthSpreadNm: number }

/** One reconstruction request in the vocabulary's quantities. */
export interface ReconstructionRequest {
  aberrations: { C10_nm: number; C12_nm: number; phi12_deg: number };
  /** Higher orders keyed `C21_nm`, `phi21_deg`, ...; a missing key is 0. */
  higherOrderAberrations: Record<string, number>;
  scanDetectorRotationDeg: number;
  upsamplingFactor: number;
  numBrightFieldPixels: number;
  /** True while fewer than all bright-field pixels are used (a drag). */
  reducedBrightField: boolean;
  computeLoss: boolean;
  sample: SampleValues | null;
}

/** The magnitude key of a higher-order term: `C21` -> `C21_nm`. */
export function higherOrderMagnitudeKey(name: string): string {
  return `${name}_nm`;
}

/** The azimuth key of a higher-order term: `C21` -> `phi21_deg`. */
export function higherOrderAzimuthKey(name: string): string {
  return `phi${name.slice(1)}_deg`;
}

/**
 * The calibration JSON the engine parses, from the vocabulary-named one ShowPtycho.export writes.
 * A calibration of another layout is refused with the reason: a page exported by an earlier
 * release names its keys differently, and reading it by position would reconstruct with wrong optics.
 */
export function engineCalibrationJson(calibrationJson: string): string {
  const calibration = JSON.parse(calibrationJson);
  if (calibration.schema !== BROWSER_CALIBRATION_SCHEMA) {
    throw new Error(`This page holds an SSB calibration of layout ${JSON.stringify(calibration.schema ?? null)}; `
      + `this viewer reads ${BROWSER_CALIBRATION_SCHEMA}. Export the folder again with ShowPtycho.export().`);
  }
  return JSON.stringify({
    gpu_memory_gb: calibration.gpu_memory_gb,
    precision: calibration.precision,
    g_shape: calibration.g_shape,
    dc_value: calibration.dc_value,
    wavelength_A: calibration.wavelength_A,
    num_bf: calibration.num_bright_field_pixels,
    kx_bf: calibration.bright_field_k_row,
    ky_bf: calibration.bright_field_k_col,
    qx_1d: calibration.scan_q_row,
    qy_1d: calibration.scan_q_col,
    semiangle_rad: calibration.semiangle_mrad * 1e-3,
    angular_sampling_rad: (calibration.detector_sampling_mrad as number[]).map((step) => step * 1e-3),
    rotation_angle_deg: calibration.scan_detector_rotation_deg,
  });
}

/** The bright-field source JSON the engine parses, from the snake_case one the export embeds. */
export function engineSourceJson(sourceJson: string): string {
  const source = JSON.parse(sourceJson);
  return JSON.stringify({
    kind: source.kind, url: source.url, dtype: source.dtype, encoding: source.encoding, plane: source.plane,
    numBf: source.num_bright_field_pixels, scanShape: source.scan_shape,
    bytesPerBf: source.bytes_per_bright_field_pixel, bitsPerValue: source.bits_per_value,
  });
}

/** Run one reconstruction on the engine, converting nm to Angstrom, degrees to radians and the key names at this boundary. */
export function reconstruct(engine: WebGPUSSBBackend, request: ReconstructionRequest) {
  const higherOrder: Record<string, number> = {};
  for (const [key, value] of Object.entries(request.higherOrderAberrations)) {
    if (key.startsWith("phi")) higherOrder[`C${key.slice(3, -4)}_angle`] = value;
    // the engine keys a rotationally symmetric term (C30, C50: no azimuth) by its bare name
    else higherOrder[key.slice(0, -3).endsWith("0") ? key.slice(0, -3) : `${key.slice(0, -3)}_mag`] = value * ANGSTROM_PER_NM;
  }
  const sample = request.sample && request.sample.depthSpreadNm > 0
    ? { tiltRowMrad: request.sample.tiltRowMrad, tiltColMrad: request.sample.tiltColMrad, thickness: request.sample.depthSpreadNm * ANGSTROM_PER_NM }
    : undefined;
  // the engine itself rejects an upsample other than 1, 2, 4 or 8, so the narrowing cast is safe
  const options = {
    preview: request.reducedBrightField, bfCount: request.numBrightFieldPixels, computeLoss: request.computeLoss,
    rotationDeg: request.scanDetectorRotationDeg, upsample: request.upsamplingFactor, higherOrder,
    ...(sample ? { sample } : {}),
  } as WebGPUReconstructionOptions;
  return engine.reconstruct(
    request.aberrations.C10_nm * ANGSTROM_PER_NM, request.aberrations.C12_nm * ANGSTROM_PER_NM,
    request.aberrations.phi12_deg * Math.PI / 180, options,
  );
}
