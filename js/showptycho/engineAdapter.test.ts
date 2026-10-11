import { describe, expect, it } from "vitest";

import type { WebGPUSSBBackend } from "../.generated/engine/ssb/webgpu/backend";
import {
  BROWSER_CALIBRATION_SCHEMA, engineCalibrationJson, engineSourceJson, higherOrderAzimuthKey, higherOrderMagnitudeKey, reconstruct,
} from "./engineAdapter";

const CALIBRATION = {
  schema: BROWSER_CALIBRATION_SCHEMA, gpu_memory_gb: 4.5, precision: { real_dtype: "float32", complex_dtype: "complex64" },
  g_shape: [3, 128, 128], dc_value: [1, 0], wavelength_A: 0.0197, num_bright_field_pixels: 3,
  bright_field_k_row: [0.1, 0.2, 0.3], bright_field_k_col: [0.4, 0.5, 0.6], scan_q_row: [0, 0.5], scan_q_col: [0, 0.25],
  semiangle_mrad: 20, detector_sampling_mrad: [5, 4], scan_detector_rotation_deg: 345.5, detector_transposed: false,
};

describe("the vocabulary at the engine boundary", () => {
  it("hands the engine its own calibration names, each value once", () => {
    const engine = JSON.parse(engineCalibrationJson(JSON.stringify(CALIBRATION)));
    expect(engine.num_bf).toBe(3);
    // the engine's x is the scan row axis
    expect([engine.kx_bf, engine.ky_bf, engine.qx_1d, engine.qy_1d]).toEqual([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0, 0.5], [0, 0.25]]);
    expect(engine.semiangle_rad).toBeCloseTo(0.02, 12);
    expect(engine.angular_sampling_rad[0]).toBeCloseTo(0.005, 12);
    expect(engine.angular_sampling_rad[1]).toBeCloseTo(0.004, 12);
    // the physical angle goes in whole: a folded angle without its flag was 180 degrees off for a reversed scan
    expect(engine.rotation_angle_deg).toBe(345.5);
    expect(engine.wavelength_A).toBe(0.0197);
  });

  it("refuses a calibration of another layout with the reason", () => {
    const earlier = { schema_version: 1, kind: "showptycho_webgpu_folder", num_bf: 3 };
    expect(() => engineCalibrationJson(JSON.stringify(earlier))).toThrow(/reads quantem\.ssb-browser-calibration\/2\. Export the folder again/);
  });

  it("names the source of the bright-field counts in snake_case on the wire", () => {
    const source = { kind: "bf_columns", url: "source/bf_columns.qem", dtype: "uint16", encoding: "qem", plane: 16384,
      num_bright_field_pixels: 3, scan_shape: [128, 128], bytes_per_bright_field_pixel: 32768, bits_per_value: 16 };
    expect(JSON.parse(engineSourceJson(JSON.stringify(source)))).toEqual({
      kind: "bf_columns", url: "source/bf_columns.qem", dtype: "uint16", encoding: "qem", plane: 16384,
      numBf: 3, scanShape: [128, 128], bytesPerBf: 32768, bitsPerValue: 16,
    });
  });

  it("converts nm to Angstrom and degrees to radians once, at the call into the engine", async () => {
    const calls: unknown[][] = [];
    const engine = { reconstruct: (...args: unknown[]) => { calls.push(args); return Promise.resolve(null); } } as unknown as WebGPUSSBBackend;
    await reconstruct(engine, {
      aberrations: { C10_nm: 3, C12_nm: 1, phi12_deg: 90 },
      higherOrderAberrations: { [higherOrderMagnitudeKey("C21")]: 5, [higherOrderAzimuthKey("C21")]: 45, [higherOrderMagnitudeKey("C30")]: 2 },
      scanDetectorRotationDeg: 10, upsamplingFactor: 2, numBrightFieldPixels: 7, reducedBrightField: true, computeLoss: false,
      sample: { tiltRowMrad: 3, tiltColMrad: -4, depthSpreadNm: 12 },
    });
    const [c10, c12, phi12, options] = calls[0] as [number, number, number, Record<string, unknown>];
    expect([c10, c12]).toEqual([30, 10]);
    expect(phi12).toBeCloseTo(Math.PI / 2, 12);
    expect(options).toEqual({
      preview: true, bfCount: 7, computeLoss: false, rotationDeg: 10, upsample: 2,
      higherOrder: { C21_mag: 50, C21_angle: 45, C30: 20 },
      // the depth spread of the model, which the engine calls thickness, in Angstrom
      sample: { tiltRowMrad: 3, tiltColMrad: -4, thickness: 120 },
    });
  });

  it("sends no sample for standard SSB (depth spread 0)", async () => {
    const calls: unknown[][] = [];
    const engine = { reconstruct: (...args: unknown[]) => { calls.push(args); return Promise.resolve(null); } } as unknown as WebGPUSSBBackend;
    await reconstruct(engine, {
      aberrations: { C10_nm: 0, C12_nm: 0, phi12_deg: 0 }, higherOrderAberrations: {}, scanDetectorRotationDeg: 0,
      upsamplingFactor: 1, numBrightFieldPixels: 7, reducedBrightField: false, computeLoss: true,
      sample: { tiltRowMrad: 3, tiltColMrad: -4, depthSpreadNm: 0 },
    });
    expect((calls[0][3] as Record<string, unknown>).sample).toBeUndefined();
  });
});
