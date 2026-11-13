import { describe, expect, it } from "vitest";
import { extractFloat32 } from "./format";

describe("extractFloat32", () => {
  it("ignores trailing base64-safe pad bytes when the expected float count is known", () => {
    const values = new Float32Array([0, 0.5, 1]);
    const padded = new Uint8Array(values.byteLength + 2);
    padded.set(new Uint8Array(values.buffer));

    const parsed = extractFloat32(new DataView(padded.buffer), values.length);

    expect(Array.from(parsed ?? [])).toEqual([0, 0.5, 1]);
  });

  it("decodes expected floats from an offset DataView with trailing pad bytes", () => {
    const values = new Float32Array([3, 4, 5]);
    const padded = new Uint8Array(3 + values.byteLength + 2);
    padded.set(new Uint8Array(values.buffer), 3);

    const parsed = extractFloat32(new DataView(padded.buffer, 3), values.length);

    expect(Array.from(parsed ?? [])).toEqual([3, 4, 5]);
  });

  it("rejects unaligned byte counts when no expected float count is provided", () => {
    const values = new Float32Array([1, 2]);
    const padded = new Uint8Array(values.byteLength + 1);
    padded.set(new Uint8Array(values.buffer));

    expect(extractFloat32(padded)).toBeNull();
  });
});

describe("extractVolumeArray", () => {
  it("views the bytes in the named integer type without copying", async () => {
    const { extractVolumeArray, asFloat32 } = await import("./format");
    const values = new Int16Array([-32768, -1, 0, 1, 32767, 12]);
    const view = extractVolumeArray(new DataView(values.buffer), "int16", 6);
    expect(view).toBeInstanceOf(Int16Array);
    expect(Array.from(view!)).toEqual(Array.from(values));
    expect(view!.buffer).toBe(values.buffer);
    // every 16-bit integer is exact in float32
    const floats = asFloat32(view!);
    expect(floats).toBeInstanceOf(Float32Array);
    expect(Array.from(floats)).toEqual(Array.from(values));
    expect(asFloat32(floats)).toBe(floats);
  });

  it("keeps uint8 as one byte per value and reads an unknown type name as float32", async () => {
    const { extractVolumeArray } = await import("./format");
    const bytes = new Uint8Array([0, 1, 254, 255]);
    const view = extractVolumeArray(bytes, "uint8");
    expect(view).toBeInstanceOf(Uint8Array);
    expect(view!.byteLength).toBe(4);
    expect(extractVolumeArray(new Float32Array([1.5, -2]).buffer, "float64")).toBeInstanceOf(Float32Array);
  });

  it("copies only when the bytes are misaligned for the type", async () => {
    const { extractVolumeArray } = await import("./format");
    const padded = new Uint8Array(9);
    new DataView(padded.buffer).setUint16(1, 513, true);
    new DataView(padded.buffer).setUint16(3, 40000, true);
    const view = extractVolumeArray(padded.subarray(1, 5), "uint16", 2);
    expect(Array.from(view!)).toEqual([513, 40000]);
  });
});
