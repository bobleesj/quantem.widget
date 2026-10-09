// @ts-expect-error Vitest raw source import is not part of the widget bundle.
import source from "./index.tsx?raw";
import { expect, test } from "vitest";

// A compare pass awaits its range readback after adopting its buffers. If a
// newer pass starts meanwhile, it adopts its own buffers and the engine retires
// these; recording them as the next drag's delta base would write destroyed
// buffers. The pass must check its generation between the two.
test("a compare pass superseded during its range readback does not record the delta base", () => {
  const body = source.slice(source.indexOf("const updateRoiCompareGpuSlots"), source.indexOf("if (await updateRoiCompareGpuSlots())"));
  const readback = body.indexOf("await engine.computeRangeBatch(batchSlots)");
  const record = body.indexOf("compareIncrementalRef.current = {");
  const guard = body.indexOf("generation !== compareViGen", readback);
  expect(readback).toBeGreaterThan(0);
  expect(guard).toBeGreaterThan(readback);
  expect(guard).toBeLessThan(record);
});
