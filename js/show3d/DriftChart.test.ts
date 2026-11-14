import { describe, expect, it } from "vitest";
import { nearestFrame } from "./DriftChart";

describe("drift chart hover", () => {
  it("reads the measured frame nearest the pointer", () => {
    const times = [0, 1.9, 4.1, 6.0];
    expect(nearestFrame(times, -3)).toBe(0);
    expect(nearestFrame(times, 1.2)).toBe(1);
    expect(nearestFrame(times, 3.2)).toBe(2);
    expect(nearestFrame(times, 99)).toBe(3);
  });
});
