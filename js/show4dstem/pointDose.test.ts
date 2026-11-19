import { describe, expect, it } from "vitest";
import { measuredPointDose } from "./pointDose";

const point = {point:true, region:[12,37,13,38]};
const snapshot = {row:12,col:37,frames:4,frame_dose_e_per_A2:3.25,dose_e_per_A2:13};
describe("Point dose", () => {
  it("reports frame and cumulative exposure at the acquired position", () => {
    expect(measuredPointDose(snapshot, point)).toEqual({row:12,col:37,frames:4,dose:3.25,total:13});
  });
  it("hides pending coordinates, ordinary scans, empty snapshots, and unavailable calibration", () => {
    expect(measuredPointDose(snapshot, {...point,region:[13,37,14,38]})).toBeNull();
    expect(measuredPointDose(snapshot, {...point,point:false})).toBeNull();
    expect(measuredPointDose({},point)).toBeNull();
    expect(measuredPointDose({...snapshot,frame_dose_e_per_A2:NaN},point)).toBeNull();
    expect(measuredPointDose({...snapshot,frames:0},point)).toBeNull();
  });
});
