import { describe, expect, it } from "vitest";
import { disagreement } from "./tiles";

describe("the ≠ mark", () => {
  it("marks a run whose held-out and image score move in opposite directions", () => {
    expect(disagreement(-0.0004, -0.02)).toBe("held-out better, image score worse");
    expect(disagreement(0.0001, 0.01)).toBe("held-out worse, image score better");
  });
  it("stays quiet when both judges agree or a change is missing", () => {
    expect(disagreement(-0.0004, 0.02)).toBeNull();
    expect(disagreement(0.0004, -0.02)).toBeNull();
    expect(disagreement(null, 0.02)).toBeNull();
    expect(disagreement(-0.0004, undefined)).toBeNull();
  });
});
