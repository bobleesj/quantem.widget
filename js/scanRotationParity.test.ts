import { describe, expect, it } from "vitest";
import fixture from "./.generated/engine/parity/scan_rotation_v1.json";
import {
  SCAN_QUARTER_TURN_WGSL,
  type ScanQuarterTurns,
  scanQuarterTurnOutputShape,
  scanQuarterTurnSourceIndex,
} from "./.generated/engine/geometry/compute/webgpu/quarter-turn";

describe("shared scan-rotation gold fixture", () => {
  it("maps every detector pattern with the canonical row-column convention", () => {
    const [scanRows, scanColumns, detectorRows, detectorColumns] = fixture.source.shape;
    const detectorPixels = detectorRows * detectorColumns;

    for (const testCase of fixture.cases) {
      const quarterTurns = testCase.quarter_turns_counterclockwise as ScanQuarterTurns;
      const [outputRows, outputColumns] = scanQuarterTurnOutputShape(
        scanRows,
        scanColumns,
        quarterTurns,
      );
      expect([outputRows, outputColumns, detectorRows, detectorColumns]).toEqual(
        testCase.output_shape,
      );
      for (let outputRow = 0; outputRow < outputRows; outputRow++) {
        for (let outputColumn = 0; outputColumn < outputColumns; outputColumn++) {
          const sourceScan = scanQuarterTurnSourceIndex(
            outputRow,
            outputColumn,
            scanRows,
            scanColumns,
            quarterTurns,
          );
          const outputScan = outputRow * outputColumns + outputColumn;
          const sourceStart = sourceScan * detectorPixels;
          const outputStart = outputScan * detectorPixels;
          expect(
            testCase.expected_values.slice(outputStart, outputStart + detectorPixels),
          ).toEqual(
            fixture.source.values.slice(sourceStart, sourceStart + detectorPixels),
          );
        }
      }
    }
  });

  it("keeps the same mapping in the hardware shader contract", () => {
    expect(SCAN_QUARTER_TURN_WGSL).toContain("sourceRow = outputColumn");
    expect(SCAN_QUARTER_TURN_WGSL).toContain(
      "sourceRow = parameters.sourceRows - 1u - outputColumn",
    );
    expect(SCAN_QUARTER_TURN_WGSL).toContain(
      "sourceScan * parameters.wordsPerScan + wordInScan",
    );
  });
});
