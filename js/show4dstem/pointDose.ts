type PointDose = {
  row?: number; col?: number; frames?: number; frame_dose_e_per_A2?: number; dose_e_per_A2?: number;
};

/** The panel samples acquisition totals separately from the viewer's frame stream. */
export function measuredPointDose(value: PointDose | undefined, frame: { point?: boolean; region?: number[] } | undefined) {
  const area = frame?.region;
  if (!frame?.point || !value || !area || area.length !== 4) return null;
  const {row, col, frames, frame_dose_e_per_A2: dose, dose_e_per_A2: total} = value;
  if (row !== area[0] || col !== area[1] || area[2] !== row + 1 || area[3] !== col + 1
      || !Number.isInteger(row) || !Number.isInteger(col) || row < 0 || col < 0
      || typeof frames !== "number" || !Number.isInteger(frames) || frames < 1
      || typeof dose !== "number" || !Number.isFinite(dose) || dose < 0
      || typeof total !== "number" || !Number.isFinite(total) || total < 0) return null;
  return {row, col, frames, dose, total};
}
