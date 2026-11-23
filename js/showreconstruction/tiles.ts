// What the stage tiles say about a run, kept apart from drawing so it is tested: whether its two judges disagree.

/** The ≠ mark: held-out (lower is better) and the image score (higher is better) moved in opposite directions against
 *  the run before, so the two judges disagree; null when they agree or a change is missing. */
export function disagreement(heldOutChange: number | null | undefined, scoreChange: number | null | undefined): string | null {
  if (heldOutChange == null || scoreChange == null || heldOutChange === 0 || scoreChange === 0) return null;
  if (heldOutChange < 0 && scoreChange < 0) return "held-out better, image score worse";
  if (heldOutChange > 0 && scoreChange > 0) return "held-out worse, image score better";
  return null;
}
