import * as React from "react";
import MuiSlider, { type SliderProps } from "@mui/material/Slider";

/**
 * Every widget slider sits under the pointer: no position easing on the thumb, track or label.
 *
 * MUI eases a slider thumb to its new position over 150 ms and only turns that off while MUI
 * itself runs the drag. Widgets that own their drags for speed (the Show3D frame slider) never
 * tell MUI a drag is active, so the thumb trailed the cursor by ~37 px while the image was already
 * on the new frame. Import this, never `@mui/material/Slider` (Slider.test.ts enforces it).
 */
export const instantSliderSx = {
  "& .MuiSlider-thumb, & .MuiSlider-track, & .MuiSlider-valueLabel": { transition: "none" },
} as const;

const Slider = React.forwardRef<HTMLSpanElement, SliderProps>(function Slider({ sx, ...props }, ref) {
  const own = sx ? (Array.isArray(sx) ? sx : [sx]) : [];
  return <MuiSlider ref={ref} {...props} sx={[instantSliderSx, ...own]} />;
});

export default Slider;
export type { SliderProps };
