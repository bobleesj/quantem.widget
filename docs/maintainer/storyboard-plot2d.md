# Plot2D storyboard

Use the [Plot2D tutorial](../tutorials/plot2d.ipynb) for the deterministic
calibration fixture. Repeat with a representative measured or computed scalar
map for scientific signoff. These are required stories, not completed results.

| ID | Action | Required evidence |
|---|---|---|
| P2D-01 | Hover low/high row and column bins | Physical bin centers and original float64 values match the input; row zero is at the bottom. |
| P2D-02 | Wheel zoom in/out, drag, then Reset View | Scientific pixels and axis limits change together; reset restores the full grid; source values stay unchanged. |
| P2D-03 | Change Color on one of two plots | Map and colorbar change together, numerical limits stay fixed, and the other plot stays unchanged. |
| P2D-04 | Move the tutorial angle slider; replace values with set_data | Reading line moves without resending data; replacement map updates without changing viewport or color limits. |
| P2D-05 | Use narrow/wide layouts, light/dark notebook themes | Labels, menus and axes remain readable; max_width is respected; no clipped controls. |
| P2D-06 | Save PNG, export a Matplotlib figure | Both exports show the current axes, values, viewport, color limits and reading line. |
| P2D-07 | Save notebook after interaction, close and reopen | Test default static preview and opt-in save_state=True separately. Default snapshot omits data_bytes; opt-in preserves float64 arrays. Saved view restores without a kernel where supported; payload size and fallback freshness are stated. |
| P2D-08 | Rapidly replace maps/colormaps while hovering; close during rendering | Hover corresponds to the visible map; stale asynchronous work cannot paint later; resources are released without errors. |

Record browser/adapter, map shape and dtype, screenshots before/after, console
errors, first-paint time and gesture-to-paint latency. Python state tests and
cell execution alone are not browser signoff.
