import * as React from "react";
import { onGPULost, webGPULostReason } from "../display/device";

/**
 * Mount `Viewer` again when the browser loses its GPU.
 *
 * A GPU process crash, a driver reset or exhausted GPU memory takes the WebGPU device and, in Chrome, the
 * pixels of every 2D canvas with it: the page is left blank. Mounting the viewer again rebuilds every
 * canvas, cache and engine from the data still in memory; with the device gone, the engines come back as
 * the JavaScript paths, which draw the same values. Synced state (slice, colormap, contrast) is in the
 * model and survives; local view state (zoom, pan) resets.
 */
export function remountOnGPULoss<Props extends object>(Viewer: React.ComponentType<Props>): React.FC<Props> {
  return function GPULossRemount(props: Props) {
    const [epoch, setEpoch] = React.useState(0);
    React.useEffect(() => {
      let timer = 0;
      const remount = () => {
        window.clearTimeout(timer);
        timer = window.setTimeout(() => setEpoch(value => value + 1), 50);
      };
      const stopWatching = onGPULost(remount);
      // The browser restores lost 2D canvases blank, a moment after the device is lost. The event does
      // not bubble, so it is caught on its way down. Only a loss of the WebGPU device counts.
      const restored = () => { if (webGPULostReason() !== null) remount(); };
      document.addEventListener("contextrestored", restored, true);
      return () => {
        window.clearTimeout(timer);
        stopWatching();
        document.removeEventListener("contextrestored", restored, true);
      };
    }, []);
    return <Viewer key={epoch} {...props} />;
  };
}
