import { afterEach, describe, expect, it, vi } from "vitest";

// The adapter options the shared WebGPU device asks for on a browser reporting `platform`.
async function adapterOptions(platform?: string): Promise<GPURequestAdapterOptions | undefined> {
  vi.resetModules();
  const requestAdapter = vi.fn().mockResolvedValue(null);
  vi.stubGlobal("navigator", { gpu: { requestAdapter }, userAgentData: platform ? { platform } : undefined });
  const { getGPUDevice } = await import("./device");
  expect(await getGPUDevice()).toBeNull();
  return requestAdapter.mock.calls[0][0];
}

afterEach(() => vi.unstubAllGlobals());

describe("WebGPU adapter request", () => {
  it("leaves powerPreference out on Windows, where Chromium ignores it and warns", async () => {
    expect(await adapterOptions("Windows")).toEqual({});
  });

  it("asks for the high-performance GPU elsewhere, as on a dual-GPU Mac", async () => {
    expect(await adapterOptions("macOS")).toEqual({ powerPreference: "high-performance" });
    expect(await adapterOptions("Linux")).toEqual({ powerPreference: "high-performance" });
    expect(await adapterOptions()).toEqual({ powerPreference: "high-performance" });  // Firefox, Safari
  });
});

// A browser whose adapter hands out devices the test can lose on demand.
async function deviceWithLoss() {
  vi.resetModules();
  const losses: ((info: { reason: string; message: string }) => void)[] = [];
  const requestDevice = vi.fn(async () => ({ lost: new Promise(resolve => losses.push(resolve)) }));
  const requestAdapter = vi.fn(async () => ({ limits: {}, features: new Set(), info: { description: "Test GPU" }, requestDevice }));
  vi.stubGlobal("navigator", { gpu: { requestAdapter } });
  const device = await import("./device");
  const lose = async (reason: string, message: string) => {
    losses[losses.length - 1]({ reason, message });
    await new Promise(resolve => setTimeout(resolve, 0));
  };
  return { device, requestAdapter, lose };
}

describe("WebGPU device loss", () => {
  it("moves the page to JavaScript after a loss it did not cause, says why once, and never asks for a new device", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { device, requestAdapter, lose } = await deviceWithLoss();
    expect(await device.getHardwareGPUDevice()).not.toBeNull();
    const dropped = vi.fn();
    device.onGPULost(dropped);
    await lose("unknown", "GPU process exited");
    expect(dropped).toHaveBeenCalledOnce();
    expect(await device.getHardwareGPUDevice()).toBeNull();
    expect(await device.getRenderPath()).toBe("CPU");
    expect(device.webGPULostReason()).toBe("GPU process exited");
    await expect(device.requireHardwareGPUDevice("Browser FFT")).rejects.toThrow(/device was lost \(GPU process exited\)/);
    expect(requestAdapter).toHaveBeenCalledOnce();
    expect(warn).toHaveBeenCalledOnce();
    expect(warn.mock.calls[0][0]).toMatch(/WebGPU device was lost \(GPU process exited\).*reload the page/);
    warn.mockRestore();
  });

  it("rebuilds a device the page destroyed itself", async () => {
    const { device, requestAdapter, lose } = await deviceWithLoss();
    expect(await device.getGPUDevice()).not.toBeNull();
    await lose("destroyed", "Device was destroyed.");
    expect(await device.getGPUDevice()).not.toBeNull();
    expect(requestAdapter).toHaveBeenCalledTimes(2);
    expect(device.webGPULostReason()).toBeNull();
  });
});

describe("WebGPU display shaders", () => {
  it("an adapter whose compiler rejects the display shader draws on the CPU instead of painting black", async () => {
    vi.resetModules();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const requestDevice = vi.fn(async () => ({
      lost: new Promise(() => {}),
      createShaderModule: vi.fn(() => ({})),
      createComputePipelineAsync: vi.fn(async () => { throw new Error("FXC compile failed with error: E_FAIL"); }),
    }));
    const requestAdapter = vi.fn(async () => ({ limits: {}, features: new Set(), info: { description: "Parallels Display Adapter" }, requestDevice }));
    vi.stubGlobal("navigator", { gpu: { requestAdapter } });
    const { createGPUColormapEngine } = await import("./colormaps");
    expect(await createGPUColormapEngine()).toBeNull();
    expect(warn.mock.calls[0][0]).toMatch(/do not build on this adapter; drawing on the CPU/);
    warn.mockRestore();
  });
});

describe("GPU memory guard", () => {
  // A device whose next allocation reports out-of-memory (or not) through its error scope.
  function fakeDevice(outOfMemory: boolean) {
    let resolveLost: (info: { reason: string; message: string }) => void = () => {};
    const device = {
      lost: new Promise<{ reason: string; message: string }>(resolve => { resolveLost = resolve; }),
      pushErrorScope: vi.fn(),
      popErrorScope: vi.fn(async () => (outOfMemory ? { message: "out of memory" } : null)),
      destroy: vi.fn(() => resolveLost({ reason: "destroyed", message: "" })),
    };
    return device;
  }

  async function loaded(outOfMemory: boolean) {
    vi.resetModules();
    const fake = fakeDevice(outOfMemory);
    const requestAdapter = vi.fn(async () => ({ limits: {}, features: new Set(), info: { description: "Test GPU" }, requestDevice: async () => fake }));
    vi.stubGlobal("navigator", { gpu: { requestAdapter } });
    const device = await import("./device");
    expect(await device.getGPUDevice()).toBe(fake);
    return { device, fake };
  }

  it("leaves WebGPU with the reason when an allocation runs out of GPU memory", async () => {
    const { device, fake } = await loaded(true);
    const lost = vi.fn();
    device.onGPULost(lost);
    const texture = device.guardGPUMemory(fake as unknown as GPUDevice, "a 512 x 512 x 512 volume", () => "texture");
    expect(texture).toBe("texture");
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(fake.destroy).toHaveBeenCalledTimes(1);
    expect(lost).toHaveBeenCalledTimes(1);
    expect(device.webGPULostReason()).toBe("the GPU has no memory for a 512 x 512 x 512 volume");
    expect(await device.getGPUDevice()).toBeNull();  // JavaScript paths until the page reloads
  });

  it("changes nothing when the allocation fits", async () => {
    const { device, fake } = await loaded(false);
    device.guardGPUMemory(fake as unknown as GPUDevice, "a small volume", () => 1);
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(fake.destroy).not.toHaveBeenCalled();
    expect(device.webGPULostReason()).toBeNull();
    expect(await device.getGPUDevice()).toBe(fake);
  });
});
