/**
 * WebGL 2 Volume Renderer — ray-casting with slice plane indicators.
 * Standalone module following the pattern of webgpu-fft.ts.
 */

// ============================================================================
// Types
// ============================================================================

export interface VolumeRenderParams {
  sliceX: number;  // 0..nx-1 (current slice positions for plane indicators)
  sliceY: number;  // 0..ny-1
  sliceZ: number;  // 0..nz-1
  nx: number;
  ny: number;
  nz: number;
  opacity: number;       // global opacity multiplier 0..1
  brightness: number;    // brightness adjustment 0.1..3
  showSlicePlanes: boolean;  // toggle slice plane indicators
}

export interface CameraState {
  yaw: number;       // radians, horizontal rotation
  pitch: number;     // radians, vertical rotation (clamped ±89°)
  distance: number;  // camera distance from volume center
  panX: number;      // horizontal pan
  panY: number;      // vertical pan
}

export const DEFAULT_CAMERA: CameraState = {
  yaw: Math.PI / 6,     // 30°
  pitch: Math.PI / 8,   // 22.5°
  distance: 1.8,
  panX: 0,
  panY: 0,
};

// ============================================================================
// Matrix math (column-major Float32Array[16])
// ============================================================================

function mat4Identity(): Float32Array {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

function mat4Inverse(m: Float32Array): Float32Array {
  const inv = new Float32Array(16);
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-10) return mat4Identity();
  det = 1.0 / det;

  inv[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  inv[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  inv[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  inv[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  inv[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  inv[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  inv[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  inv[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  inv[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  inv[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  inv[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  inv[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  inv[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  inv[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  inv[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  inv[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return inv;
}

function lookAt(
  eyeX: number, eyeY: number, eyeZ: number,
  centerX: number, centerY: number, centerZ: number,
  upX: number, upY: number, upZ: number,
): Float32Array {
  let fx = centerX - eyeX, fy = centerY - eyeY, fz = centerZ - eyeZ;
  const fLen = Math.sqrt(fx * fx + fy * fy + fz * fz);
  fx /= fLen; fy /= fLen; fz /= fLen;

  // side = forward × up
  let sx = fy * upZ - fz * upY, sy = fz * upX - fx * upZ, sz = fx * upY - fy * upX;
  const sLen = Math.sqrt(sx * sx + sy * sy + sz * sz);
  sx /= sLen; sy /= sLen; sz /= sLen;

  // recomputed up = side × forward
  const ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;

  const m = new Float32Array(16);
  m[0] = sx;  m[1] = ux;  m[2] = -fx; m[3] = 0;
  m[4] = sy;  m[5] = uy;  m[6] = -fy; m[7] = 0;
  m[8] = sz;  m[9] = uz;  m[10] = -fz; m[11] = 0;
  m[12] = -(sx * eyeX + sy * eyeY + sz * eyeZ);
  m[13] = -(ux * eyeX + uy * eyeY + uz * eyeZ);
  m[14] = (fx * eyeX + fy * eyeY + fz * eyeZ);
  m[15] = 1;
  return m;
}

function perspective(fov: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fov / 2);
  const rangeInv = 1.0 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * rangeInv;
  m[11] = -1;
  m[14] = 2 * far * near * rangeInv;
  return m;
}

// ============================================================================
// GLSL Shaders
// ============================================================================

const VERTEX_SHADER = /* glsl */`#version 300 es
out vec2 v_uv;
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  v_uv = vec2(x, y) * 0.5 + 0.5;
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = /* glsl */`#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler3D u_volume;
uniform sampler2D u_colormap;
uniform vec3 u_aspectRatio;
uniform mat4 u_invViewProj;
uniform vec3 u_cameraPos;
uniform int u_numSteps;
uniform float u_opacity;
uniform float u_brightness;
uniform float u_sliceX;
uniform float u_sliceY;
uniform float u_sliceZ;
uniform vec4 u_bgColor;
uniform int u_showSlicePlanes;

bool intersectBox(vec3 origin, vec3 dir, vec3 bmin, vec3 bmax,
                  out float tNear, out float tFar) {
  vec3 invDir = 1.0 / dir;
  vec3 t1 = (bmin - origin) * invDir;
  vec3 t2 = (bmax - origin) * invDir;
  vec3 tmin = min(t1, t2);
  vec3 tmax = max(t1, t2);
  tNear = max(max(tmin.x, tmin.y), tmin.z);
  tFar = min(min(tmax.x, tmax.y), tmax.z);
  return tNear <= tFar && tFar > 0.0;
}

vec3 worldToTex(vec3 p, vec3 bmin, vec3 bmax) {
  return (p - bmin) / (bmax - bmin);
}

// Returns t for ray-plane intersection, -1 if miss
float intersectSlicePlane(vec3 origin, vec3 dir, int axis, float pos,
                          vec3 bmin, vec3 bmax) {
  float worldPos = bmin[axis] + pos * (bmax[axis] - bmin[axis]);
  if (abs(dir[axis]) < 1e-8) return -1.0;
  float t = (worldPos - origin[axis]) / dir[axis];
  if (t < 0.0) return -1.0;
  vec3 p = origin + t * dir;
  // Check if intersection is inside box on the other two axes
  for (int i = 0; i < 3; i++) {
    if (i != axis && (p[i] < bmin[i] || p[i] > bmax[i])) return -1.0;
  }
  return t;
}

void main() {
  // Reconstruct ray from clip space
  vec2 ndc = v_uv * 2.0 - 1.0;
  vec4 worldNear = u_invViewProj * vec4(ndc, -1.0, 1.0);
  vec4 worldFar = u_invViewProj * vec4(ndc, 1.0, 1.0);
  worldNear.xyz /= worldNear.w;
  worldFar.xyz /= worldFar.w;

  vec3 rayOrigin = worldNear.xyz;
  vec3 rayDir = normalize(worldFar.xyz - worldNear.xyz);

  vec3 halfExt = u_aspectRatio * 0.5;
  vec3 bmin = -halfExt;
  vec3 bmax = halfExt;

  float tNear, tFar;
  if (!intersectBox(rayOrigin, rayDir, bmin, bmax, tNear, tFar)) {
    fragColor = u_bgColor;
    return;
  }

  tNear = max(tNear, 0.0);
  float stepSize = (tFar - tNear) / float(u_numSteps);

  // Compute slice plane intersections (only if enabled)
  float tSliceXY = -1.0, tSliceXZ = -1.0, tSliceYZ = -1.0;
  if (u_showSlicePlanes != 0) {
    tSliceXY = intersectSlicePlane(rayOrigin, rayDir, 2, u_sliceZ, bmin, bmax);
    tSliceXZ = intersectSlicePlane(rayOrigin, rayDir, 1, u_sliceY, bmin, bmax);
    tSliceYZ = intersectSlicePlane(rayOrigin, rayDir, 0, u_sliceX, bmin, bmax);
  }

  // Front-to-back compositing
  vec4 accum = vec4(0.0);

  for (int i = 0; i < 512; i++) {
    if (i >= u_numSteps) break;

    float t = tNear + (float(i) + 0.5) * stepSize;
    vec3 pos = rayOrigin + t * rayDir;
    vec3 texCoord = worldToTex(pos, bmin, bmax);

    // Composite slice planes at their depth (before volume at this step)
    // XY plane (blue)
    if (tSliceXY > 0.0 && abs(t - tSliceXY) < stepSize * 0.6) {
      vec3 slicePos = rayOrigin + tSliceXY * rayDir;
      vec3 sliceTex = worldToTex(slicePos, bmin, bmax);
      float sliceVal = texture(u_volume, sliceTex).r;
      vec3 sliceCol = texture(u_colormap, vec2(clamp(sliceVal * u_brightness, 0.0, 1.0), 0.5)).rgb;
      sliceCol = mix(sliceCol, vec3(0.3, 0.5, 1.0), 0.4);
      float sliceAlpha = 0.6 * (1.0 - accum.a);
      accum.rgb += sliceCol * sliceAlpha;
      accum.a += sliceAlpha;
      tSliceXY = -1.0;
    }
    // XZ plane (green)
    if (tSliceXZ > 0.0 && abs(t - tSliceXZ) < stepSize * 0.6) {
      vec3 slicePos = rayOrigin + tSliceXZ * rayDir;
      vec3 sliceTex = worldToTex(slicePos, bmin, bmax);
      float sliceVal = texture(u_volume, sliceTex).r;
      vec3 sliceCol = texture(u_colormap, vec2(clamp(sliceVal * u_brightness, 0.0, 1.0), 0.5)).rgb;
      sliceCol = mix(sliceCol, vec3(0.3, 1.0, 0.4), 0.4);
      float sliceAlpha = 0.6 * (1.0 - accum.a);
      accum.rgb += sliceCol * sliceAlpha;
      accum.a += sliceAlpha;
      tSliceXZ = -1.0;
    }
    // YZ plane (red)
    if (tSliceYZ > 0.0 && abs(t - tSliceYZ) < stepSize * 0.6) {
      vec3 slicePos = rayOrigin + tSliceYZ * rayDir;
      vec3 sliceTex = worldToTex(slicePos, bmin, bmax);
      float sliceVal = texture(u_volume, sliceTex).r;
      vec3 sliceCol = texture(u_colormap, vec2(clamp(sliceVal * u_brightness, 0.0, 1.0), 0.5)).rgb;
      sliceCol = mix(sliceCol, vec3(1.0, 0.3, 0.3), 0.4);
      float sliceAlpha = 0.6 * (1.0 - accum.a);
      accum.rgb += sliceCol * sliceAlpha;
      accum.a += sliceAlpha;
      tSliceYZ = -1.0;
    }

    // Sample volume
    float intensity = texture(u_volume, texCoord).r;
    intensity = clamp(intensity * u_brightness, 0.0, 1.0);

    // Colormap lookup
    vec3 color = texture(u_colormap, vec2(intensity, 0.5)).rgb;

    // Transfer function: opacity proportional to intensity
    float alpha = intensity * u_opacity * stepSize * 10.0;

    // Front-to-back compositing (emission-absorption)
    accum.rgb += (1.0 - accum.a) * color * alpha;
    accum.a += (1.0 - accum.a) * alpha;

    if (accum.a > 0.95) break;
  }

  // Blend with background
  fragColor = vec4(accum.rgb + u_bgColor.rgb * (1.0 - accum.a), 1.0);
}`;

// ============================================================================
// VolumeRenderer class
// ============================================================================

export class VolumeRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private volumeTexture: WebGLTexture;
  private colormapTexture: WebGLTexture;
  private vao: WebGLVertexArrayObject;
  private uniforms: Record<string, WebGLUniformLocation>;
  private aspectRatio: [number, number, number] = [1, 1, 1];

  static isSupported(): boolean {
    try {
      const c = document.createElement("canvas");
      const gl = c.getContext("webgl2");
      return gl !== null;
    } catch { return false; }
  }

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { alpha: true, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL 2 not available");
    this.gl = gl;

    // Compile shaders
    const vs = this.compileShader(gl.VERTEX_SHADER, VERTEX_SHADER);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error("Shader link failed: " + gl.getProgramInfoLog(this.program));
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    // Cache uniform locations
    this.uniforms = {};
    const names = [
      "u_volume", "u_colormap", "u_aspectRatio", "u_invViewProj",
      "u_cameraPos", "u_numSteps", "u_opacity", "u_brightness",
      "u_sliceX", "u_sliceY", "u_sliceZ", "u_bgColor", "u_showSlicePlanes",
    ];
    for (const name of names) {
      const loc = gl.getUniformLocation(this.program, name);
      if (loc) this.uniforms[name] = loc;
    }

    // Create empty VAO (full-screen triangle uses gl_VertexID)
    this.vao = gl.createVertexArray()!;

    // Create textures
    this.volumeTexture = gl.createTexture()!;
    this.colormapTexture = gl.createTexture()!;

    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  private compileShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error("Shader compile failed: " + info);
    }
    return shader;
  }

  uploadVolume(data: Float32Array, nx: number, ny: number, nz: number): void {
    const gl = this.gl;

    // Normalize to [0,255] uint8 — avoids R32F float texture filtering issues
    // (R32F needs OES_texture_float_linear for LINEAR filter; R8 always works)
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
    }
    const range = max - min || 1;
    const normalized = new Uint8Array(data.length);
    for (let i = 0; i < data.length; i++) {
      normalized[i] = Math.round(Math.max(0, Math.min(255, ((data[i] - min) / range) * 255)));
    }

    // Compute aspect ratio (longest axis = 1.0)
    const maxDim = Math.max(nx, ny, nz);
    this.aspectRatio = [nx / maxDim, ny / maxDim, nz / maxDim];

    // Upload 3D texture as R8 (normalized uint8, LINEAR filter always supported)
    gl.bindTexture(gl.TEXTURE_3D, this.volumeTexture);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nx, ny, nz, 0, gl.RED, gl.UNSIGNED_BYTE, normalized);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  }

  uploadColormap(lut: Uint8Array): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.colormapTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, lut);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  render(params: VolumeRenderParams, camera: CameraState, bgColor: [number, number, number]): void {
    const gl = this.gl;
    const canvas = gl.canvas as HTMLCanvasElement;

    // Handle high-DPI displays
    const dpr = window.devicePixelRatio || 1;
    const displayW = canvas.clientWidth;
    const displayH = canvas.clientHeight;
    const bufferW = Math.round(displayW * dpr);
    const bufferH = Math.round(displayH * dpr);
    if (canvas.width !== bufferW || canvas.height !== bufferH) {
      canvas.width = bufferW;
      canvas.height = bufferH;
    }
    gl.viewport(0, 0, bufferW, bufferH);

    // Camera setup
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
    const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
    const eyeX = camera.distance * cp * sy + camera.panX;
    const eyeY = camera.distance * sp + camera.panY;
    const eyeZ = camera.distance * cp * cy;

    const viewMatrix = lookAt(eyeX, eyeY, eyeZ, camera.panX, camera.panY, 0, 0, 1, 0);
    const projMatrix = perspective(Math.PI / 4, displayW / displayH, 0.01, 100.0);
    const viewProjMatrix = mat4Multiply(projMatrix, viewMatrix);
    const invViewProj = mat4Inverse(viewProjMatrix);

    // Number of steps scales with volume size
    const maxDim = Math.max(params.nx, params.ny, params.nz);
    const numSteps = Math.min(512, Math.max(128, maxDim * 2));

    // Clear and draw
    gl.clearColor(bgColor[0], bgColor[1], bgColor[2], 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.volumeTexture);
    gl.uniform1i(this.uniforms["u_volume"], 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.colormapTexture);
    gl.uniform1i(this.uniforms["u_colormap"], 1);

    // Set uniforms
    gl.uniform3f(this.uniforms["u_aspectRatio"], this.aspectRatio[0], this.aspectRatio[1], this.aspectRatio[2]);
    gl.uniformMatrix4fv(this.uniforms["u_invViewProj"], false, invViewProj);
    gl.uniform3f(this.uniforms["u_cameraPos"], eyeX, eyeY, eyeZ);
    gl.uniform1i(this.uniforms["u_numSteps"], numSteps);
    gl.uniform1f(this.uniforms["u_opacity"], params.opacity);
    gl.uniform1f(this.uniforms["u_brightness"], params.brightness);
    gl.uniform1f(this.uniforms["u_sliceX"], params.nx > 1 ? params.sliceX / (params.nx - 1) : 0.5);
    gl.uniform1f(this.uniforms["u_sliceY"], params.ny > 1 ? params.sliceY / (params.ny - 1) : 0.5);
    gl.uniform1f(this.uniforms["u_sliceZ"], params.nz > 1 ? params.sliceZ / (params.nz - 1) : 0.5);
    gl.uniform4f(this.uniforms["u_bgColor"], bgColor[0], bgColor[1], bgColor[2], 1.0);
    gl.uniform1i(this.uniforms["u_showSlicePlanes"], params.showSlicePlanes ? 1 : 0);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteTexture(this.volumeTexture);
    gl.deleteTexture(this.colormapTexture);
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
  }
}
