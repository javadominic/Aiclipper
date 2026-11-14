/**
 * ShaderGradient - High-performance WebGL implementation of ShaderGradient (waterPlane)
 * Matches @shadergradient/react properties & visuals without external dependencies.
 */
(function (global) {
  "use strict";

  function hexToRgb(hex, def) {
    if (!hex) return def;
    hex = hex.replace("#", "").trim();
    if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
    const num = parseInt(hex, 16);
    if (isNaN(num)) return def;
    return [((num >> 16) & 255) / 255, ((num >> 8) & 255) / 255, (num & 255) / 255];
  }

  const VS = `#version 300 es
  precision highp float;
  in vec2 aPos;
  out vec2 vUv;
  void main() {
    vUv = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
  }`;

  const FS = `#version 300 es
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;

  uniform vec2 uResolution;
  uniform float uTime;
  uniform vec3 uColor1;
  uniform vec3 uColor2;
  uniform vec3 uColor3;
  uniform float uSpeed;
  uniform float uFrequency;
  uniform float uStrength;
  uniform float uDensity;
  uniform float uAmplitude;
  uniform float uBrightness;
  uniform float uReflection;
  uniform float uGrain;
  uniform float uRotZ;
  uniform float uRotY;
  uniform vec2 uPosOffset;

  // Wave elevation function for waterPlane (calm, slow, ambient fluid)
  float waterPlane(vec2 p, float t) {
    // Apply position offset (positionX = -1.4)
    p += uPosOffset;

    // Apply rotationZ (50 deg)
    float cz = cos(uRotZ), sz = sin(uRotZ);
    p = mat2(cz, -sz, sz, cz) * p;

    // Apply density scaling
    p *= (uDensity * 0.75 + 0.35);

    // Controlled slow speed
    float time = t * uSpeed * 0.18;
    float freq = uFrequency * 0.48;

    // Octave 1: Primary directional water wave
    float w1 = sin(p.x * freq + time) * cos(p.y * (freq * 0.72) + time * 0.82);

    // Octave 2: Cross angular ripple
    vec2 p2 = mat2(0.8, -0.6, 0.6, 0.8) * p;
    float w2 = sin(p2.x * (freq * 1.3) - time * 0.95) * cos(p2.y * (freq * 1.2) + time * 0.88) * 0.55;

    // Octave 3: Harmonic liquid swell
    vec2 p3 = mat2(0.55, 0.83, -0.83, 0.55) * (p + vec2(w1, w2) * 0.3);
    float w3 = sin(p3.x * (freq * 1.95) + time * 1.15) * sin(p3.y * (freq * 1.85) - time * 1.05) * 0.3;

    // Octave 4: Surface fine ripples
    float w4 = cos((p.x + p.y) * (freq * 2.8) + time * 1.35) * 0.12;

    return (w1 + w2 + w3 + w4) * (uStrength / 7.0) * uAmplitude * 0.45;
  }

  void main() {
    vec2 st = (gl_FragCoord.xy - 0.5 * uResolution) / min(uResolution.x, uResolution.y);
    float t = uTime;

    // Finite difference normal
    float eps = 0.008;
    float h0 = waterPlane(st, t);
    float hx = waterPlane(st + vec2(eps, 0.0), t);
    float hy = waterPlane(st + vec2(0.0, eps), t);

    vec3 normal = normalize(vec3(-(hx - h0) / eps, -(hy - h0) / eps, 1.0));

    // Tilt normal by rotY (10 deg)
    float cy = cos(uRotY), sy = sin(uRotY);
    normal.xz = mat2(cy, -sy, sy, cy) * normal.xz;

    // 3D Lighting setup
    vec3 lightDir = normalize(vec3(0.35, 0.75, 1.1));
    float diff = clamp(dot(normal, lightDir), 0.0, 1.0);

    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(clamp(dot(normal, halfDir), 0.0, 1.0), 22.0) * uReflection;

    // Multi-color blending: color2 (red) -> color3 (purple) -> color1 (lime)
    float val = clamp((h0 + 0.38) * 1.25 + st.y * 0.15 - st.x * 0.1, 0.0, 1.0);
    vec3 col;
    if (val < 0.46) {
      float k = val / 0.46;
      col = mix(uColor2, uColor3, smoothstep(0.0, 1.0, k));
    } else {
      float k = (val - 0.46) / 0.54;
      col = mix(uColor3, uColor1, smoothstep(0.0, 1.0, k));
    }

    // Apply 3D diffuse + specular + brightness
    col = col * (0.65 + 0.35 * diff) * uBrightness + vec3(spec);

    // Film grain
    if (uGrain > 0.5) {
      float g = fract(sin(dot(gl_FragCoord.xy + fract(t * 0.015), vec2(12.9898, 78.233))) * 43758.5453);
      col += (g - 0.5) * 0.04;
    }

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
  }`;

  function ShaderGradient(host, props = {}) {
    if (!host) return () => {};

    // Defaults matching user's config with slow calm motion
    const conf = Object.assign({
      animate: "on",
      brightness: 1.1,
      color1: "#f0ffb3",
      color2: "#db3033",
      color3: "#cb83e1",
      grain: "on",
      pixelDensity: 1.5,
      positionX: -1.4,
      positionY: 0,
      reflection: 0.1,
      rotationX: 0,
      rotationY: 10,
      rotationZ: 50,
      uAmplitude: 1,
      uDensity: 0.5,
      uFrequency: 5.5,
      uSpeed: 0.12, // Much slower, gentle liquid wave motion
      uStrength: 7,
      uTime: 0
    }, props);

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "position:absolute;top:0;left:0;width:100%;height:100%;display:block;pointer-events:none;z-index:0;";
    host.innerHTML = "";
    host.appendChild(canvas);

    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, powerPreference: "high-performance" });
    if (!gl) {
      console.warn("ShaderGradient: WebGL2 not supported");
      return () => canvas.remove();
    }

    function createShader(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error("Shader compile error:", gl.getShaderInfoLog(s));
        gl.deleteShader(s);
        return null;
      }
      return s;
    }

    const vs = createShader(gl.VERTEX_SHADER, VS);
    const fs = createShader(gl.FRAGMENT_SHADER, FS);
    if (!vs || !fs) return () => canvas.remove();

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("Program link error:", gl.getProgramInfoLog(prog));
      return () => canvas.remove();
    }

    gl.useProgram(prog);

    // Quad geometry
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1
    ]), gl.STATIC_DRAW);

    const aPos = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    // Uniform locations
    const uRes = gl.getUniformLocation(prog, "uResolution");
    const uTime = gl.getUniformLocation(prog, "uTime");
    const uColor1 = gl.getUniformLocation(prog, "uColor1");
    const uColor2 = gl.getUniformLocation(prog, "uColor2");
    const uColor3 = gl.getUniformLocation(prog, "uColor3");
    const uSpeed = gl.getUniformLocation(prog, "uSpeed");
    const uFrequency = gl.getUniformLocation(prog, "uFrequency");
    const uStrength = gl.getUniformLocation(prog, "uStrength");
    const uDensity = gl.getUniformLocation(prog, "uDensity");
    const uAmplitude = gl.getUniformLocation(prog, "uAmplitude");
    const uBrightness = gl.getUniformLocation(prog, "uBrightness");
    const uReflection = gl.getUniformLocation(prog, "uReflection");
    const uGrain = gl.getUniformLocation(prog, "uGrain");
    const uRotZ = gl.getUniformLocation(prog, "uRotZ");
    const uRotY = gl.getUniformLocation(prog, "uRotY");
    const uPosOffset = gl.getUniformLocation(prog, "uPosOffset");

    // Static uniforms
    const c1 = hexToRgb(conf.color1, [0.941, 1.0, 0.702]);
    const c2 = hexToRgb(conf.color2, [0.859, 0.188, 0.200]);
    const c3 = hexToRgb(conf.color3, [0.796, 0.514, 0.882]);

    gl.uniform3f(uColor1, c1[0], c1[1], c1[2]);
    gl.uniform3f(uColor2, c2[0], c2[1], c2[2]);
    gl.uniform3f(uColor3, c3[0], c3[1], c3[2]);
    gl.uniform1f(uSpeed, Number(conf.uSpeed) || 0.12);
    gl.uniform1f(uFrequency, Number(conf.uFrequency) || 5.5);
    gl.uniform1f(uStrength, Number(conf.uStrength) || 7.0);
    gl.uniform1f(uDensity, Number(conf.uDensity) || 0.5);
    gl.uniform1f(uAmplitude, Number(conf.uAmplitude) || 1.0);
    gl.uniform1f(uBrightness, Number(conf.brightness) || 1.1);
    gl.uniform1f(uReflection, Number(conf.reflection) || 0.1);
    gl.uniform1f(uGrain, conf.grain === "on" || conf.grain === true ? 1.0 : 0.0);
    gl.uniform1f(uRotZ, (Number(conf.rotationZ) || 50) * Math.PI / 180);
    gl.uniform1f(uRotY, (Number(conf.rotationY) || 10) * Math.PI / 180);
    gl.uniform2f(uPosOffset, Number(conf.positionX) || -1.4, Number(conf.positionY) || 0);

    let raf = null;
    let running = true;
    let visible = true;
    const startT = performance.now();

    function resize() {
      const rect = host.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, conf.pixelDensity || 1.5);
      const w = Math.max(1, Math.round(rect.width * dpr));
      const h = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        gl.viewport(0, 0, w, h);
        gl.useProgram(prog);
        gl.uniform2f(uRes, w, h);
      }
    }

    function render(now) {
      if (!running) return;
      if (visible && (conf.animate === "on" || conf.animate === true)) {
        const sec = (now - startT) * 0.001;
        gl.useProgram(prog);
        gl.uniform1f(uTime, sec);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
      raf = requestAnimationFrame(render);
    }

    const ro = new ResizeObserver(resize);
    ro.observe(host);

    const io = new IntersectionObserver(entries => {
      visible = entries.some(e => e.isIntersecting);
    });
    io.observe(host);

    resize();
    raf = requestAnimationFrame(render);

    return function stop() {
      running = false;
      if (raf) cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      gl.deleteBuffer(buf);
      gl.deleteProgram(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      canvas.remove();
    };
  }

  global.ShaderGradient = ShaderGradient;
  global.GradientASCII = function (el, options) {
    return ShaderGradient(el, options);
  };

})(typeof window !== "undefined" ? window : this);
