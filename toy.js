// Thread by Thread — a tiny compute.toys-compatible WebGPU runtime and editor.
(() => {
"use strict";

const MAX_DISPATCH = 64;          // cap for #dispatch_count
const MAX_STORAGE = 4;            // compute.toys allows 2; we allow a few more
const STORAGE_BYTES = 8 << 20;    // 8 MB per #storage buffer (2M floats)
const MAX_WIDTH = 1024;           // cap on the internal screen resolution
const LS_PREFIX = "thread-by-thread:v1:";

// ------------------------------------------------------------------ highlighting
const KEYWORDS = new Set(("fn let var const override struct alias if else for while loop break continue " +
  "continuing return switch case default discard true false enable requires diagnostic").split(" "));
const TYPES = /^(f32|f16|i32|u32|bool|vec[234][fiuh]?|mat[234]x[234][fh]?|array|ptr|atomic|sampler|sampler_comparison|texture_\w+|function|private|workgroup|uniform|storage|read|write|read_write|rgba16float|rgba32float)$/;
const BUILTINS = new Set(("abs acos acosh all any arrayLength asin asinh atan atan2 atanh atomicAdd atomicAnd " +
  "atomicCompareExchangeWeak atomicExchange atomicLoad atomicMax atomicMin atomicOr atomicStore atomicSub atomicXor " +
  "bitcast ceil clamp cos cosh countLeadingZeros countOneBits countTrailingZeros cross degrees determinant distance " +
  "dot exp exp2 extractBits faceForward firstLeadingBit firstTrailingBit floor fma fract frexp insertBits " +
  "inverseSqrt ldexp length log log2 max min mix modf normalize pack2x16float pack4x8snorm pack4x8unorm pow " +
  "quantizeToF16 radians reflect refract reverseBits round saturate select sign sin sinh smoothstep sqrt step " +
  "storageBarrier tan tanh textureDimensions textureLoad textureSample textureSampleLevel textureStore transpose " +
  "trunc unpack2x16float unpack4x8snorm unpack4x8unorm workgroupBarrier workgroupUniformLoad subgroupAdd").split(" "));
const PRELUDE_NAMES = new Set("time mouse dispatch screen pass_in pass_out passLoad passStore".split(" "));
const TOKEN = /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|(^[ \t]*#[^\n]*)|(@[A-Za-z_]\w*)|(\b0x[0-9a-fA-F]+[iu]?\b|(?:\b\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fhiu]?\b)|([A-Za-z_]\w*)/gm;
const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function highlight(src) {
  let out = "", last = 0;
  src.replace(TOKEN, (m, com, pre, attr, num, id, off) => {
    out += esc(src.slice(last, off));
    last = off + m.length;
    let cls = null;
    if (com) cls = "c"; else if (pre) cls = "pp"; else if (attr) cls = "at"; else if (num) cls = "n";
    else if (KEYWORDS.has(id)) cls = "k"; else if (TYPES.test(id)) cls = "t";
    else if (BUILTINS.has(id)) cls = "f"; else if (PRELUDE_NAMES.has(id)) cls = "g";
    out += cls ? `<span class="${cls}">${esc(m)}</span>` : esc(m);
    return m;
  });
  return out + esc(src.slice(last));
}

// ------------------------------------------------------------------ preprocessing
function preprocess(src) {
  const storage = [], wgCount = {}, dispatchCount = {}, errors = [];
  const lines = src.split("\n").map((line, i) => {
    const m = line.match(/^\s*#(\w+)\s*(.*)$/);
    if (!m) return line;
    const dir = m[1], rest = m[2].trim(), args = rest.split(/\s+/);
    if (dir === "storage") {
      const name = args[0], type = rest.slice(name.length).trim();
      if (!name || !type) errors.push({ line: i + 1, message: "#storage needs a name and a type, like: #storage data array<f32>" });
      else storage.push({ name, type });
    } else if (dir === "workgroup_count") {
      const n = args.slice(1, 4).map((x) => parseInt(x, 10));
      if (!args[0] || n.some((x) => !(x >= 1))) errors.push({ line: i + 1, message: "#workgroup_count needs an entry point and 1–3 positive counts, like: #workgroup_count reduce 64 1 1" });
      else wgCount[args[0]] = [n[0], n[1] || 1, n[2] || 1];
    } else if (dir === "dispatch_count") {
      const n = parseInt(args[1], 10);
      if (!args[0] || !(n >= 1)) errors.push({ line: i + 1, message: "#dispatch_count needs an entry point and a count, like: #dispatch_count matmul 20" });
      else dispatchCount[args[0]] = Math.min(n, MAX_DISPATCH);
    } else {
      errors.push({ line: i + 1, message: `Unknown directive #${dir}. This playground supports #storage, #workgroup_count and #dispatch_count.` });
    }
    return "";                       // keep line numbers stable
  });
  if (storage.length > MAX_STORAGE) errors.push({ line: 1, message: `At most ${MAX_STORAGE} #storage buffers are supported here (compute.toys allows 2).` });
  return { body: lines.join("\n"), storage, wgCount, dispatchCount, errors };
}

function stripComments(s) {
  return s.replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, " "))
          .replace(/\/\/[^\n]*/g, (m) => " ".repeat(m.length));
}

function findEntryPoints(src) {
  const s = stripComments(src);
  const consts = {};
  for (const m of s.matchAll(/\b(?:const|override)\s+(\w+)\s*(?::\s*\w+)?\s*=\s*(\d+)[ui]?\s*;/g)) consts[m[1]] = +m[2];
  const eps = [];
  for (const m of s.matchAll(/((?:@\w+\s*(?:\([^)]*\))?\s*)+)fn\s+(\w+)/g)) {
    if (!/@compute\b/.test(m[1])) continue;
    const size = [1, 1, 1];
    const ws = m[1].match(/@workgroup_size\s*\(([^)]*)\)/);
    if (ws) ws[1].split(",").map((x) => x.trim()).filter(Boolean).forEach((x, i) => {
      const v = /^\d+[ui]?$/.test(x) ? parseInt(x, 10) : consts[x];
      if (i < 3) size[i] = v || 16;
    });
    eps.push({ name: m[2], size });
  }
  return eps;
}

function buildPrelude(storage) {
  let s =
`struct Time { frame: u32, elapsed: f32, delta: f32 }
struct Mouse { pos: vec2i, click: i32 }
struct DispatchInfo { id: u32 }
@group(0) @binding(0) var<uniform> time: Time;
@group(0) @binding(1) var<uniform> mouse: Mouse;
@group(0) @binding(2) var<uniform> dispatch: DispatchInfo;
@group(0) @binding(3) var screen: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var pass_in: texture_2d_array<f32>;
@group(0) @binding(5) var pass_out: texture_storage_2d_array<rgba16float, write>;
fn passLoad(layer: i32, coord: vec2i, lod: i32) -> vec4f { return textureLoad(pass_in, coord, layer, lod); }
fn passStore(layer: i32, coord: vec2i, value: vec4f) { textureStore(pass_out, coord, layer, value); }
`;
  storage.forEach((b, i) => { s += `@group(0) @binding(${6 + i}) var<storage, read_write> ${b.name}: ${b.type};\n`; });
  return s;
}

// ------------------------------------------------------------------ GPU context
const G = { device: null, format: null, blit: null, ready: null, failure: null,
  noPresent: /[?&]nopresent\b/.test(location.search) };   // test mode: skip canvas presentation
const toys = [];

const BLIT_WGSL = `
@group(0) @binding(0) var tex: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var c = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
  return vec4f(c[i], 0., 1.);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let lin = clamp(textureLoad(tex, vec2i(p.xy), 0).rgb, vec3f(0.), vec3f(1.));
  let srgb = select(1.055 * pow(lin, vec3f(1. / 2.4)) - .055, lin * 12.92, lin <= vec3f(.0031308));
  return vec4f(srgb, 1.);
}`;

function envReport() {
  let framed = false;
  try { framed = window.self !== window.top; } catch (_) { framed = true; }
  return `[secure context: ${window.isSecureContext ? "yes" : "no"} · embedded frame: ${framed ? "yes" : "no"} · navigator.gpu: ${navigator.gpu ? "yes" : "no"}]`;
}

async function initGPU() {
  let framed = false;
  try { framed = window.self !== window.top; } catch (_) { framed = true; }
  if (!window.isSecureContext) {
    throw new Error(`WebGPU only works on secure pages (https, localhost, or a file opened directly), and ${location.host || "this address"} isn't one. If you're serving this file locally, open it at http://localhost:${location.port || "PORT"} instead of an IP address.`);
  }
  if (!navigator.gpu) {
    throw new Error((framed && window.isSecureContext
      ? "WebGPU is hidden inside this embedded view, even though your browser may support it. Open the page as a standalone file to run the exercises."
      : "This browser doesn't expose WebGPU. Try a recent Chrome, Edge or Safari.") + " " + envReport());
  }
  let adapter = null;
  try { adapter = await navigator.gpu.requestAdapter(); }
  catch (e) { throw new Error(`requestAdapter() failed: ${e.message} ${envReport()}`); }
  if (!adapter) {
    throw new Error((framed
      ? "WebGPU is present, but the browser refused a GPU adapter inside this embedded view. Open the page as a standalone file to run the exercises."
      : "WebGPU is present, but no GPU adapter is available. Check that hardware acceleration is on (chrome://gpu).") + " " + envReport());
  }
  let device;
  try { device = await adapter.requestDevice(); }
  catch (e) { throw new Error(`requestDevice() failed: ${e.message} ${envReport()}`); }
  G.adapter = adapter;                // keep the adapter alive for the device's lifetime
  G.device = device;
  G.format = navigator.gpu.getPreferredCanvasFormat();
  const mod = device.createShaderModule({ code: BLIT_WGSL });
  G.blit = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: mod, entryPoint: "vs" },
    fragment: { module: mod, entryPoint: "fs", targets: [{ format: G.format }] },
    primitive: { topology: "triangle-list" },
  });
  G.dispatchIds = device.createBuffer({ size: 256 * MAX_DISPATCH, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const ids = new Uint32Array(64 * MAX_DISPATCH);
  for (let i = 0; i < MAX_DISPATCH; i++) ids[i * 64] = i;
  device.queue.writeBuffer(G.dispatchIds, 0, ids);
  device.lost.then((info) => {
    if (info.reason === "destroyed") return;
    console.warn("GPU device lost:", info.reason, info.message);
    G.lastLoss = info.message;
    G.device = null;
    toys.forEach((t) => t.onDeviceLost());
    const now = performance.now();
    G.losses = (G.losses || []).filter((t) => now - t < 20000).concat(now);
    if (G.losses.length > 3) {
      G.failure = `The GPU keeps resetting (${G.lastLoss || "no reason given"}). Reload the page to try again.`;
      toys.forEach((t) => t.fail(G.failure));
      return;
    }
    G.ready = initGPU().then(() => toys.forEach((t) => t.onDeviceRestored()));
  });
  device.addEventListener("uncapturederror", (e) => console.warn("WebGPU:", e.error.message));
}

// ------------------------------------------------------------------ editor
const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
const MOD = isMac ? "⌘" : "Ctrl";

class Editor {
  constructor(host, onChange, onRun) {
    host.innerHTML = `<div class="ed-gutter" aria-hidden="true"><div class="ed-nums"></div></div>
      <div class="ed-view" aria-hidden="true"><div class="ed-scrollbox"><div class="ed-bands"></div><pre class="ed-pre"></pre></div></div>
      <textarea class="ed-input" spellcheck="false" autocapitalize="off" autocomplete="off" autocorrect="off" wrap="off" aria-label="Shader code"></textarea>`;
    this.nums = host.querySelector(".ed-nums");
    this.box = host.querySelector(".ed-scrollbox");
    this.bands = host.querySelector(".ed-bands");
    this.pre = host.querySelector(".ed-pre");
    this.ta = host.querySelector("textarea");
    this.onChange = onChange;
    this.escaped = false;
    this.ta.addEventListener("input", () => { this.render(); onChange(this.ta.value); });
    this.ta.addEventListener("scroll", () => this.sync());
    this.ta.addEventListener("keydown", (e) => this.key(e, onRun));
    this.ta.addEventListener("blur", () => { this.escaped = false; });
  }
  get value() { return this.ta.value; }
  set value(v) { this.ta.value = v; this.render(); }
  // replace the whole text in an undoable way
  replaceAll(v) {
    this.ta.focus();
    this.ta.select();
    if (!document.execCommand || !document.execCommand("insertText", false, v)) this.ta.value = v;
    this.ta.setSelectionRange(0, 0);
    this.ta.scrollTop = 0;
    this.render();
    this.onChange(this.ta.value);
  }
  render() {
    const v = this.ta.value;
    this.pre.innerHTML = highlight(v) + "\n";
    const n = v.split("\n").length;
    if (n !== this.lineCount) {
      this.lineCount = n;
      let s = "";
      for (let i = 1; i <= n; i++) s += i + "\n";
      this.nums.textContent = s;
    }
    this.sync();
  }
  sync() {
    this.box.style.transform = `translate(${-this.ta.scrollLeft}px, ${-this.ta.scrollTop}px)`;
    this.nums.style.transform = `translateY(${-this.ta.scrollTop}px)`;
  }
  markErrors(lines) {
    this.bands.innerHTML = "";
    const lh = parseFloat(getComputedStyle(this.pre).lineHeight) || 20;
    for (const l of new Set(lines)) {
      if (l < 1) continue;
      const b = document.createElement("div");
      b.className = "ed-band";
      b.style.top = `calc(var(--ed-pad) + ${(l - 1) * lh}px)`;
      b.style.height = lh + "px";
      this.bands.appendChild(b);
    }
  }
  goto(line) {
    const lines = this.ta.value.split("\n");
    let pos = 0;
    for (let i = 0; i < line - 1 && i < lines.length; i++) pos += lines[i].length + 1;
    this.ta.focus();
    this.ta.setSelectionRange(pos, pos);
    const lh = parseFloat(getComputedStyle(this.pre).lineHeight) || 20;
    this.ta.scrollTop = Math.max(0, (line - 4) * lh);
  }
  insert(text) {
    if (!document.execCommand || !document.execCommand("insertText", false, text)) this.ta.setRangeText(text, this.ta.selectionStart, this.ta.selectionEnd, "end");
    this.render();
    this.onChange(this.ta.value);
  }
  key(e, onRun) {
    const ta = this.ta;
    if ((e.ctrlKey || e.metaKey) && (e.key === "Enter" || e.key === "s")) { e.preventDefault(); onRun(); return; }
    if (e.key === "Escape") { this.escaped = true; return; }
    if (e.key === "Tab" && !this.escaped && !e.ctrlKey && !e.altKey && !e.metaKey) {
      e.preventDefault();
      const v = ta.value, s = ta.selectionStart, en = ta.selectionEnd;
      const ls = v.lastIndexOf("\n", s - 1) + 1;
      if (s === en && !e.shiftKey) { this.insert("    "); return; }
      let le = v.indexOf("\n", en - (en > s && v[en - 1] === "\n" ? 1 : 0));
      if (le < 0) le = v.length;
      const block = v.slice(ls, le);
      const next = e.shiftKey ? block.replace(/^ {1,4}/gm, "") : block.replace(/^/gm, "    ");
      ta.setSelectionRange(ls, le);
      this.insert(next);
      ta.setSelectionRange(ls, ls + next.length);
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const v = ta.value, s = ta.selectionStart;
      const ls = v.lastIndexOf("\n", s - 1) + 1;
      const indent = v.slice(ls, s).match(/^\s*/)[0];
      const extra = /[{(\[]\s*$/.test(v.slice(ls, s)) ? "    " : "";
      this.insert("\n" + indent + extra);
      return;
    }
    if (e.key === "}") {
      const v = ta.value, s = ta.selectionStart;
      const ls = v.lastIndexOf("\n", s - 1) + 1;
      if (ta.selectionStart === ta.selectionEnd && /^ {4,}$/.test(v.slice(ls, s))) {
        e.preventDefault();
        ta.setSelectionRange(s - 4, s);
        this.insert("}");
      }
    }
  }
}

// ------------------------------------------------------------------ toy
class Toy {
  constructor(el, opts = {}) {
    this.el = el;
    this.bare = !!opts.bare;
    this.id = el.id || `toy-${toys.length}`;
    const pres = [...el.querySelectorAll("pre")];
    const starter = opts.code ?? (pres.find((p) => !p.classList.contains("solution"))?.textContent ?? "");
    const solution = pres.find((p) => p.classList.contains("solution"))?.textContent ?? null;
    this.starter = starter.replace(/\s+$/, "") + "\n";
    this.solution = solution ? solution.replace(/\s+$/, "") + "\n" : null;
    this.mine = this.starter;
    if (!this.bare) { try { const s = localStorage.getItem(LS_PREFIX + this.id); if (s) this.mine = s; } catch (_) {} }
    this.mode = "mine";
    this.running = !this.bare || !matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.visible = false;
    this.elapsed = 0; this.frameNo = 0;
    this.w = 0; this.h = 0;
    this.eps = null;
    this.buffers = [];
    this.mouse = { x: 0, y: 0, click: 0 };
    this.fpsCount = 0; this.fpsTime = 0; this.fps = 0;
    this.buildDOM(el.dataset.title || "", el.dataset.label || "");
    toys.push(this);
  }

  buildDOM(title, label) {
    const el = this.el;
    el.innerHTML = "";
    el.classList.add("toy-ready");
    if (this.bare) {
      el.innerHTML = `<canvas class="toy-canvas"></canvas>`;
      this.canvas = el.querySelector("canvas");
      this.stage = el;
    } else {
      el.innerHTML = `
      <div class="toy-head">
        <div class="toy-name">${label ? `<span class="toy-label">${esc(label)}</span>` : ""}<span class="toy-title">${esc(title)}</span></div>
        <span class="toy-state" data-state="idle">Waiting</span>
      </div>
      <div class="toy-body">
        <div class="toy-stage">
          <canvas class="toy-canvas"></canvas>
          <div class="toy-overlay" hidden></div>
        </div>
        <div class="toy-code">
          <div class="toy-bar">
            <div class="toy-bar-group">
              <button type="button" class="btn btn-run" title="Compile and run (${MOD}+Enter)">Run <kbd>${MOD}↵</kbd></button>
              <button type="button" class="btn btn-pause">Pause</button>
              <button type="button" class="btn btn-restart" title="Reset time, frame count and buffers">Restart</button>
            </div>
            <div class="toy-bar-group">
              ${this.solution ? `<button type="button" class="btn btn-sol">Show solution</button>` : ""}
              <button type="button" class="btn btn-reset" title="Put the starting code back (${MOD}+Z undoes this)">Reset code</button>
            </div>
          </div>
          <div class="ed"></div>
        </div>
      </div>
      <div class="toy-foot">
        <div class="toy-foot-left">
          <div class="hud-stats" aria-label="Frame rate, frame count and resolution">—</div>
          <div class="toy-dispatch" aria-label="Dispatches per frame"></div>
        </div>
        <ul class="toy-msgs" aria-live="polite"></ul>
      </div>`;
      this.canvas = el.querySelector("canvas");
      this.stage = el.querySelector(".toy-stage");
      this.overlay = el.querySelector(".toy-overlay");
      this.stats = el.querySelector(".hud-stats");
      this.stateChip = el.querySelector(".toy-state");
      this.dispatchEl = el.querySelector(".toy-dispatch");
      this.msgs = el.querySelector(".toy-msgs");
      this.pauseBtn = el.querySelector(".btn-pause");
      this.solBtn = el.querySelector(".btn-sol");
      this.editor = new Editor(el.querySelector(".ed"), (v) => this.edited(v), () => this.compile());
      this.editor.value = this.mine;
      el.querySelector(".btn-run").onclick = () => this.compile();
      this.pauseBtn.onclick = () => this.setRunning(!this.running);
      el.querySelector(".btn-restart").onclick = () => this.restart();
      el.querySelector(".btn-reset").onclick = () => {
        if (this.mode === "solution") this.toggleSolution();
        this.editor.replaceAll(this.starter);
        this.compile();
      };
      if (this.solBtn) this.solBtn.onclick = () => this.toggleSolution();
      this.msgs.addEventListener("click", (e) => {
        const li = e.target.closest("[data-line]");
        if (li) this.editor.goto(+li.dataset.line);
      });
    }
    const setMouse = (e, click) => {
      const r = this.canvas.getBoundingClientRect();
      this.mouse.x = Math.floor((e.clientX - r.left) / r.width * this.w);
      this.mouse.y = Math.floor((e.clientY - r.top) / r.height * this.h);
      if (click !== undefined) this.mouse.click = click;
    };
    this.canvas.addEventListener("pointermove", (e) => setMouse(e));
    this.canvas.addEventListener("pointerdown", (e) => { setMouse(e, 1); this.canvas.setPointerCapture?.(e.pointerId); });
    this.canvas.addEventListener("pointerup", (e) => setMouse(e, 0));
    this.canvas.addEventListener("pointercancel", () => { this.mouse.click = 0; });
  }

  get code() { return this.bare ? this.mine : this.editor.value; }

  edited(v) {
    if (this.mode === "mine") {
      this.mine = v;
      try {
        if (v === this.starter) localStorage.removeItem(LS_PREFIX + this.id);
        else localStorage.setItem(LS_PREFIX + this.id, v);
      } catch (_) {}
    }
    clearTimeout(this.debounce);
    this.debounce = setTimeout(() => this.compile(), 650);
  }

  toggleSolution() {
    if (this.mode === "mine") {
      this.mode = "solution";
      this.editor.value = this.solution;
      this.solBtn.textContent = "Back to my code";
      this.el.classList.add("showing-solution");
    } else {
      this.mode = "mine";
      this.editor.value = this.mine;
      this.solBtn.textContent = "Show solution";
      this.el.classList.remove("showing-solution");
    }
    this.editor.ta.scrollTop = 0; this.editor.sync();
    this.compile();
  }

  setRunning(r) {
    this.running = r;
    if (this.pauseBtn) this.pauseBtn.textContent = r ? "Pause" : "Play";
    this.updateChip();
  }

  setState(state, text) {
    this.state = state;
    if (!this.stateChip) return;
    this.stateChip.dataset.state = state;
    this.stateChip.textContent = text;
  }
  updateChip() {
    if (this.state === "error" || this.state === "fail") return;
    if (!this.eps) return;
    this.setState(this.running ? "running" : "paused", this.running ? "Running" : "Paused");
  }

  // The overlay only ever covers a canvas that has never been drawn to.
  // Once a shader has produced pixels, status goes to the message panel instead,
  // so nothing is ever composited over shader output.
  showOverlay(text) {
    if (!this.overlay) return;
    if (text && this.hasRendered) { this.report([{ type: "warning", message: text }]); text = ""; }
    this.overlay.hidden = !text;
    this.overlay.textContent = text || "";
  }


  report(messages) {
    if (this.bare) { messages.forEach((m) => console.warn(m.message)); return; }
    this.msgs.innerHTML = "";
    const errLines = [];
    for (const m of messages) {
      const li = document.createElement("li");
      li.className = `msg msg-${m.type || "error"}`;
      const where = m.line >= 1 ? `Line ${m.line}${m.col ? ":" + m.col : ""}` : (m.line === undefined ? "" : "Prelude");
      if (m.line >= 1) { li.dataset.line = m.line; li.tabIndex = 0; }
      li.innerHTML = `${where ? `<span class="msg-where">${where}</span>` : ""}<span class="msg-text"></span>`;
      li.querySelector(".msg-text").textContent = m.message;
      this.msgs.appendChild(li);
      if ((m.type || "error") === "error") errLines.push(m.line);
    }
    this.editor.markErrors(errLines);
  }

  async compile() {
    clearTimeout(this.debounce);
    if (G.failure) return this.fail(G.failure);
    try { await G.ready; } catch (e) { return this.fail(e.message); }
    const d = G.device;
    if (!d) return;
    const src = this.code;
    const seq = (this.compileSeq = (this.compileSeq || 0) + 1);
    const pp = preprocess(src);
    const report = (msgs) => {
      if (seq !== this.compileSeq) return;
      this.report(msgs);
      if (msgs.some((m) => (m.type || "error") === "error")) {
        this.setState("error", this.eps ? "Error · showing last good build" : "Compile error");
        return true;
      }
      return false;
    };
    if (pp.errors.length) return report(pp.errors);
    const eps = findEntryPoints(pp.body);
    if (!eps.length) return report([{ line: undefined, message: "No @compute entry points found. Add a function marked @compute @workgroup_size(...)." }]);
    const warnings = [];
    for (const n of [...Object.keys(pp.wgCount), ...Object.keys(pp.dispatchCount)]) {
      if (!eps.some((e) => e.name === n)) warnings.push({ type: "warning", line: undefined, message: `A directive names "${n}", but there is no @compute function with that name.` });
    }
    const prelude = buildPrelude(pp.storage);
    const offset = prelude.split("\n").length - 1;
    const module = d.createShaderModule({ code: prelude + pp.body });
    const info = await module.getCompilationInfo();
    const msgs = info.messages.map((m) => ({
      type: m.type, line: m.lineNum ? m.lineNum - offset : undefined, col: m.linePos, message: m.message,
    }));
    if (msgs.some((m) => m.type === "error")) return report(msgs.concat(warnings));

    const layout = this.layoutFor(pp.storage.length);
    let pipelines;
    d.pushErrorScope("validation");
    try {
      pipelines = await Promise.all(eps.map((e) => d.createComputePipelineAsync({
        layout: layout.pipelineLayout, compute: { module, entryPoint: e.name },
      })));
    } catch (e) {
      await d.popErrorScope();
      return report([{ line: undefined, message: e.message }]);
    }
    const err = await d.popErrorScope();
    if (err) return report([{ line: undefined, message: err.message }]);
    if (seq !== this.compileSeq) return;

    eps.forEach((e, i) => { e.pipeline = pipelines[i]; });
    this.eps = eps;
    this.pp = pp;
    this.usesPass = /\bpass(Load|Store)\b|\bpass_(in|out)\b/.test(stripComments(pp.body));
    this.layout = layout;
    this.ensureStorage(pp.storage.length);
    this.bindGroup = null;
    this.checkFrames = 2;
    report(msgs.concat(warnings));
    this.setState("running", "Running");
    this.updateChip();
    this.showOverlay("");
    this.needsDraw = true;
    this.updateDispatchInfo();
    return false;
  }

  layoutFor(nStorage) {
    const d = G.device;
    G.layouts = G.layouts || {};
    if (G.layouts[nStorage]) return G.layouts[nStorage];
    const C = GPUShaderStage.COMPUTE;
    const entries = [
      { binding: 0, visibility: C, buffer: { type: "uniform" } },
      { binding: 1, visibility: C, buffer: { type: "uniform" } },
      { binding: 2, visibility: C, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 } },
      { binding: 3, visibility: C, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" } },
      { binding: 4, visibility: C, texture: { sampleType: "float", viewDimension: "2d-array" } },
      { binding: 5, visibility: C, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d-array" } },
    ];
    for (let i = 0; i < nStorage; i++) entries.push({ binding: 6 + i, visibility: C, buffer: { type: "storage" } });
    const bgl = d.createBindGroupLayout({ entries });
    return (G.layouts[nStorage] = { bgl, pipelineLayout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }) });
  }

  ensureStorage(n) {
    const d = G.device;
    while (this.buffers.length < n) {
      this.buffers.push(d.createBuffer({ size: STORAGE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }));
    }
  }

  ensureSurface() {
    const d = G.device;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = Math.max(2, Math.round(this.canvas.clientWidth * dpr));
    let h = Math.max(2, Math.round(this.canvas.clientHeight * dpr));
    if (w > MAX_WIDTH) { h = Math.round(h * MAX_WIDTH / w); w = MAX_WIDTH; }
    const passNeeded = !!this.usesPass;
    if (w === this.w && h === this.h && this.ctx && passNeeded === this.passAllocated) return;
    this.w = w; this.h = h;
    this.canvas.width = w; this.canvas.height = h;
    if (!G.noPresent) {
      if (!this.ctx) this.ctx = this.canvas.getContext("webgpu");
      this.ctx.configure({ device: d, format: G.format, alphaMode: "opaque" });
    } else this.ctx = true;
    this.screen?.destroy(); this.passIn?.destroy(); this.passOut?.destroy();
    this.screen = d.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    const ps = passNeeded ? [w, h, 4] : [1, 1, 4];
    this.passIn = d.createTexture({ size: ps, format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.passOut = d.createTexture({ size: ps, format: "rgba16float", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    this.passAllocated = passNeeded;
    this.blitGroup = d.createBindGroup({ layout: G.blit.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.screen.createView() }] });
    this.bindGroup = null;
    this.frameNo = 0;                 // new surface: stateful shaders re-initialize
    this.updateDispatchInfo();
  }

  makeBindGroup() {
    const d = G.device;
    if (!this.timeBuf) {
      this.timeBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.mouseBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    const entries = [
      { binding: 0, resource: { buffer: this.timeBuf } },
      { binding: 1, resource: { buffer: this.mouseBuf } },
      { binding: 2, resource: { buffer: G.dispatchIds, size: 16 } },
      { binding: 3, resource: this.screen.createView() },
      { binding: 4, resource: this.passIn.createView({ dimension: "2d-array" }) },
      { binding: 5, resource: this.passOut.createView({ dimension: "2d-array" }) },
    ];
    for (let i = 0; i < this.pp.storage.length; i++) entries.push({ binding: 6 + i, resource: { buffer: this.buffers[i] } });
    this.bindGroup = d.createBindGroup({ layout: this.layout.bgl, entries });
  }

  workgroupsFor(ep) {
    return this.pp.wgCount[ep.name] || [Math.ceil(this.w / ep.size[0]), Math.ceil(this.h / ep.size[1]), 1];
  }

  updateDispatchInfo() {
    if (!this.dispatchEl || !this.eps || !this.w) return;
    this.dispatchEl.innerHTML = this.eps.map((ep) => {
      const wc = this.workgroupsFor(ep);
      const n = this.pp.dispatchCount[ep.name] || 1;
      const threads = wc[0] * wc[1] * wc[2] * ep.size[0] * ep.size[1] * ep.size[2];
      return `<span class="dp" title="${threads.toLocaleString()} invocations per dispatch"><b>${esc(ep.name)}</b> ${wc.join("×")}${n > 1 ? ` <i>×${n}</i>` : ""}</span>`;
    }).join('<span class="dp-arrow" aria-hidden="true">→</span>');
  }

  restart() {
    this.elapsed = 0; this.frameNo = 0;
    if (G.device && this.buffers.length) {
      const enc = G.device.createCommandEncoder();
      this.buffers.forEach((b) => enc.clearBuffer(b));
      G.device.queue.submit([enc.finish()]);
    }
    if (this.w) { this.w = 0; }        // forces new pass textures (cleared)
    this.needsDraw = true;
  }

  frame(dt) {
    const d = G.device;
    if (!d || !this.eps) return;
    this.ensureSurface();
    if (!this.bindGroup) this.makeBindGroup();
    const t = new ArrayBuffer(16), tv = new DataView(t);
    tv.setUint32(0, this.frameNo, true); tv.setFloat32(4, this.elapsed, true); tv.setFloat32(8, dt, true);
    d.queue.writeBuffer(this.timeBuf, 0, t);
    d.queue.writeBuffer(this.mouseBuf, 0, new Int32Array([this.mouse.x, this.mouse.y, this.mouse.click, 0]));
    const checking = this.checkFrames > 0;
    if (checking) { this.checkFrames--; d.pushErrorScope("validation"); }
    const enc = d.createCommandEncoder();
    for (const ep of this.eps) {
      const wc = this.workgroupsFor(ep);
      const n = this.pp.dispatchCount[ep.name] || 1;
      for (let i = 0; i < n; i++) {
        const pass = enc.beginComputePass();
        pass.setPipeline(ep.pipeline);
        pass.setBindGroup(0, this.bindGroup, [i * 256]);
        pass.dispatchWorkgroups(wc[0], wc[1], wc[2]);
        pass.end();
        if (this.passAllocated) enc.copyTextureToTexture({ texture: this.passOut }, { texture: this.passIn }, [this.w, this.h, 4]);
      }
    }
    if (!G.noPresent) {
    const rp = enc.beginRenderPass({ colorAttachments: [{
      view: this.ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
    rp.setPipeline(G.blit); rp.setBindGroup(0, this.blitGroup); rp.draw(3); rp.end();
    }
    d.queue.submit([enc.finish()]);
    if (checking) d.popErrorScope().then((e) => { if (e) { this.report([{ message: e.message }]); this.setState("error", "Runtime error"); } });
    if (!this.hasRendered) { this.hasRendered = true; if (this.overlay) this.overlay.hidden = true; }
    this.frameNo++;
    this.elapsed += dt;
    this.fpsCount++;
  }

  hud(now) {
    if (!this.stats) return;
    if (now - this.fpsTime >= 500) {
      this.fps = Math.round(this.fpsCount * 1000 / (now - this.fpsTime || 1));
      this.fpsCount = 0; this.fpsTime = now;
      this.stats.textContent = this.eps
        ? `${this.running ? this.fps : 0} fps · frame ${this.frameNo} · ${this.w}×${this.h}`
        : "—";
    }
  }

  fail(msg) {
    this.setState("fail", /resetting/.test(msg) ? "GPU reset" : "WebGPU unavailable");
    this.showOverlay(msg);
  }
  onDeviceLost() {
    this.eps = null; this.ctx = null; this.bindGroup = null; this.buffers = []; this.timeBuf = null;
    this.screen = this.passIn = this.passOut = null; this.w = 0;
    G.layouts = {};
    this.setState("error", "GPU reset");
    this.showOverlay("The GPU was reset, often because a shader ran too long (an infinite loop, for example). Recompiling…");
  }
  onDeviceRestored() { if (this.visible || this.compiledOnce) this.compile(); }

  // ---- test hooks
  async readScreen() {
    const d = G.device, w = this.w, h = this.h, bpr = Math.ceil(w * 8 / 256) * 256;
    const rb = d.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.screen }, { buffer: rb, bytesPerRow: bpr }, [w, h]);
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const src = new Uint16Array(rb.getMappedRange());
    const f16 = (x) => { const e = (x >> 10) & 31, m = x & 1023, sg = x >> 15 ? -1 : 1;
      return e === 0 ? sg * m * 2 ** -24 : e === 31 ? NaN : sg * (1 + m / 1024) * 2 ** (e - 15); };
    const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
    const c2 = cv.getContext("2d"), img = c2.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let k = 0; k < 4; k++) {
      let v = Math.min(1, Math.max(0, f16(src[y * bpr / 2 + x * 4 + k]) || 0));
      if (k < 3) v = v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
      img.data[(y * w + x) * 4 + k] = k === 3 ? 255 : Math.round(v * 255);
    }
    rb.unmap(); rb.destroy();
    c2.putImageData(img, 0, 0);
    return cv.toDataURL("image/png");
  }
  async readStorage(index, count) {
    const d = G.device, src = this.buffers[index];
    const bytes = Math.min(STORAGE_BYTES, count * 4);
    const rb = d.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder(); enc.copyBufferToBuffer(src, 0, rb, 0, bytes); d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = Array.from(new Float32Array(rb.getMappedRange()));
    rb.unmap(); rb.destroy();
    return out;
  }
}

// ------------------------------------------------------------------ loop
let autoRun = true, lastT = performance.now();
function loop(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  if (autoRun && G.device) {
    for (const t of toys) {
      if (!t.visible || !t.eps) continue;
      if (t.running) t.frame(dt);
      else if (t.needsDraw) t.frame(0);
      t.needsDraw = false;
      t.hud(now);
    }
  }
  requestAnimationFrame(loop);
}

function boot() {
  G.ready = initGPU().catch((e) => { G.failure = e.message; throw e; });
  G.ready.catch((e) => {
    console.error("Thread by Thread: WebGPU unavailable:", e.message);
    document.documentElement.classList.add("no-webgpu");
    const b = document.querySelector(".webgpu-banner");
    if (b) { b.hidden = false; b.querySelector(".webgpu-reason").textContent = G.failure; }
  });
  const hero = document.querySelector(".hero-toy");
  if (hero) {
    const code = document.getElementById("hero-shader").textContent;
    new Toy(hero, { bare: true, code });
  }
  document.querySelectorAll(".toy").forEach((el) => new Toy(el));
  // static code blocks
  document.querySelectorAll("pre.wgsl:not(.toy pre)").forEach((pre) => {
    const code = pre.querySelector("code") || pre;
    code.innerHTML = highlight(code.textContent);
  });
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const t = toys.find((x) => x.el === e.target);
      if (!t) continue;
      t.visible = e.isIntersecting;
      if (t.visible && !t.compiledOnce) { t.compiledOnce = true; t.setState("idle", "Compiling"); t.compile(); }
    }
  }, { rootMargin: "300px 0px" });
  toys.forEach((t) => io.observe(t.el));
  requestAnimationFrame(loop);
}

window.gpuBook = {
  toys, G,
  setAutoRun(v) { autoRun = v; },
  get ready() { return G.ready; },
};
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
