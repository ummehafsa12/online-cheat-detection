var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
import * as ort from "/static/proctor_engine/ort/ort.min.mjs";
import { FaceLandmarker, FilesetResolver } from "/static/proctor_engine/mediapipe/vision_bundle.mjs";
const COCO_CLASSES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush"
];
const BANNED_LABELS = /* @__PURE__ */ new Set([
  "cell phone",
  "clock",
  "book",
  "paper"
]);
const ACCESSORY_LABELS = /* @__PURE__ */ new Set([]);
const MONITORED_OBJECT_LABELS = Array.from(/* @__PURE__ */ new Set([...BANNED_LABELS, ...ACCESSORY_LABELS]));
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const YOLO_INPUT_SIZE = 640;
const NIQAB_MODE_ENABLED = true;
const EYE_PAIR_MIN_SEPARATION = 40;
const EYE_PAIR_MAX_SEPARATION = 200;
const GAZE_HORIZONTAL_THRESHOLD = 0.32;
const GAZE_VERTICAL_DOWN_THRESHOLD = 0.25;
const GAZE_VERTICAL_UP_THRESHOLD = 0.25;
const GAZE_HORIZONTAL_SUSTAIN_MS = 1500;
const GAZE_VERTICAL_SUSTAIN_MS = 2e3;
const BOOK_LABELS = ["book", "notebook", "paper", "magazine", "journal", "document", "folder", "textbook", "copy", "register"];
const BOOK_CONFIDENCE_THRESHOLD = 0.3;
const BOOK_AREA_MIN_RATIO = 0.02;
const LIGHTING_MIN_SCORE = 0.52;
const OBJECT_STABLE_FRAMES = 1;
const OBJECT_EMA_ALPHA = 0.34;
const OBJECT_EMA_DECAY = 0.78;
const OBJECT_HIGH_CONF_MARGIN = 0.2;
const ACCESSORY_STABLE_FRAMES = 1;
const ACCESSORY_EMA_ALPHA = 0.35;
const LIGHTING_EMA_ALPHA = 0.28;
const CLASS_CONF_THRESHOLDS = {
  person: 0.5,
  "cell phone": 0.28, // Lowered to detect phones more reliably while keeping false alarms low
  // Book/notebook/journal/register: balanced sensitivity vs false positives
  "book": 0.15,
  "clock": 0.2,
  "paper": 0.15
};
const MIN_AREA_RATIO_BY_LABEL = {
  person: 0.01,
  "cell phone": 1e-3,
  // Aggressively low min area to detect parts of books or tilted ones
  "book": 1e-4,
  "clock": 12e-4,
  "paper": 1e-4
};
const MIN_SHORT_SIDE_PX_BY_LABEL = {
  person: 40,
  "cell phone": 8,
  book: 10,
  clock: 12,
  paper: 12
};
const ACCESSORY_SCORE_THRESHOLDS = {
  wire: 0.12,
  earphone: 0.35,
  headphone: 0.82
};
let wasmReady = false;
let ProctorEngineCtor = null;
async function initRustEngine() {
  if (wasmReady && ProctorEngineCtor) {
    return;
  }
  const mod = await import("/static/proctor_engine/pkg/proctor_wasm.js");
  await mod.default();
  ProctorEngineCtor = mod.ProctorEngine;
  wasmReady = true;
}
class ProctorCore {
  constructor(opts = {}) {
    __publicField(this, "modelPath");
    __publicField(this, "faceModelPath");
    __publicField(this, "mediapipeWasmPath");
    __publicField(this, "ortWasmPath");
    __publicField(this, "yoloSession", null);
    __publicField(this, "faceLandmarker", null);
    __publicField(this, "engine", null);
    __publicField(this, "scratch");
    __publicField(this, "scratchCtx");
    __publicField(this, "frameCanvas");
    __publicField(this, "frameCtx");
    __publicField(this, "eyeScratch");
    __publicField(this, "eyeScratchCtx");
    __publicField(this, "lastEyeApiAt", 0);
    __publicField(this, "lastEyeApiPairs", 0);
    __publicField(this, "gazeEmaH", 0.5);
    __publicField(this, "gazeEmaV", 0.5);
    __publicField(this, "gazeStartH", null);
    __publicField(this, "gazeStartV", null);
    __publicField(this, "gazeInitAt", Date.now());
    __publicField(this, "bookHeuristicStreak", 0);
    __publicField(this, "yoloChwBuffer");
    __publicField(this, "objectTemporalState", /* @__PURE__ */ new Map());
    __publicField(this, "accessoryTemporalState", {
      wire: { ema: 0, streak: 0 },
      earphone: { ema: 0, streak: 0 },
      headphone: { ema: 0, streak: 0 }
    });
    __publicField(this, "lightingTemporalState", {
      scoreEma: 1,
      brightnessEma: 0.55,
      contrastEma: 0.12
    });
    this.modelPath = opts.modelPath || "/static/proctor_engine/models/proctor_yolo.onnx";
    this.faceModelPath = opts.faceModelPath || "/static/proctor_engine/models/face_landmarker.task";
    this.mediapipeWasmPath = opts.mediapipeWasmPath || "/static/proctor_engine/mediapipe/wasm";
    this.ortWasmPath = opts.ortWasmPath || "/static/proctor_engine/ort/";
    this.scratch = document.createElement("canvas");
    this.scratch.width = YOLO_INPUT_SIZE;
    this.scratch.height = YOLO_INPUT_SIZE;
    const scratchCtx = this.scratch.getContext("2d", { willReadFrequently: true });
    if (!scratchCtx) {
      throw new Error("Could not create scratch canvas context");
    }
    this.scratchCtx = scratchCtx;
    this.frameCanvas = document.createElement("canvas");
    const frameCtx = this.frameCanvas.getContext("2d", { willReadFrequently: true });
    if (!frameCtx) {
      throw new Error("Could not create frame canvas context");
    }
    this.frameCtx = frameCtx;
    this.yoloChwBuffer = new Float32Array(YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * 3);
  }
  async init(modelOverride) {
    await initRustEngine();
    if (!ProctorEngineCtor) {
      throw new Error("Rust engine constructor missing");
    }
    this.engine = new ProctorEngineCtor(void 0);
    ort.env.wasm.wasmPaths = this.ortWasmPath;
    const vision = await FilesetResolver.forVisionTasks(this.mediapipeWasmPath);
    this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: this.faceModelPath
      },
      runningMode: "VIDEO",
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
      numFaces: 3
    });
    await this.loadModel(modelOverride || this.modelPath);
    this.resetTemporalState();
  }
  async loadModel(modelPath) {
    const modelBytes = await this.fetchModelBytes(modelPath);
    this.assertModelBytes(modelBytes);
    this.yoloSession = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all"
    });
  }
  resetState() {
    if (this.engine) {
      this.engine.reset_state();
    }
    this.resetTemporalState();
  }
  async analyze(source, timestampMs) {
    if (!this.engine || !this.yoloSession) {
      return null;
    }
    const now = timestampMs ?? performance.now();
    const dims = this.getSourceSize(source);
    if (!dims || !dims.width || !dims.height) {
      return null;
    }
    const vw = dims.width;
    const vh = dims.height;
    if (this.frameCanvas.width !== vw || this.frameCanvas.height !== vh) {
      this.frameCanvas.width = vw;
      this.frameCanvas.height = vh;
    }
    this.frameCtx.drawImage(source, 0, 0, vw, vh);
    const inferStart = performance.now();
    const yoloOut = await this.runYolo(vw, vh);
    const detections = yoloOut.detections;
    const lighting = this.stabilizeLightingSignal(yoloOut.lighting);
    const face = this.runFaceSignals(source, now, vw, vh);
    // Geometry-based book heuristic (desk region)
    const bookHeuristic = this.detectDeskBookHeuristic(vw, vh);
    if (bookHeuristic) {
      this.bookHeuristicStreak = Math.min(10, this.bookHeuristicStreak + 1);
    } else {
      this.bookHeuristicStreak = 0;
    }
    const lightingVisibility = this.assessLightingVisibilityCompromise(lighting, detections, face, vw, vh);
    const personCount = detections.filter((d) => d.label === "person").length;
    const stabilizedLabels = this.stabilizeObjectLabels(detections);
    const heuristicAccessoryLabels = [];
    if (face.accessory.wire_detected) {
      heuristicAccessoryLabels.push("wire_heuristic");
    }
    if (face.accessory.earphone_detected) {
      heuristicAccessoryLabels.push("earphone_heuristic");
    }
    if (face.accessory.headphone_detected) {
      heuristicAccessoryLabels.push("headphone_heuristic");
    }
    const rawBannedLabels = Array.from(
      /* @__PURE__ */ new Set([...stabilizedLabels.banned, ...stabilizedLabels.accessory, ...heuristicAccessoryLabels])
    );
    if (this.bookHeuristicStreak >= 1) {
      rawBannedLabels.push("book_heuristic");
    }
    const labelDisplayMap = {
      clock: "smartwatch",
      book: "book/paper",
      paper: "paper"
    };
    const allowedDisplay = /* @__PURE__ */ new Set(["cell phone", "smartwatch", "book/paper", "paper"]);
    const bannedLabels = rawBannedLabels.map((label) => labelDisplayMap[label] || label).filter((label) => allowedDisplay.has(label)).sort();
    const analysis = this.engine.process({
      timestamp_ms: Date.now(),
      person_count: personCount,
      face_count: face.face_count,
      banned_labels: bannedLabels,
      yaw_deg: face.yaw_deg,
      pitch_deg: face.pitch_deg,
      ear_mean: face.ear_mean,
      gaze_offset: face.gaze_offset,
      lighting_score: lighting.score,
      lighting_visibility_compromised: lightingVisibility.compromised
    });
    const inferMs = performance.now() - inferStart;
    const evaluation = this.evaluateRealtime(analysis, face, lighting, lightingVisibility);
    return {
      analysis,
      evaluation,
      detections,
      face,
      lighting,
      lightingVisibility,
      infer_ms: inferMs
    };
  }
  getSourceSize(source) {
    if (source instanceof HTMLVideoElement) {
      return {
        width: source.videoWidth,
        height: source.videoHeight
      };
    }
    return {
      width: source.width,
      height: source.height
    };
  }
  async fetchModelBytes(modelPath) {
    const res = await fetch(modelPath, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`Model request failed (${res.status} ${res.statusText})`);
    }
    const contentType = (res.headers.get("content-type") || "").toLowerCase();
    if (contentType.includes("text/html")) {
      throw new Error("Model URL returned HTML, not ONNX.");
    }
    const bytes = await res.arrayBuffer();
    this.assertModelBytes(bytes);
    return bytes;
  }
  assertModelBytes(bytes) {
    if (bytes.byteLength < 4096) {
      throw new Error(`Model file too small (${bytes.byteLength} bytes).`);
    }
    const prefix = new TextDecoder().decode(new Uint8Array(bytes, 0, Math.min(256, bytes.byteLength)));
    const lowerPrefix = prefix.toLowerCase();
    if (lowerPrefix.includes("<!doctype html") || lowerPrefix.includes("<html")) {
      throw new Error("Model content is HTML, not ONNX.");
    }
    if (prefix.startsWith("version https://git-lfs.github.com/spec/v1")) {
      throw new Error("Model file is a Git LFS pointer, not the real binary model.");
    }
  }
  async runYolo(frameW, frameH) {
    if (!this.yoloSession) {
      return { detections: [], lighting: this.emptyLightingSignal() };
    }
    const { inputTensor, meta, lighting } = this.prepareYoloInput(frameW, frameH);
    const feed = {
      [this.yoloSession.inputNames[0]]: inputTensor
    };
    const outputs = await this.yoloSession.run(feed);
    const outName = this.yoloSession.outputNames[0];
    const tensor = outputs[outName];
    return {
      detections: this.parseYoloOutput(tensor, frameW, frameH, meta),
      lighting
    };
  }
  prepareYoloInput(frameW, frameH) {
    const scale = Math.min(YOLO_INPUT_SIZE / frameW, YOLO_INPUT_SIZE / frameH);
    const resizedW = Math.max(1, Math.round(frameW * scale));
    const resizedH = Math.max(1, Math.round(frameH * scale));
    const padX = Math.floor((YOLO_INPUT_SIZE - resizedW) / 2);
    const padY = Math.floor((YOLO_INPUT_SIZE - resizedH) / 2);
    this.scratchCtx.fillStyle = "#000";
    this.scratchCtx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
    this.scratchCtx.drawImage(this.frameCanvas, 0, 0, frameW, frameH, padX, padY, resizedW, resizedH);
    const imageData = this.scratchCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
    const { chw, lighting } = this.toCHWFloat32(imageData.data, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, this.yoloChwBuffer);
    const inputTensor = new ort.Tensor("float32", chw, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);
    return {
      inputTensor,
      meta: {
        inputSize: YOLO_INPUT_SIZE,
        scale,
        padX,
        padY
      },
      lighting
    };
  }
  toCHWFloat32(rgba, width, height, out) {
    const area = width * height;
    let luminanceSum = 0;
    let luminanceSqSum = 0;
    let underCount = 0;
    let overCount = 0;
    for (let i = 0; i < area; i += 1) {
      const base = i * 4;
      const r = rgba[base] / 255;
      const g = rgba[base + 1] / 255;
      const b = rgba[base + 2] / 255;
      out[i] = r;
      out[area + i] = g;
      out[2 * area + i] = b;
      const lum = r * 0.2126 + g * 0.7152 + b * 0.0722;
      luminanceSum += lum;
      luminanceSqSum += lum * lum;
      if (lum < 0.13) {
        underCount += 1;
      } else if (lum > 0.92) {
        overCount += 1;
      }
    }
    const mean = luminanceSum / Math.max(1, area);
    const variance = Math.max(0, luminanceSqSum / Math.max(1, area) - mean * mean);
    const contrast = Math.sqrt(variance);
    const underRatio = underCount / Math.max(1, area);
    const overRatio = overCount / Math.max(1, area);
    return {
      chw: out,
      lighting: this.deriveLightingSignal(mean, contrast, underRatio, overRatio)
    };
  }
  parseYoloOutput(tensor, frameW, frameH, meta) {
    const dims = tensor.dims;
    if (dims.length !== 2 && dims.length !== 3) {
      return [];
    }
    const data = tensor.data;
    let numPred = 0;
    let attrs = 0;
    let getter;
    if (dims.length === 2) {
      numPred = dims[0];
      attrs = dims[1];
      getter = (predIdx, attrIdx) => data[predIdx * attrs + attrIdx];
    } else if (dims[1] < dims[2]) {
      attrs = dims[1];
      numPred = dims[2];
      getter = (predIdx, attrIdx) => data[attrIdx * numPred + predIdx];
    } else {
      numPred = dims[1];
      attrs = dims[2];
      getter = (predIdx, attrIdx) => data[predIdx * attrs + attrIdx];
    }
    if (attrs < 6) {
      return [];
    }
    const hasObjectness = attrs >= 85;
    const classStart = hasObjectness ? 5 : 4;
    const raw = [];
    for (let i = 0; i < numPred; i += 1) {
      const cx = getter(i, 0);
      const cy = getter(i, 1);
      const w = getter(i, 2);
      const h = getter(i, 3);
      const obj = hasObjectness ? getter(i, 4) : 1;
      let bestClass = -1;
      let bestClassScore = 0;
      for (let c = classStart; c < attrs; c += 1) {
        const clsScore = getter(i, c);
        if (clsScore > bestClassScore) {
          bestClassScore = clsScore;
          bestClass = c - classStart;
        }
      }
      const confidence = obj * bestClassScore;
      if (bestClass < 0) {
        continue;
      }
      const label = COCO_CLASSES[bestClass] ?? `class_${bestClass}`;
      const minConf = CLASS_CONF_THRESHOLDS[label] ?? 0.4;
      if (confidence < minConf) {
        continue;
      }
      if (label !== "person" && !BANNED_LABELS.has(label) && !ACCESSORY_LABELS.has(label)) {
        continue;
      }
      const x1 = this.clamp((cx - w / 2 - meta.padX) / meta.scale, 0, frameW);
      const y1 = this.clamp((cy - h / 2 - meta.padY) / meta.scale, 0, frameH);
      const x2 = this.clamp((cx + w / 2 - meta.padX) / meta.scale, 0, frameW);
      const y2 = this.clamp((cy + h / 2 - meta.padY) / meta.scale, 0, frameH);
      const boxW = Math.max(0, x2 - x1);
      const boxH = Math.max(0, y2 - y1);
      if (!this.passesBoxSizeGate(label, boxW, boxH, frameW, frameH)) {
        continue;
      }
      raw.push({ label, confidence, bbox: [x1, y1, x2, y2] });
    }
    return this.nms(raw, 0.5);
  }
  passesBoxSizeGate(label, boxW, boxH, frameW, frameH) {
    if (boxW <= 1 || boxH <= 1) {
      return false;
    }
    const areaRatio = boxW * boxH / Math.max(1, frameW * frameH);
    const shortSide = Math.min(boxW, boxH);
    const minAreaRatio = MIN_AREA_RATIO_BY_LABEL[label] ?? 9e-4;
    const minShortSide = MIN_SHORT_SIDE_PX_BY_LABEL[label] ?? 10;
    if (areaRatio < minAreaRatio || shortSide < minShortSide) {
      return false;
    }
    return true;
  }
  nms(detections, iouThreshold) {
    const byScore = [...detections].sort((a, b) => b.confidence - a.confidence);
    const keep = [];
    while (byScore.length > 0) {
      const current = byScore.shift();
      keep.push(current);
      for (let i = byScore.length - 1; i >= 0; i -= 1) {
        const other = byScore[i];
        if (other.label !== current.label) {
          continue;
        }
        if (this.iou(current.bbox, other.bbox) > iouThreshold) {
          byScore.splice(i, 1);
        }
      }
    }
    return keep;
  }
  iou(a, b) {
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    const interW = Math.max(0, x2 - x1);
    const interH = Math.max(0, y2 - y1);
    const interArea = interW * interH;
    const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
    const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
    return interArea / Math.max(areaA + areaB - interArea, 1e-6);
  }
  detectEyePairsLocal(source, frameW, frameH) {
    if (!NIQAB_MODE_ENABLED) return 0;
    if (!this.eyeScratch) {
      this.eyeScratch = document.createElement("canvas");
      this.eyeScratchCtx = this.eyeScratch.getContext("2d");
    }
    const w = 200;
    const h = Math.max(80, Math.round(frameH * (w / Math.max(frameW, 1))));
    this.eyeScratch.width = w;
    this.eyeScratch.height = h;
    this.eyeScratchCtx.drawImage(source, 0, 0, w, h);
    const data = this.eyeScratchCtx.getImageData(0, 0, w, h).data;
    const yStart = Math.floor(h * 0.2);
    const yEnd = Math.floor(h * 0.6);
    const colEnergy = new Array(w).fill(0);
    let samples = 0;
    for (let y = yStart; y < yEnd; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        if (lum < 90) {
          colEnergy[x] += (90 - lum);
        }
        samples++;
      }
    }
    const smoothed = colEnergy.map((_, i) => {
      let sum = 0;
      let count = 0;
      for (let k = -3; k <= 3; k++) {
        const idx = i + k;
        if (idx >= 0 && idx < w) {
          sum += colEnergy[idx];
          count++;
        }
      }
      return count ? sum / count : 0;
    });
    let peaks = [];
    const meanEnergy = smoothed.reduce((a, b) => a + b, 0) / Math.max(smoothed.length, 1);
    for (let i = 2; i < smoothed.length - 2; i++) {
      const v = smoothed[i];
      if (v > meanEnergy * 1.4 && v > smoothed[i - 1] && v > smoothed[i + 1]) {
        peaks.push({ x: i, v });
      }
    }
    if (peaks.length < 2) return 0;
    peaks = peaks.sort((a, b) => b.v - a.v).slice(0, 4).sort((a, b) => a.x - b.x);
    for (let i = 0; i < peaks.length - 1; i++) {
      const sep = Math.abs(peaks[i + 1].x - peaks[i].x);
      if (sep >= EYE_PAIR_MIN_SEPARATION && sep <= EYE_PAIR_MAX_SEPARATION) {
        return 1;
      }
    }
    return 0;
  }
  fetchEyePairsFromServer(source) {
    if (!NIQAB_MODE_ENABLED) return;
    const now = Date.now();
    if (now - this.lastEyeApiAt < 900) return;
    this.lastEyeApiAt = now;
    if (!this.eyeScratch) {
      this.eyeScratch = document.createElement("canvas");
      this.eyeScratchCtx = this.eyeScratch.getContext("2d");
    }
    const w = 320;
    const h = 240;
    this.eyeScratch.width = w;
    this.eyeScratch.height = h;
    this.eyeScratchCtx.drawImage(source, 0, 0, w, h);
    const jpeg = this.eyeScratch.toDataURL("image/jpeg", 0.5);
    fetch("/api/detect-eyes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: jpeg })
    }).then((r) => r.json()).then((resp) => {
      if (resp && typeof resp.eye_pairs === "number") {
        this.lastEyeApiPairs = resp.eye_pairs;
      }
    }).catch(() => {
    });
  }
  detectEyePairs(source, frameW, frameH) {
    if (!NIQAB_MODE_ENABLED) return 0;
    const localPairs = this.detectEyePairsLocal(source, frameW, frameH);
    if (localPairs === 0) {
      this.fetchEyePairsFromServer(source);
    }
    return Math.max(localPairs, this.lastEyeApiPairs || 0);
  }
  detectDeskBookHeuristic(frameW, frameH) {
    try {
      const regionTop = Math.floor(frameH * 0.4);
      const regionH = frameH - regionTop;
      const regionW = frameW;
      const targetW = 160;
      const targetH = Math.max(80, Math.round(regionH * (targetW / Math.max(regionW, 1))));
      if (!this.eyeScratch) {
        this.eyeScratch = document.createElement("canvas");
        this.eyeScratchCtx = this.eyeScratch.getContext("2d");
      }
      this.eyeScratch.width = targetW;
      this.eyeScratch.height = targetH;
      this.eyeScratchCtx.drawImage(this.frameCanvas, 0, regionTop, regionW, regionH, 0, 0, targetW, targetH);
      const data = this.eyeScratchCtx.getImageData(0, 0, targetW, targetH).data;
      const len = data.length / 4;
      let sum = 0;
      let sumSq = 0;
      for (let i = 0; i < len; i++) {
        const idx = i * 4;
        const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        sum += lum;
        sumSq += lum * lum;
      }
      const mean = sum / Math.max(1, len);
      const variance = sumSq / Math.max(1, len) - mean * mean;
      const areaRatio = regionW * regionH / Math.max(1, frameW * frameH);
      return mean > 105 && mean < 235 && variance < 1400 && areaRatio >= BOOK_AREA_MIN_RATIO;
    } catch (e) {
      return false;
    }
  }
  runFaceSignals(source, timeMs, frameW, frameH) {
    if (!this.faceLandmarker) {
      return this.emptyFace();
    }
    const result = this.faceLandmarker.detectForVideo(source, timeMs);
    const faces = result.faceLandmarks;
    let faceCount = faces ? faces.length : 0;
    let eyePairs = 0;
    if ((!faces || faceCount <= 1) && NIQAB_MODE_ENABLED) {
      eyePairs = this.detectEyePairs(source, frameW, frameH);
    }
    const effectiveFaceCount = Math.max(faceCount, eyePairs);
    if (!faces || faces.length === 0) {
      this.decayAccessoryTemporalState();
      const empty = this.emptyFace();
      empty.face_count = effectiveFaceCount;
      return empty;
    }
    const lm = faces[0];
    const yaw = this.estimateYaw(lm);
    const pitch = this.estimatePitch(lm);
    const roll = this.estimateRoll(lm);
    const earLeft = this.eyeAspectRatio(lm, LEFT_EYE);
    const earRight = this.eyeAspectRatio(lm, RIGHT_EYE);
    const ear = earLeft !== null && earRight !== null ? (earLeft + earRight) / 2 : null;
    const gaze = this.estimateGazeSignals(lm);
    const accessory = this.stabilizeAccessorySignal(this.detectEarAccessories(lm, frameW, frameH));
    return {
      face_count: effectiveFaceCount,
      yaw_deg: yaw,
      pitch_deg: pitch,
      roll_deg: roll,
      ear_mean: ear,
      gaze_offset: gaze.offset,
      gaze_yaw_deg: gaze.yaw_deg,
      gaze_pitch_deg: gaze.pitch_deg,
      gaze_horiz_ratio: gaze.horiz_ratio,
      gaze_vert_ratio: gaze.vert_ratio,
      accessory
    };
  }
  emptyFace() {
    return {
      face_count: 0,
      yaw_deg: null,
      pitch_deg: null,
      roll_deg: null,
      ear_mean: null,
      gaze_offset: null,
      gaze_yaw_deg: null,
      gaze_pitch_deg: null,
      accessory: this.emptyAccessorySignal()
    };
  }
  emptyAccessorySignal() {
    return {
      wire_score: 0,
      earphone_score: 0,
      headphone_score: 0,
      wire_detected: false,
      earphone_detected: false,
      headphone_detected: false
    };
  }
  emptyLightingSignal() {
    return {
      brightness: 0.55,
      contrast: 0.12,
      under_ratio: 0,
      over_ratio: 0,
      score: 1,
      good: true,
      reason: "ok"
    };
  }
  dist2(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.hypot(dx, dy);
  }
  eyeAspectRatio(points, idx) {
    if (points.length <= idx[5]) {
      return null;
    }
    const p1 = points[idx[0]];
    const p2 = points[idx[1]];
    const p3 = points[idx[2]];
    const p4 = points[idx[3]];
    const p5 = points[idx[4]];
    const p6 = points[idx[5]];
    const horiz = this.dist2(p1, p4);
    if (horiz < 1e-6) {
      return null;
    }
    const vert = this.dist2(p2, p6) + this.dist2(p3, p5);
    return vert / (2 * horiz);
  }
  estimateYaw(points) {
    if (points.length < 264) {
      return null;
    }
    const leftEye = points[33];
    const rightEye = points[263];
    const nose = points[1];
    const eyeDist = Math.max(Math.abs(rightEye.x - leftEye.x), 1e-6);
    const midEyeX = (leftEye.x + rightEye.x) / 2;
    return Math.atan2(nose.x - midEyeX, eyeDist) * 180 * 2.25 / Math.PI;
  }
  estimatePitch(points) {
    if (points.length < 292) {
      return null;
    }
    const leftEye = points[33];
    const rightEye = points[263];
    const nose = points[1];
    const chin = points[152];
    const midEyeY = (leftEye.y + rightEye.y) / 2;
    const faceH = Math.max(chin.y - midEyeY, 1e-6);
    return Math.atan2(nose.y - midEyeY, faceH) * 180 * 3.1 / Math.PI;
  }
  estimateRoll(points) {
    if (points.length < 264) {
      return null;
    }
    const leftEye = points[33];
    const rightEye = points[263];
    return Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180 / Math.PI;
  }
  estimateGazeSignals(points) {
    if (points.length < 478) {
      return { offset: null, yaw_deg: null, pitch_deg: null, horiz_ratio: null, vert_ratio: null };
    }
    const leftOuter = points[33].x;
    const leftInner = points[133].x;
    const rightInner = points[362].x;
    const rightOuter = points[263].x;
    const leftTop = points[159].y;
    const leftBottom = points[145].y;
    const rightTop = points[386].y;
    const rightBottom = points[374].y;
    const leftIris = this.meanX(points, [468, 469, 470, 471, 472]);
    const rightIris = this.meanX(points, [473, 474, 475, 476, 477]);
    const leftIrisY = this.meanY(points, [468, 469, 470, 471, 472]);
    const rightIrisY = this.meanY(points, [473, 474, 475, 476, 477]);
    const leftRatio = this.safeRatio(leftIris - leftOuter, leftInner - leftOuter);
    const rightRatio = this.safeRatio(rightIris - rightInner, rightOuter - rightInner);
    const leftRatioV = this.safeRatio(leftIrisY - leftTop, leftBottom - leftTop);
    const rightRatioV = this.safeRatio(rightIrisY - rightTop, rightBottom - rightTop);
    if (leftRatio === null || rightRatio === null || leftRatioV === null || rightRatioV === null) {
      return { offset: null, yaw_deg: null, pitch_deg: null, horiz_ratio: null, vert_ratio: null };
    }
    const yawRatio = ((leftRatio - 0.5) * 2 + (rightRatio - 0.5) * 2) / 2;
    const pitchRatio = ((leftRatioV - 0.5) * 2 + (rightRatioV - 0.5) * 2) / 2;
    const yawDeg = this.clamp(yawRatio * 38, -40, 40);
    const pitchDeg = this.clamp(-pitchRatio * 28, -30, 30);
    const offset = this.clamp((Math.abs(yawRatio) * 0.75 + Math.abs(pitchRatio) * 0.55) * 1.4, 0, 1.5);
    const horizRatio = this.clamp(yawRatio * 0.5 + 0.5, 0, 1);
    const vertRatio = this.clamp(pitchRatio * 0.5 + 0.5, 0, 1);
    return {
      offset,
      yaw_deg: yawDeg,
      pitch_deg: pitchDeg,
      horiz_ratio: horizRatio,
      vert_ratio: vertRatio
    };
  }
  estimateEyeOnlyGaze(points) {
    if (points.length < 374) return { offset: null, yaw_deg: null, pitch_deg: null };
    const leftOuter = points[33].x;
    const leftInner = points[133].x;
    const rightInner = points[362].x;
    const rightOuter = points[263].x;
    const leftTop = points[159] ? points[159].y : points[33].y;
    const leftBottom = points[145] ? points[145].y : points[33].y;
    const rightTop = points[386] ? points[386].y : points[362].y;
    const rightBottom = points[374] ? points[374].y : points[362].y;
    const leftCenter = (leftInner + leftOuter) / 2;
    const rightCenter = (rightInner + rightOuter) / 2;
    const yawRatio = this.safeRatio((leftCenter - leftOuter), leftInner - leftOuter) - this.safeRatio((rightOuter - rightCenter), rightOuter - rightInner);
    const pitchRatio = this.safeRatio((leftBottom - leftTop), Math.max(leftBottom - leftTop, 1e-6));
    const horizRatio = this.clamp((yawRatio || 0) * 0.5 + 0.5, 0, 1);
    const vertRatio = this.clamp(pitchRatio || 0, 0, 1);
    const yawDeg = this.clamp((horizRatio - 0.5) * 80, -35, 35);
    const pitchDeg = this.clamp((vertRatio - 0.5) * -40, -20, 20);
    const offset = this.clamp((Math.abs(yawDeg) / 40 + Math.abs(pitchDeg) / 25), 0, 1.5);
    return { offset, yaw_deg: yawDeg, pitch_deg: pitchDeg, horiz_ratio: horizRatio, vert_ratio: vertRatio };
  }
  detectEarAccessories(points, frameW, frameH) {
    if (points.length < 478) {
      return this.emptyAccessorySignal();
    }
    const rois = this.computeAccessoryRois(points, frameW, frameH);
    const leftWire = this.roiStats(rois.leftWire);
    const rightWire = this.roiStats(rois.rightWire);
    const leftEarPad = this.roiStats(rois.leftEarPad);
    const rightEarPad = this.roiStats(rois.rightEarPad);
    const headBand = this.roiStats(rois.headBand);
    const wireScoreHeuristic = this.clamp(
      Math.max(leftWire.vertical_ratio, rightWire.vertical_ratio) * 1.65 + Math.max(leftWire.edge_ratio, rightWire.edge_ratio) * 0.45,
      0,
      1
    );
    const headphoneScoreHeuristic = this.clamp(
      Math.min(leftEarPad.dark_ratio, rightEarPad.dark_ratio) * 0.50 + headBand.dark_ratio * 0.30 + (leftEarPad.edge_ratio + rightEarPad.edge_ratio) * 0.85,
      0,
      1
    );
    const earphoneScoreHeuristic = this.clamp(
      wireScoreHeuristic * 0.72 + Math.max(leftEarPad.vertical_ratio, rightEarPad.vertical_ratio) * 0.55 + Math.max(leftEarPad.edge_ratio, rightEarPad.edge_ratio) * 0.18,
      0,
      1
    );
    return {
      wire_score: wireScoreHeuristic,
      earphone_score: earphoneScoreHeuristic,
      headphone_score: headphoneScoreHeuristic,
      wire_detected: wireScoreHeuristic >= ACCESSORY_SCORE_THRESHOLDS.wire,
      earphone_detected: earphoneScoreHeuristic >= ACCESSORY_SCORE_THRESHOLDS.earphone,
      headphone_detected: headphoneScoreHeuristic >= ACCESSORY_SCORE_THRESHOLDS.headphone
    };
  }
  stabilizeAccessorySignal(raw) {
    const wire = this.updateTemporalScore("wire", raw.wire_score, ACCESSORY_SCORE_THRESHOLDS.wire);
    const earphone = this.updateTemporalScore("earphone", raw.earphone_score, ACCESSORY_SCORE_THRESHOLDS.earphone);
    const headphone = this.updateTemporalScore("headphone", raw.headphone_score, ACCESSORY_SCORE_THRESHOLDS.headphone);
    return {
      wire_score: wire.ema,
      earphone_score: earphone.ema,
      headphone_score: headphone.ema,
      wire_detected: wire.detected,
      earphone_detected: earphone.detected,
      headphone_detected: headphone.detected
    };
  }
  updateTemporalScore(key, rawScore, threshold) {
    const state = this.accessoryTemporalState[key];
    state.ema = state.ema === 0 ? rawScore : state.ema * (1 - ACCESSORY_EMA_ALPHA) + rawScore * ACCESSORY_EMA_ALPHA;
    if (rawScore >= threshold) {
      state.streak += 1;
    } else {
      state.streak = 0;
    }
    const highConfidenceHit = rawScore >= threshold + 0.2;
    const detected = highConfidenceHit || state.streak >= ACCESSORY_STABLE_FRAMES && state.ema >= threshold;
    return { ema: state.ema, detected };
  }
  computeAccessoryRois(points, frameW, frameH) {
    const leftEar = this.pointPx(points, 234, frameW, frameH);
    const rightEar = this.pointPx(points, 454, frameW, frameH);
    const topHead = this.pointPx(points, 10, frameW, frameH);
    const chin = this.pointPx(points, 152, frameW, frameH);
    const faceW = Math.max(1, Math.abs(rightEar.x - leftEar.x));
    const faceH = Math.max(1, Math.abs(chin.y - topHead.y));
    return {
      leftWire: this.centerRect(leftEar.x, leftEar.y + faceH * 0.28, faceW * 0.18, faceH * 0.92, frameW, frameH),
      rightWire: this.centerRect(rightEar.x, rightEar.y + faceH * 0.28, faceW * 0.18, faceH * 0.92, frameW, frameH),
      leftEarPad: this.centerRect(leftEar.x, leftEar.y, faceW * 0.24, faceH * 0.38, frameW, frameH),
      rightEarPad: this.centerRect(rightEar.x, rightEar.y, faceW * 0.24, faceH * 0.38, frameW, frameH),
      headBand: this.centerRect((leftEar.x + rightEar.x) / 2, topHead.y + faceH * 0.08, faceW * 0.64, faceH * 0.16, frameW, frameH)
    };
  }
  pointPx(points, idx, frameW, frameH) {
    return {
      x: points[idx].x * frameW,
      y: points[idx].y * frameH
    };
  }
  centerRect(cx, cy, width, height, frameW, frameH) {
    const w = Math.max(4, Math.round(width));
    const h = Math.max(4, Math.round(height));
    const x = Math.round(this.clamp(cx - w / 2, 0, frameW - 2));
    const y = Math.round(this.clamp(cy - h / 2, 0, frameH - 2));
    const maxW = Math.max(2, frameW - x);
    const maxH = Math.max(2, frameH - y);
    return {
      x,
      y,
      w: Math.min(w, maxW),
      h: Math.min(h, maxH)
    };
  }
  roiStats(rect) {
    if (rect.w < 3 || rect.h < 3) {
      return { edge_ratio: 0, dark_ratio: 0, vertical_ratio: 0 };
    }
    const pixels = this.frameCtx.getImageData(rect.x, rect.y, rect.w, rect.h).data;
    const area = rect.w * rect.h;
    const gray = new Float32Array(area);
    let darkCount = 0;
    for (let i = 0; i < area; i += 1) {
      const base = i * 4;
      const g = pixels[base] * 0.299 + pixels[base + 1] * 0.587 + pixels[base + 2] * 0.114;
      gray[i] = g;
      if (g < 64) {
        darkCount += 1;
      }
    }
    let edgeCount = 0;
    let verticalLikeCount = 0;
    for (let y = 1; y < rect.h - 1; y += 1) {
      for (let x = 1; x < rect.w - 1; x += 1) {
        const idx = y * rect.w + x;
        const gx = gray[idx + 1] - gray[idx - 1];
        const gy = gray[idx + rect.w] - gray[idx - rect.w];
        const mag = Math.abs(gx) + Math.abs(gy);
        if (mag > 60) {
          edgeCount += 1;
        }
        if (Math.abs(gx) > 70 && Math.abs(gx) > Math.abs(gy) * 1.28) {
          verticalLikeCount += 1;
        }
      }
    }
    const coreArea = Math.max(1, (rect.w - 2) * (rect.h - 2));
    return {
      edge_ratio: edgeCount / coreArea,
      dark_ratio: darkCount / area,
      vertical_ratio: verticalLikeCount / coreArea
    };
  }
  stabilizeObjectLabels(detections) {
    const maxConfidenceByLabel = /* @__PURE__ */ new Map();
    for (const det of detections) {
      if (!BANNED_LABELS.has(det.label) && !ACCESSORY_LABELS.has(det.label)) {
        continue;
      }
      const prev = maxConfidenceByLabel.get(det.label) ?? 0;
      if (det.confidence > prev) {
        maxConfidenceByLabel.set(det.label, det.confidence);
      }
    }
    const banned = [];
    const accessory = [];
    for (const label of MONITORED_OBJECT_LABELS) {
      const seenConfidence = maxConfidenceByLabel.get(label) ?? 0;
      const state = this.objectTemporalState.get(label) ?? { ema: 0, streak: 0 };
      if (seenConfidence > 0) {
        state.ema = state.ema === 0 ? seenConfidence : state.ema * (1 - OBJECT_EMA_ALPHA) + seenConfidence * OBJECT_EMA_ALPHA;
        state.streak += 1;
      } else {
        state.ema *= OBJECT_EMA_DECAY;
        state.streak = 0;
      }
      this.objectTemporalState.set(label, state);
      const minConf = CLASS_CONF_THRESHOLDS[label] ?? 0.4;
      const highConfidenceHit = seenConfidence >= minConf + OBJECT_HIGH_CONF_MARGIN;
      const stableHit = state.streak >= OBJECT_STABLE_FRAMES && state.ema >= minConf;
      if (!highConfidenceHit && !stableHit) {
        continue;
      }
      if (BANNED_LABELS.has(label)) {
        banned.push(label);
      } else if (ACCESSORY_LABELS.has(label)) {
        accessory.push(label);
      }
    }
    banned.sort();
    accessory.sort();
    return { banned, accessory };
  }
  resetTemporalState() {
    this.objectTemporalState.clear();
    this.accessoryTemporalState.wire = { ema: 0, streak: 0 };
    this.accessoryTemporalState.earphone = { ema: 0, streak: 0 };
    this.accessoryTemporalState.headphone = { ema: 0, streak: 0 };
    this.lightingTemporalState.scoreEma = 1;
    this.lightingTemporalState.brightnessEma = 0.55;
    this.lightingTemporalState.contrastEma = 0.12;
  }
  decayAccessoryTemporalState() {
    this.accessoryTemporalState.wire.ema *= 0.72;
    this.accessoryTemporalState.earphone.ema *= 0.72;
    this.accessoryTemporalState.headphone.ema *= 0.72;
    this.accessoryTemporalState.wire.streak = 0;
    this.accessoryTemporalState.earphone.streak = 0;
    this.accessoryTemporalState.headphone.streak = 0;
  }
  meanX(points, idx) {
    let sum = 0;
    for (const i of idx) {
      sum += points[i].x;
    }
    return sum / idx.length;
  }
  meanY(points, idx) {
    let sum = 0;
    for (const i of idx) {
      sum += points[i].y;
    }
    return sum / idx.length;
  }
  safeRatio(num, den) {
    if (Math.abs(den) < 1e-6) {
      return null;
    }
    return num / den;
  }
  clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }
  deriveLightingSignal(brightness, contrast, underRatio, overRatio) {
    const brightnessScore = 1 - this.clamp(Math.abs(brightness - 0.54) / 0.34, 0, 1);
    const contrastScore = this.clamp((contrast - 0.08) / 0.17, 0, 1);
    const underPenalty = this.clamp(underRatio / 0.5, 0, 1);
    const overPenalty = this.clamp(overRatio / 0.4, 0, 1);
    const score = this.clamp(brightnessScore * 0.46 + contrastScore * 0.54 - underPenalty * 0.28 - overPenalty * 0.28, 0, 1);
    let reason = "ok";
    if (underRatio > 0.45 || brightness < 0.25) {
      reason = "too_dark";
    } else if (overRatio > 0.35 || brightness > 0.85) {
      reason = "too_bright";
    } else if (contrast < 0.09) {
      reason = "low_contrast";
    }
    return {
      brightness,
      contrast,
      under_ratio: underRatio,
      over_ratio: overRatio,
      score,
      good: score >= LIGHTING_MIN_SCORE && reason === "ok",
      reason
    };
  }
  stabilizeLightingSignal(raw) {
    this.lightingTemporalState.scoreEma = this.lightingTemporalState.scoreEma * (1 - LIGHTING_EMA_ALPHA) + raw.score * LIGHTING_EMA_ALPHA;
    this.lightingTemporalState.brightnessEma = this.lightingTemporalState.brightnessEma * (1 - LIGHTING_EMA_ALPHA) + raw.brightness * LIGHTING_EMA_ALPHA;
    this.lightingTemporalState.contrastEma = this.lightingTemporalState.contrastEma * (1 - LIGHTING_EMA_ALPHA) + raw.contrast * LIGHTING_EMA_ALPHA;
    return this.deriveLightingSignal(
      this.lightingTemporalState.brightnessEma,
      this.lightingTemporalState.contrastEma,
      raw.under_ratio,
      raw.over_ratio
    );
  }
  assessLightingVisibilityCompromise(lighting, detections, face, frameW, frameH) {
    if (lighting.score >= LIGHTING_MIN_SCORE) {
      return { compromised: false, reason: "none", severity: 0 };
    }
    const persons = detections.filter((d) => d.label === "person");
    const monitoredObjects = detections.filter((d) => d.label !== "person" && (BANNED_LABELS.has(d.label) || ACCESSORY_LABELS.has(d.label)));
    const avgPersonConf = persons.length > 0 ? persons.reduce((s, d) => s + d.confidence, 0) / persons.length : 0;
    const avgObjectConf = monitoredObjects.length > 0 ? monitoredObjects.reduce((s, d) => s + d.confidence, 0) / monitoredObjects.length : 0;
    const frameArea = Math.max(1, frameW * frameH);
    const maxPersonAreaRatio = persons.reduce((m, d) => {
      const area = Math.max(0, d.bbox[2] - d.bbox[0]) * Math.max(0, d.bbox[3] - d.bbox[1]);
      return Math.max(m, area / frameArea);
    }, 0);
    const faceVisibilityLost = face.face_count === 0 && maxPersonAreaRatio >= 0.055;
    const objectVisibilityLow = persons.length > 0 && avgPersonConf < 0.58 || monitoredObjects.length > 0 && avgObjectConf < 0.46;
    const globalVisibilityLost = face.face_count === 0 && persons.length === 0 && lighting.score < 0.4;
    if (faceVisibilityLost) {
      return { compromised: true, reason: "face_visibility_lost", severity: this.clamp(1 - lighting.score, 0, 1) };
    }
    if (objectVisibilityLow) {
      return { compromised: true, reason: "object_visibility_low", severity: this.clamp(1 - lighting.score, 0, 1) };
    }
    if (globalVisibilityLost) {
      return { compromised: true, reason: "global_visibility_lost", severity: this.clamp(1 - lighting.score, 0, 1) };
    }
    return { compromised: false, reason: "none", severity: 0 };
  }
  evaluateRealtime(analysis, face, lighting, lightingVisibility) {
    let risk = analysis.suspicion_score;
    const reasons = [];
    const yaw = face.yaw_deg ?? 0;
    const pitch = face.pitch_deg ?? 0;
    const roll = face.roll_deg ?? 0;
    const gazeYaw = face.gaze_yaw_deg ?? 0;
    const gazePitch = face.gaze_pitch_deg ?? 0;
    const gazeHorizRatio = face.gaze_horiz_ratio ?? 0.5;
    const gazeVertRatio = face.gaze_vert_ratio ?? 0.5;
    const now = Date.now();
    const allowGaze = now - this.gazeInitAt > 1200;
    // Smooth gaze ratios
    this.gazeEmaH = this.gazeEmaH * 0.6 + gazeHorizRatio * 0.4;
    this.gazeEmaV = this.gazeEmaV * 0.6 + gazeVertRatio * 0.4;
    const horizDev = Math.abs(this.gazeEmaH - 0.5);
    const vertDevDown = this.gazeEmaV - 0.5;
    const vertDevUp = 0.5 - this.gazeEmaV;
    const horizExceeded = horizDev > GAZE_HORIZONTAL_THRESHOLD;
    const vertDownExceeded = vertDevDown > GAZE_VERTICAL_DOWN_THRESHOLD;
    const vertUpExceeded = vertDevUp > GAZE_VERTICAL_UP_THRESHOLD;
    if (!horizExceeded) this.gazeStartH = null;
    if (!vertDownExceeded && !vertUpExceeded) this.gazeStartV = null;
    if (horizExceeded && !this.gazeStartH) this.gazeStartH = now;
    if ((vertDownExceeded || vertUpExceeded) && !this.gazeStartV) this.gazeStartV = now;
    const horizSustained = this.gazeStartH && now - this.gazeStartH >= GAZE_HORIZONTAL_SUSTAIN_MS;
    const vertSustained = this.gazeStartV && now - this.gazeStartV >= GAZE_VERTICAL_SUSTAIN_MS;

    // Set specific vertical gaze flags for the server
    if (allowGaze && vertSustained) {
      if (vertDevUp > GAZE_VERTICAL_UP_THRESHOLD) {
        analysis.active_flags.GAZE_UP = true;
      } else if (vertDevDown > GAZE_VERTICAL_DOWN_THRESHOLD) {
        analysis.active_flags.GAZE_DOWN = true;
      }
    }
    if (allowGaze && horizSustained) {
      analysis.active_flags.LOOKING_AWAY = true;
    }

    if (Math.abs(yaw) > 25) {
      risk += 10;
      reasons.push("Head yaw off-axis");
    }
    if (Math.abs(pitch) > 35) {
      risk += 9;
      reasons.push("Head pitch off-axis");
    }
    if (Math.abs(roll) > 12) {
      risk += 6;
      reasons.push("Head roll tilt");
    }
    if (allowGaze && horizSustained) {
      risk += 11;
      reasons.push("Eye gaze horizontal drift");
    }
    if (allowGaze && vertSustained) {
      risk += 9;
      reasons.push("Eye gaze vertical drift");
    }
    if (analysis.active_flags.bad_lighting && lightingVisibility.compromised) {
      risk += 12;
      if (lightingVisibility.reason === "face_visibility_lost") {
        reasons.push("Bad lighting is hiding face details");
      } else if (lightingVisibility.reason === "object_visibility_low") {
        reasons.push("Bad lighting is reducing object visibility");
      } else if (lightingVisibility.reason === "global_visibility_lost") {
        reasons.push("Bad lighting is reducing overall scene visibility");
      } else if (lighting.reason === "too_dark") {
        reasons.push("Bad lighting: too dark");
      } else if (lighting.reason === "too_bright") {
        reasons.push("Bad lighting: too bright");
      } else {
        reasons.push("Bad lighting: low contrast");
      }
    }
    if (face.accessory.wire_detected) {
      risk += 14;
      reasons.push("Wire-like pattern near ears");
    }
    if (face.accessory.earphone_detected) {
      risk += 18;
      reasons.push("Earphone-like cue near ears");
    }
    if (face.accessory.headphone_detected) {
      risk += 16;
      reasons.push("Headphone-like cue around head");
    }
    if (analysis.metrics.banned_labels.length > 0) {
      reasons.push(`Objects: ${analysis.metrics.banned_labels.join(", ")}`);
    }
    if (analysis.active_flags.no_face) {
      reasons.push("No face detected");
    }
    if (analysis.active_flags.multiple_faces) {
      reasons.push("Multiple faces detected");
    }
    const riskScore = Math.round(this.clamp(risk, 0, 100));
    const safetyLevel = 100 - riskScore;
    const hardBlock = analysis.active_flags.no_face || analysis.active_flags.multiple_faces || analysis.active_flags.banned_object || face.accessory.earphone_detected || face.accessory.headphone_detected || face.accessory.wire_detected;
    const goodToGo = !hardBlock && safetyLevel >= 72;
    let verdict = "GOOD_TO_GO";
    if (!goodToGo && safetyLevel >= 50) {
      verdict = "WATCH_CLOSELY";
    } else if (!goodToGo) {
      verdict = "NOT_SAFE";
    }
    return {
      safety_level: safetyLevel,
      risk_score: riskScore,
      good_to_go: goodToGo,
      verdict,
      reasons
    };
  }
}
export {
  ProctorCore
};
