var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
import { ProctorCore } from "./proctor_core.js";
class StudentProctorEngine {
  constructor(opts) {
    __publicField(this, "core");
    __publicField(this, "studentId");
    __publicField(this, "socket");
    __publicField(this, "inferIntervalMs");
    __publicField(this, "manifestUrl");
    __publicField(this, "modelPath");
    __publicField(this, "initialized", false);
    __publicField(this, "frameCounter", 0);
    __publicField(this, "lastRunAt", 0);
    __publicField(this, "lastProcessedAt", 0);
    __publicField(this, "emaEffectiveFps", 0);
    __publicField(this, "lastTelemetry", null);
    __publicField(this, "lastBookCheckAt", 0);
    __publicField(this, "bookDetectedUntil", 0);
    __publicField(this, "bookBridgeHoldMs", 2500);
    __publicField(this, "bookCheckCanvas", null);
    __publicField(this, "bookCheckCtx", null);
    __publicField(this, "integrity", {
      checked: false,
      verified: false,
      version: "unknown",
      failed_assets: []
    });
    this.studentId = String(opts.studentId);
    this.socket = opts.socket || null;
    this.modelPath = opts.modelPath || "/static/proctor_engine/models/proctor_yolo.onnx";
    this.manifestUrl = opts.manifestUrl || "/api/proctor/manifest";
    const targetFps = Math.max(1, Math.min(30, Number(opts.inferFps) || 18));
    this.inferIntervalMs = Math.round(1e3 / targetFps);
    this.core = new ProctorCore({
      modelPath: this.modelPath,
      faceModelPath: "/static/proctor_engine/models/face_landmarker.task",
      mediapipeWasmPath: "/static/proctor_engine/mediapipe/wasm",
      ortWasmPath: "/static/proctor_engine/ort/"
    });
  }
  async init() {
    await this.core.init(this.modelPath);
    this.initialized = true;
  }
  isReady() {
    return this.initialized;
  }
  getLastTelemetry() {
    return this.lastTelemetry;
  }
  getIntegrityState() {
    return { ...this.integrity, failed_assets: [...this.integrity.failed_assets] };
  }
  async verifyIntegrity() {
    try {
      const manifestRes = await fetch(this.manifestUrl, { cache: "no-store" });
      if (!manifestRes.ok) {
        throw new Error(`Manifest request failed (${manifestRes.status})`);
      }
      const manifest = await manifestRes.json();
      const assets = manifest && manifest.assets || {};
      const failed = [];
      for (const [assetPath, expectedHash] of Object.entries(assets)) {
        const assetUrl = `/static/proctor_engine/${assetPath}`;
        const bytes = await this.fetchAssetBytes(assetUrl);
        const actual = await this.sha256Hex(bytes);
        const expected = String(expectedHash || "").replace(/^sha256:/i, "").toLowerCase();
        if (actual !== expected) {
          failed.push(assetPath);
        }
      }
      this.integrity = {
        checked: true,
        verified: failed.length === 0,
        version: String(manifest?.version || "unknown"),
        failed_assets: failed
      };
      return this.getIntegrityState();
    } catch (err) {
      this.integrity = {
        checked: true,
        verified: false,
        version: "unknown",
        failed_assets: [],
        error: err instanceof Error ? err.message : String(err)
      };
      return this.getIntegrityState();
    }
  }
  async processFrame(source) {
    if (!this.initialized) {
      return null;
    }
    const now = performance.now();
    if (this.lastRunAt > 0 && now - this.lastRunAt < this.inferIntervalMs) {
      return this.lastTelemetry;
    }
    this.lastRunAt = now;
    const out = await this.core.analyze(source, now);
    if (!out) {
      return null;
    }
    this.frameCounter += 1;
    const cycleMs = this.lastProcessedAt > 0 ? Math.max(1, now - this.lastProcessedAt) : this.inferIntervalMs;
    this.lastProcessedAt = now;
    const effectiveFps = 1e3 / cycleMs;
    this.emaEffectiveFps = this.emaEffectiveFps === 0 ? effectiveFps : this.emaEffectiveFps * 0.9 + effectiveFps * 0.1;
    const payload = this.buildTelemetry(out, this.emaEffectiveFps);
    this.applyBookBridge(payload);
    this.lastTelemetry = payload;
    this.emitTelemetry(payload);
    this.maybeServerBookCheck(source, payload.metrics?.banned_labels || []);
    return payload;
  }
  applyBookBridge(payload) {
    if (!payload || !payload.metrics) {
      return;
    }
    const now = Date.now();
    if (this.bookDetectedUntil > now) {
      const labels = Array.isArray(payload.metrics.banned_labels) ? payload.metrics.banned_labels.slice() : [];
      if (!labels.includes("book")) {
        labels.push("book");
      }
      payload.metrics.banned_labels = labels;
      if (payload.analysis) {
        const flags = Array.isArray(payload.analysis.active_flags) ? payload.analysis.active_flags.slice() : [];
        if (!flags.includes("BOOK_DETECTED")) {
          flags.push("BOOK_DETECTED");
        }
        payload.analysis.active_flags = flags;
      }
    }
  }
  buildTelemetry(out, effectiveFps) {
    return {
      student_id: this.studentId,
      timestamp_ms: Date.now(),
      frame_id: this.frameCounter,
      engine: {
        version: "proctor_wasm_v1",
        model: "proctor_yolo.onnx",
        integrity_verified: this.integrity.verified,
        manifest_version: this.integrity.version
      },
      analysis: {
        suspicion_score: out.analysis.suspicion_score,
        safety_level: out.evaluation.safety_level,
        verdict: out.evaluation.verdict,
        raw_flags: out.analysis.raw_flags,
        active_flags: out.analysis.active_flags
      },
      metrics: {
        person_count: out.analysis.metrics.person_count,
        face_count: out.analysis.metrics.face_count,
        banned_labels: out.analysis.metrics.banned_labels,
        yaw_deg: out.face.yaw_deg,
        pitch_deg: out.face.pitch_deg,
        roll_deg: out.face.roll_deg,
        gaze_yaw_deg: out.face.gaze_yaw_deg,
        gaze_pitch_deg: out.face.gaze_pitch_deg,
        gaze_horiz_ratio: out.face.gaze_horiz_ratio,
        gaze_vert_ratio: out.face.gaze_vert_ratio,
        ear_mean: out.analysis.metrics.ear_mean,
        gaze_offset: out.analysis.metrics.gaze_offset,
        lighting_score: out.lighting.score,
        lighting_visibility_compromised: out.lightingVisibility.compromised,
        accessory: {
          wire_detected: out.face.accessory.wire_detected,
          earphone_detected: out.face.accessory.earphone_detected,
          headphone_detected: out.face.accessory.headphone_detected
        }
      },
      perf: {
        infer_ms: Number(out.infer_ms.toFixed(2)),
        effective_fps: Number(effectiveFps.toFixed(2))
      }
    };
  }
  emitTelemetry(payload) {
    if (!this.socket) {
      return;
    }
    this.socket.emit("telemetry_update_v2", payload);
    const banned = payload.metrics.banned_labels || [];
    const legacyObjects = {
      phone: banned.includes("cell phone"),
      laptop: banned.includes("laptop"),
      book: banned.includes("book") || banned.includes("book_heuristic")
    };
    this.socket.emit("telemetry_update", {
      student_id: payload.student_id,
      score: payload.analysis.suspicion_score,
      faces: payload.metrics.face_count,
      objects: legacyObjects
    });
  }
  maybeServerBookCheck(source, bannedLabels) {
    const now = Date.now();
    if (bannedLabels.some((l) => ["book", "book_heuristic", "paper", "notebook"].includes(l))) return;
    if (this.bookDetectedUntil > now) return;
    if (now - this.lastBookCheckAt < 2000) return;
    this.lastBookCheckAt = now;
    try {
      if (!this.bookCheckCanvas) {
        this.bookCheckCanvas = document.createElement("canvas");
        this.bookCheckCtx = this.bookCheckCanvas.getContext("2d");
      }
      const w = 320;
      const h = 240;
      this.bookCheckCanvas.width = w;
      this.bookCheckCanvas.height = h;
      this.bookCheckCtx.drawImage(source, 0, 0, w, h);
      const jpeg = this.bookCheckCanvas.toDataURL("image/jpeg", 0.6);
      fetch("/api/detect-book", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: jpeg })
      }).then((r) => r.json()).then((resp) => {
        if (resp && resp.book_detected) {
          const until = Date.now() + this.bookBridgeHoldMs;
          this.bookDetectedUntil = Math.max(this.bookDetectedUntil, until);
          if (this.lastTelemetry) {
            const updated = JSON.parse(JSON.stringify(this.lastTelemetry));
            this.applyBookBridge(updated);
            updated.timestamp_ms = Date.now();
            this.lastTelemetry = updated;
            this.emitTelemetry(updated);
          }
        }
      }).catch(() => {});
    } catch (e) {
      // ignore
    }
  }
  async fetchAssetBytes(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`Asset fetch failed (${res.status}): ${url}`);
    }
    return res.arrayBuffer();
  }
  async sha256Hex(bytes) {
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const arr = Array.from(new Uint8Array(digest));
    return arr.map((b) => b.toString(16).padStart(2, "0")).join("");
  }
}
export {
  StudentProctorEngine
};
