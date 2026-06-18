var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
import { ProctorCore } from "./proctor_core.js";
const DEFAULTS = {
  workers: 3,
  checksPerSecond: 9,
  windowSize: 4,
  highThreshold: 58,
  flagAvgThreshold: 62,
  cooldownSec: 120,
  telemetryLagMs: 450,
  modelPath: "/static/proctor_engine/models/proctor_yolo.onnx"
};
class AdminTamperVerifier {
  constructor(options = {}) {
    __publicField(this, "opts");
    __publicField(this, "workerCores", []);
    __publicField(this, "callbacks", null);
    __publicField(this, "workerTimers", []);
    __publicField(this, "workerBusy", []);
    __publicField(this, "initialized", false);
    __publicField(this, "history", /* @__PURE__ */ new Map());
    __publicField(this, "cooldownUntil", /* @__PURE__ */ new Map());
    this.opts = {
      workers: options.workers ?? DEFAULTS.workers,
      checksPerSecond: options.checksPerSecond ?? DEFAULTS.checksPerSecond,
      windowSize: options.windowSize ?? DEFAULTS.windowSize,
      highThreshold: options.highThreshold ?? DEFAULTS.highThreshold,
      flagAvgThreshold: options.flagAvgThreshold ?? DEFAULTS.flagAvgThreshold,
      cooldownSec: options.cooldownSec ?? DEFAULTS.cooldownSec,
      telemetryLagMs: options.telemetryLagMs ?? DEFAULTS.telemetryLagMs,
      modelPath: options.modelPath ?? DEFAULTS.modelPath
    };
  }
  async init() {
    this.workerCores = [];
    for (let i = 0; i < this.opts.workers; i += 1) {
      const core = new ProctorCore({
        modelPath: this.opts.modelPath,
        faceModelPath: "/static/proctor_engine/models/face_landmarker.task",
        mediapipeWasmPath: "/static/proctor_engine/mediapipe/wasm",
        ortWasmPath: "/static/proctor_engine/ort/"
      });
      await core.init(this.opts.modelPath);
      this.workerCores.push(core);
    }
    this.workerBusy = new Array(this.opts.workers).fill(false);
    this.initialized = true;
  }
  start(callbacks) {
    if (!this.initialized) {
      throw new Error("AdminTamperVerifier not initialized");
    }
    this.stop();
    this.callbacks = callbacks;
    const perWorkerChecks = this.opts.checksPerSecond / this.opts.workers;
    const tickMs = Math.max(220, Math.round(1e3 / Math.max(0.1, perWorkerChecks)));
    for (let workerId = 0; workerId < this.opts.workers; workerId += 1) {
      const jitter = Math.floor(Math.random() * 50);
      const timer = window.setInterval(() => {
        void this.runWorkerCheck(workerId);
      }, tickMs + jitter);
      this.workerTimers.push(timer);
    }
  }
  stop() {
    for (const timer of this.workerTimers) {
      clearInterval(timer);
    }
    this.workerTimers = [];
  }
  async runWorkerCheck(workerId) {
    if (!this.callbacks || this.workerBusy[workerId]) {
      return;
    }
    this.workerBusy[workerId] = true;
    try {
      const students = this.callbacks.getActiveStudentIds();
      if (!students || students.length === 0) {
        return;
      }
      const studentId = students[Math.floor(Math.random() * students.length)];
      const video = this.callbacks.getVideoElement(studentId);
      if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
        return;
      }
      const core = this.workerCores[workerId];
      const analyzed = await core.analyze(video, performance.now());
      if (!analyzed) {
        return;
      }
      const synthetic = this.toTelemetryShape(studentId, analyzed);
      const telemetryHistory = this.callbacks.getTelemetryHistory(studentId);
      const matched = this.pickNearestTelemetry(telemetryHistory, synthetic.timestamp_ms, this.opts.telemetryLagMs);
      if (!matched) {
        return;
      }
      const mismatch = this.computeMismatchScore(synthetic, matched);
      const now = Date.now();
      this.callbacks.onCheck?.({
        student_id: studentId,
        mismatch_score: mismatch,
        sample_time_ms: synthetic.timestamp_ms,
        matched_telemetry_time_ms: matched.timestamp_ms,
        worker_id: workerId
      });
      const list = this.history.get(studentId) ?? [];
      list.push({ ts: now, score: mismatch });
      while (list.length > this.opts.windowSize) {
        list.shift();
      }
      this.history.set(studentId, list);
      if (list.length < this.opts.windowSize) {
        return;
      }
      const highMismatchCount = list.filter((s) => s.score >= this.opts.highThreshold).length;
      const avg = list.reduce((acc, cur) => acc + cur.score, 0) / Math.max(1, list.length);
      const cooldown = this.cooldownUntil.get(studentId) || 0;
      if (now < cooldown) {
        return;
      }
      if (highMismatchCount >= 3 && avg >= this.opts.flagAvgThreshold) {
        this.cooldownUntil.set(studentId, now + this.opts.cooldownSec * 1e3);
        this.callbacks.onFlag?.({
          student_id: studentId,
          mismatch_score: Math.round(avg),
          window_size: this.opts.windowSize,
          high_mismatch_count: highMismatchCount,
          details: `Tamper mismatch window exceeded: ${highMismatchCount}/${this.opts.windowSize} high mismatches, avg=${avg.toFixed(1)}`,
          samples: list.map((s) => ({ ...s }))
        });
      }
    } catch (err) {
      console.warn("Admin verifier worker error", workerId, err);
    } finally {
      this.workerBusy[workerId] = false;
    }
  }
  toTelemetryShape(studentId, analyzed) {
    if (!analyzed) {
      throw new Error("Missing analyzed payload");
    }
    return {
      student_id: studentId,
      timestamp_ms: analyzed.analysis.timestamp_ms,
      frame_id: analyzed.analysis.frame_id,
      engine: {
        version: "proctor_wasm_v1",
        model: "proctor_yolo.onnx",
        integrity_verified: true,
        manifest_version: "admin"
      },
      analysis: {
        suspicion_score: analyzed.analysis.suspicion_score,
        safety_level: analyzed.evaluation.safety_level,
        verdict: analyzed.evaluation.verdict,
        raw_flags: analyzed.analysis.raw_flags,
        active_flags: analyzed.analysis.active_flags
      },
      metrics: {
        person_count: analyzed.analysis.metrics.person_count,
        face_count: analyzed.analysis.metrics.face_count,
        banned_labels: analyzed.analysis.metrics.banned_labels,
        yaw_deg: analyzed.face.yaw_deg,
        pitch_deg: analyzed.face.pitch_deg,
        roll_deg: analyzed.face.roll_deg,
        gaze_yaw_deg: analyzed.face.gaze_yaw_deg,
        gaze_pitch_deg: analyzed.face.gaze_pitch_deg,
        ear_mean: analyzed.analysis.metrics.ear_mean,
        gaze_offset: analyzed.analysis.metrics.gaze_offset,
        lighting_score: analyzed.lighting.score,
        lighting_visibility_compromised: analyzed.lightingVisibility.compromised,
        accessory: {
          wire_detected: analyzed.face.accessory.wire_detected,
          earphone_detected: analyzed.face.accessory.earphone_detected,
          headphone_detected: analyzed.face.accessory.headphone_detected
        }
      },
      perf: {
        infer_ms: analyzed.infer_ms,
        effective_fps: 0
      }
    };
  }
  pickNearestTelemetry(history, targetTs, maxLagMs) {
    if (!history || history.length === 0) {
      return null;
    }
    let best = null;
    let bestLag = Number.POSITIVE_INFINITY;
    for (const item of history) {
      const lag = Math.abs(Number(item.timestamp_ms || 0) - targetTs);
      if (lag < bestLag) {
        bestLag = lag;
        best = item;
      }
    }
    if (!best || bestLag > maxLagMs) {
      return null;
    }
    return best;
  }
  computeMismatchScore(adminData, studentData) {
    const adminSuspicion = Number(adminData.analysis?.suspicion_score || 0);
    const studentSuspicion = Number(studentData.analysis?.suspicion_score || 0);
    const suspicionDelta = this.clamp(Math.abs(adminSuspicion - studentSuspicion) / 100, 0, 1);
    const faceDelta = this.clamp(Math.abs((adminData.metrics?.face_count || 0) - (studentData.metrics?.face_count || 0)) / 3, 0, 1);
    const personDelta = this.clamp(Math.abs((adminData.metrics?.person_count || 0) - (studentData.metrics?.person_count || 0)) / 3, 0, 1);
    const adminLabels = new Set(adminData.metrics?.banned_labels || []);
    const studentLabels = new Set(studentData.metrics?.banned_labels || []);
    const union = /* @__PURE__ */ new Set([...adminLabels, ...studentLabels]);
    const intersectCount = [...union].filter((label) => adminLabels.has(label) && studentLabels.has(label)).length;
    const bannedMismatch = union.size === 0 ? 0 : this.clamp(1 - intersectCount / union.size, 0, 1);
    const yawDelta = this.clamp(Math.abs((adminData.metrics?.yaw_deg || 0) - (studentData.metrics?.yaw_deg || 0)) / 45, 0, 1);
    const pitchDelta = this.clamp(Math.abs((adminData.metrics?.pitch_deg || 0) - (studentData.metrics?.pitch_deg || 0)) / 35, 0, 1);
    const headPoseMismatch = this.clamp((yawDelta + pitchDelta) / 2, 0, 1);
    const gazeYawDelta = this.clamp(Math.abs((adminData.metrics?.gaze_yaw_deg || 0) - (studentData.metrics?.gaze_yaw_deg || 0)) / 40, 0, 1);
    const gazePitchDelta = this.clamp(Math.abs((adminData.metrics?.gaze_pitch_deg || 0) - (studentData.metrics?.gaze_pitch_deg || 0)) / 30, 0, 1);
    const gazeMismatch = this.clamp((gazeYawDelta + gazePitchDelta) / 2, 0, 1);
    const lightAdmin = Number(adminData.metrics?.lighting_score || 0);
    const lightStudent = Number(studentData.metrics?.lighting_score || 0);
    const lightingMismatch = this.clamp(Math.abs(lightAdmin - lightStudent), 0, 1);
    const weighted = suspicionDelta * 0.3 + faceDelta * 0.2 + personDelta * 0.1 + bannedMismatch * 0.15 + headPoseMismatch * 0.1 + gazeMismatch * 0.07 + lightingMismatch * 0.08;
    return Math.round(this.clamp(weighted, 0, 1) * 100);
  }
  clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }
}
export {
  AdminTamperVerifier
};
