import { ProctorCore, type ProctorAnalyzeResult } from "./proctor_core.js";

type SocketLike = {
  emit: (event: string, payload: unknown) => void;
};

export type StudentTelemetryV2 = {
  student_id: string;
  timestamp_ms: number;
  frame_id: number;
  engine: {
    version: string;
    model: string;
    integrity_verified: boolean;
    manifest_version: string;
  };
  analysis: {
    suspicion_score: number;
    safety_level: number;
    verdict: "GOOD_TO_GO" | "WATCH_CLOSELY" | "NOT_SAFE";
    raw_flags: Record<string, boolean>;
    active_flags: Record<string, boolean>;
  };
  metrics: {
    person_count: number;
    face_count: number;
    banned_labels: string[];
    yaw_deg: number | null;
    pitch_deg: number | null;
    roll_deg: number | null;
    gaze_yaw_deg: number | null;
    gaze_pitch_deg: number | null;
    ear_mean: number | null;
    gaze_offset: number | null;
    lighting_score: number;
    lighting_visibility_compromised: boolean;
    accessory: {
      wire_detected: boolean;
      earphone_detected: boolean;
      headphone_detected: boolean;
    };
  };
  perf: {
    infer_ms: number;
    effective_fps: number;
  };
};

export type IntegrityState = {
  checked: boolean;
  verified: boolean;
  version: string;
  failed_assets: string[];
  error?: string;
};

export type StudentEngineOptions = {
  studentId: string | number;
  socket?: SocketLike | null;
  modelPath?: string;
  manifestUrl?: string;
  inferFps?: number;
};

export class StudentProctorEngine {
  private core: ProctorCore;
  private studentId: string;
  private socket: SocketLike | null;
  private inferIntervalMs: number;
  private manifestUrl: string;
  private modelPath: string;

  private initialized = false;
  private frameCounter = 0;
  private lastRunAt = 0;
  private lastProcessedAt = 0;
  private emaEffectiveFps = 0;
  private lastTelemetry: StudentTelemetryV2 | null = null;

  private integrity: IntegrityState = {
    checked: false,
    verified: false,
    version: "unknown",
    failed_assets: []
  };

  constructor(opts: StudentEngineOptions) {
    this.studentId = String(opts.studentId);
    this.socket = opts.socket || null;
    this.modelPath = opts.modelPath || "/static/proctor_engine/models/proctor_yolo.onnx";
    this.manifestUrl = opts.manifestUrl || "/api/proctor/manifest";

    const targetFps = Math.max(1, Math.min(30, Number(opts.inferFps) || 18));
    this.inferIntervalMs = Math.round(1000 / targetFps);

    this.core = new ProctorCore({
      modelPath: this.modelPath,
      faceModelPath: "/static/proctor_engine/models/face_landmarker.task",
      mediapipeWasmPath: "/static/proctor_engine/mediapipe/wasm",
      ortWasmPath: "/static/proctor_engine/ort/"
    });
  }

  async init(): Promise<void> {
    await this.core.init(this.modelPath);
    this.initialized = true;
  }

  isReady(): boolean {
    return this.initialized;
  }

  getLastTelemetry(): StudentTelemetryV2 | null {
    return this.lastTelemetry;
  }

  getIntegrityState(): IntegrityState {
    return { ...this.integrity, failed_assets: [...this.integrity.failed_assets] };
  }

  async verifyIntegrity(): Promise<IntegrityState> {
    try {
      const manifestRes = await fetch(this.manifestUrl, { cache: "no-store" });
      if (!manifestRes.ok) {
        throw new Error(`Manifest request failed (${manifestRes.status})`);
      }

      const manifest = await manifestRes.json();
      const assets = (manifest && manifest.assets) || {};
      const failed: string[] = [];

      for (const [assetPath, expectedHash] of Object.entries<string>(assets)) {
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

  async processFrame(source: HTMLCanvasElement | HTMLVideoElement): Promise<StudentTelemetryV2 | null> {
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

    const effectiveFps = 1000 / cycleMs;
    this.emaEffectiveFps = this.emaEffectiveFps === 0 ? effectiveFps : this.emaEffectiveFps * 0.9 + effectiveFps * 0.1;

    const payload = this.buildTelemetry(out, this.emaEffectiveFps);
    this.lastTelemetry = payload;

    this.emitTelemetry(payload);
    return payload;
  }

  private buildTelemetry(out: ProctorAnalyzeResult, effectiveFps: number): StudentTelemetryV2 {
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

  private emitTelemetry(payload: StudentTelemetryV2): void {
    if (!this.socket) {
      return;
    }

    this.socket.emit("telemetry_update_v2", payload);

    const banned = payload.metrics.banned_labels || [];
    const legacyObjects = {
      phone: banned.includes("cell phone"),
      laptop: banned.includes("laptop")
    };

    this.socket.emit("telemetry_update", {
      student_id: payload.student_id,
      score: payload.analysis.suspicion_score,
      faces: payload.metrics.face_count,
      objects: legacyObjects
    });
  }

  private async fetchAssetBytes(url: string): Promise<ArrayBuffer> {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`Asset fetch failed (${res.status}): ${url}`);
    }
    return res.arrayBuffer();
  }

  private async sha256Hex(bytes: ArrayBuffer): Promise<string> {
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const arr = Array.from(new Uint8Array(digest));
    return arr.map((b) => b.toString(16).padStart(2, "0")).join("");
  }
}
