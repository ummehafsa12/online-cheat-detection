/**
 * ═══════════════════════════════════════════════════════════
 *  PROCTOR DETECTION CONFIG  —  v1.0 LOCKED
 * ═══════════════════════════════════════════════════════════
 *
 *  Central configuration for the entire detection pipeline.
 *  Every threshold, weight, and label list lives here.
 *  Both student_engine.js and admin_verifier.js import from
 *  this single source of truth.
 *
 *  To recalibrate: edit ONLY this file.
 * ═══════════════════════════════════════════════════════════
 */

// ── YOLO Model ──────────────────────────────────────────
export const YOLO_INPUT_SIZE = 640;

export const COCO_CLASSES = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
  "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
  "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
  "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
  "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
  "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
  "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
  "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
];

// ── Banned & Accessory Labels ───────────────────────────
export const BANNED_LABELS = new Set([
  'cell phone',
  'clock',      // Smartwatch (COCO uses "clock")
  'book',       // Book/paper notes
  'paper'       // Included for UI mapping; model won't detect directly
]);

export const ACCESSORY_LABELS = new Set([]);

export const MONITORED_OBJECT_LABELS = Array.from(new Set([...BANNED_LABELS, ...ACCESSORY_LABELS]));

// ── Per-class Confidence Thresholds ─────────────────────
export const CLASS_CONF_THRESHOLDS = {
  person: 0.50,
  'cell phone': 0.28,
  'book': 0.15,
  'clock': 0.20,
  'paper': 0.15
};

// ── Minimum Area Ratio (bbox area / frame area) ─────────
export const MIN_AREA_RATIO_BY_LABEL = {
  person: 0.01,
  "cell phone": 4e-4,
  book: 15e-4,
  clock: 12e-4,
  paper: 10e-4
};

// ── Minimum Short-side Pixels ───────────────────────────
export const MIN_SHORT_SIDE_PX_BY_LABEL = {
  person: 40,
  "cell phone": 8,
  book: 15,
  clock: 12,
  paper: 12
};

// ── Face Landmark Indices ───────────────────────────────
export const LEFT_EYE  = [33, 160, 158, 133, 153, 144];
export const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

// ── Temporal Stabilization ──────────────────────────────
export const OBJECT_STABLE_FRAMES    = 1;
export const OBJECT_EMA_ALPHA        = 0.34;
export const OBJECT_EMA_DECAY        = 0.78;
export const OBJECT_HIGH_CONF_MARGIN = 0.20;
export const ACCESSORY_STABLE_FRAMES = 1;
export const ACCESSORY_EMA_ALPHA     = 0.35;
export const LIGHTING_EMA_ALPHA      = 0.28;
export const LIGHTING_MIN_SCORE      = 0.52;

// ── Accessory Heuristic Thresholds ──────────────────────
export const ACCESSORY_SCORE_THRESHOLDS = {
  wire:      2.0,
  earphone:  2.0,
  headphone: 2.0
};

// ── Evaluation: Risk Weights (evaluateRealtime) ─────────
export const EVAL = {
  yaw_threshold:        24,    // degrees — head yaw off-axis
  yaw_risk:             10,
  pitch_threshold:      18,    // degrees — head pitch off-axis
  pitch_risk:            9,
  roll_threshold:       15,    // degrees — head roll tilt
  roll_risk:             6,
  gaze_yaw_threshold:   22,    // degrees — eye gaze horizontal
  gaze_yaw_risk:        11,
  gaze_pitch_threshold: 16,    // degrees — eye gaze vertical
  gaze_pitch_risk:       9,
  bad_lighting_risk:    12,
  wire_risk:            0,
  earphone_risk:        0,
  headphone_risk:       0,
  no_face_risk:         35,    // CRITICAL — hiding face
  multi_face_risk:      25,    // multiple faces
  safety_good_threshold: 72    // safetyLevel >= this → GOOD_TO_GO
};

// Voice detection (admin WebAudio pipeline)
export const VOICE_RMS_THRESHOLD = 0.016;
export const VOICE_ZCR_MIN = 0.08;
export const VOICE_ZCR_MAX = 0.45;
export const VOICE_SPECTRAL_FLUX_THRESHOLD = 0.012;
export const VOICE_EMA_ALPHA = 0.3;
export const VOICE_COOLDOWN_MS = 350;
export const VOICE_HIGHPASS_HZ = 300;
export const VOICE_LOWPASS_HZ = 3400;
export const VOICE_FILTER_Q = 0.7;

// Niqab/abaya fallback (eyes-only)
export const NIQAB_MODE_ENABLED = true;
export const EYE_PAIR_MIN_SEPARATION = 40;   // pixels between eye centers (downscaled frame)
export const EYE_PAIR_MAX_SEPARATION = 200;

// Gaze tuning (horizontal vs vertical with sustain)
export const GAZE_HORIZONTAL_THRESHOLD = 0.32;       // offset from center (more sensitive)
export const GAZE_VERTICAL_DOWN_THRESHOLD = 0.34;    // lenient down-gaze detection
export const GAZE_VERTICAL_UP_THRESHOLD = 0.24;      // up-gaze is unusual, catch sooner
export const GAZE_HORIZONTAL_SUSTAIN_MS = 1500;      // ~1.5s sustain
export const GAZE_VERTICAL_SUSTAIN_MS = 2000;        // ~2s sustain

// Book/notebook detection
export const BOOK_LABELS = ['book','notebook','paper','magazine','journal','document','folder','textbook','copy','register'];
export const BOOK_CONFIDENCE_THRESHOLD = 0.30;
export const BOOK_AREA_MIN_RATIO = 0.02;             // allow smaller books/papers
