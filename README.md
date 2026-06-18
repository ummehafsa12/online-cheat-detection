# AI-Vision Smart Proctor (Online CheatBuster)

End-to-end, browser-first exam proctoring built with Flask, Socket.IO, and a WebAssembly vision stack (YOLO ONNX + MediaPipe). The server coordinates auth, exam lifecycle, telemetry relays, recordings, and warning/termination logic while the heavy vision work runs on the student/admin browsers.

---

## Contents
- Overview
- Architecture
- Feature Set
- Repository Layout
- Core Components
- Data & Storage
- Setup and Running
- Student and Admin Flows
- Configuration (env vars)
- Operations and Monitoring
- Known Limits / Troubleshooting

---

## Overview
- Browser-based proctoring: Student runs a WASM vision engine; telemetry is pushed to the server for aggregation and to admins for review.
- Low-latency monitoring: Socket.IO transports telemetry, warnings, and WebRTC signalling; MJPEG fallback streams are cached on the server.
- Warning pipeline: Configurable thresholds; three warnings auto-terminate by default with per-type cooldowns and admin overrides.
- Record-keeping: Exam sessions, results, and violations are stored in MySQL; audio/video recordings are saved under `static/recording/`.
- Integrity aids: Proctor asset manifest, tamper cross-check between admin-rendered video and student telemetry, and tab/shortcut/voice traps.

---

## Architecture
```
Student Browser
  - WebRTC video, WASM ProctorCore (YOLO + MediaPipe)
  - Emits telemetry, warnings, audio chunks, optional recordings
           | Socket.IO + REST + WebRTC signalling
           v
Flask Backend (app.py)
  - Auth, sessions, CSRF helper, rate limiting
  - Exam lifecycle, warning ledger, DB persistence
  - Telemetry relay, MJPEG fallback cache, file storage
           | Socket.IO + HTTPS
           v
Admin Dashboard (browser)
  - Grid of live students, WebRTC viewer, warning controls
  - Local voice detection + AdminTamperVerifier (WASM replay)
           |
           v
MySQL + disk
  - Tables: students, profiles, exam_sessions, exam_results, violations
  - Recordings: static/recording/, static/recording/audio/
```

---

## Feature Set
- Authentication and account flows (register, login, forgot/reset password with signed tokens).
- Student pre-exam checks: rules page, system check API, optional face snapshot verification (currently pass-through for WASM-only flow).
- Live proctoring:
  - StudentProctorEngine (client) runs YOLO ONNX + MediaPipe face landmarks, evaluates gaze, head pose, banned objects (phone, book/paper, smartwatch/clock), accessories (wire/earphone/headphone heuristics), lighting, and face count.
  - Telemetry v2 carries detailed metrics, warnings, and performance; server relays to admins and caches for dashboards.
  - WebRTC signalling (offer/answer/ICE) for P2P video; MJPEG fallback via cached base64 frames.
  - Voice: admin browser performs RMS-based detection on relayed raw PCM chunks; emits `voice_alert` back to student and flags telemetry.
- Warning/termination pipeline:
  - `WarningSystem` with per-type cooldowns, max warnings (default 3), and optional auto-terminate (env-driven).
  - `TabSwitchDetector` increments warnings on focus loss/tab changes.
  - Mapped violation types are persisted to DB during live events and on exam submission.
- Exam lifecycle:
  - `/api/exam-session/start` marks session IN_PROGRESS, primes runtime state, and notifies admins.
  - `/api/exam-session/end` or termination events close sessions and emit summary.
  - Submission (`/exam` POST) computes score from submitted questions or provided totals; derives pass/fail/terminated status and records results.
- Admin console:
  - `admin_live_dashboard.html` displays suspicion scores, warnings, metrics, and live video tiles.
  - Actions: notify student, clear warnings, toggle enforcement, request frame captures, force terminate, trigger voice warning.
  - Cross-check: AdminTamperVerifier re-runs ProctorCore on live video to compare against telemetry (detects spoofed feeds).
- Recording and playback:
  - Student uploads audio snippets and optional full session recordings; metadata JSON saved alongside media.
  - Admin downloads video/audio via `/download/recording/...` routes and views in `Recordings.html`.
- Health and safety:
  - `/api/system-health` reports DB/socket status, cached frame counts, uptime.
  - Rate limits on sensitive routes; session cookies are HTTPOnly and optionally Secure/SameSite.

---

## Repository Layout
- `app.py` - Flask app with all routes, Socket.IO handlers, warning logic, session/result persistence, and system health.
- `warning_system.py` - Warning and tab-switch management with per-type cooldowns and auto-terminate timer.
- `templates/` - Jinja HTML pages (login, signup, rules, system check, Exam UI, results, admin students/results/recordings/profile, live dashboard).
- `static/`
  - `proctor_engine/` - WASM runtime (ONNXRuntime Web, MediaPipe), proctor core JS, admin verifier, student engine, detection config, manifest.
  - `css/`, `js/`, `img/` - UI assets for exam and admin pages.
  - `recording/` and `recording/audio/` - Stored WebM/MP4 recordings and audio uploads.
  - `Profiles/` - Uploaded/verified student face images.
- `models/` - Legacy CV models (YOLOv4-tiny, MobileNetSSD, coco names) retained for reference.
- `Haarcascades/` - Haar face model (legacy fallback).
- `scripts/sync_proctor_assets.sh` - Copies/builds proctor WASM assets from a companion computer-vision repo and regenerates `static/proctor_engine/manifest.json`.
- `create_users.py` - Seeds test admin/student accounts using bcrypt hashing.
- `examproctordb.sql` - Full schema plus sample data for MySQL `examproctordb`.
- `START_EXAM_PROCTOR.bat` / `RUN_INSTRUCTIONS.txt` - Windows quick-start helpers.

---

## Core Components
### Backend (Flask, `app.py`)
- Auth + sessions: per-role session slots (`student_user`, `admin_user`), CSRF token injector, rate limiting keyed by client IP.
- DB bootstrap: `ensure_db_schema()` creates/aligns `students`, `profiles`, `exam_sessions`, `exam_results`, `violations`.
- Password flows: signed reset tokens via `itsdangerous`, SMTP config for reset emails.
- Exam lifecycle endpoints: start/end, submission scoring, rules/system check, pre-exam face verify (pass-through gate for WASM flow).
- Media handling: receives base64 frames (for MJPEG fallback), audio chunks, and combined session recordings; stores under `static/recording/`.
- Telemetry + WebRTC: Socket.IO namespaces `/student` and `/admin` handle telemetry v1/v2, live frames, audio relay, warning events, and WebRTC signalling.
- Health: `/api/system-health` snapshot, startup health checks, optional watchdog.

### Client Proctor Engine (`static/proctor_engine/runtime/*.js`)
- `proctor_core.js`: Runs YOLO ONNX (proctor_yolo.onnx) via onnxruntime-web + MediaPipe FaceLandmarker; stabilizes detections, scores lighting, head/gaze pose, accessories; produces suspicion/risk metrics and verdict.
- `student_engine.js`: Wraps ProctorCore, enforces inference FPS, emits telemetry v2 over Socket.IO, performs integrity check against `manifest.json`.
- `admin_verifier.js`: Spins worker copies of ProctorCore to re-run inference on admin-side video frames and compare against telemetry history (tamper detection).
- `config/detection_config.js`: Single source of truth for thresholds, banned labels (cell phone, clock/smartwatch, book/paper), EMA weights, risk weights.
- `manifest.json`: Generated by `sync_proctor_assets.sh`; student integrity check compares SHA-256 hashes of engine assets.

### Warning and Enforcement (`warning_system.py`)
- Default max warnings: 3; global gap 1.5 s plus per-type gaps (faster for tab switch/shortcuts, slower for gaze).
- Auto-terminate scheduled ~2 s after hitting threshold to surface the final warning; emits `exam_terminated` to student and `student_exam_terminated` to admin.
- TabSwitchDetector triggers warnings after configurable count (default 1).

### Admin UX
- Live dashboard shows counts of active students, warnings, suspicion levels, flagged objects, face counts, and last update times.
- Controls to clear warnings, force terminate, request extra frames, notify students, toggle enforcement, and trigger voice warnings.
- Records and results pages show DB-backed history; profile page updates admin details.

---

## Data & Storage
- **Database (MySQL, default examproctordb)**
  - `students`: ID, Name, Email, Password (hash), Profile image path, Role (ADMIN/STUDENT).
  - `profiles`: student_id, profile_image_path, image_type, face_detected flag, timestamps.
  - `exam_sessions`: SessionID, StudentID, StartTime, EndTime, Status (IN_PROGRESS/COMPLETED/TERMINATED).
  - `exam_results`: ResultID, StudentID, SessionID, Score, totals, Status (PASS/FAIL/TERMINATED).
  - `violations`: StudentID, SessionID, ViolationType, Details, Timestamp.
- **File storage**
  - Face captures: `static/Profiles/face_<id>_*.jpg`.
  - Audio: `static/recording/audio/<student>_<name>_<timestamp>.(webm|ogg|wav|m4a)`.
  - Session video: `static/recording/<student>_<name>_<timestamp>.(webm|mp4|ogg|mkv)` plus sidecar JSON metadata.
  - Proctor assets: `static/proctor_engine/*` (models, wasm, runtime bundles, manifest).

---

## Setup and Running
### Prerequisites
- Python 3.11+ (matching the bundled `venv` layout).
- MySQL or MariaDB running locally (defaults: host 127.0.0.1, user root, no password, db `examproctordb`).
- Optional: Node.js + pnpm/npm only if you need to rebuild proctor engine assets via `scripts/sync_proctor_assets.sh`.

### Quickstart (Windows, PowerShell)
```
python -m venv .venv
.\venv\Scripts\activate
pip install -r requirements.txt
python app.py   # serves at http://127.0.0.1:5001
```
Or run `START_EXAM_PROCTOR.bat` which bootstraps the venv, installs deps, opens the browser, and starts the server.

### Database bootstrap
- Import `examproctordb.sql` into MySQL **or** let `ensure_db_schema()` create tables on first run.
- Seed test accounts with `python create_users.py` (creates `student@test.com` / `password123` and `admin@test.com` / `admin123`).

### Proctor engine assets
- The repo already contains ONNX and WASM assets. If you regenerate from a source CV repo, run:
```
bash scripts/sync_proctor_assets.sh /path/to/comp_vision_repo
```
This copies models/wasm bundles and rebuilds `static/proctor_engine/manifest.json` with SHA-256 hashes for integrity checks.

---

## Student and Admin Flows
### Student
1) Register or sign in at `/` (role STUDENT).  
2) View rules (`/rules`) and run system check (`/systemCheck`).  
3) Pre-exam face verify (`/api/pre-exam-face-verify`) - currently passes after receiving a valid frame, as ML runs on the client.  
4) Enter exam (`/exam`), click Start Exam to create an IN_PROGRESS session, begin telemetry and warning tracking.  
5) Warnings appear in-page; after three warnings the exam auto-terminates.  
6) Submit exam (`/exam` POST). Results are computed server-side; violations/warnings are persisted.

### Admin
1) Sign in as ADMIN.  
2) Manage students (`/adminStudents`), face profiles, results (`/adminResults`), recordings (`/adminRecordings`), profile page.  
3) Live monitor via `/adminLiveMonitoring` (grid dashboard) or `/admin/live/<id>` for focused view.  
4) Actions: notify student, clear warnings, force terminate, request frame captures, toggle enforcement, trigger voice warnings, watch WebRTC or MJPEG streams, and review telemetry history (`/api/admin/student-telemetry/<id>`).

---

## Configuration (Environment Variables)
- `MYSQL_HOST` (default 127.0.0.1), `MYSQL_PORT` (3306), `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`.
- `FLASK_SECRET_KEY` (else stored in `.flask_secret_key`), `FLASK_DEBUG` (0/1).
- `COOKIE_SECURE` (1 to enforce Secure cookies), `AUTO_RESTART` (1 to restart on crash).
- Monitoring/tuning: `FAST_FACE_ONLY_MODE`, `RUN_POSE_ANALYSIS`, `OBJECT_ANALYSIS_INTERVAL_SEC`, `OBJECT_CONSEC_FRAMES`, `EYES_CLOSED_SECONDS`, `LOOKING_AWAY_SECONDS`, `NO_FACE_SECONDS`, `LEFT_SEAT_SECONDS`, `AUTO_TERMINATE_DEFAULT`, `ALLOW_SERVER_CAMERA_FALLBACK`.
- SMTP for password reset: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`, `SMTP_USE_TLS`, `SMTP_USE_SSL`, `PASSWORD_RESET_SALT`, `PASSWORD_RESET_MAX_AGE_SEC`.
- WebRTC/SocketIO: uses eventlet if available; falls back to Werkzeug threads if monitoring is disabled or SocketIO import fails.

---

## Operations and Monitoring
- Logs show availability of OpenCV, Socket.IO mode, and monitoring status at startup.
- `/api/system-health` returns DB connectivity, cache counts, and uptime for dashboards or external probes.
- Active students and warnings can be polled via `/api/admin-active-students` and `/api/all-student-warnings`.
- Runtime warning cache is kept in-memory; DB writes are best-effort to avoid blocking telemetry.

---

## Known Limits / Troubleshooting
- Server-side CV (OpenCV/Torch) is disabled by default; detection runs in browser WASM. Ensure modern browsers with WebGPU/WebAssembly support.
- `manifest.json` ships empty hashes; regenerate with `scripts/sync_proctor_assets.sh` to enforce integrity checks.
- Some legacy accounts in `examproctordb.sql` store plaintext passwords; log in once to trigger automatic hash upgrade.
- If Socket.IO/eventlet is missing, monitoring will be disabled and the app runs in basic Flask mode (no live dashboard).
- WebRTC depends on network traversal; fallback MJPEG frames continue via Socket.IO if P2P fails.
- Large model files (`yolov8n.onnx`, `face_landmarker.task`, `proctor_yolo.onnx`) must remain in place; avoid committing new weights without updating the manifest.
- Audio voice detection runs on the admin browser; ensure microphone permissions are granted on the admin side for accurate alerts.

---

---

## Face Verification Formula (Backend Logic)
The system uses the `DeepFace` framework with the `VGG-Face` model for identity verification. During the pre-exam phase, a live snapshot is compared against the student's registered profile image.

**Logic & Thresholds:**
- **Model**: `VGG-Face` (Pre-trained CNN)
- **Distance Metric**: `Cosine Similarity`
- **Verification Formula**:  
  $$Verified = \begin{cases} True, & \text{if } Distance \leq 0.40 \\ False, & \text{if } Distance > 0.40 \end{cases}$$
- **Processing Flow**:
  1. Extract face embedding (128-d or 2622-d vector) from registered image ($V_{reg}$).
  2. Extract face embedding from live snapshot ($V_{live}$).
  3. Calculate Cosine Distance: $D = 1 - \frac{V_{reg} \cdot V_{live}}{\|V_{reg}\| \|V_{live}\|}$.
  4. If $D \leq 0.40$, access is granted.

---

## License and Credits
Internal academic project (Online CheatBuster) using open-source components: Flask, Flask-SocketIO, onnxruntime-web, MediaPipe Tasks Vision, and YOLO-based ONNX models.

