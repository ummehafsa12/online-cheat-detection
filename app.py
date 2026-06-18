# app.py -- Final Merged & Fixed Version
# Features: Complete CRUD, COCO Object Detection (Phone/Book), Enhanced Monitoring

import os
import io
import struct
import numpy as np
import time
import random
import math
import base64
import json
import re
import smtplib
import secrets
import threading
import traceback
import logging
import hashlib
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    # Warm up DeepFace by building the model once
    # This will ensure the model is loaded into memory at startup
    # DeepFace.build_model("Facenet") 
except Exception:
    DEEPFACE_AVAILABLE = False

_profile_embeddings_cache = {}
FACE_VERIFY_DISTANCE_THRESHOLD = 0.45
FACE_VERIFY_MAX_AGE_SECONDS = 240
try:
    from scipy.spatial.distance import cosine as _cosine_dist
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
try:
    import eventlet
    eventlet.monkey_patch()
    EVENTLET_AVAILABLE = True
except Exception:
    EVENTLET_AVAILABLE = False
from functools import wraps
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from email.message import EmailMessage
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, flash, Response, send_from_directory, abort, session, send_file, make_response
)

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except Exception:
    bcrypt = None
    BCRYPT_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pymysql.cursors

# -------------------------
# Feature Flags (ML removed)
# -------------------------
CV2_AVAILABLE = False
cv2 = None
MEDIAPIPE_AVAILABLE = False
face_mesh_detector = None
pose_detector = None
TORCH_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False
FACE_REC_AVAILABLE = False
OCR_AVAILABLE = False
object_net_enabled = False
yolo_loaded_model_path = None
yolo_device = 'cpu'
prohibited_class_ids = []

# Fallback 1x1 BLACK JPEG (valid image) for MJPEG streaming when no frame is available
BLANK_JPEG = base64.b64decode(
    b'/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD8qqKKKAP/2Q=='
)

try:
    import requests
except ImportError:
    pass

# Try importing flask_socketio
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    SocketIO = None
    emit = None
    logger.warning("Flask-SocketIO not available. Install with: pip install flask-socketio")

class MySQL:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
            
    def init_app(self, app):
        self.app = app

    @property
    def connection(self):
        from flask import g
        if 'mysql_db' not in g:
            g.mysql_db = pymysql.connect(
                host=self.app.config.get('MYSQL_HOST', '127.0.0.1'),
                user=self.app.config.get('MYSQL_USER', 'root'),
                password=self.app.config.get('MYSQL_PASSWORD', ''),
                db=self.app.config.get('MYSQL_DB', 'examproctordb'),
                port=self.app.config.get('MYSQL_PORT', 3306),
                autocommit=True
            )
        return g.mysql_db


# -------------------------
# WebRTC / Telemetry Engine (WASM-based)
# -------------------------
profileName = None
# -------------------------
# Configuration & Globals
# -------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def _static_path(*parts):
    """Return an absolute path under the Flask static folder."""
    return os.path.join(app.static_folder, *parts)
def _get_or_create_secret_key():
    """Get a stable secret key - persists across restarts to keep sessions alive."""
    env_key = os.getenv('FLASK_SECRET_KEY', '').strip()
    if env_key:
        return env_key
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.flask_secret_key')
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            return f.read().strip()
    new_key = secrets.token_hex(32)
    try:
        with open(key_file, 'w') as f:
            f.write(new_key)
    except Exception:
        pass
    return new_key

app.secret_key = _get_or_create_secret_key()
app.config['SECRET_KEY'] = app.secret_key
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = (os.getenv('COOKIE_SECURE', '0') == '1')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

# Ensure recording folders exist so admin panel can list files immediately.
try:
    os.makedirs(_static_path('recording', 'audio'), exist_ok=True)
    os.makedirs(_static_path('Profiles'), exist_ok=True)
except Exception:
    pass

@app.before_request
def make_session_permanent():
    session.permanent = True


# Serve a tiny fallback favicon to avoid browser 404 requests for /favicon.ico
# Returns a 1x1 transparent PNG in-memory so no static file is required.
from flask import make_response
import base64

_ONE_PIXEL_PNG_B64 = (
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII='
)


@app.route('/favicon.ico')
def favicon():
    png = base64.b64decode(_ONE_PIXEL_PNG_B64)
    resp = make_response(png)
    resp.headers.set('Content-Type', 'image/png')
    resp.headers.set('Content-Length', len(png))
    return resp

# MySQL config
# Use 127.0.0.1 default to force TCP and avoid local socket/pipe resolution issues.
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', '127.0.0.1')
app.config['MYSQL_PORT'] = int(os.getenv('MYSQL_PORT', '3306'))
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'examproctordb')
mysql = MySQL(app)

# Password reset / email config
PASSWORD_RESET_SALT = os.getenv('PASSWORD_RESET_SALT', 'password-reset-salt-v1')
PASSWORD_RESET_MAX_AGE_SEC = 15 * 60
SMTP_HOST = os.getenv('SMTP_HOST', '')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USERNAME', '')
SMTP_PASS = os.getenv('SMTP_PASSWORD', '')
SMTP_FROM = os.getenv('SMTP_FROM_EMAIL', SMTP_USER or 'no-reply@example.com')
SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', '1') == '1'
SMTP_USE_SSL = os.getenv('SMTP_USE_SSL', '0') == '1'

# -------------------------
# Auth Helpers
# -------------------------
def _is_hashed_password(value):
    if not value:
        return False
    return (
        value.startswith('pbkdf2:') or
        value.startswith('scrypt:') or
        value.startswith('$2a$') or
        value.startswith('$2b$') or
        value.startswith('$2y$')
    )

def _verify_password(stored_password, candidate):
    if not stored_password:
        return False
    if stored_password.startswith('$2a$') or stored_password.startswith('$2b$') or stored_password.startswith('$2y$'):
        if not BCRYPT_AVAILABLE:
            logger.error("bcrypt is required to verify bcrypt-hashed passwords.")
            return False
        try:
            return bool(bcrypt.checkpw(candidate.encode('utf-8'), stored_password.encode('utf-8')))
        except Exception:
            return False
    if _is_hashed_password(stored_password):
        return check_password_hash(stored_password, candidate)
    # Legacy plaintext fallback
    return stored_password == candidate

def _hash_password_bcrypt(password):
    if not BCRYPT_AVAILABLE:
        raise RuntimeError("bcrypt dependency unavailable. Install with: pip install bcrypt")
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')

def _password_reset_serializer():
    return URLSafeTimedSerializer(app.secret_key)

def _build_password_reset_token(user_id, email, role):
    payload = {'uid': int(user_id), 'email': str(email), 'role': str(role)}
    return _password_reset_serializer().dumps(payload, salt=PASSWORD_RESET_SALT)

def _load_password_reset_token(token):
    return _password_reset_serializer().loads(
        token,
        salt=PASSWORD_RESET_SALT,
        max_age=PASSWORD_RESET_MAX_AGE_SEC
    )

def _send_password_reset_email(to_email, display_name, reset_link):
    if not SMTP_HOST:
        logger.warning("SMTP_HOST not configured; skipping reset email send.")
        return False
    msg = EmailMessage()
    msg['Subject'] = 'Password reset request'
    msg['From'] = SMTP_FROM
    msg['To'] = to_email
    safe_name = display_name or 'User'
    msg.set_content(
        f"Hi {safe_name},\n\n"
        f"Use this link to reset your password:\n{reset_link}\n\n"
        "This link expires in 15 minutes.\n"
        "If you did not request this, you can ignore this email.\n"
    )
    try:
        if SMTP_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=12) as server:
                if SMTP_USER:
                    server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
                if SMTP_USE_TLS:
                    server.starttls()
                if SMTP_USER:
                    server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
        return True
    except Exception as e:
        logger.error(f"Password reset email send failed: {e}")
        return False

def current_user():
    """Return the currently authenticated user from the role-keyed session slot.
    Admin and student each have their own slot so one can never overwrite the other.
    """
    return session.get('admin_user') or session.get('student_user')

def current_admin():
    """Return admin user if logged in, else None."""
    return session.get('admin_user')

def current_student():
    """Return student user if logged in, else None."""
    return session.get('student_user')

def require_role(role):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Check the role-specific session slot directly to avoid cross-role checks
            role_upper = (role or '').upper()
            if role_upper == 'ADMIN':
                user = session.get('admin_user')
            elif role_upper == 'STUDENT':
                user = session.get('student_user')
            else:
                user = current_user()

            if not user:
                if request.path.startswith('/api/'):
                    return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
                flash('Please login first.', 'error')
                return redirect(url_for('main'))
            if role_upper and (user.get('Role') or '').upper() != role_upper:
                if request.path.startswith('/api/'):
                    return jsonify({'ok': False, 'error': 'Forbidden'}), 403
                flash('Unauthorized access.', 'error')
                return redirect(url_for('main'))
            return fn(*args, **kwargs)
        return wrapper
    return decorator

# -------------------------
# CSRF + Rate Limit
# -------------------------
_rate_limit_store = {}
_rate_limit_lock = threading.Lock()

def _client_ip():
    xff = request.headers.get('X-Forwarded-For', '')
    if xff:
        return xff.split(',')[0].strip()
    return request.remote_addr or 'unknown'

def rate_limit(bucket, max_requests, window_seconds):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = f"{bucket}:{_client_ip()}"
            with _rate_limit_lock:
                rec = _rate_limit_store.get(key)
                if not rec or now >= rec['reset_at']:
                    rec = {'count': 0, 'reset_at': now + window_seconds}
                    _rate_limit_store[key] = rec
                rec['count'] += 1
                if rec['count'] > max_requests:
                    return jsonify({'error': 'Too many requests'}) if request.path.startswith('/api/') else ("Too many requests", 429)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def _ensure_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['csrf_token'] = token
    return token

@app.context_processor
def inject_csrf_token():
    return {'csrf_token': _ensure_csrf_token}

def _same_origin():
    host = request.host_url.rstrip('/')
    origin = request.headers.get('Origin')
    referer = request.headers.get('Referer')
    if origin:
        return origin.rstrip('/') == host
    if referer:
        try:
            pr = urlparse(referer)
            return f"{pr.scheme}://{pr.netloc}" == host
        except Exception:
            return False
    return False

def _get_active_session_id(student_id):
    """Retrieve the currently active session ID for a student, returning 0 if none found."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT SessionID FROM exam_sessions WHERE StudentID=%s AND Status='IN_PROGRESS' ORDER BY StartTime DESC LIMIT 1", (student_id,))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else 0
    except Exception:
        return 0

def _get_latest_session_id(student_id):
    """Retrieve the most recent session ID for a student (any status), returning 0 if none found."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT SessionID
            FROM exam_sessions
            WHERE StudentID=%s
            ORDER BY StartTime DESC, SessionID DESC
            LIMIT 1
        """, (student_id,))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else 0
    except Exception:
        return 0

def _get_active_or_latest_session_id(student_id):
    """Prefer IN_PROGRESS session; fallback to the latest session for the student."""
    sid = _get_active_session_id(student_id)
    if sid:
        return sid
    return _get_latest_session_id(student_id)

def _ensure_session_id(student_id, status='IN_PROGRESS', start_dt=None):
    """Create a fallback session if none exists (best-effort). Returns new SessionID or 0."""
    try:
        cur = mysql.connection.cursor()
        if start_dt is not None:
            cur.execute("""
                INSERT INTO exam_sessions (StudentID, StartTime, Status)
                VALUES (%s, %s, %s)
            """, (student_id, start_dt, status))
        else:
            cur.execute("""
                INSERT INTO exam_sessions (StudentID, StartTime, Status)
                VALUES (%s, NOW(), %s)
            """, (student_id, status))
        mysql.connection.commit()
        sid = cur.lastrowid
        cur.close()
        return sid
    except Exception:
        return 0

def _build_stream_placeholder(student_id, message):
    """Build a static placeholder frame for M-JPEG stream when real video isn't available."""
    if np is None or cv2 is None:
        return None
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return frame

def _overlay_status_snapshot(frame, snapshot, overlay_item):
    """Placeholder to overlay warning information onto an M-JPEG frame without ML inference."""
    if frame is None or cv2 is None:
        return frame
    
    score = snapshot.get("suspicion_score", 0)
    warnings = snapshot.get("warning_count", 0)
    
    cv2.putText(frame, f"Warnings: {warnings}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if warnings > 0 else (0, 255, 0), 2)
    cv2.putText(frame, f"Suspicion Score: {score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if score > 50 else (0, 255, 0), 2)
    
    return frame


def _bytes_has_single_face(img_bytes):
    """Return True if the provided image bytes contain exactly one face.
    Uses OpenCV Haar cascade when available; if OpenCV or cascade isn't
    available the function falls back to permissive behavior (returns True)
    to avoid blocking registrations on missing optional dependencies.
    """
    try:
        if not CV2_AVAILABLE or cv2 is None or np is None:
            return True
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = os.path.join('Haarcascades', 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            return True
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return isinstance(faces, (list, tuple, np.ndarray)) and len(faces) == 1
    except Exception:
        return True

@app.before_request
def csrf_protect():
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return
    # Exempt socket polling/engine routes
    if request.path.startswith('/socket.io'):
        return
    session_token = session.get('csrf_token')
    req_token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
    if session_token and req_token and secrets.compare_digest(session_token, req_token):
        return
    if _same_origin():
        return
    return ("CSRF validation failed", 403)

# -------------------------
# DB Schema Guard
# -------------------------
def ensure_db_schema():
    """Ensure required tables exist with the expected schema."""
    try:
        cur = mysql.connection.cursor()

        # Password hashes can exceed 100 chars (e.g., pbkdf2/scrypt).
        # Ensure column length is sufficient to prevent silent truncation.
        try:
            cur.execute("ALTER TABLE students MODIFY COLUMN Password VARCHAR(255) NOT NULL")
        except Exception:
            # Keep startup resilient if table/schema differs temporarily.
            pass

        # Exam sessions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS exam_sessions (
                SessionID INT AUTO_INCREMENT PRIMARY KEY,
                StudentID INT NOT NULL,
                StartTime DATETIME DEFAULT CURRENT_TIMESTAMP,
                EndTime DATETIME NULL,
                Status ENUM('IN_PROGRESS','COMPLETED','TERMINATED') DEFAULT 'IN_PROGRESS',
                INDEX idx_exam_sessions_student (StudentID)
            )
        """)

        # Exam results
        cur.execute("""
            CREATE TABLE IF NOT EXISTS exam_results (
                ResultID INT AUTO_INCREMENT PRIMARY KEY,
                StudentID INT NOT NULL,
                SessionID INT NOT NULL,
                Score DECIMAL(5,2) DEFAULT 0,
                TotalQuestions INT DEFAULT 125,
                CorrectAnswers INT DEFAULT 0,
                SubmissionTime DATETIME DEFAULT CURRENT_TIMESTAMP,
                Status ENUM('PASS','FAIL','TERMINATED') DEFAULT 'FAIL',
                Attempts INT DEFAULT 1,
                FailureReasons TEXT,
                INDEX idx_exam_results_student (StudentID),
                INDEX idx_exam_results_session (SessionID)
            )
        """)
        
        # Patch attempts logic without deleting old data
        try:
            cur.execute("ALTER TABLE exam_results ADD COLUMN IF NOT EXISTS Attempts INT DEFAULT 1")
            cur.execute("ALTER TABLE exam_results ADD COLUMN IF NOT EXISTS FailureReasons TEXT")
            cur.execute("ALTER TABLE exam_results MODIFY COLUMN Status ENUM('PASS','FAIL','TERMINATED') DEFAULT 'FAIL'")
            cur.execute("ALTER TABLE exam_sessions MODIFY COLUMN Status ENUM('IN_PROGRESS','COMPLETED','TERMINATED') DEFAULT 'IN_PROGRESS'")
        except Exception as patch_err:
            pass

        # Violations
        cur.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                ViolationID INT AUTO_INCREMENT PRIMARY KEY,
                StudentID INT NOT NULL,
                SessionID INT NOT NULL,
                ViolationType VARCHAR(64) NOT NULL,
                Details TEXT,
                Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_violations_student (StudentID),
                INDEX idx_violations_session (SessionID)
            )
        """)

        mysql.connection.commit()
        cur.close()
    except Exception as e:
        logger.error(f"DB schema ensure failed: {e}", exc_info=True)

# SocketIO
if SOCKETIO_AVAILABLE:
    try:
        socketio = SocketIO(
            app,
            cors_allowed_origins="*",
            async_mode="threading",
            ping_timeout=60,
            ping_interval=20,
            max_http_buffer_size=20_000_000,
        )
        MONITORING_ENABLED = True
        logger.info("SocketIO initialized successfully")
    except Exception as e:
        logger.error(f"SocketIO init failed: {e}")
        socketio = None
        MONITORING_ENABLED = False
else:
    socketio = None
    MONITORING_ENABLED = False

# Camera & detection globals
CAMERA_INDEX = 0

# Object detection logic is now handled strictly in WASM on the client side.
# Import monitoring modules
try:
    from warning_system import WarningSystem, TabSwitchDetector
    MONITORING_MODULES_AVAILABLE = True
    logger.info("Monitoring modules imported successfully")
except Exception as e:
    logger.warning(f"Monitoring modules import failed: {e}")
    MONITORING_MODULES_AVAILABLE = False
    # Define fallback classes
    class WarningSystem:
        def __init__(self, *args, **kwargs):
            self.warnings = {}
            self.student_names = {}
            self.max_warnings = 3
            self.lock = threading.Lock()
        def initialize_student(self, *args, **kwargs): pass
        def add_warning(self, *args, **kwargs): return False
    class TabSwitchDetector:
        def __init__(self, *args, **kwargs): pass
        def initialize_student(self, *args, **kwargs): pass
        def detect_tab_switch(self, *args, **kwargs): return {'terminated': False, 'count': 0}

# Instantiate monitoring helpers (real or fallback)
warning_system = WarningSystem(socketio, max_warnings=3) if MONITORING_MODULES_AVAILABLE else WarningSystem()
tab_switch_detector = TabSwitchDetector(warning_system)

studentInfo = None
detection_threads_started = False
latest_student_frames = {}
latest_student_frames_lock = threading.Lock()
# Throttled debug tracker for live-frame reception
_last_frame_log_at = {}
student_detection_state = {}
student_detection_state_lock = threading.Lock()
active_exam_students = set()
active_exam_students_lock = threading.Lock()
student_frame_rx_counts = {}
student_frame_rx_lock = threading.Lock()
student_stale_violation_at = {}
student_stale_violation_lock = threading.Lock()
runtime_warning_state = {}
runtime_warning_state_lock = threading.Lock()
# Track active student socket session ids for targeted emits
_student_socket_sids = {}
_student_socket_sids_lock = threading.Lock()
# Throttle server-originated object alerts per student/label
# Throttle server-originated object alerts per student/label
_last_object_alert_at = {}
# Track latest v2 telemetry timestamps to avoid double-processing v1/v2
_last_v2_telemetry_at = {}
# Cache for admin-active-students API (10s TTL)
_student_cache = {'data': None, 'ts': 0}
# Track which students have already received a 'feed_started' emit so we only send it once per session
_feed_started_for = set()
_feed_started_lock = threading.Lock()
# Application start time for uptime reporting
_APP_START_TIME = time.time()

# Thresholds for Eye Tracking
EAR_THRESHOLD = 0.22                          # Balanced: tolerant of natural blinks, flags sustained closure
EYES_CLOSED_SECONDS = float(os.getenv('EYES_CLOSED_SECONDS', '1.2'))   # warn if eyes closed >1.2s
LOOKING_AWAY_SECONDS = float(os.getenv('LOOKING_AWAY_SECONDS', '3.0'))  # warn if gaze away >3s
NO_FACE_SECONDS = float(os.getenv('NO_FACE_SECONDS', '1.0'))           # warn if no face >1s
SEAT_RISE_RATIO_THRESHOLD = 0.38
LEAN_RATIO_THRESHOLD = 0.28
MOTION_AREA_RATIO_THRESHOLD = 0.020          # moderate sensitivity to movement
LEFT_SEAT_SECONDS = 4.0                       # allow brief posture shifts
MOVEMENT_DISTRACTION_SECONDS = 3.5           # sustained movement before warning
POSE_ANALYSIS_FPS = 12.0
CAMERA_BLOCKED_BRIGHTNESS = 32               # catch covered camera without punishing low light
CAMERA_BLOCKED_SECONDS = 1.0                 # seconds of dark frame before CAMERA_OFF warning
# Priority mode to stabilize live stream + face detection first.
FAST_FACE_ONLY_MODE = (os.getenv('FAST_FACE_ONLY_MODE', '0') == '1')
RUN_POSE_ANALYSIS = (os.getenv('RUN_POSE_ANALYSIS', '1') == '1')
OBJECT_ANALYSIS_INTERVAL_SEC = float(os.getenv('OBJECT_ANALYSIS_INTERVAL_SEC', '0.10'))
OBJECT_CONSEC_FRAMES = int(os.getenv('OBJECT_CONSEC_FRAMES', '3'))

logger.info(
    f"Detection config: fast_face_only={FAST_FACE_ONLY_MODE}"
)

# ------------------------------------------------------------------
# Health check utilities (startup + live watchdog)
# ------------------------------------------------------------------
CRITICAL_ASSETS = [
    ("face_landmarker.task", os.path.join(BASE_DIR, "face_landmarker.task")),
    ("blaze_face_short_range.tflite", os.path.join(BASE_DIR, "blaze_face_short_range.tflite")),
    ("yolov8n.onnx", os.path.join(BASE_DIR, "yolov8n.onnx")),
]
health_watchdog_started = False


def _hash_file(path):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.debug(f"Health hash failed for {path}: {e}")
        return None


def _check_db_connection():
    try:
        conn = mysql.connection
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        return True, "Database reachable"
    except Exception as e:
        return False, f"DB check failed: {e}"


def run_startup_health_checks():
    logger.info("=" * 60)
    logger.info("Health: running startup checks")
    checks = []

    checks.append(("SocketIO monitoring", MONITORING_ENABLED and socketio is not None,
                   "SocketIO initialized" if MONITORING_ENABLED and socketio else "SocketIO disabled"))

    checks.append(("WebRTC signaling", MONITORING_ENABLED and socketio is not None,
                   "request_webrtc_stream / ICE handlers loaded"))

    db_ok, db_msg = _check_db_connection()
    checks.append(("Database connectivity", db_ok, db_msg))

    checks.append(("Pose thresholds sane", 0.18 <= EAR_THRESHOLD <= 0.28 and 0.5 <= EYES_CLOSED_SECONDS <= 3.0,
                   f"EAR={EAR_THRESHOLD}, eyes_closed={EYES_CLOSED_SECONDS}s, away={LOOKING_AWAY_SECONDS}s"))

    for name, ok, detail in checks:
        if ok:
            logger.info(f"[PASS] {name}: {detail}")
        else:
            logger.warning(f"[FAIL] {name}: {detail}")

    # Asset integrity
    for label, path in CRITICAL_ASSETS:
        h = _hash_file(path)
        if h:
            logger.info(f"[ASSET] {label} md5={h}")
        else:
            logger.warning(f"[ASSET] {label} missing or unreadable at {path}")

    logger.info("Health: startup checks complete")
    logger.info("=" * 60)


def start_health_watchdog(interval_seconds: float = 5.0):
    global health_watchdog_started
    if health_watchdog_started or not MONITORING_ENABLED or not socketio:
        return
    health_watchdog_started = True

    def loop():
        last_info = 0
        while True:
            now = time.time()
            stale = []
            with latest_student_frames_lock:
                for sid, entry in latest_student_frames.items():
                    ts = entry.get('frame_timestamp') or entry.get('timestamp') or 0
                    if ts and now - ts > 5:
                        stale.append(sid)
            if stale:
                logger.warning(f"[HEALTH] No live frames from {', '.join(stale)} in last 5s")

            # periodic heartbeat
            if now - last_info > 30:
                last_info = now
                with active_exam_students_lock:
                    active = len(active_exam_students)
                logger.info(f"[HEALTH] heartbeat: active_students={active}, frames_cached={len(latest_student_frames)}")
            time.sleep(interval_seconds)

    threading.Thread(target=loop, daemon=True, name="health-watchdog").start()

def _record_runtime_warning(student_id, student_name, violation_type, details):
    sid = str(student_id)
    if warning_system is None:
        # If the warning system isn't ready, avoid client-side auto-terminate by keeping count at 0.
        return 0, {
            'type': str(violation_type or 'UNKNOWN').upper(),
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'details': str(details or '').strip()
        }
    with runtime_warning_state_lock:
        rec = runtime_warning_state.setdefault(sid, {
            'warnings': 0,
            'student_name': str(student_name or 'Unknown'),
            'violations': [],
            'start_time': int(time.time() * 1000)
        })
        rec['student_name'] = str(student_name or rec.get('student_name') or 'Unknown')
        if rec['warnings'] < 3:
            rec['warnings'] += 1
            violation = {
                'type': str(violation_type or 'UNKNOWN').upper(),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'details': str(details or '').strip()
            }
            rec['violations'].append(violation)
            count = int(rec['warnings'])
            return count, violation
        else:
            count = 3
            # Return last violation or dummy
            dummy = rec['violations'][-1] if rec['violations'] else {
                'type': str(violation_type or 'UNKNOWN').upper(),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'details': str(details or '').strip()
            }
            return count, dummy

def _get_runtime_warning_state(student_id):
    sid = str(student_id)
    with runtime_warning_state_lock:
        rec = dict(runtime_warning_state.get(sid) or {})
        return {
            'warnings': int(rec.get('warnings') or 0),
            'student_name': str(rec.get('student_name') or 'Unknown'),
            'violations': list(rec.get('violations') or []),
            'start_time': rec.get('start_time')
        }

# Friendly labels for UI rendering
VIOLATION_LABELS = {
    'TAB_SWITCH': 'Tab switching',
    'PROHIBITED_SHORTCUT': 'Prohibited shortcut',
    'PROHIBITED_OBJECT': 'Prohibited object',
    'MULTIPLE_FACES': 'Multiple faces',
    'NO_FACE': 'No face detected',
    'EYES_CLOSED': 'Eyes closed',
    'DISTRACTION': 'Distraction',
    'HEAD_MOVEMENT': 'Head movement',
    'HEAD_DOWN': 'Head down',
    'HEAD_POSE': 'Head pose',
    'GAZE_AWAY': 'Gaze away',
    'VOICE_DETECTED': 'Voice detected',
    'IDENTITY_MISMATCH': 'Identity mismatch',
    'TERMINATED_BY_ADMIN': 'Terminated by admin',
}

def _friendly_violation_type(vtype):
    key = str(vtype or 'UNKNOWN').upper()
    return VIOLATION_LABELS.get(key, key.replace('_', ' ').title())

def _maybe_issue_object_warning(student_id, student_name, label):
    """Server-side object detection -> warning + student notification (book/paper)."""
    sid = str(student_id)
    lbl = str(label or '').strip().lower()
    if not sid or not lbl:
        return
    now = time.time()
    key = f"{sid}:{lbl}"
    last = float(_last_object_alert_at.get(key, 0.0))
    # Small throttle to avoid repeated warnings on every frame
    if now - last < 0.8:
        return
    _last_object_alert_at[key] = now

    details = f"Prohibited item detected: {lbl}"
    target_room = f"student:{sid}"
    try:
        with _student_socket_sids_lock:
            target_sid = _student_socket_sids.get(sid)
    except Exception:
        target_sid = None
    # Ensure warning_system student initialized
    if warning_system:
        if sid not in warning_system.warnings:
            warning_system.initialize_student(sid, student_name or f"Student {sid}")
        pre = warning_system.get_warnings(sid)
        warning_system.add_warning(sid, 'PROHIBITED_OBJECT', details, emit_to_student=True)
        post = warning_system.get_warnings(sid)
        if post > pre:
            _record_runtime_warning(sid, student_name, 'PROHIBITED_OBJECT', details)
            # Push a direct UI warning to student + admin for visibility
            if socketio:
                payload = {
                    'student_id': sid,
                    'label': lbl,
                    'details': details,
                    'total_warnings': post
                }
                target = target_sid or target_room
                socketio.emit('server_object_detected', payload, namespace='/student', to=target)
                socketio.emit('server_object_detected', payload, namespace='/admin')
                violation_payload = {
                    'student_id': sid,
                    'student_name': student_name,
                    'total_warnings': post,
                    'violation': {'type': 'PROHIBITED_OBJECT', 'details': details},
                    'type': 'PROHIBITED_OBJECT',
                    'details': details,
                    'source': 'server'
                }
                target = target_sid or target_room
                socketio.emit('student_violation', violation_payload, namespace='/student', to=target)
                socketio.emit('student_violation', violation_payload, namespace='/admin')
    else:
        count, _ = _record_runtime_warning(sid, student_name, 'PROHIBITED_OBJECT', details)
        if socketio:
            payload = {
                'student_id': sid,
                'label': lbl,
                'details': details,
                'total_warnings': count
            }
            target = target_sid or target_room
            socketio.emit('server_object_detected', payload, namespace='/student', to=target)
            socketio.emit('server_object_detected', payload, namespace='/admin')
            violation_payload = {
                'student_id': sid,
                'student_name': student_name,
                'total_warnings': count,
                'violation': {'type': 'PROHIBITED_OBJECT', 'details': details},
                'type': 'PROHIBITED_OBJECT',
                'details': details,
                'source': 'server'
            }
            target = target_sid or target_room
            socketio.emit('student_violation', violation_payload, namespace='/student', to=target)
            socketio.emit('student_violation', violation_payload, namespace='/admin')

def _maybe_issue_behavioral_warning(student_id, student_name, vtype, details):
    """Server-side behavioral warning + student notification."""
    sid = str(student_id)
    if not sid: return
    now = time.time()
    key = f"{sid}:{vtype}"
    # Throttle between consecutive warnings of the same type
    # Increased to 3.0 to prevent 'false' spam while remaining responsive
    throttle = 3.0 if vtype == 'NO_FACE' else 4.0
    last = float(_last_object_alert_at.get(key, 0.0))
    if now - last < throttle:
        return
    _last_object_alert_at[key] = now

    if warning_system:
        if sid not in warning_system.warnings:
            warning_system.initialize_student(sid, student_name or f"Student {sid}")
        pre = warning_system.get_warnings(sid)
        warning_system.add_warning(sid, vtype, details, emit_to_student=True)
        post = warning_system.get_warnings(sid)
        if post > pre:
             _record_runtime_warning(sid, student_name, vtype, details)


def _reset_exam_runtime_state(student_id, started_at_ms=None, student_name=None):
    sid = str(student_id)
    with runtime_warning_state_lock:
        prev = runtime_warning_state.get(sid, {})
        name = student_name or prev.get('student_name') or 'Unknown'
        runtime_warning_state[sid] = {
            'warnings': 0,
            'student_name': name,
            'violations': [],
            'start_time': int(started_at_ms) if started_at_ms is not None else prev.get('start_time')
        }
    if warning_system:
        warning_system.reset_student(sid)

def write_violation_async(student_id, session_id, vtype, details):
    """Non-blocking violation insert to keep telemetry path fast."""
    def _task():
        try:
            with app.app_context():
                sid = int(session_id or 0) if session_id else 0
                if not sid:
                    sid = int(_get_active_or_latest_session_id(student_id) or 0)
                if not sid:
                    # Last resort: create a session so violations can be persisted.
                    sid = int(_ensure_session_id(student_id, status='IN_PROGRESS') or 0)
                if not sid:
                    return
                vtype_norm = str(vtype or 'UNKNOWN').upper()
                details_norm = str(details or '')[:500]
                cur = mysql.connection.cursor()
                cur.execute("""
                    INSERT INTO violations (StudentID, SessionID, ViolationType, Details, Timestamp)
                    VALUES (%s, %s, %s, %s, NOW())
                """, (student_id, sid, vtype_norm, details_norm))
                mysql.connection.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Violation write failed for student {student_id}: {e}")
    try:
        if EVENTLET_AVAILABLE:
            eventlet.spawn(_task)
        else:
            threading.Thread(target=_task, daemon=True).start()
    except Exception as e:
        logger.error(f"Failed to schedule violation write: {e}")

# Wire violation writer into warning system for persistence/flush
try:
    if warning_system:
        warning_system.violation_writer = write_violation_async
except Exception:
    pass

def _end_exam_runtime_state(student_id, clear_warning_cache=True):
    """Remove student from live monitoring state and optionally reset warning cache."""
    sid = str(student_id)
    with active_exam_students_lock:
        active_exam_students.discard(sid)
    with latest_student_frames_lock:
        latest_student_frames.pop(sid, None)
    with student_detection_state_lock:
        student_detection_state.pop(sid, None)
    if clear_warning_cache:
        with runtime_warning_state_lock:
            runtime_warning_state.pop(sid, None)
    # Reset feed_started tracking so the event fires again if student reconnects
    with _feed_started_lock:
        _feed_started_for.discard(sid)
    if warning_system and clear_warning_cache:
        warning_system.reset_student(sid)

# -------------------------
@app.route('/')
def main():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
@rate_limit('login', max_requests=12, window_seconds=60)
def login():
    email = (request.form.get('username') or '').strip()  # This is actually email
    password = request.form.get('password') or ''

    if not email or not password:
        flash('Please enter both email and password.', 'login_error')
        return redirect(url_for('main'))
    
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT ID, Name, Email, Password, Role FROM students WHERE Email=%s", (email,))
        data = cur.fetchone()

        if not data:
            cur.close()
            flash('No account found with this email. Please register first.', 'login_error')
            return redirect(url_for('main'))

        if not _verify_password(data[3], password):
            cur.close()
            flash('Invalid password. Please try again.', 'login_error')
            return redirect(url_for('main'))
        
        student_id, name, email, password_db, role = data
        # Upgrade legacy plaintext password to hash on successful login
        if not _is_hashed_password(password_db):
            try:
                cur.execute("UPDATE students SET Password=%s WHERE ID=%s", (generate_password_hash(password), student_id))
                mysql.connection.commit()
            except Exception:
                mysql.connection.rollback()

        # Attempt limit feature disabled
        # if role == 'STUDENT':
        #     try:
        #         cur.execute("SELECT MAX(Attempts) FROM exam_results WHERE StudentID=%s", (student_id,))
        #         row = cur.fetchone()
        #         max_attempts = int(row[0]) if row and row[0] is not None else 0
        #         if max_attempts >= 5:
        #             cur.close()
        #             flash('You have used all 5 exam attempts. You are permanently dismissed and cannot retake this exam.', 'login_error')
        #             return redirect(url_for('main'))
        #     except Exception as e:
        #         logger.error(f"Error checking attempts limit during login: {e}")

        session.permanent = True
        user_data = {
            "Id": str(student_id),
            "Name": name,
            "Email": email,
            "Role": role
        }
        if role == 'ADMIN':
            # Admin gets its own slot — never overrides student session
            session['admin_user'] = user_data
        else:
            # Student gets its own slot — never overrides admin session
            session['student_user'] = user_data
        cur.close()
        
        if role == 'STUDENT':
            return redirect(url_for('rules'))
        else:
            return redirect(url_for('adminStudents'))
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        flash('Login failed due to a server error. Please try again.', 'login_error')
        return redirect(url_for('main'))

@app.route('/forgot-password', methods=['GET', 'POST'])
@rate_limit('forgot_password', max_requests=6, window_seconds=300)
def forgot_password():
    generic_msg = 'If an account with that email exists, a reset link has been sent.'
    if request.method == 'GET':
        return render_template('forgot_password.html')

    email = (request.form.get('email') or '').strip().lower()
    try:
        if email:
            cur = mysql.connection.cursor()
            cur.execute("SELECT ID, Name, Email, Role FROM students WHERE Email=%s LIMIT 1", (email,))
            row = cur.fetchone()
            cur.close()
            if row:
                uid, name, user_email, role = row
                token = _build_password_reset_token(uid, user_email, role)
                reset_url = url_for('reset_password', token=token, _external=True)
                _send_password_reset_email(user_email, name, reset_url)
    except Exception as e:
        logger.error(f"Forgot password flow error: {e}")

    # Generic response regardless of whether account exists (prevents enumeration).
    flash(generic_msg, 'login_success')
    return redirect(url_for('main'))

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
@rate_limit('reset_password', max_requests=20, window_seconds=300)
def reset_password(token):
    token_data = None
    try:
        token_data = _load_password_reset_token(token)
    except SignatureExpired:
        flash('This password reset link has expired. Please request a new one.', 'login_error')
        return redirect(url_for('forgot_password'))
    except BadSignature:
        flash('Invalid password reset link.', 'login_error')
        return redirect(url_for('forgot_password'))
    except Exception as e:
        logger.error(f"Reset token validation error: {e}")
        flash('Invalid password reset link.', 'login_error')
        return redirect(url_for('forgot_password'))

    if request.method == 'GET':
        return render_template('reset_password.html', token=token)

    new_password = request.form.get('password') or ''
    confirm_password = request.form.get('confirm_password') or ''
    if len(new_password) < 8:
        flash('Password must be at least 8 characters long.', 'login_error')
        return render_template('reset_password.html', token=token)
    if new_password != confirm_password:
        flash('Passwords do not match.', 'login_error')
        return render_template('reset_password.html', token=token)

    try:
        new_hash = _hash_password_bcrypt(new_password)
    except Exception as e:
        logger.error(f"Password hashing failed: {e}")
        flash('Unable to reset password right now. Please try again later.', 'login_error')
        return render_template('reset_password.html', token=token)

    try:
        uid = int(token_data.get('uid'))
        email = str(token_data.get('email') or '').strip().lower()
        role = str(token_data.get('role') or '').upper()
        if role not in ('STUDENT', 'ADMIN'):
            flash('Invalid password reset link.', 'login_error')
            return redirect(url_for('forgot_password'))

        cur = mysql.connection.cursor()
        cur.execute("UPDATE students SET Password=%s WHERE ID=%s AND Email=%s AND Role=%s", (new_hash, uid, email, role))
        mysql.connection.commit()
        changed = int(cur.rowcount or 0)
        cur.close()
        if changed < 1:
            flash('Unable to reset password. Please request a new reset link.', 'login_error')
            return redirect(url_for('forgot_password'))
    except Exception as e:
        mysql.connection.rollback()
        logger.error(f"Password reset DB update failed: {e}")
        flash('Unable to reset password right now. Please try again later.', 'login_error')
        return render_template('reset_password.html', token=token)

    flash('Password reset successful. Please sign in with your new password.', 'login_success')
    return redirect(url_for('main'))

@app.route('/register', methods=['POST'])
@rate_limit('register', max_requests=8, window_seconds=300)
def register():
    if request.method == 'POST':
        fullname = (request.form.get('fullname') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''
        profile_picture = request.files.get('profile_picture')
        webcam_image = request.form.get('webcam_image')

        if not fullname or not email or not password:
            flash('Name, email, and password are required.', 'register_error')
            return redirect(url_for('main', register='true'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'register_error')
            return redirect(url_for('main', register='true'))
        if confirm_password and password != confirm_password:
            flash('Passwords do not match.', 'register_error')
            return redirect(url_for('main', register='true'))
        
        # Check if email already exists
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM students WHERE Email = %s", (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            cursor.close()
            flash(f'Email "{email}" is already registered. Please login or use another email.', 'register_error')
            return redirect(url_for('main', register='true'))
        
        # Handle profile picture
        profile_filename = None
        
        if profile_picture and profile_picture.filename:
            # Save uploaded file
            filename = secure_filename(profile_picture.filename)
            profile_filename = f"{email}_{filename}"
            img_bytes = profile_picture.read()
            profile_picture.seek(0)
            if not _bytes_has_single_face(img_bytes):
                flash('Profile image must contain exactly one clear human face.', 'register_error')
                return redirect(url_for('main', register='true'))
            os.makedirs('static/Profiles', exist_ok=True)
            with open(os.path.join('static/Profiles', profile_filename), 'wb') as f:
                f.write(img_bytes)
            
        elif webcam_image:
            # Save webcam captured image
            img_data = webcam_image.split(',', 1)[1] if ',' in webcam_image else webcam_image
            img_bytes = base64.b64decode(img_data)
            if not _bytes_has_single_face(img_bytes):
                flash('Captured image must contain exactly one clear human face.', 'register_error')
                return redirect(url_for('main', register='true'))
            profile_filename = f"{email}_webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            
            os.makedirs('static/Profiles', exist_ok=True)
            with open(os.path.join('static/Profiles', profile_filename), 'wb') as f:
                f.write(img_bytes)
        
        if not profile_filename:
            flash('Profile image is required. Please upload a photo or capture from webcam.', 'register_error')
            return redirect(url_for('main', register='true'))

        try:
            # Insert into students table - handle both with and without Profile column
            try:
                # Try with Profile column
                cursor.execute(
                    "INSERT INTO students (Name, Email, Password, Profile, Role) VALUES (%s, %s, %s, %s, %s)",
                        (fullname, email, generate_password_hash(password), profile_filename, 'STUDENT')
                )
            except Exception as col_error:
                error_str = str(col_error)
                logger.warning(f"Initial register attempt failed: {error_str}")
                
                # If Profile column doesn't exist, insert without it
                if "Unknown column 'Profile'" in error_str or "field list" in error_str.lower():
                    logger.info("Retrying registration without Profile column...")
                    cursor.execute(
                        "INSERT INTO students (Name, Email, Password, Role) VALUES (%s, %s, %s, %s)",
                        (fullname, email, generate_password_hash(password), 'STUDENT')
                    )
                else:
                    raise col_error
            
            mysql.connection.commit()
            cursor.close()
            
            logger.info(f"Registration successful for email: {email}")
            flash('Registration successful! Please sign in now.', 'register_success')
            return redirect(url_for('main'))
            
        except Exception as e:
            if mysql.connection:
                mysql.connection.rollback()
            if cursor:
                cursor.close()
            logger.error(f"Registration error for {email}: {e}", exc_info=True)
            flash(f'Error during registration: {str(e)}', 'register_error')
            return redirect(url_for('main', register='true'))
    
    return redirect(url_for('main'))

@app.route('/logout')
def logout():
    global detection_threads_started
    # Determine who is logging out so we only pop the right session slot
    student = session.get('student_user')
    admin = session.get('admin_user')
    
    if student:
        # Student logout: clean up monitoring threads then pop only the student slot
        sid_int = None
        try:
            sid_int = int(student['Id'])
        except Exception:
            pass
        if sid_int is not None:
            # legacy python monitor disabled
            with active_exam_students_lock:
                active_exam_students.discard(sid_int)
            with latest_student_frames_lock:
                latest_student_frames.pop(sid_int, None)
            with student_detection_state_lock:
                student_detection_state.pop(sid_int, None)
        detection_threads_started = False
        # camera_streamer.release() # This line was commented out in the original code, keeping it that way.
        # Only clear student-related keys — admin session preserved
        session.pop('student_user', None)
        session.pop('student_face_verified_at', None)
        session.pop('face_verified_at', None)

    elif admin:
        # Admin logout: only remove admin slot — student exam session preserved
        session.pop('admin_user', None)

    return redirect(url_for('main'))


@app.route('/admin-logout')
def admin_logout():
    """Dedicated admin logout — ONLY clears the admin session slot."""
    session.pop('admin_user', None)
    return redirect(url_for('main'))

@app.route('/rules')
@require_role('STUDENT')
def rules():
    return render_template('ExamRules.html')

# @app.route('/faceInput')
# @require_role('STUDENT')
# def faceInput():
#     # Release any server-held webcam so browser capture can open camera reliably.
#     try:
#         # camera_streamer.release() # This line was commented out in the original code, keeping it that way.
#         pass
#     except Exception:
#         pass
#     user = current_user()
#     # legacy python monitor disabled
#     with active_exam_students_lock:
#         if user:
#             active_exam_students.discard(int(user['Id']))
#     return render_template('ExamFaceInput.html')

@app.route('/video_capture')
def video_capture():
    """Stream MJPEG for face capture page (simple preview)."""
    def gen():
        try:
            # camera_streamer.start() # This line was commented out in the original code, keeping it that way.
            pass
        except Exception as e:
            logger.error(f"video_capture start error: {e}")
            return
        while True:
            try:
                # frame = camera_streamer.read() # This line was commented out in the original code, keeping it that way.
                frame = np.zeros((480, 640, 3), dtype=np.uint8) # Placeholder frame
            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                break
            # draw rectangles for preview optionally
            if CV2_AVAILABLE:
                # faces = detect_faces(frame) # This line was commented out in the original code, keeping it that way.
                # for f in faces:
                #     cv2.rectangle(frame, (f['x'], f['y']), (f['x'] + f['w'], f['y'] + f['h']), (0,255,0), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/saveFaceInput', methods=['POST'])
def saveFaceInput():
    global profileName
    
    try:
        # Client se JSON data receive karein
        data = request.get_json()
        # Hum maan rahe hain ki client 'image_data' key se Base64 string bhej raha hai
        image_data_b64 = data.get('image_data') 

        if not image_data_b64:
            flash('No image data received.', 'error')
            # Frontend ko bata dein ki error hai
            return jsonify({'status': 'error', 'message': 'No image data'}), 400

        # Data URL prefix remove karein (e.g., 'data:image/png;base64,')
        if ',' in image_data_b64:
            image_data_b64 = image_data_b64.split(',', 1)[1]

        # Base64 data ko decode karke image mein badlein
        image_bytes = base64.b64decode(image_data_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # This line was commented out in the original code, keeping it that way.
        frame = np.zeros((480, 640, 3), dtype=np.uint8) # Placeholder frame

        if frame is None:
            raise Exception("Could not decode image data.")
            
        # File name banayein aur save karein (assuming 'static/profiles' folder exists)
        profileName = f"profile_{int(time.time())}.jpg"
        save_path = os.path.join('static', 'profiles', profileName) 
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        # cv2.imwrite(save_path, frame) # This line was commented out in the original code, keeping it that way.
        
        flash('Face captured successfully and saved.', 'success')
        
        # Seedha Exam System Check page par redirect karein (User ki zaroorat ke mutabik)
        return redirect(url_for('systemCheck'))

    except Exception as e:
        logger.error(f"saveFaceInput error: {e}")
        flash(f'Failed to process or save image: {e}', 'error')
        # Error hone par wapis face input page par bhej dein
        return redirect(url_for('faceInput'))

# @app.route('/confirmFaceInput')
# def confirmFaceInput():
#     profile = profileName
#     # return render_template('ExamConfirmFaceInput.html', profile=profile)
#     return redirect(url_for('systemCheck'))

@app.route('/systemCheck')
@require_role('STUDENT')
def systemCheck():
    return render_template('ExamSystemCheck.html')

@app.route('/systemCheck', methods=['POST'])
@require_role('STUDENT')
def systemCheckRoute():
    examData = request.json or {}
    output = 'exam'
    # simple example check:
    inputs = examData.get('input', '')
    if 'Not available' in inputs:
        output = 'systemCheckError'
    return jsonify({"output": output})

@app.route('/exam')
@require_role('STUDENT')
def exam():
    """Load exam page and prepare camera; monitoring starts when student clicks Start Exam."""
    global detection_threads_started
    user = current_user()
    
    ensure_db_schema()
    
    try:
        # Do not grab physical webcam on server by default.
        # Browser-based capture is used for pre-exam verification and monitoring frames.
        # Enabling server camera can conflict with browser camera access on Windows.
        use_server_camera = (os.getenv('ALLOW_SERVER_CAMERA_FALLBACK', '0') == '1')
        if use_server_camera:
            # camera_streamer.start() # This line was commented out in the original code, keeping it that way.
            print("✅ Server camera started (fallback mode)")
            # print(f"✅ Camera running: {camera_streamer.running}") # This line was commented out in the original code, keeping it that way.
        else:
            # camera_streamer.release() # This line was commented out in the original code, keeping it that way.
            print("✅ Browser camera mode active (server camera disabled)")
    except Exception as e:
        if os.getenv('ALLOW_SERVER_CAMERA_FALLBACK', '0') == '1':
            logger.error(f"Exam camera start error: {e}")
            flash('Camera not accessible. Please check camera permissions.', 'error')
            return redirect(url_for('systemCheck'))
        logger.warning(f"Server camera release/start warning ignored in browser camera mode: {e}")

    # Prepare monitoring identity
    student_id = user['Id']
    student_name = user['Name']
    
    # Attempt limit feature disabled
    # try:
    #     cur = mysql.connection.cursor()
    #     cur.execute("SELECT MAX(Attempts) FROM exam_results WHERE StudentID=%s", (student_id,))
    #     row = cur.fetchone()
    #     cur.close()
    #     max_attempts = int(row[0]) if row and row[0] is not None else 0
    #     if max_attempts >= 5:
    #         flash('You have used all 5 exam attempts. You are permanently dismissed and cannot retake this exam.', 'error')
    #         return redirect(url_for('showResultFail', reason='Maximum attempts reached.'))
    # except Exception as db_err:
    #     logger.error(f"Error checking attempts limit: {db_err}")
    
    print(f"🎯 Exam page ready for {student_name} (ID: {student_id})")
    # Strict gate: student must complete pre-exam face verification before exam session can start.
    session['face_verified_for_exam'] = False
    session.pop('student_face_verified_at', None)
    
    return render_template('Exam.html', 
                         student_id=student_id, 
                         max_warnings=3, 
                         monitoring_enabled=MONITORING_ENABLED,
                         wasm_proctor_enabled=True)

@app.route('/api/exam-session/start', methods=['POST'])
@require_role('STUDENT')
@rate_limit('exam_start', max_requests=10, window_seconds=60)
def examSessionStart():
    """Start monitoring/warnings only after student explicitly starts exam."""
    global detection_threads_started
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401

    student_id = str(user['Id'])
    student_name = user['Name']
    started_at_ms = int(time.time() * 1000)
    face_verified = bool(session.get('face_verified_for_exam'))
    verified_at = session.get('student_face_verified_at')
    if not face_verified:
        return jsonify({'ok': False, 'error': 'Face verification required before starting exam.'}), 403

    try:
        verify_age = time.time() - float(verified_at or 0)
    except Exception:
        verify_age = FACE_VERIFY_MAX_AGE_SECONDS + 1

    if verify_age > FACE_VERIFY_MAX_AGE_SECONDS:
        session['face_verified_for_exam'] = False
        session.pop('student_face_verified_at', None)
        return jsonify({'ok': False, 'error': 'Face verification expired. Please verify again.'}), 403

    try:
        with active_exam_students_lock:
            already_active = student_id in active_exam_students
            if not already_active:
                active_exam_students.add(student_id)

        _reset_exam_runtime_state(student_id, started_at_ms, student_name)
        # legacy python monitor disabled

        # Create a fresh IN_PROGRESS session (best-effort; don't block monitoring if DB hiccups)
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                UPDATE exam_sessions SET Status='COMPLETED', EndTime=NOW()
                WHERE StudentID=%s AND Status='IN_PROGRESS'
            """, (student_id,))
            cur.execute("""
                INSERT INTO exam_sessions (StudentID, StartTime, Status)
                VALUES (%s, NOW(), 'IN_PROGRESS')
            """, (student_id,))
            mysql.connection.commit()
            cur.close()
        except Exception as db_err:
            logger.error(f"examSessionStart DB error (continuing without DB): {db_err}", exc_info=True)

        if warning_system:
            warning_system.initialize_student(student_id, student_name)
        try:
            if socketio:
                socketio.emit('student_exam_started', {
                    'student_id': student_id,
                    'student_name': student_name,
                    'start_time': started_at_ms,
                    'warnings': 0,
                    'violations': []
                }, namespace='/admin')
        except Exception as emit_err:
            logger.debug(f"student_exam_started emit failed: {emit_err}")

        detection_threads_started = True
        # One-time token: consume verification once exam session starts.
        session['face_verified_for_exam'] = False
        session.pop('student_face_verified_at', None)
        return jsonify({'ok': True, 'started': True, 'start_time': started_at_ms})
    except Exception as e:
        logger.error(f"examSessionStart error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Failed to start exam session'}), 500


@app.route('/api/exam-session/end', methods=['POST'])
@require_role('STUDENT')
@rate_limit('exam_end', max_requests=10, window_seconds=60)
def examSessionEnd():
    """End monitoring and mark exam session finished/terminated."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401

    data = request.get_json(silent=True) or {}
    student_id = str(user['Id'])
    student_name = user.get('Name', 'Unknown')
    terminated = bool(data.get('terminated'))
    reason = str(data.get('reason') or '')

    try:
        if warning_system:
            try:
                warning_system.flush_violations_to_db(student_id, _get_active_or_latest_session_id(student_id))
            except Exception as e:
                logger.warning(f"flush_violations_to_db failed on exam end for {student_id}: {e}")
        _end_exam_runtime_state(student_id, clear_warning_cache=False)
        try:
            cur = mysql.connection.cursor()
            status = 'TERMINATED' if terminated else 'COMPLETED'
            cur.execute("""
                UPDATE exam_sessions
                SET Status=%s, EndTime=NOW()
                WHERE StudentID=%s AND Status='IN_PROGRESS'
            """, (status, student_id))
            mysql.connection.commit()
            cur.close()
        except Exception as db_err:
            logger.warning(f"examSessionEnd DB error (continuing): {db_err}")

        try:
            if socketio:
                socketio.emit('student_exam_completed', {
                    'student_id': student_id,
                    'student_name': student_name,
                    'terminated': terminated,
                    'reason': reason,
                    'ended_at': int(time.time() * 1000)
                }, namespace='/admin')
        except Exception as emit_err:
            logger.debug(f"student_exam_completed emit failed: {emit_err}")

        return jsonify({'ok': True, 'terminated': terminated, 'reason': reason})
    except Exception as e:
        logger.error(f"examSessionEnd error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Failed to end exam session'}), 500


@app.route('/api/proctor/manifest', methods=['GET'])
def proctorManifest():
    """Serve the proctor engine integrity manifest."""
    manifest_path = os.path.join(app.static_folder, 'proctor_engine', 'manifest.json')
    if not os.path.isfile(manifest_path):
        return jsonify({'error': 'Manifest not found'}), 404
    return send_file(manifest_path, mimetype='application/json')

@app.route('/api/pre-exam-face-verify', methods=['POST'])
@require_role('STUDENT')
@rate_limit('pre_exam_face_verify', max_requests=200, window_seconds=60)
def preExamFaceVerify():
    """Verify live captured face using the new AI Vision Engine before exam starts."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'matched': False, 'error': 'Unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    image_data = payload.get('image_data') or payload.get('frame')
    if not image_data:
        return jsonify({'ok': False, 'matched': False, 'error': 'Missing image_data'}), 400

    try:
        student_id = int(user['Id'])
        
        # Load profile picture from DB
        cur = mysql.connection.cursor()
        cur.execute("SELECT Profile FROM students WHERE ID=%s", (student_id,))
        row = cur.fetchone()
        cur.close()
        
        if not row or not row[0]:
            return jsonify({'ok': False, 'matched': False, 'error': 'Profile image not found in database'}), 400
            
        profile_filename = row[0]
        profile_path = os.path.join(app.root_path, 'static', 'Profiles', profile_filename)
        if not os.path.exists(profile_path):
            profile_path = os.path.join(app.root_path, 'static', 'profiles', profile_filename)
            
        if not os.path.exists(profile_path):
            return jsonify({'ok': False, 'matched': False, 'error': 'Profile image file missing on server'}), 400
            
        if not DEEPFACE_AVAILABLE:
            return jsonify({'ok': False, 'matched': False, 'error': 'Face verification service unavailable'}), 500
            
        import cv2 as _cv2
        import numpy as _np
        
        image_b64 = image_data.split(',', 1)[1] if ',' in image_data else image_data
        img_bytes = base64.b64decode(image_b64)
        nparr = _np.frombuffer(img_bytes, _np.uint8)
        frame = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'ok': False, 'matched': False, 'error': 'Invalid image format'}), 400
            
        # Optimization: Use cached profile embeddings and manual distance calculation
        # This avoids re-processing the profile image every single time.
        try:
            # 1. Get live frame embedding
            objs = DeepFace.represent(
                img_path=frame, 
                model_name="Facenet", 
                detector_backend="opencv",
                enforce_detection=True
            )
            
            if not objs:
                return jsonify({'ok': False, 'matched': False, 'error': 'No face detected in webcam. Please align your face.'}), 200
            
            live_embedding = objs[0]["embedding"]
            
            # 2. Get/Cache profile embedding
            profile_embedding = _profile_embeddings_cache.get(profile_path)
            if not profile_embedding:
                logger.info(f"Generating new profile embedding for: {profile_path}")
                p_objs = DeepFace.represent(
                    img_path=profile_path, 
                    model_name="Facenet", 
                    detector_backend="opencv",
                    enforce_detection=True
                )
                if p_objs:
                    profile_embedding = p_objs[0]["embedding"]
                    _profile_embeddings_cache[profile_path] = profile_embedding
            
            if not profile_embedding:
                return jsonify({'ok': False, 'matched': False, 'error': 'Could not process profile image'}), 500

            # 3. Calculate distance
            distance = 1.0 # default
            if SCIPY_AVAILABLE:
                distance = _cosine_dist(live_embedding, profile_embedding)
            else:
                # fallback manual cosine similarity
                a = _np.array(live_embedding)
                b = _np.array(profile_embedding)
                distance = 1.0 - (_np.dot(a, b) / (_np.linalg.norm(a) * _np.linalg.norm(b)))

            verified = distance <= FACE_VERIFY_DISTANCE_THRESHOLD
            
            if verified:
                session['face_verified_for_exam'] = True
                session['student_face_verified_at'] = time.time()
                return jsonify({
                    'ok': True,
                    'matched': True,
                    'distance': float(distance),
                    'threshold': float(FACE_VERIFY_DISTANCE_THRESHOLD)
                }), 200
            else:
                logger.info(
                    f"Face verification mismatch for student {student_id}: "
                    f"distance={float(distance):.4f}, threshold={FACE_VERIFY_DISTANCE_THRESHOLD:.2f}"
                )
                return jsonify({
                    'ok': True,
                    'matched': False,
                    'distance': float(distance),
                    'threshold': float(FACE_VERIFY_DISTANCE_THRESHOLD),
                    'error': 'Face does not match registered profile. Please ensure proper lighting and alignment.'
                }), 200
                
        except ValueError:
            return jsonify({'ok': False, 'matched': False, 'error': 'No face detected. Please align your face.'}), 200

    except Exception as e:
        logger.error(f"preExamFaceVerify error: {e}", exc_info=True)
        return jsonify({'ok': False, 'matched': False, 'error': 'Verification failed'}), 500

                         
@app.route('/exam', methods=['POST'])
@require_role('STUDENT')
@rate_limit('exam_submit', max_requests=20, window_seconds=60)
def examAction():
    """Handle exam submission; stop detection, camera, and save result to DB."""
    global detection_threads_started
    ensure_db_schema()
    data = request.json or {}
    
    # stop detection
    detection_threads_started = False
    # camera_streamer.release() # This line was commented out in the original code, keeping it that way.
    
    user = current_user()
    student_id = user['Id'] if user else None
    student_name = user['Name'] if user else 'Unknown'
    
    # stop admin monitoring for student
    sid_str = str(student_id) if student_id else (str(user['Id']) if user else None)
    # legacy python monitor disabled
    runtime_state = _get_runtime_warning_state(sid_str) if sid_str else {}
    
    # Calculate results (prefer server-side derivation from submitted question list)
    time_spent = data.get('time_spent', 0)
    auto_terminated = data.get('auto_terminated', False)

    questions_payload = data.get('questions') if isinstance(data.get('questions'), list) else None
    if questions_payload is not None and len(questions_payload) > 0:
        total_questions = len(questions_payload)
        correct_answers = sum(1 for q in questions_payload if bool(q.get('is_correct')))
        score = correct_answers * 2
    else:
        tq = data.get('total_questions')
        if tq is None:
            tq = data.get('question_count')
        if tq is None and data.get('total') is not None:
            try:
                tq = int(float(data.get('total')) // 2)
            except Exception:
                tq = None
        try:
            total_questions = int(tq) if tq is not None else 125
        except Exception:
            total_questions = 125
        total_questions = max(1, total_questions)

        try:
            submitted_score = int(float(data.get('score', 0)))
        except Exception:
            submitted_score = 0
        max_score = total_questions * 2
        score = max(0, min(submitted_score, max_score))
        correct_answers = int(round(score / 2.0))

    # Calculate percentage
    max_score = total_questions * 2
    percentage = round((correct_answers / total_questions) * 100, 2) if total_questions > 0 else 0
    
    # Determine DB status (must match ENUM: 'PASS','FAIL','TERMINATED')
    if auto_terminated:
        db_status = 'TERMINATED'
    elif percentage >= 50:
        db_status = 'PASS'
    else:
        db_status = 'FAIL'
    
    # Get warnings & violations from warning_system (in-memory)
    warnings_count = 0
    violations_list = []
    if warning_system and student_id:
        sid_str = str(student_id)
        warnings_count = warning_system.get_warnings(sid_str)
        violations_list = warning_system.get_violations(sid_str)
    runtime_violations = runtime_state.get('violations') or []
    if runtime_violations and len(runtime_violations) > len(violations_list):
        violations_list = runtime_violations
    violations_list = violations_list[:3]
    warnings_count = min(max(warnings_count, int(runtime_state.get('warnings') or 0)), 3)
    
    # ---- Save to correct DB schema ----
    # Violation type map: frontend/warning_system types -> DB ENUM values
    VTYPE_MAP = {
        'multiple_faces': 'MULTIPLE_FACES', 'MULTIPLE_FACES': 'MULTIPLE_FACES',
        'no_face': 'NO_FACE', 'NO_FACE': 'NO_FACE',
        'FACE_OBSCURED': 'NO_FACE', 'face_obscured': 'NO_FACE',
        'eyes_closed': 'EYES_CLOSED', 'EYES_CLOSED': 'EYES_CLOSED',
        'gaze_left': 'GAZE_LEFT', 'GAZE_LEFT': 'GAZE_LEFT',
        'gaze_right': 'GAZE_RIGHT', 'GAZE_RIGHT': 'GAZE_RIGHT',
        'gaze_up': 'GAZE_UP', 'GAZE_UP': 'GAZE_UP',
        'gaze_down': 'GAZE_DOWN', 'GAZE_DOWN': 'GAZE_DOWN',
        'gaze_up_left': 'GAZE_UP_LEFT', 'GAZE_UP_LEFT': 'GAZE_UP_LEFT',
        'gaze_up_right': 'GAZE_UP_RIGHT', 'GAZE_UP_RIGHT': 'GAZE_UP_RIGHT',
        'gaze_down_left': 'GAZE_DOWN_LEFT', 'GAZE_DOWN_LEFT': 'GAZE_DOWN_LEFT',
        'gaze_down_right': 'GAZE_DOWN_RIGHT', 'GAZE_DOWN_RIGHT': 'GAZE_DOWN_RIGHT',
        'voice_detected': 'VOICE_DETECTED', 'VOICE_DETECTED': 'VOICE_DETECTED',
        'DISTRACTION': 'DISTRACTION', 'distraction': 'DISTRACTION',
        'NOT_FORWARD': 'DISTRACTION', 'not_forward': 'DISTRACTION',
        'GAZE_AWAY': 'DISTRACTION', 'gaze_away': 'DISTRACTION',
        'STUDENT_LEFT_SEAT': 'STUDENT_LEFT_SEAT', 'student_left_seat': 'STUDENT_LEFT_SEAT',
        'mic_off': 'VOICE_DETECTED', 'MIC_OFF': 'VOICE_DETECTED',
        'head_movement': 'HEAD_MOVEMENT', 'HEAD_MOVEMENT': 'HEAD_MOVEMENT',
        'head_down': 'HEAD_DOWN', 'HEAD_DOWN': 'HEAD_DOWN',
        'identity_mismatch': 'IDENTITY_MISMATCH', 'IDENTITY_MISMATCH': 'IDENTITY_MISMATCH',
        'camera_off': 'NO_FACE', 'CAMERA_OFF': 'NO_FACE',
        'camera_blocked': 'NO_FACE', 'CAMERA_BLOCKED': 'NO_FACE',
        'prohibited_object': 'PROHIBITED_OBJECT', 'PROHIBITED_OBJECT': 'PROHIBITED_OBJECT',
        'tab_switch': 'TAB_SWITCH', 'TAB_SWITCH': 'TAB_SWITCH',
        'FULLSCREEN_EXIT': 'TAB_SWITCH', 'fullscreen_exit': 'TAB_SWITCH',
        'prohibited_shortcut': 'PROHIBITED_SHORTCUT', 'PROHIBITED_SHORTCUT': 'PROHIBITED_SHORTCUT',
        'KEYBOARD_SHORTCUT': 'PROHIBITED_SHORTCUT', 'DEVTOOLS_OPEN': 'PROHIBITED_SHORTCUT',
        'DEVTOOLS_SHORTCUT': 'PROHIBITED_SHORTCUT', 'DEVTOOLS_OPENED': 'PROHIBITED_SHORTCUT',
        'COPY_PASTE': 'PROHIBITED_SHORTCUT',
        'terminated_by_admin': 'TERMINATED_BY_ADMIN', 'TERMINATED_BY_ADMIN': 'TERMINATED_BY_ADMIN',
        'HEAD_POSE': 'HEAD_MOVEMENT', 'head_pose': 'HEAD_MOVEMENT',
        'Book/Notebook': 'PROHIBITED_OBJECT', 'book/notebook': 'PROHIBITED_OBJECT'
    }
    
    try:
        cur = mysql.connection.cursor()

        # Best-effort fallback for missing session start time
        fallback_start_dt = None
        try:
            runtime_start_ms = int(runtime_state.get('start_time') or 0)
        except Exception:
            runtime_start_ms = 0
        if runtime_start_ms > 0:
            fallback_start_dt = datetime.fromtimestamp(runtime_start_ms / 1000.0)
        else:
            try:
                ts = int(float(time_spent or 0))
            except Exception:
                ts = 0
            if ts > 0:
                fallback_start_dt = datetime.now() - timedelta(seconds=ts)

        # 1. Find the most recent session (start/end endpoints may already have closed it)
        cur.execute("""
            SELECT SessionID, StartTime, Status
            FROM exam_sessions
            WHERE StudentID=%s
            ORDER BY (Status='IN_PROGRESS') DESC, StartTime DESC, SessionID DESC
            LIMIT 1
        """, (student_id,))
        session_row = cur.fetchone()
        session_end_status = 'TERMINATED' if auto_terminated else 'COMPLETED'

        if session_row:
            session_id = session_row[0]
            start_dt = session_row[1]
            if (start_dt is None) and fallback_start_dt:
                cur.execute("UPDATE exam_sessions SET StartTime=%s WHERE SessionID=%s", (fallback_start_dt, session_id))
            # Ensure session is closed with the final status
            cur.execute("""
                UPDATE exam_sessions
                SET EndTime=NOW(), Status=%s
                WHERE SessionID=%s
            """, (session_end_status, session_id))
        else:
            # Fallback: create session now if missing
            start_dt = fallback_start_dt or datetime.now()
            cur.execute("""
                INSERT INTO exam_sessions (StudentID, StartTime, EndTime, Status)
                VALUES (%s, %s, NOW(), %s)
            """, (student_id, start_dt, session_end_status))
            session_id = cur.lastrowid
            logger.warning(f"No session found for student {student_id}. Created fallback session {session_id}.")
        
        # 2. Keep track of Attempts and insert new result row
        try:
            cur.execute("SELECT MAX(Attempts) FROM exam_results WHERE StudentID=%s", (student_id,))
            row = cur.fetchone()
            current_attempts = int(row[0]) if row and row[0] is not None else 0
        except Exception:
            current_attempts = 0
            
        new_attempt = current_attempts + 1

        # 3. Insert into exam_results
        failure_reasons = ""
        if db_status == 'TERMINATED':
            failure_reasons = request.args.get('reason') or "Maximum warnings reached or exam terminated by system."
        elif db_status == 'FAIL':
            failure_reasons = "Failed to achieve the passing score of 50%."

        cur.execute("""
            INSERT INTO exam_results
                (StudentID, SessionID, Score, TotalQuestions, CorrectAnswers, SubmissionTime, Status, Attempts, FailureReasons)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s)
        """, (student_id, session_id, percentage, total_questions, correct_answers, db_status, new_attempt, failure_reasons))
        
        # 4. Persist violations for this session (rewrite to keep result pages consistent)
        if violations_list:
            cur.execute("DELETE FROM violations WHERE SessionID=%s", (session_id,))
            for v in violations_list:
                raw_type = v.get('type', 'TAB_SWITCH')
                db_vtype = VTYPE_MAP.get(raw_type, VTYPE_MAP.get(str(raw_type).upper(), 'TAB_SWITCH'))
                details = str(v.get('details', '') or '')[:500]
                ts_raw = v.get('time') or v.get('timestamp') or None
                ts_val = None
                if ts_raw:
                    try:
                        ts_s = str(ts_raw)
                        if 'T' in ts_s:
                            # ISO from browser
                            ts_val = datetime.fromisoformat(ts_s.replace('Z', '+00:00'))
                            # MySQL adapter expects naive datetime
                            if getattr(ts_val, 'tzinfo', None) is not None:
                                ts_val = ts_val.replace(tzinfo=None)
                        else:
                            ts_val = datetime.strptime(ts_s, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        ts_val = None
                if not ts_val:
                    ts_val = datetime.now()
                cur.execute("""
                    INSERT INTO violations (StudentID, SessionID, ViolationType, Details, Timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                """, (student_id, session_id, str(db_vtype or 'UNKNOWN').upper(), details, ts_val))
        
        mysql.connection.commit()
        cur.close()
        logger.info(f"✅ Result saved: StudentID={student_id} SessionID={session_id} Score={percentage}% Status={db_status} Warnings={warnings_count}")
    except Exception as e:
        logger.error(f"Error saving exam result to DB: {e}", exc_info=True)
        try:
            mysql.connection.rollback()
        except:
            pass
    
    # Emit result to admin dashboard
    if socketio and student_id:
        socketio.emit('student_exam_ended', {
            'student_id': student_id,
            'student_name': student_name,
            'score': score,
            'percentage': percentage,
            'status': db_status,
            'auto_terminated': auto_terminated
        }, namespace='/admin')

    if sid_str:
        _end_exam_runtime_state(sid_str)
    
    return jsonify({
        "output": "submitted",
        "score": score,
        "percentage": percentage,
        "status": db_status,
        "link": "showResultPass" if db_status == 'PASS' else "showResultFail"
    })

@app.route('/showResultPass')
@app.route('/showResultFail')
@require_role('STUDENT')
def showResult():
    """Show student exam result page after exam submission - fetch from DB"""
    ensure_db_schema()
    user = current_user()
    result_data = None
    
    if user:
        student_id = user.get('Id')
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                SELECT er.Score, er.TotalQuestions, er.CorrectAnswers,
                       er.SubmissionTime, er.Status, er.SessionID,
                       es.StartTime, es.EndTime,
                       (SELECT COUNT(*) FROM violations v WHERE v.SessionID = er.SessionID) AS warnings_count,
                       er.Attempts, er.FailureReasons
                FROM exam_results er
                JOIN exam_sessions es ON es.SessionID = er.SessionID
                WHERE er.StudentID = %s
                ORDER BY er.SubmissionTime DESC
                LIMIT 1
            """, (student_id,))
            row = cur.fetchone()
            cur.close()
            if row:
                total_q = int(row[1] or 0)
                correct_q = int(row[2] or 0)
                if total_q > 0:
                    percentage = round((correct_q / total_q) * 100.0, 2)
                else:
                    percentage = float(row[0]) if row[0] else 0
                db_status  = row[4]   # PASS / FAIL / TERMINATED
                # Time spent (seconds) - prefer submission/end time, fallback to now
                start_time = row[6]
                end_time = row[3] or row[7] or None
                time_spent = 0
                if start_time and end_time:
                    try:
                        time_spent = max(0, int((end_time - start_time).total_seconds()))
                    except:
                        time_spent = 0
                # Grade from percentage
                if percentage >= 90:   grade = 'A'
                elif percentage >= 75: grade = 'B'
                elif percentage >= 60: grade = 'C'
                elif percentage >= 50: grade = 'D'
                else:                  grade = 'F'
                
                # Fetch violations for the latest session to build a breakdown for the report
                violations = []
                violations_breakdown = {}
                try:
                    vcur = mysql.connection.cursor()
                    vcur.execute("""
                        SELECT ViolationType, Details, Timestamp
                        FROM violations
                        WHERE SessionID = %s
                        ORDER BY Timestamp ASC
                    """, (row[5],))
                    vrows = vcur.fetchall()
                    vcur.close()
                    if vrows:
                        for vrow in vrows:
                            vtype_raw = str(vrow[0] or 'UNKNOWN')
                            vtype = _friendly_violation_type(vtype_raw)
                            violations.append({
                                'type': vtype,
                                'details': str(vrow[1] or ''),
                                'time': str(vrow[2] or '')
                            })
                            violations_breakdown[vtype] = violations_breakdown.get(vtype, 0) + 1
                except Exception as v_err:
                    logger.warning(f"Violations fetch warning: {v_err}")

                # Fallback to in-memory violations if DB rows are missing
                if not violations:
                    try:
                        fallback_violations = []
                        if warning_system and student_id:
                            fallback_violations = warning_system.get_violations(str(student_id)) or []
                        if fallback_violations:
                            for vrow in fallback_violations:
                                vtype_raw = str(vrow.get('type') or 'UNKNOWN')
                                vtype = _friendly_violation_type(vtype_raw)
                                violations.append({
                                    'type': vtype,
                                    'details': str(vrow.get('details') or ''),
                                    'time': str(vrow.get('time') or '')
                                })
                                violations_breakdown[vtype] = violations_breakdown.get(vtype, 0) + 1
                    except Exception as v_fallback_err:
                        logger.warning(f"Violations fallback warning: {v_fallback_err}")

                warnings_db = int(row[8]) if row[8] else 0
                warnings_live = 0
                if warning_system and student_id:
                    try:
                        warnings_live = int(warning_system.get_warnings(str(student_id)) or 0)
                    except Exception:
                        warnings_live = 0
                warnings_issued = max(warnings_db, len(violations), warnings_live)
                if db_status == 'TERMINATED' and warnings_issued < 3:
                    warnings_issued = 3

                total_violations = max(len(violations), warnings_issued if db_status == 'TERMINATED' else len(violations))

                attempts_count = int(row[9]) if len(row) > 9 and row[9] is not None else 1
                failure_reason = row[10] if len(row) > 10 else ""

                result_data = {
                    'percentage':        percentage,
                    'score':             (row[2] or 0) * 2,  # CorrectAnswers * 2

                    'correct_answers':   int(row[2] or 0),
                    'total_questions':   row[1] or 125,
                    'grade':             grade,
                    'time_spent':        time_spent,
                    'warnings_issued':   warnings_issued,
                    'auto_terminated':   (db_status == 'TERMINATED'),
                    'status':            db_status,
                    'reason':            failure_reason or request.args.get('reason') or ('Maximum warnings reached' if db_status == 'TERMINATED' else ''),
                    'submission_time':   row[3] or row[7],
                    'exam_title':        'Final Examination',
                    'violations':        violations,
                    'violations_breakdown': violations_breakdown,
                    'total_violations':  total_violations,
                    'attempts':          attempts_count,
                    'max_attempts':      5
                }
        except Exception as e:
            logger.error(f"Error fetching student result: {e}", exc_info=True)
    
    # Build studentInfo dict for template
    student_ctx = None
    if user:
        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT Profile FROM students WHERE ID=%s", (user.get('Id'),))
            pr = cur.fetchone()
            cur.close()
            student_ctx = {
                'Id':      user.get('Id'),
                'Name':    user.get('Name'),
                'Email':   user.get('Email'),
                'Profile': pr[0] if pr and pr[0] else None
            }
        except Exception:
            student_ctx = user
    
    template_name = 'showResultPass.html'
    if result_data and result_data.get('status') and result_data.get('status') != 'PASS':
        template_name = 'showResultPass.html'
    elif not result_data:
        template_name = 'ExamResultFail.html'
    return render_template(template_name, result=result_data, studentInfo=student_ctx)

@app.route('/adminResultDetails/<int:resultId>')
@require_role('ADMIN')
def adminResultDetails(resultId):
    """Show detailed attempt history for a student - resultId is StudentID"""
    try:
        ensure_db_schema()
        cur = mysql.connection.cursor()
        
        # 1. Fetch student info
        cur.execute("SELECT ID, Name, Email, Profile FROM students WHERE ID = %s", (resultId,))
        student_info = cur.fetchone()
        if not student_info:
            flash("Student not found.", "warning")
            return redirect(url_for('adminResults'))
            
        student_dict = {
            'student_id': student_info[0],
            'student_name': student_info[1],
            'student_email': student_info[2],
            'student_profile': student_info[3],
        }

        # 2. Fetch all attempts (exam_results)
        cur.execute("""
            SELECT er.ResultID, er.Score, er.TotalQuestions, er.CorrectAnswers,
                   er.SubmissionTime, er.Status, er.SessionID,
                   es.StartTime, es.EndTime, er.Attempts, er.FailureReasons
            FROM exam_results er
            JOIN exam_sessions es ON es.SessionID = er.SessionID
            WHERE er.StudentID = %s
            ORDER BY er.Attempts DESC
        """, (resultId,))
        attempts_rows = cur.fetchall()
        
        attempts = []
        total_terminated = 0
        total_passed = 0
        best_percentage = 0.0
        total_violations_all = 0
        
        for row in attempts_rows:
            res_id = row[0]
            total_q = int(row[2] or 0)
            correct_q = int(row[3] or 0)
            percentage = round((correct_q / total_q) * 100.0, 2) if total_q > 0 else (float(row[1]) if row[1] else 0.0)
            db_status = row[5]
            session_id = row[6]
            start_time = row[7]
            end_time = row[4] or row[8]
            
            attempt_num = row[9] if row[9] is not None else 1
            failure_reasons = row[10] if len(row) > 10 and row[10] else ""
            
            if db_status == 'TERMINATED':
                total_terminated += 1
            elif db_status == 'PASS':
                total_passed += 1
            if percentage > best_percentage:
                best_percentage = percentage

            time_spent = 0
            if start_time and end_time:
                try:
                    time_spent = max(0, int((end_time - start_time).total_seconds()))
                except: pass
                
            if percentage >= 90:   grade = 'A'
            elif percentage >= 75: grade = 'B'
            elif percentage >= 60: grade = 'C'
            elif percentage >= 50: grade = 'D'
            else:                  grade = 'F'
            
            # Fetch violations for this session
            cur.execute("""
                SELECT ViolationType, Details, Timestamp
                FROM violations WHERE SessionID=%s ORDER BY Timestamp ASC
            """, (session_id,))
            vrows = cur.fetchall()
            violations = []
            for r in vrows:
                vtype = _friendly_violation_type(r[0])
                violations.append({'type': vtype, 'details': r[1], 'time': str(r[2])})
                
            if not violations and warning_system:
                # Provide fallback for latest attempt maybe
                try:
                    fallback = warning_system.get_violations(str(resultId)) or []
                    for v in fallback:
                        vtype = _friendly_violation_type(v.get('type'))
                        violations.append({
                            'type': vtype,
                            'details': v.get('details'),
                            'time': str(v.get('time') or '')
                        })
                except Exception:
                    pass
                    
            warnings_live = 0
            if warning_system:
                try:
                    warnings_live = int(warning_system.get_warnings(str(resultId)) or 0)
                except Exception: pass
                
            warnings_issued = max(len(violations), warnings_live)
            if db_status == 'TERMINATED' and warnings_issued < 3:
                warnings_issued = 3
                
            total_violations_all += len(violations)
            
            attempts.append({
                'attempt_num': attempt_num,
                'score': (row[3] or 0) * 2,
                'total_questions': row[2] or 125,
                'percentage': percentage,
                'grade': grade,
                'time_spent': time_spent,
                'warnings_issued': warnings_issued,
                'auto_terminated': (db_status == 'TERMINATED'),
                'status': db_status,
                'submission_time': row[4],
                'failure_reasons': failure_reasons,
                'violations': violations
            })

        cur.close()
        
        # Summary stats
        summary = {
            'total_attempts': len(attempts),
            'total_terminated': total_terminated,
            'total_passed': total_passed,
            'best_percentage': best_percentage,
            'total_violations': total_violations_all
        }
        
        return render_template('ResultDetails.html', student=student_dict, attempts=attempts, summary=summary)
    except Exception as e:
        logger.error(f"Error fetching result details: {e}", exc_info=True)
        flash(f"Error loading result details: {e}", "danger")
        return redirect(url_for('adminResults'))

@app.route('/adminResults')
@require_role('ADMIN')
def adminResults():
    """Fetch all exam results from correct DB schema and render ExamResult.html"""
    results = []
    try:
        ensure_db_schema()
        page = max(1, int(request.args.get('page', 1) or 1))
        limit = max(1, min(100, int(request.args.get('limit', 20) or 20)))
        offset = (page - 1) * limit
        cur = mysql.connection.cursor()
        cur.execute("SELECT COUNT(DISTINCT StudentID) FROM exam_results")
        total_results = int(cur.fetchone()[0] or 0)
        pages = max(1, math.ceil(total_results / limit)) if limit else 1
        # Fetch latest result per student with violation count from DB
        cur.execute("""
            SELECT
                er.ResultID,
                er.StudentID,
                s.Name        AS student_name,
                s.Email       AS student_email,
                s.Profile     AS student_profile,
                er.Score,
                er.TotalQuestions,
                er.CorrectAnswers,
                er.SubmissionTime,
                er.Status,
                er.SessionID,
                es.StartTime,
                es.EndTime,
                (SELECT COUNT(*) FROM violations v WHERE v.SessionID = er.SessionID) AS violations_count
            FROM exam_results er
            JOIN students      s  ON s.ID         = er.StudentID
            LEFT JOIN exam_sessions es ON es.SessionID  = er.SessionID
            WHERE er.ResultID IN (
                SELECT er2.ResultID
                FROM exam_results er2
                INNER JOIN (
                    SELECT StudentID, MAX(SubmissionTime) AS latest_time
                    FROM exam_results
                    GROUP BY StudentID
                ) latest ON latest.StudentID = er2.StudentID
                         AND latest.latest_time = er2.SubmissionTime
            )
            ORDER BY er.SubmissionTime DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        rows = cur.fetchall()
        cur.close()
        
        for row in rows:
            total_q   = int(row[6] or 0)
            correct_q = int(row[7] or 0)
            sid_str   = str(row[1])
            if total_q > 0:
                percentage = round((correct_q / total_q) * 100.0, 2)
            else:
                percentage = float(row[5]) if row[5] else 0
            db_status  = row[9]   # PASS / FAIL / TERMINATED
            # Time spent in seconds
            time_spent = 0
            start_time = row[11]
            end_time = row[8] or row[12]
            if start_time and end_time:
                try:
                    time_spent = max(0, int((end_time - start_time).total_seconds()))
                except: pass
            # Grade
            if percentage >= 90:   grade = 'A'
            elif percentage >= 75: grade = 'B'
            elif percentage >= 60: grade = 'C'
            elif percentage >= 50: grade = 'D'
            else:                  grade = 'F'

            # Warnings count: use violations DB count as primary source
            violations_db_count = int(row[13]) if row[13] else 0
            # Also check in-memory warning_system for real-time accuracy
            mem_warnings = 0
            if warning_system and sid_str:
                try:
                    mem_warnings = int(warning_system.get_warnings(sid_str) or 0)
                except:
                    pass
            # Use the maximum of DB violations and in-memory warnings — whichever is higher is more accurate
            warnings_issued = max(violations_db_count, mem_warnings)
            if db_status == 'TERMINATED' and warnings_issued < 3:
                warnings_issued = 3
            
            # student_profile: URL or filename for student photo
            student_profile = row[4] or ''
            
            # For display, align violations with warnings to avoid showing 0 when terminated
            display_violations = max(violations_db_count, warnings_issued)

            results.append({
                'result_id':       row[0],
                'student_id':      row[1],
                'student_name':    row[2],
                'student_email':   row[3],
                'student_profile': student_profile,
                'exam_title':      'Final Examination',
                'score':           correct_q,
                'total_questions': total_q if total_q > 0 else 0,
                'percentage':      percentage,
                'grade':           grade,
                'time_spent':      time_spent,
                'warnings_issued': warnings_issued,
                'violations':      display_violations,
                'auto_terminated': (db_status == 'TERMINATED'),
                'submission_time': row[8],
            })
    except Exception as e:
        logger.error(f"Error fetching results: {e}", exc_info=True)
        flash(f"Error loading results: {e}", "danger")
    # Add no-cache headers so admin panel always fetches fresh data (not stale browser cache)
    resp = make_response(render_template(
        'ExamResult.html',
        results=results,
        total_results=total_results if 'total_results' in locals() else len(results),
        page=page if 'page' in locals() else 1,
        pages=pages if 'pages' in locals() else 1,
        limit=limit if 'limit' in locals() else 20
    ))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/adminRecordings')
@require_role('ADMIN')
def adminRecordings():
    """List saved exam session videos and audio recordings."""
    video_dir = _static_path('recording')
    audio_dir = _static_path('recording', 'audio')
    videos = []
    audios = []

    def infer_from_name(name):
        """
        Infer student metadata from filename patterns:
        - <student>_YYYYMMDD_HHMMSS.ext
        - <student_id>_<student>_YYYYMMDD_HHMMSS.ext
        """
        stem = os.path.splitext(name)[0]
        parts = stem.split('_')
        student_id = None
        student_name = None
        session_start = None
        if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) == 8 and parts[-1].isdigit() and len(parts[-1]) == 6:
            date_token = parts[-2]
            time_token = parts[-1]
            body = parts[:-2]
            if body and body[0].isdigit():
                student_id = int(body[0])  # e.g. "42" -> 42
                body = body[1:]
            if body:
                student_name = ' '.join(body)
            session_start = f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]} {time_token[:2]}:{time_token[2:4]}:{time_token[4:6]}"
        return student_name, student_id, session_start

    def compact_to_epoch(compact_ts):
        if not compact_ts:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H%M%S", "%Y%m%d_%H%M%S"):
            try:
                return datetime.strptime(compact_ts, fmt).timestamp()
            except Exception:
                continue
        return None

    try:
        # Load session metadata if available
        session_meta = {}
        if os.path.isdir(video_dir):
            for name in os.listdir(video_dir):
                if name.lower().endswith('.json'):
                    try:
                        with open(os.path.join(video_dir, name), 'r') as f:
                            data = json.load(f)
                        video_path = data.get('video_path')
                        if video_path:
                            base = os.path.basename(video_path)
                            session_meta[base] = {
                                'student_name': data.get('student_name'),
                                'student_id': data.get('student_id'),
                                'session_start': data.get('session_start'),
                                'session_end': data.get('session_end'),
                                'total_violations': data.get('total_violations')
                            }
                    except Exception:
                        continue
        if os.path.isdir(video_dir):
            for name in os.listdir(video_dir):
                if name.lower().endswith(('.mp4', '.webm', '.ogg')):
                    full = os.path.join(video_dir, name)
                    meta = session_meta.get(name, {})
                    inf_name, inf_id, inf_start = infer_from_name(name)
                    videos.append({
                        'name': name,
                        'static_path': f"recording/{name}",
                        'mime_type': 'video/webm' if name.lower().endswith('.webm') else ('video/ogg' if name.lower().endswith('.ogg') else 'video/mp4'),
                        'size': os.path.getsize(full),
                        'mtime': os.path.getmtime(full),
                        'student_name': meta.get('student_name') or inf_name,
                        'student_id': meta.get('student_id') or inf_id,
                        'session_start': meta.get('session_start') or inf_start,
                        'session_start_epoch': compact_to_epoch(meta.get('session_start') or inf_start),
                        'session_end': meta.get('session_end'),
                        'total_violations': meta.get('total_violations'),
                        'matched_audio': None
                    })
        if os.path.isdir(audio_dir):
            for name in os.listdir(audio_dir):
                if name.lower().endswith(('.wav', '.mp3', '.ogg', '.webm', '.m4a')):
                    full = os.path.join(audio_dir, name)
                    inferred_student, inferred_id, inferred_start = infer_from_name(name)
                    audios.append({
                        'name': name,
                        'static_path': f"recording/audio/{name}",
                        'size': os.path.getsize(full),
                        'mtime': os.path.getmtime(full),
                        'student_name': inferred_student,
                        'student_id': inferred_id,
                        'session_start': inferred_start,
                        'session_start_epoch': compact_to_epoch(inferred_start)
                    })

        # Match each video with nearest audio (same student, closest timestamp).
        max_delta_sec = 180
        for v in videos:
            v_epoch = v.get('session_start_epoch') or v.get('mtime')
            v_sid = v.get('student_id')
            v_sname = (v.get('student_name') or '').strip().lower()
            best = None
            best_delta = None
            for a in audios:
                a_sid = a.get('student_id')
                a_sname = (a.get('student_name') or '').strip().lower()
                same_student = (v_sid is not None and a_sid is not None and int(v_sid) == int(a_sid))
                if not same_student and v_sname and a_sname:
                    same_student = (v_sname == a_sname)
                if not same_student:
                    continue

                a_epoch = a.get('session_start_epoch') or a.get('mtime')
                if a_epoch is None or v_epoch is None:
                    continue

                delta = abs(float(v_epoch) - float(a_epoch))
                if delta <= max_delta_sec and (best is None or delta < best_delta):
                    best = a
                    best_delta = delta

            if best:
                v['matched_audio'] = best
    except Exception as e:
        logger.error(f"Error listing recordings: {e}", exc_info=True)
        flash(f"Error loading recordings: {e}", "danger")
    videos.sort(key=lambda x: x['mtime'], reverse=True)
    audios.sort(key=lambda x: x['mtime'], reverse=True)
    return render_template('Recordings.html', videos=videos, audios=audios)

@app.route('/adminProfile')
@require_role('ADMIN')
def adminProfile():
    """Admin profile page."""
    admin = current_admin() or {}
    admin_id = admin.get('Id')
    admin_info = {
        'id': admin_id,
        'name': admin.get('Name') or 'Admin',
        'email': admin.get('Email') or '',
        'profile': None
    }
    try:
        if admin_id:
            cur = mysql.connection.cursor()
            cur.execute("SELECT ID, Name, Email, Profile FROM students WHERE ID=%s LIMIT 1", (admin_id,))
            row = cur.fetchone()
            cur.close()
            if row:
                admin_info['id'] = row[0]
                admin_info['name'] = row[1] or admin_info['name']
                admin_info['email'] = row[2] or admin_info['email']
                admin_info['profile'] = row[3]
    except Exception as e:
        logger.error(f"adminProfile load error: {e}", exc_info=True)
    return render_template('admin_profile.html', admin=admin_info)

def _safe_send_from_dir(base_dir, filename):
    """Send file from a base directory safely."""
    if not filename or '..' in filename or filename.startswith(('/', '\\')):
        return abort(400)
    if not os.path.isdir(base_dir):
        return abort(404)
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return abort(404)
    return send_from_directory(base_dir, filename, as_attachment=True)

@app.route('/download/recording/video/<path:filename>')
@require_role('ADMIN')
def download_recording_video(filename):
    return _safe_send_from_dir(_static_path('recording'), filename)

@app.route('/download/recording/audio/<path:filename>')
@require_role('ADMIN')
def download_recording_audio(filename):
    return _safe_send_from_dir(_static_path('recording', 'audio'), filename)

def _safe_delete_from_dir(base_dir, filename):
    """Delete a file from a base directory safely."""
    if not filename or '..' in filename or filename.startswith(('/', '\\')):
        return False, 'Invalid filename'
    if not os.path.isdir(base_dir):
        return False, 'Directory not found'

    path = os.path.join(base_dir, filename)
    if not os.path.isfile(path):
        return False, 'File not found'

    try:
        os.remove(path)
        return True, None
    except Exception as e:
        logger.error(f"Failed deleting file {path}: {e}", exc_info=True)
        return False, 'Delete failed'

@app.route('/adminRecordings/delete/video/<path:filename>', methods=['POST'])
@require_role('ADMIN')
def delete_recording_video(filename):
    ok, err = _safe_delete_from_dir(_static_path('recording'), filename)
    if not ok:
        flash(err or 'Unable to delete video recording.', 'danger')
        return redirect(url_for('adminRecordings'))

    try:
        meta_name = f"{os.path.splitext(filename)[0]}.json"
        meta_path = os.path.join(_static_path('recording'), meta_name)
        if os.path.isfile(meta_path):
            os.remove(meta_path)
    except Exception as meta_err:
        logger.warning(f"Recording metadata cleanup failed for {filename}: {meta_err}")

    flash('Video recording deleted successfully.', 'success')
    return redirect(url_for('adminRecordings'))

@app.route('/adminRecordings/delete/audio/<path:filename>', methods=['POST'])
@require_role('ADMIN')
def delete_recording_audio(filename):
    ok, err = _safe_delete_from_dir(_static_path('recording', 'audio'), filename)
    if not ok:
        flash(err or 'Unable to delete audio recording.', 'danger')
        return redirect(url_for('adminRecordings'))

    flash('Audio recording deleted successfully.', 'success')
    return redirect(url_for('adminRecordings'))

@app.route('/adminStudents')
@require_role('ADMIN')
def adminStudents():
    """Fetch and display all students with profile images and exam results"""
    try:
        logger.info("=== ADMIN STUDENTS PAGE LOAD ===")
        ensure_db_schema()
        page = max(1, int(request.args.get('page', 1) or 1))
        limit = max(1, min(100, int(request.args.get('limit', 20) or 20)))
        offset = (page - 1) * limit
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT COUNT(*) FROM students WHERE Role='STUDENT'")
        total_students = int(cur.fetchone()[0] or 0)
        pages = max(1, math.ceil(total_students / limit)) if limit else 1

        cur.execute("SELECT id, name, email, password, profile FROM students WHERE Role='STUDENT' ORDER BY id LIMIT %s OFFSET %s", (limit, offset))
        rows = cur.fetchall()
        
        logger.info(f"Number of student records found: {len(rows)}")
        
        # Fetch latest result per student using correct schema
        results_map = {}
        try:
            student_ids = [row[0] for row in rows]
            if student_ids:
                placeholders = ','.join(['%s'] * len(student_ids))
                cur.execute(f"""
                    SELECT er.StudentID, er.Score, er.TotalQuestions, er.CorrectAnswers,
                           er.SubmissionTime, er.Status,
                           (SELECT COUNT(*) FROM violations v WHERE v.SessionID = er.SessionID) AS warnings_count
                    FROM exam_results er
                    WHERE er.ResultID IN (
                        SELECT MAX(ResultID) FROM exam_results WHERE StudentID IN ({placeholders}) GROUP BY StudentID
                    )
                """, tuple(student_ids))
                result_rows = cur.fetchall()
                for r in result_rows:
                    total_q = int(r[2] or 0)
                    correct_q = int(r[3] or 0)
                    if total_q > 0:
                        pct = round((correct_q / total_q) * 100.0, 2)
                    else:
                        pct = float(r[1]) if r[1] else 0
                    status = r[5]
                    if pct >= 90:   grade = 'A'
                    elif pct >= 75: grade = 'B'
                    elif pct >= 60: grade = 'C'
                    elif pct >= 50: grade = 'D'
                    else:           grade = 'F'
                    results_map[r[0]] = {
                        'score':           (r[3] or 0) * 2,
                        'total_questions': r[2] or 125,
                        'percentage':      pct,
                        'grade':           grade,
                        'warnings_issued': int(r[6]) if r[6] else 0,
                        'auto_terminated': (status == 'TERMINATED'),
                        'submission_time': r[4],
                    }
        except Exception as re:
            logger.warning(f"Results fetch warning: {re}")
        
        students = []
        for idx, row in enumerate(rows):
            student = {
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "password": row[3],
                "profile": row[4],
                "result": results_map.get(row[0])  # attach latest result if exists
            }
            students.append(student)
        
        cur.close()
        
        # Count students with profile images
        registered_count = sum(1 for s in students if s["profile"] and s["profile"].strip())
        
        logger.info(f"Students with profile images: {registered_count}/{len(students)}")
        
        # Check if profile images exist in filesystem
        if registered_count > 0:
            for student in students:
                if student["profile"]:
                    profile_path = os.path.join("static", "Profiles", student["profile"])
                    if os.path.exists(profile_path):
                        logger.debug(f"Profile image exists: {profile_path}")
                    else:
                        logger.warning(f"Profile image NOT FOUND: {profile_path}")
        
        return render_template(
            "Students.html",  # Make sure this matches your template filename
            students=students,
            registered_count=registered_count,
            MONITORING_ENABLED=MONITORING_ENABLED,
            page=page,
            pages=pages,
            total_students=total_students,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Error in adminStudents route: {str(e)}", exc_info=True)
        flash(f"Database error: {str(e)}", "danger")
        
        # Return empty data but still render template
        return render_template(
            "Students.html",
            students=[],
            registered_count=0,
            MONITORING_ENABLED=False
        )

@app.route('/adminLiveMonitoring')
@require_role('ADMIN')
def adminLiveMonitoring():
    if not MONITORING_ENABLED:
        flash('Live monitoring not available. Ensure flask-socketio is installed.', 'error')
        return redirect(url_for('adminStudents'))
    return render_template('admin_live_dashboard.html')

@app.route('/admin/live/<int:student_id>')
@app.route('/admin/live-stream/<int:student_id>')
@require_role('ADMIN')
def admin_live_stream(student_id):
    """
    Continuous MJPEG stream for a specific student.

    OpenCV is optional now that the heavy lifting happens on the client. If cv2 is
    missing we still stream the latest JPEG bytes we have (base64 sent by the
    student). This keeps the admin preview alive instead of returning HTTP 503.
    """

    frame_interval = 0.12  # ~8 FPS is enough for the admin wallboard
    stream_debug = (os.getenv('STREAM_DEBUG', '0') == '1')
    sid_str = str(student_id)

    def _frame_bytes():
        """Fetch freshest JPEG bytes for the student or return a placeholder."""
        with latest_student_frames_lock:
            item = latest_student_frames.get(sid_str)
        if not item:
            return BLANK_JPEG

        frame_obj = (
            item.get('processed_frame')
            or item.get('frame_bytes')
            or item.get('frame')
        )

        # If we already have bytes, return directly.
        if isinstance(frame_obj, (bytes, bytearray, memoryview)):
            return bytes(frame_obj)

        # If it's a NumPy array and cv2 is available, encode to JPEG.
        if CV2_AVAILABLE and cv2 is not None and isinstance(frame_obj, np.ndarray):
            try:
                ok, buffer = cv2.imencode('.jpg', frame_obj, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    return buffer.tobytes()
            except Exception as enc_err:
                logger.debug(f"admin_live_stream encode failed for {sid_str}: {enc_err}")

        # If it's a base64 string, decode.
        if isinstance(frame_obj, str):
            try:
                b64 = frame_obj.split(',', 1)[1] if ',' in frame_obj else frame_obj
                return base64.b64decode(b64)
            except Exception as dec_err:
                logger.debug(f"admin_live_stream base64 decode failed for {sid_str}: {dec_err}")

        return BLANK_JPEG

    def generate():
        frame_counter = 0
        while True:
            try:
                frame_bytes = _frame_bytes()
                frame_counter += 1
                if stream_debug and (frame_counter % 40 == 1):
                    logger.info(f"Streaming MJPEG fallback frame for student={student_id} count={frame_counter}")

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    frame_bytes +
                    b'\r\n'
                )
            except GeneratorExit:
                break
            except Exception as e:
                logger.error(f"admin_live_stream({student_id}) generator error: {e}", exc_info=True)
            time.sleep(frame_interval)

    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    resp.headers['X-Accel-Buffering'] = 'no'
    # Same-origin by default, add permissive CORS header for embedded stream clients.
    resp.headers['Access-Control-Allow-Credentials'] = 'true'
    origin = request.headers.get('Origin')
    if origin:
        resp.headers['Access-Control-Allow-Origin'] = origin
        resp.headers['Vary'] = 'Origin'
    else:
        resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# CRUD student endpoints
@app.route('/insertStudent', methods=['POST'])
@require_role('ADMIN')
def insertStudent():
    if request.method == "POST":
        try:
            name = request.form['username']
            email = request.form['email']
            password = request.form['password']
            profile_image = request.files.get('profile_image')
            profile_image_data = request.form.get('profile_image_data')
            filename = None

            if profile_image and profile_image.filename:
                img_bytes = profile_image.read()
                profile_image.seek(0)
                filename = secure_filename(profile_image.filename)
            elif profile_image_data:
                image_b64 = profile_image_data.split(',', 1)[1] if ',' in profile_image_data else profile_image_data
                img_bytes = base64.b64decode(image_b64)
                filename = "profile_upload.jpg"
            else:
                flash('Profile image is required when creating a student.', 'error')
                return redirect(url_for('adminStudents'))

            safe_email = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in email.strip().lower())
            profile_filename = f"{safe_email}_{filename}"
            os.makedirs('static/Profiles', exist_ok=True)
            with open(os.path.join('static/Profiles', profile_filename), 'wb') as f:
                f.write(img_bytes)

            cur = mysql.connection.cursor()
            try:
                cur.execute(
                    "INSERT INTO students (Name, Email, Password, Profile, Role) VALUES (%s, %s, %s, %s, %s)",
                    (name, email, generate_password_hash(password), profile_filename, 'STUDENT')
                )
            except Exception as col_error:
                if "Unknown column 'Profile'" in str(col_error):
                    cur.execute(
                        "INSERT INTO students (Name, Email, Password, Role) VALUES (%s, %s, %s, %s)",
                        (name, email, generate_password_hash(password), 'STUDENT')
                    )
                else:
                    raise col_error
            mysql.connection.commit()
            cur.close()
            flash('Student added successfully', 'success')
        except Exception as e:
            logger.error(f"Error inserting student: {e}")
            flash('Error adding student', 'error')
        return redirect(url_for('adminStudents'))

@app.route('/deleteStudent/<string:stdId>', methods=['GET'])
@require_role('ADMIN')
def deleteStudent(stdId):
    try:
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM students WHERE ID=%s", (stdId,))
        mysql.connection.commit()
        cur.close()
        flash("Record deleted successfully", 'success')
    except Exception as e:
        logger.error(f"Error deleting student: {e}")
        flash('Error deleting student', 'error')
    return redirect(url_for('adminStudents'))

@app.route('/updateStudent', methods=['POST'])
@require_role('ADMIN')
def updateStudent():
    if request.method == 'POST':
        try:
            id_data = request.form['id']
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            cur = mysql.connection.cursor()
            final_password = None
            if password and password.strip():
                final_password = generate_password_hash(password)
            else:
                cur.execute("SELECT Password FROM students WHERE ID=%s", (id_data,))
                old_row = cur.fetchone()
                final_password = old_row[0] if old_row else generate_password_hash("123456")
            cur.execute("""
                UPDATE students
                SET Name=%s, Email=%s, Password=%s
                WHERE ID=%s
            """, (name, email, final_password, id_data))
            mysql.connection.commit()
            cur.close()
            flash('Student updated successfully', 'success')
        except Exception as e:
            logger.error(f"Error updating student: {e}")
            flash('Error updating student', 'error')
        return redirect(url_for('adminStudents'))

@app.route('/registerFace', methods=['POST'])
@require_role('ADMIN')
def registerFace():
    try:
        student_id = request.form.get('student_id')
        student_name = request.form.get('student_name')
        file = request.files.get('face_image')
        webcam_image = request.form.get('webcam_image')

        filename = f"face_{student_id}_{int(time.time())}.jpg"
        os.makedirs('static/Profiles', exist_ok=True)
        save_path = os.path.join('static', 'Profiles', filename)

        # Accept either uploaded image file OR captured webcam base64 image.
        if file and file.filename:
            img_bytes = file.read()
            file.seek(0)
            with open(save_path, 'wb') as out:
                out.write(img_bytes)
        elif webcam_image:
            try:
                image_b64 = webcam_image.split(',', 1)[1] if ',' in webcam_image else webcam_image
                img_bytes = base64.b64decode(image_b64)
                with open(save_path, 'wb') as out:
                    out.write(img_bytes)
            except Exception:
                flash("Invalid webcam image data", 'error')
                return redirect(url_for('adminStudents'))
        elif request.is_json and request.json.get('image_data'):
            try:
                image_b64 = request.json.get('image_data', '')
                image_b64 = image_b64.split(',', 1)[1] if ',' in image_b64 else image_b64
                img_bytes = base64.b64decode(image_b64)
                with open(save_path, 'wb') as out:
                    out.write(img_bytes)
            except Exception:
                flash("Invalid image payload", 'error')
                return redirect(url_for('adminStudents'))
        else:
            cur = mysql.connection.cursor()
            cur.execute("SELECT Profile FROM students WHERE ID=%s", (student_id,))
            row = cur.fetchone()
            cur.close()
            existing_profile = None
            if row:
                existing_profile = row[0] if isinstance(row, (list, tuple)) else row.get('Profile')
            if existing_profile:
                flash(f"Face already registered for {student_name}", 'success')
                return redirect(url_for('adminStudents'))
            flash("Please upload a photo or capture from webcam", 'error')
            return redirect(url_for('adminStudents'))

        # Update database - handle both with and without Profile column
        cur = mysql.connection.cursor()
        try:
            cur.execute("UPDATE students SET Profile=%s WHERE ID=%s", (filename, student_id))
        except Exception as col_error:
            if "Unknown column 'Profile'" in str(col_error):
                # If Profile column doesn't exist, skip the update
                flash("Profile column not available in database", 'error')
            else:
                raise col_error
        
        mysql.connection.commit()
        cur.close()

        flash(f"Face registered for {student_name}", 'success')
        return redirect(url_for('adminStudents'))

    except Exception as e:
        logger.error(f"registerFace error: {e}")
        flash("Error registering face", 'error')
        return redirect(url_for('adminStudents'))

@app.route('/api/detect-eyes', methods=['POST'])
@rate_limit('detect_eyes', max_requests=60, window_seconds=60)
def api_detect_eyes():
    """Optional helper: detect eye pairs using Haar cascade for niqab fallback."""
    try:
        import cv2 as _cv2
        import numpy as _np
    except Exception:
        return jsonify({'error': 'cv2_unavailable'}), 503
    data = request.get_json(silent=True) or {}
    image_b64 = data.get('image', '')
    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400
    try:
        image_b64 = image_b64.split(',', 1)[1] if ',' in image_b64 else image_b64
        img_bytes = base64.b64decode(image_b64)
        nparr = _np.frombuffer(img_bytes, _np.uint8)
        img = _cv2.imdecode(nparr, _cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'decode_failed'}), 400
        cascade_path = os.path.join('Haarcascades', 'haarcascade_eye.xml')
        if not os.path.exists(cascade_path):
            fallback_dir = getattr(_cv2, 'data', None) and getattr(_cv2.data, 'haarcascades', None)
            if fallback_dir:
                fallback_path = os.path.join(fallback_dir, 'haarcascade_eye.xml')
                if os.path.exists(fallback_path):
                    cascade_path = fallback_path
        if not os.path.exists(cascade_path):
            logger.warning(f"Haarcascade eye file missing at {cascade_path}. Returning empty detections.")
            return jsonify({'eye_pairs': 0, 'bboxes': [], 'warning': 'cascade_missing'})
        eye_cascade = _cv2.CascadeClassifier(cascade_path)
        detections = eye_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        bboxes = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in detections]
        eye_pairs = max(1, len(bboxes) // 2) if len(bboxes) >= 2 else len(bboxes) // 2
        return jsonify({'eye_pairs': eye_pairs, 'bboxes': bboxes})
    except Exception as e:
        logger.warning(f"/api/detect-eyes failed: {e}")
        return jsonify({'error': 'processing_failed'}), 500

@app.route('/api/detect-book', methods=['POST'])
@rate_limit('detect_book', max_requests=40, window_seconds=60)
def api_detect_book():
    """Detect large rectangular book/notebook using OpenCV contour heuristic."""
    try:
        import cv2 as _cv2
        import numpy as _np
    except Exception:
        return jsonify({'book_detected': False, 'error': 'cv2_unavailable'}), 503
    data = request.get_json(silent=True) or {}
    image_b64 = data.get('image', '')
    if not image_b64:
        return jsonify({'book_detected': False, 'error': 'No image provided'}), 400
    try:
        image_b64 = image_b64.split(',', 1)[1] if ',' in image_b64 else image_b64
        img_bytes = base64.b64decode(image_b64)
        nparr = _np.frombuffer(img_bytes, _np.uint8)
        img = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'book_detected': False, 'error': 'decode_failed'}), 400
        h, w = img.shape[:2]
        gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
        gray = _cv2.GaussianBlur(gray, (5, 5), 0)
        edges = _cv2.Canny(gray, 40, 120)
        contours, _ = _cv2.findContours(edges, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        detected = False
        for cnt in contours:
            # Normal, robust thresholds for accurate book/paper detection
            area = _cv2.contourArea(cnt)
            if area < 800: # Lowered threshold to detect smaller/distant books
                continue
            rect = _cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), _ = rect
            if rh == 0 or rw == 0:
                continue
            aspect = max(rw, rh) / max(1e-3, min(rw, rh))
            # Relaxed aspect ratio for tilted books
            if aspect > 6.0: 
                continue
            hull = _cv2.convexHull(cnt)
            hull_area = _cv2.contourArea(hull)
            solidity = area / max(hull_area, 1e-3)
            if solidity < 0.35: # Lowered solidity to reliably detect books held by hands
                continue
            detected = True
            break

        return jsonify({'book_detected': detected})
    except Exception as e:
        logger.warning(f"/api/detect-book failed: {e}")
        return jsonify({'book_detected': False, 'error': 'processing_failed'}), 500

# -------------------------
# API Endpoints for Real-Time Data
# -------------------------
# Voice tracking dict: Dict[student_id, bool] - indicates if voice was currently heard
_student_voice_activity = {}
@app.route('/api/student-frame', methods=['POST'])
@require_role('STUDENT')
@rate_limit('student_frame', max_requests=900, window_seconds=60)
def api_student_frame():
    """Receive student browser frame quickly and defer heavy detection to background workers."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    if not CV2_AVAILABLE or np is None:
        return jsonify({'ok': False, 'error': 'OpenCV unavailable'}), 503

    payload = request.get_json(silent=True) or {}
    image_data = payload.get('image_data') or payload.get('frame')
    if not image_data:
        return jsonify({'ok': False, 'error': 'Missing image_data'}), 400

    try:
        logger.debug("Frame received in /api/student-frame")
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # This line was commented out in the original code, keeping it that way.
        frame = np.zeros((480, 640, 3), dtype=np.uint8) # Placeholder frame
        if frame is None:
            return jsonify({'ok': False, 'error': 'Bad frame'}), 400
        # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR) # This line was commented out in the original code, keeping it that way.

        student_id = str(user['Id'])
        student_name = str(user.get('Name') or f'student_{student_id}')
        
        sid_str = student_id
        with student_frame_rx_lock:
            count = int(student_frame_rx_counts.get(sid_str, 0)) + 1
            student_frame_rx_counts[sid_str] = count

        if (count % 15) == 1:
            logger.info(f"Frame received from: {sid_str} (count={count}, shape={getattr(frame, 'shape', None)})")

        with latest_student_frames_lock:
            prev = latest_student_frames.get(sid_str, {})
            raw_ts = time.time()
            latest_student_frames[sid_str] = {
                'frame': frame,
                'processed_frame': prev.get('processed_frame'),
                'timestamp': raw_ts,
                'frame_timestamp': raw_ts,
                'processed_timestamp': prev.get('processed_timestamp', 0.0),
                'detections': prev.get('detections', []),
                'processed_frame_b64': prev.get('processed_frame_b64'),
                'status_snapshot': prev.get('status_snapshot', {}),
                'last_visible_object_labels': prev.get('last_visible_object_labels', []),
                'last_prohibited_object_labels': prev.get('last_prohibited_object_labels', []),
                'last_person_count': prev.get('last_person_count', 0),
            }
        with student_stale_violation_lock:
            student_stale_violation_at.pop(sid_str, None)

        # The WASM engine now processes everything client side. 
        # We still accept frames to stream to the admin view via `admin_live_stream` MJPEG.
        
        return jsonify({'ok': True, 'queued': True, 'student_id': student_id})

    except Exception as e:
        logger.error(f"api_student_frame error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Decode failed'}), 400

@app.route('/api/upload-audio', methods=['POST'])
@require_role('STUDENT')
@rate_limit('student_audio_upload', max_requests=20, window_seconds=300)
def api_upload_audio():
    """Receive browser-recorded student audio and store it for admin recordings."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401

    file = request.files.get('audio')
    if not file or not file.filename:
        return jsonify({'ok': False, 'error': 'Missing audio file'}), 400

    try:
        student_id = int(user['Id'])
        student_name = ''.join(ch if ch.isalnum() else '_' for ch in str(user.get('Name') or 'student')).strip('_') or f"student_{student_id}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        content_type = (file.content_type or '').lower()
        ext = '.webm'
        if 'ogg' in content_type:
            ext = '.ogg'
        elif 'wav' in content_type or 'wave' in content_type:
            ext = '.wav'
        elif 'mp4' in content_type or 'm4a' in content_type:
            ext = '.m4a'

        audio_dir = _static_path('recording', 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        filename = f"{student_id}_{student_name}_{timestamp}{ext}"
        file.save(os.path.join(audio_dir, filename))
        return jsonify({'ok': True, 'filename': filename})
    except Exception as e:
        logger.error(f"api_upload_audio error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Audio upload failed'}), 500

@app.route('/api/upload-session-recording', methods=['POST'])
@require_role('STUDENT')
@rate_limit('student_session_recording_upload', max_requests=12, window_seconds=300)
def api_upload_session_recording():
    """Receive a combined browser-recorded exam session video with embedded audio."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401

    file = request.files.get('recording')
    if not file or not file.filename:
        return jsonify({'ok': False, 'error': 'Missing recording file'}), 400

    try:
        student_id = int(user['Id'])
        student_name = ''.join(ch if ch.isalnum() else '_' for ch in str(user.get('Name') or 'student')).strip('_') or f"student_{student_id}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        started_at_raw = (request.form.get('started_at') or '').strip()
        session_start = timestamp
        if started_at_raw:
            try:
                session_start = datetime.fromtimestamp(float(started_at_raw) / 1000.0).strftime("%Y%m%d_%H%M%S")
            except Exception:
                session_start = timestamp

        content_type = (file.content_type or '').lower()
        ext = '.webm'
        if 'mp4' in content_type:
            ext = '.mp4'
        elif 'ogg' in content_type:
            ext = '.ogg'
        elif file.filename and '.' in file.filename:
            guessed_ext = os.path.splitext(file.filename)[1].lower()
            if guessed_ext in ('.webm', '.mp4', '.ogg', '.mkv'):
                ext = guessed_ext

        video_dir = _static_path('recording')
        os.makedirs(video_dir, exist_ok=True)
        filename = f"{student_id}_{student_name}_{session_start}{ext}"
        output_path = os.path.join(video_dir, filename)
        file.save(output_path)

        session_id = _get_active_session_id(student_id)
        meta_path = os.path.join(video_dir, f"{student_id}_{student_name}_{session_start}.json")
        metadata = {
            'student_id': student_id,
            'student_name': str(user.get('Name') or f"Student {student_id}"),
            'session_id': session_id,
            'session_start': session_start,
            'video_path': output_path,
            'embedded_audio': True,
            'content_type': content_type or 'video/webm',
            'file_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
        }
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as meta_err:
            logger.warning(f"session recording metadata save failed: {meta_err}")

        return jsonify({'ok': True, 'filename': filename})
    except Exception as e:
        logger.error(f"api_upload_session_recording error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Session recording upload failed'}), 500

@app.route('/api/student-exit-signal', methods=['POST'])
@require_role('STUDENT')
@rate_limit('student_exit_signal', max_requests=20, window_seconds=60)
def api_student_exit_signal():
    """Receive keepalive beacon when student closes/hides exam tab."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    try:
        student_id = str(user['Id'])
        student_name = str(user.get('Name') or 'Unknown')
        event_type = (request.form.get('event_type') or request.args.get('event_type') or 'TAB_CLOSE').upper()
        details = (request.form.get('details') or request.args.get('details') or 'Tab/window closed during exam').strip()
        details = details[:500]

        # Enforce tab switch as a violation (strict — short cooldown so repeats are counted)
        warning_added = True
        terminated = False
        if warning_system:
            warning_added, terminated = warning_system.add_warning(student_id, 'TAB_SWITCH', f"{event_type}: {details}")
        if warning_added:
            _record_runtime_warning(student_id, student_name, 'TAB_SWITCH', f"{event_type}: {details}")
        logger.info(f"[TAB_SWITCH ENFORCED] student={student_id} details={details} terminated={terminated}")

        # Best-effort immediate DB persistence for close events
        try:
            if warning_added and warning_system:
                cur = mysql.connection.cursor()
                cur.execute("""
                    SELECT SessionID FROM exam_sessions
                    WHERE StudentID=%s AND Status='IN_PROGRESS'
                    ORDER BY StartTime DESC LIMIT 1
                """, (student_id,))
                sess = cur.fetchone()
                if sess:
                    write_violation_async(student_id, sess[0], 'TAB_SWITCH', f"{event_type}: {details}")
                cur.close()
            cur = mysql.connection.cursor()
            cur.execute("""
                SELECT SessionID FROM exam_sessions
                WHERE StudentID=%s AND Status='IN_PROGRESS'
                ORDER BY StartTime DESC LIMIT 1
            """, (student_id,))
            sess = cur.fetchone()
            if sess:
                write_violation_async(student_id, sess[0], 'TAB_SWITCH', f"{event_type}: {details}")
            cur.close()
        except Exception as db_err:
            logger.warning(f"student_exit_signal DB save failed: {db_err}")

        return jsonify({'ok': True})
    except Exception as e:
        logger.error(f"api_student_exit_signal error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Exit signal failed'}), 500

@app.route('/api/my-warnings')
@require_role('STUDENT')
@rate_limit('student_warning_state', max_requests=120, window_seconds=60)
def api_my_warnings():
    """Return the current student's live warning state for UI sync."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    try:
        student_id = str(user['Id'])
        warnings_count = 0
        violations = []
        if warning_system:
            warnings_count = int(warning_system.get_warnings(student_id) or 0)
            violations = warning_system.get_violations(student_id) or []
        else:
            runtime_state = _get_runtime_warning_state(student_id)
            warnings_count = int(runtime_state.get('warnings') or 0)
            violations = runtime_state.get('violations') or []
        latest_violation = violations[-1] if violations else None
        return jsonify({
            'ok': True,
            'student_id': int(student_id),
            'warnings': min(warnings_count, 3),
            'violations': violations,
            'latest_violation': latest_violation
        })
    except Exception as e:
        logger.error(f"api_my_warnings error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Warning state fetch failed'}), 500

@app.route('/api/acknowledge-warning', methods=['POST'])
@require_role('STUDENT')
def api_acknowledge_warning():
    """Triggered by student browser when they click 'I Understand' on a warning.
    Enforces a strict, guaranteed 3-second quiet gap across the entire system.
    """
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    
    student_id = str(user['Id'])
    now = time.time()
    
    # 1. Update the warning system global gap and type gap logs
    if warning_system:
        warning_system.acknowledge_warning(student_id)
        
    # 2. Reset the server-side telemetry warning alert throttles to now (which enforces a new cooldown)
    for k in list(_last_object_alert_at.keys()):
        if k.startswith(f"{student_id}:"):
            _last_object_alert_at[k] = now
            
    logger.info(f"✓ Warning acknowledged for student {student_id}. Resetting all server-side alert throttles.")
    return jsonify({'ok': True})

@app.route('/api/today-violations')
@require_role('ADMIN')
def api_today_violations():
    """Return total violations count today from violations table"""
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT COUNT(*) FROM violations WHERE DATE(Timestamp) = CURDATE()")
        row = cur.fetchone()
        cur.close()
        count = int(row[0]) if row else 0
        return jsonify({'count': count})
    except Exception as e:
        logger.error(f"API today-violations error: {e}")
        return jsonify({'count': 0})

@app.route('/api/student-warnings/<int:student_id>')
@require_role('ADMIN')
def api_student_warnings(student_id):
    """Return current live warnings for a student from warning_system"""
    if warning_system:
        count = int(warning_system.get_warnings(student_id) or 0)
        violations = warning_system.get_violations(student_id) or []
    else:
        count = 0
        violations = []
    runtime_state = _get_runtime_warning_state(student_id)
    count = max(count, int(runtime_state.get('warnings') or 0))
    if len(runtime_state.get('violations') or []) > len(violations):
        violations = runtime_state.get('violations') or violations
    return jsonify({'student_id': student_id, 'warnings': count, 'violations': violations})

@app.route('/api/all-student-warnings')
@require_role('ADMIN')
def api_all_student_warnings():
    """Return warnings for all active students"""
    result = {}
    active_ids = set()
    if warning_system:
        with warning_system.lock:
            for sid, count in warning_system.warnings.items():
                active_ids.add(str(sid))
                result[str(sid)] = {
                    'warnings': count,
                    'name': warning_system.student_names.get(sid, 'Unknown'),
                    'violations': warning_system.violations.get(sid, [])
                }
    with runtime_warning_state_lock:
        for sid, rec in runtime_warning_state.items():
            active_ids.add(str(sid))
            current = result.setdefault(str(sid), {'warnings': 0, 'name': rec.get('student_name', 'Unknown'), 'violations': []})
            current['warnings'] = max(int(current.get('warnings') or 0), int(rec.get('warnings') or 0))
            if len(rec.get('violations') or []) > len(current.get('violations') or []):
                current['violations'] = list(rec.get('violations') or [])
            if not current.get('name'):
                current['name'] = rec.get('student_name', 'Unknown')
            if rec.get('start_time') and not current.get('start_time'):
                current['start_time'] = int(rec.get('start_time'))
    # Attach start_time (epoch ms) when available so admin UI can show accurate duration.
    try:
        if active_ids:
            cur = mysql.connection.cursor()
            placeholders = ','.join(['%s'] * len(active_ids))
            cur.execute(
                f"SELECT StudentID, StartTime FROM exam_sessions WHERE Status='IN_PROGRESS' AND StudentID IN ({placeholders})",
                tuple(active_ids)
            )
            for sid, start_time in cur.fetchall():
                sid_str = str(sid)
                if sid_str in result and start_time:
                    result[sid_str]['start_time'] = int(start_time.timestamp() * 1000)
            cur.close()
    except Exception as e:
        logger.warning(f"api_all_student_warnings start_time fetch failed: {e}")
    return jsonify(result)

@app.route('/api/admin-active-students')
@require_role('ADMIN')
def api_admin_active_students():
    """Return a list of active students with live warnings and start times."""
    now = time.time()
    if _student_cache['data'] and (now - _student_cache['ts']) < 10:
        return jsonify(_student_cache['data'])

    ensure_db_schema()
    active_ids = set()

    with active_exam_students_lock:
        active_ids.update(str(sid) for sid in active_exam_students)
    with latest_student_frames_lock:
        active_ids.update(str(sid) for sid in latest_student_frames.keys())
    with runtime_warning_state_lock:
        active_ids.update(str(sid) for sid in runtime_warning_state.keys())
    if warning_system:
        with warning_system.lock:
            active_ids.update(str(sid) for sid in warning_system.warnings.keys())

    if not active_ids:
        _student_cache['data'] = {'students': []}
        _student_cache['ts'] = now
        return jsonify(_student_cache['data'])

    names = {}
    start_times = {}
    try:
        cur = mysql.connection.cursor()
        placeholders = ','.join(['%s'] * len(active_ids))
        cur.execute(f"SELECT ID, Name FROM students WHERE ID IN ({placeholders})", tuple(active_ids))
        for sid, name in cur.fetchall():
            names[str(sid)] = name or f"Student {sid}"
        cur.execute(
            f"SELECT StudentID, StartTime FROM exam_sessions WHERE Status='IN_PROGRESS' AND StudentID IN ({placeholders})",
            tuple(active_ids)
        )
        for sid, start_time in cur.fetchall():
            if start_time:
                start_times[str(sid)] = int(start_time.timestamp() * 1000)
        cur.close()
    except Exception as e:
        logger.warning(f"admin-active-students DB fetch failed: {e}")

    students = []
    for sid in active_ids:
        warnings_count = 0
        violations = []
        if warning_system:
            warnings_count = int(warning_system.get_warnings(sid) or 0)
            violations = warning_system.get_violations(sid) or []
        runtime_state = _get_runtime_warning_state(sid)
        warnings_count = max(warnings_count, int(runtime_state.get('warnings') or 0))
        if len(runtime_state.get('violations') or []) > len(violations):
            violations = runtime_state.get('violations') or violations
        start_ms = start_times.get(sid) or runtime_state.get('start_time')

        last_update_ms = None
        telemetry_metrics = None
        telemetry_analysis = None
        telemetry_snapshot = None
        with latest_student_frames_lock:
            entry = latest_student_frames.get(sid)
            if entry and entry.get('timestamp'):
                last_update_ms = int(float(entry['timestamp']) * 1000)
            telemetry_snapshot = entry.get('wasm_telemetry') if entry else None
            if telemetry_snapshot:
                telemetry_metrics = telemetry_snapshot.get('metrics') or {}
                telemetry_analysis = telemetry_snapshot.get('analysis') or {}
            elif entry and entry.get('status_snapshot'):
                telemetry_metrics = {
                    'face_count': entry['status_snapshot'].get('faces_detected'),
                    'banned_labels': entry.get('last_prohibited_object_labels') or []
                }
                telemetry_analysis = {
                    'suspicion_score': entry['status_snapshot'].get('suspicion_score', 0),
                    'active_flags': entry['status_snapshot'].get('active_flags', [])
                }

        students.append({
            'student_id': sid,
            'student_name': names.get(sid) or runtime_state.get('student_name') or f"Student {sid}",
            'warnings': min(warnings_count, 3),
            'violations': violations,
            'start_time': start_ms,
            'last_update': last_update_ms,
            'metrics': telemetry_metrics or {},
            'suspicion_score': (telemetry_analysis or {}).get('suspicion_score', 0),
            'active_flags': (telemetry_analysis or {}).get('active_flags') or [],
            'banned_labels': (telemetry_metrics or {}).get('banned_labels') or [],
            'face_count': (telemetry_metrics or {}).get('face_count')
        })

    response = {'students': students}
    _student_cache['data'] = response
    _student_cache['ts'] = time.time()
    return jsonify(response)

# Pipeline API endpoints were removed since inference runs in client WASM

# -------------------------
# SocketIO handlers
# -------------------------
if MONITORING_ENABLED and socketio:
    @socketio.on('connect', namespace='/student')
    def student_connect():
        sid = request.sid
        user = current_user()
        user_id = str((user or {}).get('Id') or request.args.get('student_id') or '')
        if user_id:
            with _student_socket_sids_lock:
                _student_socket_sids[user_id] = sid
        try:
            if user_id:
                join_room(f"student:{user_id}")
        except Exception:
            pass
        logger.info(f"Student socket connected: sid={sid} user={user_id or 'anon'} role={(user or {}).get('Role')}")

    @socketio.on('disconnect', namespace='/student')
    def student_disconnect():
        sid = request.sid
        try:
            with _student_socket_sids_lock:
                for uid, ssid in list(_student_socket_sids.items()):
                    if ssid == sid:
                        _student_socket_sids.pop(uid, None)
                        break
        except Exception:
            pass
        logger.info(f'Student socket disconnected: {sid}')

    @socketio.on('connect', namespace='/admin')
    def admin_connect():
        sid = request.sid
        logger.info(f"Admin socket connected: sid={sid}")
        try:
            join_room('admin_room')
        except Exception:
            pass

    @socketio.on('disconnect', namespace='/admin')
    def admin_disconnect():
        sid = request.sid
        logger.info(f"Admin socket disconnected: sid={sid}")

    @socketio.on('request_student_feed', namespace='/student')
    def handle_request_student_feed(data):
        student_id = data.get('student_id')
        emit('request_ack', {'student_id': student_id})

    # ══════════════════════════════════════════════════════════════════
    # CRITICAL FIX: 'warning_issued' handler
    # Exam.html emits this every time student gets a warning.
    # Without this handler warnings NEVER reach admin dashboard.
    # ══════════════════════════════════════════════════════════════════
    @socketio.on('warning_issued', namespace='/student')
    def handle_warning_issued(data):
        student_id   = str(data.get('student_id'))
        student_name = data.get('student_name', 'Unknown')
        violation    = data.get('violation', {})
        vtype        = violation.get('type', 'TAB_SWITCH')
        details      = violation.get('details', str(vtype))
        if str(vtype).upper() == 'TAB_SWITCH':
            dlow = str(details).lower()
            if 'lost focus' in dlow or 'window' in dlow or 'hidden' in dlow:
                details = 'Tab switching detected'
        runtime_count, runtime_violation = _record_runtime_warning(student_id, student_name, vtype, details)
        
        logger.info(f"⚠️  warning_issued received: student={student_id} type={vtype}")
        
        # 1. Update in-memory warning_system so count stays accurate
        if warning_system and student_id:
            if student_id not in warning_system.warnings:
                warning_system.initialize_student(student_id, student_name)
            warning_added, terminated = warning_system.add_warning(student_id, vtype, details, emit_to_student=False)
            # add_warning already emits 'student_violation' to /admin — done!
            if terminated:
                emit('auto_terminated', {'student_id': student_id, 'reason': 'Max warnings reached'})
        else:
            # warning_system unavailable — manually forward to admin
            socketio.emit('student_violation', {
                'student_id':     student_id,
                'student_name':   student_name,
                'total_warnings': max(int(data.get('warning_number', 1) or 1), runtime_count),
                'violation':      runtime_violation,
            }, namespace='/admin')
        
        # 2. Also save violation immediately to DB (live persistence)
        VTYPE_MAP = {
            'TAB_SWITCH': 'TAB_SWITCH', 'tab_switch': 'TAB_SWITCH',
            'FULLSCREEN_EXIT': 'TAB_SWITCH', 'fullscreen_exit': 'TAB_SWITCH',
            'PROHIBITED_SHORTCUT': 'PROHIBITED_SHORTCUT', 'prohibited_shortcut': 'PROHIBITED_SHORTCUT',
            'KEYBOARD_SHORTCUT': 'PROHIBITED_SHORTCUT', 'DEVTOOLS_OPEN': 'PROHIBITED_SHORTCUT',
            'DEVTOOLS_SHORTCUT': 'PROHIBITED_SHORTCUT', 'DEVTOOLS_OPENED': 'PROHIBITED_SHORTCUT',
            'COPY_PASTE': 'PROHIBITED_SHORTCUT',
            'MULTIPLE_FACES': 'MULTIPLE_FACES', 'multiple_faces': 'MULTIPLE_FACES',
            'NO_FACE': 'NO_FACE', 'no_face': 'NO_FACE',
            'FACE_OBSCURED': 'NO_FACE', 'face_obscured': 'NO_FACE',
            'EYES_CLOSED': 'EYES_CLOSED', 'eyes_closed': 'EYES_CLOSED',
            'GAZE_LEFT': 'GAZE_LEFT', 'gaze_left': 'GAZE_LEFT',
            'GAZE_RIGHT': 'GAZE_RIGHT', 'gaze_right': 'GAZE_RIGHT',
            'GAZE_UP': 'GAZE_UP', 'gaze_up': 'GAZE_UP',
            'GAZE_DOWN': 'GAZE_DOWN', 'gaze_down': 'GAZE_DOWN',
            'GAZE_UP_LEFT': 'GAZE_UP_LEFT', 'gaze_up_left': 'GAZE_UP_LEFT',
            'GAZE_UP_RIGHT': 'GAZE_UP_RIGHT', 'gaze_up_right': 'GAZE_UP_RIGHT',
            'GAZE_DOWN_LEFT': 'GAZE_DOWN_LEFT', 'gaze_down_left': 'GAZE_DOWN_LEFT',
            'GAZE_DOWN_RIGHT': 'GAZE_DOWN_RIGHT', 'gaze_down_right': 'GAZE_DOWN_RIGHT',
            'VOICE_DETECTED': 'VOICE_DETECTED', 'voice_detected': 'VOICE_DETECTED',
            'DISTRACTION': 'DISTRACTION', 'distraction': 'DISTRACTION',
            'NOT_FORWARD': 'DISTRACTION', 'not_forward': 'DISTRACTION',
            'GAZE_AWAY': 'DISTRACTION', 'gaze_away': 'DISTRACTION',
            'STUDENT_LEFT_SEAT': 'STUDENT_LEFT_SEAT', 'student_left_seat': 'STUDENT_LEFT_SEAT',
            'MIC_OFF': 'VOICE_DETECTED', 'mic_off': 'VOICE_DETECTED',
            'HEAD_MOVEMENT': 'HEAD_MOVEMENT', 'head_movement': 'HEAD_MOVEMENT',
            # HEAD_DOWN: new head-down detection feature — maps to its own violation type
            'HEAD_DOWN': 'HEAD_DOWN', 'head_down': 'HEAD_DOWN',
            'HEAD_POSE': 'HEAD_MOVEMENT', 'head_pose': 'HEAD_MOVEMENT',
            'IDENTITY_MISMATCH': 'IDENTITY_MISMATCH', 'identity_mismatch': 'IDENTITY_MISMATCH',
            'CAMERA_OFF': 'NO_FACE', 'camera_off': 'NO_FACE',
            'CAMERA_BLOCKED': 'NO_FACE', 'camera_blocked': 'NO_FACE',
            'PROHIBITED_OBJECT': 'PROHIBITED_OBJECT', 'prohibited_object': 'PROHIBITED_OBJECT',
            'TERMINATED_BY_ADMIN': 'TERMINATED_BY_ADMIN', 'terminated_by_admin': 'TERMINATED_BY_ADMIN',
        }
        db_vtype = VTYPE_MAP.get(vtype, VTYPE_MAP.get(str(vtype).upper(), 'DISTRACTION'))
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                SELECT SessionID FROM exam_sessions
                WHERE StudentID=%s AND Status='IN_PROGRESS'
                ORDER BY StartTime DESC LIMIT 1
            """, (student_id,))
            sess = cur.fetchone()
            # DB persistence is handled via warning_system.violation_writer; avoid double-inserts.
            if sess and not warning_system:
                write_violation_async(student_id, sess[0], db_vtype, str(details)[:500])
            cur.close()
        except Exception as db_err:
            logger.warning(f"Live violation DB save failed: {db_err}")

    @socketio.on('exam_auto_terminated', namespace='/student')
    def handle_exam_auto_terminated(data):
        """Student exam terminated due to max warnings"""
        student_id   = data.get('student_id')
        student_name = data.get('student_name', 'Unknown')
        reason       = data.get('reason', 'Max warnings reached')
        logger.info(f"🚫 Exam auto-terminated: student={student_id}")
        if student_id:
            if warning_system:
                try:
                    warning_system.flush_violations_to_db(student_id, _get_active_or_latest_session_id(student_id))
                except Exception as e:
                    logger.warning(f"flush_violations_to_db failed for {student_id}: {e}")
            _end_exam_runtime_state(str(student_id), clear_warning_cache=False)
        socketio.emit('student_exam_terminated', {
            'student_id':   student_id,
            'student_name': student_name,
            'reason':       reason,
        }, namespace='/admin')
    
    @socketio.on('terminate_exam', namespace='/student')
    def handle_terminate_exam(data):
        student_id = str(data.get('student_id'))
        reason = data.get('reason', 'Manual termination by admin')
        if warning_system:
            warning_system.add_warning(student_id, 'TERMINATED_BY_ADMIN', reason, emit_to_student=False)
        emit('terminated_ack', {'student_id': student_id, 'reason': reason})
    
    @socketio.on('prohibited_action', namespace='/student')
    def handle_prohibited_action(data):
        student_id = str(data.get('student_id'))
        action = data.get('action')
        if student_id:
            student_name = data.get('student_name') or 'Unknown'
            # Record runtime warning but omit termination logic since that's handled client-side or natively now
            _record_runtime_warning(student_id, student_name, 'PROHIBITED_SHORTCUT', str(action))
            if warning_system:
                _, terminated = warning_system.add_warning(student_id, 'PROHIBITED_SHORTCUT', str(action))
            else:
                terminated = False
            logger.info(f"[SHORTCUT] student={student_id} action={action} terminated={terminated}")
            if terminated:
                emit('auto_terminated', {'student_id': student_id})
        else:
            logger.info(f"[SHORTCUT IGNORED] student={student_id} action={action}")
    
    @socketio.on('tab_switch_detected', namespace='/student')
    def handle_tab_switch(data):
        student_id = str(data.get('student_id'))
        student_name = str(data.get('student_name') or (current_user() or {}).get('Name') or 'Unknown')
        details = str(data.get('details') or 'Tab switch detected').strip()
        dlow = details.lower()
        if 'lost focus' in dlow or 'window' in dlow or 'hidden' in dlow:
            details = 'Tab switching detected'
        if student_id:
            warning_added = True
            terminated = False
            if warning_system:
                warning_added, terminated = warning_system.add_warning(student_id, 'TAB_SWITCH', details)
            if warning_added:
                _record_runtime_warning(student_id, student_name, 'TAB_SWITCH', details)
            logger.info(f"[TAB_SWITCH] student={student_id} details={details} terminated={terminated}")
        else:
            logger.info(f"[TAB_SWITCH IGNORED] missing student_id details={details}")
    
    # --- WASM TELEMETRY LAYER ---
    @socketio.on('telemetry_update', namespace='/student')
    def handle_telemetry_update(data):
        student_id = str(data.get('student_id'))
        score = data.get('score', 0)
        faces = data.get('faces', 0)
        objects = data.get('objects', {})
        allowed_objects = []
        if objects.get('phone'):
            allowed_objects.append('cell phone')
        if objects.get('smartwatch'):
            allowed_objects.append('smartwatch')
        if objects.get('book'):
            allowed_objects.append('book')
        if objects.get('paper'):
            allowed_objects.append('paper')
        
        sid_str = student_id
        # If v2 telemetry is already flowing, avoid duplicating object alerts
        last_v2 = float(_last_v2_telemetry_at.get(sid_str, 0.0))
        use_legacy_alerts = (time.time() - last_v2) > 2.0
        now_ts = time.time()
        
        with latest_student_frames_lock:
            prev = latest_student_frames.get(sid_str, {})
            # Keep student alive in the active students polling, and update their telemetry
            if score >= 50:
                logger.info(f"🚨 [WASM TELEMETRY] High Suspicion for {sid_str}: Score {score}")
                
            latest_student_frames[sid_str] = {
                'frame': prev.get('frame'),
                'frame_bytes': prev.get('frame_bytes'),
                'processed_frame': prev.get('processed_frame'),
                'timestamp': prev.get('timestamp', now_ts), 
                'frame_timestamp': prev.get('frame_timestamp', now_ts),
                'processed_timestamp': now_ts,
                'detections': prev.get('detections', []),
                'status_snapshot': {
                    'warning_count': int(warning_system.get_warnings(sid_str) if warning_system else 0),
                    'suspicion_score': score,
                    'faces_detected': faces,
                    'phone_detected': 'cell phone' in allowed_objects,
                    'smartwatch_detected': 'smartwatch' in allowed_objects,
                    'book_detected': 'book' in allowed_objects,
                    'paper_detected': 'paper' in allowed_objects
                },
                'last_visible_object_labels': prev.get('last_visible_object_labels', []),
                'last_prohibited_object_labels': allowed_objects,
                'last_person_count': faces,
            }
    
        # Server-side object warning from legacy telemetry when v2 is absent
        if use_legacy_alerts and ('book' in allowed_objects or 'paper' in allowed_objects):
            student_name = _get_runtime_warning_state(sid_str).get('student_name') or f"Student {sid_str}"
            label = 'book' if 'book' in allowed_objects else 'paper'
            _maybe_issue_object_warning(sid_str, student_name, label)
    
    # --- WEBRTC SIGNALING (STUDENT -> ADMIN) ---
    
    # ── Telemetry buffering & history ──
    _telemetry_history = {}  # { student_id: deque(maxlen=200) }
    _telemetry_history_lock = threading.Lock()
    _telemetry_buffer = {}   # { student_id: latest telemetry payload }
    _telemetry_buffer_lock = threading.Lock()
    
    def flush_telemetry_loop():
        while True:
            try:
                if EVENTLET_AVAILABLE:
                    eventlet.sleep(0.25)
                else:
                    time.sleep(0.25)
                with _telemetry_buffer_lock:
                    snapshot = dict(_telemetry_buffer)
                    _telemetry_buffer.clear()
                if not snapshot:
                    continue
                for sid, payload in snapshot.items():
                    try:
                        socketio.emit('telemetry_update', {'student_id': sid, 'data': payload},
                                      room='admin_room', namespace='/admin')
                    except Exception as emit_err:
                        logger.warning(f"telemetry flush emit failed for {sid}: {emit_err}")
            except Exception as loop_err:
                logger.error(f"telemetry flush loop error: {loop_err}", exc_info=True)
                if not EVENTLET_AVAILABLE:
                    time.sleep(0.25)
    
    if EVENTLET_AVAILABLE:
        eventlet.spawn(flush_telemetry_loop)
    else:
        threading.Thread(target=flush_telemetry_loop, daemon=True).start()
    
    @socketio.on('student_live_frame', namespace='/student')
    def handle_student_live_frame(data):
        """Relay live camera frame from student to admin via socket (no OpenCV needed)."""
        student_id = str(data.get('student_id', ''))
        if not student_id or not data.get('frame'):
            return
        now_ts = time.time()
        # Log once every 5 seconds per student to confirm frames are arriving
        try:
            last_log = _last_frame_log_at.get(student_id, 0)
            if now_ts - last_log >= 5:
                _last_frame_log_at[student_id] = now_ts
                logger.info(f"[FRAME] Received live frame from student {student_id} (len={len(data['frame'])})")
        except Exception:
            pass
        # Register student as active
        with active_exam_students_lock:
            active_exam_students.add(student_id)
        # Update timestamp in latest_student_frames so polling finds them
        warnings_count = int(warning_system.get_warnings(student_id) if warning_system else 0)
        violations_live = warning_system.get_violations(student_id) if warning_system else []
        runtime_state = _get_runtime_warning_state(student_id)
        warnings_count = max(warnings_count, int(runtime_state.get('warnings') or 0))
        if len(runtime_state.get('violations') or []) > len(violations_live):
            violations_live = runtime_state.get('violations') or violations_live
        # Decode base64 frame to numpy for MJPEG fallback and store raw bytes
        decoded_frame = None
        frame_b64 = str(data.get('frame') or '')
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',', 1)[1]
        frame_bytes = None
        try:
            import base64 as _b64
            frame_bytes = _b64.b64decode(frame_b64)
            if cv2:
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                decoded_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as dec_err:
            logger.debug(f"[FRAME] decode failed for student {student_id}: {dec_err}")
        with latest_student_frames_lock:
            prev = latest_student_frames.get(student_id, {})
            latest_student_frames[student_id] = {
                **prev,
                'frame': frame_b64,
                'frame_bytes': frame_bytes if frame_bytes is not None else prev.get('frame_bytes'),
                'processed_frame': decoded_frame if decoded_frame is not None else prev.get('processed_frame'),
                'timestamp': now_ts,
                'frame_timestamp': now_ts,
                'processed_timestamp': now_ts,
                'status_snapshot': {
                    **(prev.get('status_snapshot') or {}),
                    'warning_count': warnings_count,
                    'suspicion_score': data.get('score', 0),
                }
            }
        # Relay to admin namespace as 'student_frame' (which admin already listens for)
        socketio.emit('student_frame', {
            'student_id': student_id,
            'frame': frame_b64,
            'score': data.get('score', 0),
            'timestamp_ms': data.get('timestamp_ms', int(now_ts * 1000)),
            'warnings': warnings_count,
            'violations': violations_live,
        }, namespace='/admin')
        # Notify admin once per student session that frames are arriving (hides loading spinner)
        with _feed_started_lock:
            if student_id not in _feed_started_for:
                _feed_started_for.add(student_id)
                socketio.emit('feed_started', {'student_id': student_id}, namespace='/admin')
    
    @socketio.on('student_audio_chunk', namespace='/student')
    def handle_audio_chunk(data):
        """
        Relays raw PCM Int16 audio chunk to the admin namespace for local processing.
        """
        try:
            user = current_user()
            if not user or user.get('Role') != 'STUDENT':
                return
            student_id = str(user['Id'])
            
            # Relay raw JS ArrayBuffer bytes to Admin Dashboard
            socketio.emit('relay_student_audio', {
                'student_id': student_id,
                'audio': data  # data is the binary buffer
            }, namespace='/admin')
                
        except Exception as e:
            logger.error(f"Error relaying audio chunk: {e}")
    
    @socketio.on('admin_trigger_voice_warning', namespace='/admin')
    def handle_admin_voice_detection_trigger(data):
        """
        Triggered by the Admin Dashboard's local voice detection logic.
        Sends a voice_alert directly to the student.
        """
        try:
            student_id = data.get('student_id')
            if not student_id: return
            
            logger.warning(f"🎙️ Admin-side system DETECTED VOICE for student {student_id}")
            
            # Mark as active for metrics
            _student_voice_activity[student_id] = {"active": True, "rms": data.get('rms', 100)}
            
            # Notify student
            target_room = f"student:{student_id}"
            socketio.emit('voice_alert', {'detected': True, 'rms': data.get('rms', 100)}, namespace='/student', to=target_room)
        except Exception as e:
            logger.error(f"Error triggering voice warning: {e}")
    
    @socketio.on('telemetry_update_v2', namespace='/student')
    def handle_telemetry_update_v2(data):
        """Receive hyper-detailed telemetry from student WASM engine and relay to admin."""
        student_id = str(data.get('student_id', ''))
        if not student_id:
            return
        _last_v2_telemetry_at[student_id] = time.time()
        
        analysis = data.get('analysis', {})
        metrics = data.get('metrics', {})
        
        # Normalize active_flags from client (handle both list and dict)
        raw_flags = analysis.get('active_flags', [])
        active_flags = []
        if isinstance(raw_flags, dict):
            for k, v in raw_flags.items():
                if v: active_flags.append(k.upper())
        elif isinstance(raw_flags, list):
            active_flags = [str(f).upper() for f in raw_flags]

        # Normalize & filter labels to only the objects we care about.
        raw_labels = [str(l).lower() for l in (metrics.get('banned_labels') or [])]
        allowed_labels = {
            'cell phone', 'phone',
            'clock', 'smartwatch',
            'book', 'book_heuristic', 'notebook', 'textbook', 'copy', 'register',
            'journal', 'notepad', 'notes', 'binder', 'folder', 'document', 'magazine',
            'paper'
        }
        filtered = [l for l in raw_labels if l in allowed_labels]
        normalized = []
        for label in filtered:
            if label in ('cell phone', 'phone'):
                normalized.append('cell phone')
            elif label in ('clock', 'smartwatch'):
                normalized.append('smartwatch')
            elif label in ('book', 'book_heuristic', 'book/paper', 'notebook', 'textbook', 'copy', 'register', 'journal', 'notepad', 'notes', 'binder', 'folder', 'document', 'magazine'):
                normalized.append('book')
            elif label == 'paper':
                normalized.append('paper')
        
        # Inject accessory detections into normalized labels for admin UI
        acc = metrics.get('accessory', {})
        if acc.get('headphone_detected') and 'headphone' not in normalized:
            normalized.append('headphone')
        if acc.get('earphone_detected') and 'earphone' not in normalized:
            normalized.append('earphone')

        metrics['banned_labels'] = list(dict.fromkeys(normalized))
        data['metrics'] = metrics

    
        runtime_state = _get_runtime_warning_state(student_id)
        student_name = runtime_state.get('student_name') or f"Student {student_id}"
        warnings_count = int(runtime_state.get('warnings') or 0)
        if warning_system:
            try:
                warnings_count = max(warnings_count, int(warning_system.get_warnings(student_id) or 0))
            except Exception:
                pass
        violations = runtime_state.get('violations') or []
        start_time = runtime_state.get('start_time')
        data['warnings'] = warnings_count
        data['violations'] = violations[-5:]
        if start_time:
            data['start_time'] = int(start_time)
    
        # ─── BEHAVIORAL & OBJECT WARNINGS ───
        # If server sees book/paper in telemetry, issue a server warning immediately
        banned_lower = [str(x).lower() for x in metrics.get('banned_labels', [])]
        if 'book' in banned_lower or 'paper' in banned_lower or 'book/notebook' in banned_lower:
            label = 'Book' if 'book' in banned_lower or 'book/notebook' in banned_lower else 'Paper'
            _maybe_issue_object_warning(student_id, student_name, label)

        # 1. Face Presence - now respects client-side 3s delay
        if 'NO_FACE' in active_flags:
            _maybe_issue_behavioral_warning(student_id, student_name, 'NO_FACE', "No face detected. Please ensure you are visible to the camera.")

        # 2. Eye Gaze (Up/Down/Left/Right)
        if any(f in active_flags for f in ['GAZE_UP', 'GAZE_UP_LEFT', 'GAZE_UP_RIGHT']):
            _maybe_issue_behavioral_warning(student_id, student_name, 'GAZE_UP', "Looking UP detected")
        elif any(f in active_flags for f in ['GAZE_DOWN', 'GAZE_DOWN_LEFT', 'GAZE_DOWN_RIGHT']):
            _maybe_issue_behavioral_warning(student_id, student_name, 'GAZE_DOWN', "Looking DOWN detected")
        elif 'LOOKING_AWAY' in active_flags:
            _maybe_issue_behavioral_warning(student_id, student_name, 'DISTRACTION', "Looking away from screen")

        # 3. Head Pose (Up/Down)
        if 'HEAD_POSE' in active_flags:
            _maybe_issue_behavioral_warning(student_id, student_name, 'HEAD_POSE', "Head turned away from screen")



        # 4. Accessories
        if acc.get('headphone_detected'):
            _maybe_issue_behavioral_warning(student_id, student_name, 'PROHIBITED_OBJECT', "Headphones detected")
        elif acc.get('earphone_detected'):
            _maybe_issue_behavioral_warning(student_id, student_name, 'PROHIBITED_OBJECT', "Earphones detected")
        # elif acc.get('wire_detected'):
        #     _maybe_issue_behavioral_warning(student_id, student_name, 'PROHIBITED_OBJECT', "Prohibited wire pattern detected")

    
        # Ensure student appears in admin polling
        with active_exam_students_lock:
            active_exam_students.add(student_id)

    
        # Inject server-side voice_detected flag into active_flags
        voice_info = _student_voice_activity.get(student_id, {"active": False, "rms": 0})

        is_voice = voice_info.get("active", False)
        voice_rms = voice_info.get("rms", 0)
        
        # Always add voice metadata to metrics for real-time admin display
        metrics['voice_rms'] = float(voice_rms)
        metrics['voice_threat_level'] = min(100, int((voice_rms / 1000) * 100)) # Scale 0-100%
        data['metrics'] = metrics
    
        if is_voice:
            if 'voice_detected' not in active_flags:
                active_flags.append('voice_detected')
            
        analysis['active_flags'] = active_flags
        data['analysis'] = analysis
            
        # Reset the voice activity trap for the next telemetry window
        _student_voice_activity[student_id] = {"active": False, "rms": 0}
    
        # Store in ring buffer for cross-verification
        from collections import deque
        with _telemetry_history_lock:
            if student_id not in _telemetry_history:
                _telemetry_history[student_id] = deque(maxlen=200)
            _telemetry_history[student_id].append(data)
    
        # Update existing tracking dict
        now_ts = time.time()
        with latest_student_frames_lock:
            prev = latest_student_frames.get(student_id, {})
            latest_student_frames[student_id] = {
                'frame': prev.get('frame'),
                'frame_bytes': prev.get('frame_bytes'),
                'processed_frame': prev.get('processed_frame'),
                'timestamp': prev.get('timestamp', now_ts),
                'frame_timestamp': prev.get('frame_timestamp', now_ts),
                'processed_timestamp': now_ts,
                'detections': prev.get('detections', []),
                'status_snapshot': {
                    'warning_count': warnings_count,
                    'suspicion_score': analysis.get('suspicion_score', 0),
                    'faces_detected': metrics.get('face_count', 0),
                    'phone_detected': 'cell phone' in metrics.get('banned_labels', []),
                    'smartwatch_detected': 'smartwatch' in metrics.get('banned_labels', []),
                    'book_detected': 'book' in metrics.get('banned_labels', []),
                    'paper_detected': 'paper' in metrics.get('banned_labels', []),
                },
                'last_visible_object_labels': metrics.get('banned_labels', []),
                'last_prohibited_object_labels': metrics.get('banned_labels', []),
                'last_person_count': metrics.get('person_count', 0),
                'wasm_telemetry': data,  # Store full telemetry for admin
            }
    
        if analysis.get('suspicion_score', 0) >= 50:
            logger.info(f"🚨 [WASM TELEMETRY v2] High Suspicion for {student_id}: Score {analysis.get('suspicion_score', 0)}")
    
        # Buffer for batched admin delivery
        with _telemetry_buffer_lock:
            _telemetry_buffer[student_id] = data
        # Bust cached admin-active-students snapshot
        _student_cache['data'] = None
    
    @socketio.on('admin_notify_student', namespace='/admin')
    def handle_admin_notify_student(data):
        """Relay an observation notification from admin to a specific student."""
        student_id = str(data.get('student_id', ''))
        if not student_id:
            return
        
        # We broadcast the specific metrics that the admin is seeing to the student
        # so the student knows EXACTLY what was flagged.
        notification_payload = {
            'message': 'You are being observed by the Proctor.',
            'timestamp': time.time(),
            'metrics': data.get('metrics', {}) # admin sends current snapshot
        }
        
        target_room = f"student:{student_id}"
        logger.info(f"🔔 Admin notifying student in room {target_room}")
        socketio.emit('admin_notification', notification_payload, namespace='/student', to=target_room)
    
    @socketio.on('force_terminate_exam', namespace='/admin')
    def handle_force_terminate_exam(data):
        """Admin force terminates a student's exam with a summary report."""
        student_id = str(data.get('student_id', ''))
        if not student_id:
            return
        reason = data.get('reason', 'Administrative decision')
        metrics_summary = data.get('metrics_summary', {})
    
        termination_payload = {
            'terminated': True,
            'reason': reason,
            'metrics_summary': metrics_summary,
            'timestamp': time.time()
        }
    
        target_room = f"student:{student_id}"
        logger.warning(f'? Admin FORCE TERMINATED student in room {target_room}: {reason}')
        socketio.emit('exam_terminated', termination_payload, namespace='/student', to=target_room)
        if warning_system:
            try:
                warning_system.flush_violations_to_db(student_id, _get_active_or_latest_session_id(student_id))
            except Exception as e:
                logger.warning(f"flush_violations_to_db failed for {student_id}: {e}")
        _end_exam_runtime_state(student_id)
    
        if warning_system:
            warning_system.reset_student(student_id)

    @socketio.on('admin_book_detected', namespace='/admin')
    def handle_admin_book_detected(data):
        """Admin-side detection (dashboard) triggers a book warning to the student."""
        student_id = str(data.get('student_id') or '')
        label = data.get('label') or 'book'
        student_name = data.get('student_name') or f"Student {student_id}"
        if not student_id:
            return
        _maybe_issue_object_warning(student_id, student_name, label)

    @socketio.on('request_student_frames', namespace='/admin')
    def handle_request_student_frames(data):
        """Admin requests random frame snapshots from a student for cross-verification."""
        student_id = str(data.get('student_id', ''))
        count = min(int(data.get('count', 6)), 10)
        socketio.emit('capture_frames', {'count': count, 'request_id': data.get('request_id')}, namespace='/student', to=student_id)
    
    @socketio.on('student_frame_response', namespace='/student')
    def handle_student_frame_response(data):
        """Student sends captured frames back to admin for cross-verification."""
        socketio.emit('student_frame_captured', data, namespace='/admin')
    
    @app.route('/api/admin/student-telemetry/<student_id>', methods=['GET'])
    @require_role('ADMIN')
    def get_student_telemetry_history(student_id):
        """Admin API: get recent telemetry history for a student."""
        with _telemetry_history_lock:
            history = list(_telemetry_history.get(student_id, []))
        return jsonify({'student_id': student_id, 'history': history[-50:]})  # Last 50 entries
    
    @socketio.on('webrtc_offer', namespace='/student')
    def handle_webrtc_offer(data):
        socketio.emit('webrtc_offer', data, namespace='/admin')
    
    @socketio.on('webrtc_ice_candidate', namespace='/student')
    def handle_webrtc_ice_student(data):
        socketio.emit('webrtc_ice_candidate', data, namespace='/admin')
    
    # --- ADMIN CONTROL ACTIONS ---
    @socketio.on('admin_clear_warnings', namespace='/admin')
    def handle_admin_clear_warnings(data):
        student_id = str(data.get('student_id'))
        _reset_exam_runtime_state(student_id)
        emit('warnings_cleared', {'student_id': student_id, 'warnings': 0, 'violations': []}, namespace='/admin')
        socketio.emit(
            'warnings_cleared',
            {'student_id': student_id, 'warnings': 0, 'violations': []},
            namespace='/student',
            room=f"student:{student_id}"
        )
    
    @socketio.on('admin_force_terminate', namespace='/admin')
    def handle_admin_force_terminate(data):
        student_id = str(data.get('student_id'))
        reason = data.get('reason', 'Manual termination by Admin')
        if warning_system:
            warning_system.manually_terminate_student(student_id, reason)
        if student_id:
            _end_exam_runtime_state(student_id)
    
    @socketio.on('admin_toggle_enforcement', namespace='/admin')
    def handle_admin_toggle_enforcement(data):
        enabled = bool(data.get('enabled', True))
        if warning_system:
            warning_system.set_auto_terminate(enabled)
            emit('enforcement_toggled', {'enabled': enabled}, namespace='/admin')
    
    # --- WEBRTC SIGNALING (ADMIN -> STUDENT) ---
    @socketio.on('request_webrtc_stream', namespace='/admin')
    def handle_request_webrtc_stream(data):
        student_id = data.get('student_id')
        socketio.emit('request_webrtc_stream', data, namespace='/student', room=f"student:{student_id}")
    
    @socketio.on('webrtc_answer', namespace='/admin')
    def handle_webrtc_answer(data):
        student_id = data.get('student_id')
        socketio.emit('webrtc_answer', data, namespace='/student', room=f"student:{student_id}")
    
    @socketio.on('webrtc_ice_candidate', namespace='/admin')
    def handle_webrtc_ice_admin(data):
        student_id = data.get('student_id')
        socketio.emit('webrtc_ice_candidate', data, namespace='/student', room=f"student:{student_id}")
    
# -------------------------
# System Health API
# -------------------------
@app.route('/api/system-health')
def api_system_health():
    """Returns a JSON snapshot of key system health indicators for admin diagnostics."""
    admin = current_admin()
    if not admin:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    db_ok = False
    try:
        conn = mysql.connection
        with conn.cursor() as cur:
            cur.execute('SELECT 1')
        db_ok = True
    except Exception:
        db_ok = False
    with active_exam_students_lock:
        active_count = len(active_exam_students)
    with latest_student_frames_lock:
        frames_cached = len(latest_student_frames)
    uptime = int(time.time() - _APP_START_TIME)
    return jsonify({
        'ok': True,
        'socketio_enabled': bool(MONITORING_ENABLED and socketio),
        'cv2_available': bool(CV2_AVAILABLE),
        'db_connected': db_ok,
        'active_students': active_count,
        'frames_cached': frames_cached,
        'webrtc_handlers_registered': True,  # handlers registered at startup inside register_socketio_handlers
        'uptime_seconds': uptime,
    })

# -------------------------
# App entrypoint
# -------------------------
if __name__ == '__main__':
    try:
        debug_mode = (os.getenv('FLASK_DEBUG', '0') == '1')
        logger.info("=" * 60)
        logger.info("🚀 Starting Exam Proctoring System")
        logger.info(f"  - OpenCV: {'✓ Available' if CV2_AVAILABLE else '✗ Not available'}")
        logger.info(f"  - Flask-SocketIO: {'✓ Available' if SOCKETIO_AVAILABLE else '✗ Not available'}")
        logger.info(f"  - Live Monitoring: {'✓ ENABLED' if MONITORING_ENABLED else '✗ DISABLED'}")
        logger.info("  - URL: http://127.0.0.1:5001")
        logger.info(f"  - SocketIO async_mode: {socketio.async_mode if socketio else 'none'}")
        logger.info("=" * 60)
        
        with app.app_context():
            ensure_db_schema()
            run_startup_health_checks()

        if MONITORING_ENABLED and socketio:
            start_health_watchdog()

        auto_restart = (os.getenv('AUTO_RESTART', '1') == '1')
        while True:
            try:
                if MONITORING_ENABLED:
                    # use socketio.run when monitoring enabled
                    socketio.run(
                        app,
                        debug=debug_mode,
                        use_reloader=False,  # Windows: avoid socket teardown race (WinError 10038)
                        host='0.0.0.0',
                        port=5001,
                        allow_unsafe_werkzeug=True
                    )
                else:
                    logger.warning("Starting in BASIC MODE (No live monitoring)")
                    logger.info("To enable monitoring, install: pip install flask-socketio")
                    app.run(debug=debug_mode, use_reloader=False, host='0.0.0.0', port=5001, threaded=True)
                break  # clean exit
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user.")
                break
            except Exception as e:
                logger.error(f"Fatal error launching app: {e}", exc_info=True)
                if not auto_restart:
                    break
                logger.info("Auto-restart enabled. Restarting in 3 seconds...")
                time.sleep(3)
    except Exception as e:
        logger.error(f"Fatal error launching app: {e}", exc_info=True)
