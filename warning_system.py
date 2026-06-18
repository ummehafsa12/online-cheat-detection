"""
Warning System Module
Handles student warnings and exam termination
"""

import os
import threading
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

class WarningSystem:
    """Track warnings per student and emit events when thresholds reached."""
    
    def __init__(self, socketio_instance, admin_monitor=None, max_warnings=3, violation_writer=None):
        self.socketio = socketio_instance
        self.admin_monitor = admin_monitor
        self.max_warnings = max_warnings
        self.violation_writer = violation_writer  # callable(student_id, session_id, vtype, details)
        auto_env = os.getenv('AUTO_TERMINATE_DEFAULT', '1')
        # Default ON so the Nth warning is shown then termination occurs after ~2 seconds.
        self.auto_terminate = str(auto_env).strip() not in ('0', 'false', 'False')
        self.lock = threading.Lock()
        self.warnings = {}  # student_id -> count
        self.violations = {}  # student_id -> list of violations
        self._persisted = set()  # (sid, vtype, time_str)
        self.student_names = {}  # student_id -> name
        self.last_warning_at = {}  # student_id -> epoch seconds
        self.last_warning_type_at = {}  # student_id -> {type: epoch seconds}
        self.termination_timer = {}  # student_id -> Timer handle

        # Global minimum gap between any two warnings (seconds)
        self.global_gap_seconds = 3.0
        # Per-type gaps — critical items fire faster; minor distractions fire slower
        self.type_gap_seconds = {
            # Immediate threats — fire quickly but not spam
            'PROHIBITED_OBJECT':    3.0,
            'MULTIPLE_FACES':       3.0,
            'TAB_SWITCH':           3.0,
            'PROHIBITED_SHORTCUT':  3.0,
            'VOICE_DETECTED':       6.0,   # audio: only after 8s continuous
            'IDENTITY_MISMATCH':    4.0,
            # Behavioural distractions — need longer persistence
            'NO_FACE':              3.0,   # detect quickly
            'DISTRACTION':          3.0,   # gaze/head away — needs repetition
            'HEAD_MOVEMENT':        3.0,
            'HEAD_DOWN':            3.0,
            'HEAD_UP':              3.0,

            'STUDENT_LEFT_SEAT':    4.0,
            'EYES_CLOSED':          6.0,
            'GAZE_LEFT':            3.0,
            'GAZE_RIGHT':           3.0,
            'GAZE_UP':              3.0,
            'GAZE_DOWN':            3.0,
            'GAZE_UP_LEFT':         3.0,
            'GAZE_UP_RIGHT':        3.0,
            'GAZE_DOWN_LEFT':       3.0,
            'GAZE_DOWN_RIGHT':      3.0,
        }
        self.type_gap_seconds['TERMINATED_BY_ADMIN'] = 0.0

    def set_auto_terminate(self, enabled: bool):
        with self.lock:
            self.auto_terminate = enabled
        print(f"[WarningSystem] Auto-Terminate configured: {self.auto_terminate}")

    def initialize_student(self, student_id, student_name):
        """Initialize tracking for a new student"""
        sid = str(student_id)
        with self.lock:
            self.warnings[sid] = 0
            self.violations[sid] = []
            self.student_names[sid] = student_name
            self.last_warning_at[sid] = 0.0
            self.last_warning_type_at[sid] = {}
        
        print(f"✓ Warning system initialized for student {student_id} - {student_name}")
        
        # Emit initial state to admin
        if self.socketio:
            self.socketio.emit('students_list', {'students': [
                {
                    'student_id': student_id, 
                    'student_name': student_name, 
                    'warnings': 0, 
                    'violations': []
                }
            ]}, namespace='/admin')

    def add_warning(self, student_id, vtype, details=None, emit_to_student=True):
        """Add warning and check if exam should be terminated"""
        sid = str(student_id)
        with self.lock:
            self.warnings.setdefault(sid, 0)
            self.violations.setdefault(sid, [])
            self.last_warning_at.setdefault(sid, 0.0)
            self.last_warning_type_at.setdefault(sid, {})

            now = time.time()
            vtype_norm = str(vtype or 'UNKNOWN').upper()
            g_gap = float(self.global_gap_seconds)
            t_gap = float(self.type_gap_seconds.get(vtype_norm, self.global_gap_seconds))
            last_global = float(self.last_warning_at.get(sid, 0.0))
            last_type = float(self.last_warning_type_at[sid].get(vtype_norm, 0.0))

            # Hard gap for all warnings: don't increment if warning is too soon.
            if (now - last_global) < g_gap or (now - last_type) < t_gap:
                return False, False
            
            # STOP adding any more warnings if already at or above limit
            if self.warnings[sid] >= self.max_warnings:
                return False, False

            # Increment warning count. If auto-terminate is true, cap at max_warnings + 1 so it doesn't inflate endlessly.
            if self.auto_terminate:
                if self.warnings[sid] <= self.max_warnings:
                    self.warnings[sid] += 1
            else:
                self.warnings[sid] += 1
                
            self.last_warning_at[sid] = now
            self.last_warning_type_at[sid][vtype_norm] = now
            
            # Create violation record
            violation = {
                'type': vtype, 
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                'details': details
            }
            self.violations[sid].append(violation)

            self._persist_violation(sid, violation)
            
            count = self.warnings[sid]
            student_name = self.student_names.get(sid, 'Unknown')

        print(f"⚠ Warning #{count} for student {student_id}: {vtype} - {details}")

        # Emit to admin UI
        if self.socketio:
            self.socketio.emit('student_violation', {
                'student_id': student_id,
                'student_name': student_name,
                'total_warnings': min(count, self.max_warnings),
                'violation': violation,
                'type': vtype,
                'details': details,
                'source': 'server'
            }, namespace='/admin')
            
            # Emit to student UI (avoid client re-emitting)
            if emit_to_student:
                self.socketio.emit('student_violation', {
                    'student_id': student_id,
                    'student_name': student_name,
                    'total_warnings': min(count, self.max_warnings),
                    'violation': violation,
                    'type': vtype,
                    'details': details,
                    'source': 'server'
                }, namespace='/student')

        # If threshold exceeded, emit termination
        if count >= self.max_warnings:
            if not self.auto_terminate:
                # Let admin decide; notify them that student reached threshold
                if self.socketio and count == self.max_warnings:
                    self.socketio.emit('student_needs_review', {
                        'student_id': student_id,
                        'student_name': student_name,
                        'warnings': count
                    }, namespace='/admin')
                return True, False
            # Auto-terminate, but after a short grace so the UI can catch up
            def do_term():
                reason = f"Reached {self.max_warnings} warnings for violations: {vtype}"
                print(f"🚨 TERMINATING EXAM for student {student_id}: {reason}")
                self.flush_violations_to_db(sid)
                if self.socketio:
                    self.socketio.emit('student_exam_terminated', {
                        'student_id': student_id,
                        'student_name': student_name,
                        'reason': reason
                    }, namespace='/admin')
                    self.socketio.emit('exam_terminated', {
                        'student_id': student_id,
                        'reason': reason,
                        'auto_terminated': True
                    }, namespace='/student')
            try:
                if sid in self.termination_timer and self.termination_timer[sid]:
                    self.termination_timer[sid].cancel()
            except Exception:
                pass
            import threading
            t = threading.Timer(15.0, do_term)
            self.termination_timer[sid] = t
            t.start()
            return True, True

        return True, False

    def get_warnings(self, student_id):
        """Get current warning count for student"""
        sid = str(student_id)
        with self.lock:
            return min(self.warnings.get(sid, 0), self.max_warnings)

    def get_violations(self, student_id):
        """Get all violations for student"""
        sid = str(student_id)
        with self.lock:
            return list(self.violations.get(sid, []))

    def acknowledge_warning(self, student_id):
        """Set quiet period for the student after acknowledgment"""
        sid = str(student_id)
        with self.lock:
            self.last_warning_at[sid] = time.time()
            if sid in self.last_warning_type_at:
                for vtype in self.last_warning_type_at[sid]:
                    self.last_warning_type_at[sid][vtype] = time.time()
        print(f"✓ Acknowledged warning for student {student_id}, global gap reset.")

    def reset_student(self, student_id):
        """Reset warnings for student"""
        sid = str(student_id)
        with self.lock:
            self.warnings[sid] = 0
            self.violations[sid] = []
            self.last_warning_at[sid] = 0.0
            self.last_warning_type_at[sid] = {}
            self._persisted = {t for t in self._persisted if t[0] != sid}
        print(f"🧹 Cleared warnings for student {student_id}")

    def manually_terminate_student(self, student_id, reason="Manual termination by Admin"):
        """Instantly terminate a student exam regardless of warning count."""
        sid = str(student_id)
        with self.lock:
            student_name = self.student_names.get(sid, 'Unknown')
        self.flush_violations_to_db(sid)
        
        print(f"🚨 MANUAL TERMINATE for student {student_id}: {reason}")
        if self.socketio:
            self.socketio.emit('student_exam_terminated', {
                'student_id': sid,
                'student_name': student_name,
                'reason': reason
            }, namespace='/admin')
            
            self.socketio.emit('exam_terminated', {
                'student_id': sid,
                'reason': reason,
                'auto_terminated': False
            }, namespace='/student')
        return True

    # --------------------- Persistence helpers ---------------------
    def _persist_violation(self, student_id, violation, session_id=None):
        """Write violation via injected writer if available; dedupe via _persisted set."""
        key = (student_id, str(violation.get('type')).upper(), violation.get('time'))
        if key in self._persisted:
            return
        if not self.violation_writer:
            return
        try:
            self.violation_writer(student_id, session_id, str(violation.get('type')).upper(), violation.get('details') or '')
            self._persisted.add(key)
        except Exception as e:
            print(f"[WarningSystem] violation persist failed for {student_id}: {e}")

    def flush_violations_to_db(self, student_id, session_id=None):
        """Persist all stored violations for a student that haven't been written yet."""
        sid = str(student_id)
        with self.lock:
            violations = list(self.violations.get(sid, []))
        for vio in violations:
            self._persist_violation(sid, vio, session_id=session_id)


class TabSwitchDetector:
    """Detects tab switching and adds warnings"""
    
    def __init__(self, warning_system, threshold=1):
        self.warning_system = warning_system
        self.threshold = max(1, threshold)  # fire on first event by default
        self.lock = threading.Lock()
        self.switch_counts = {}  # student_id -> count

    def initialize_student(self, student_id):
        """Initialize tab switch tracking for student"""
        with self.lock:
            self.switch_counts[student_id] = 0
        print(f"✓ Tab switch detector initialized for student {student_id}")

    def detect_tab_switch(self, student_id):
        """Detect tab switch and add warning if threshold reached"""
        sid = str(student_id)
        with self.lock:
            self.switch_counts.setdefault(sid, 0)
            self.switch_counts[sid] += 1
            count = self.switch_counts[sid]

        print(f"🔄 Tab switch detected for student {student_id}. Count: {count}/{self.threshold}")

        if count >= self.threshold:
            warning_added, terminated = self.warning_system.add_warning(
                student_id, 
                'tab_switch', 
                f'{count} tab switches detected'
            )
            return {'accepted': warning_added, 'terminated': terminated, 'count': count}
        
        return {'terminated': False, 'count': count}


print("=" * 60)
print("✓ Warning system module loaded successfully")
print("  - WarningSystem: ✓ Available")
print("  - TabSwitchDetector: ✓ Available") 
print("=" * 60)
