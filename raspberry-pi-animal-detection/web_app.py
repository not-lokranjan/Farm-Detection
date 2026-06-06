import os
import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import wraps
from urllib import parse, request as urlrequest

import cv2
import psutil
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, session
from ultralytics import YOLO


APP_HOST = "0.0.0.0"
APP_PORT = 8080
MODEL_PATH = "yolov8m-oiv7.pt"
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
CONFIDENCE = 0.30
DETECT_EVERY_SECONDS = 0.75
DETECTION_CLEAR_SECONDS = 12.0
STREAM_FPS = 60
JPEG_QUALITY = 72
DB_PATH = os.environ.get("DETECTFIELD_DB", "detectfield.db")
DEFAULT_ADMIN_USER = os.environ.get("DETECTFIELD_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASSWORD = os.environ.get("DETECTFIELD_ADMIN_PASSWORD", "detector")
SESSION_SECRET_PATH = os.environ.get("DETECTFIELD_SECRET_FILE", ".detectfield_secret")
PRESENCE_EVENT_SECONDS = 15.0
CLIP_DIR = os.environ.get("DETECTFIELD_CLIP_DIR", "clips")
SETTINGS_DEFAULTS = {
    "notifications_enabled": "1",
    "recording_enabled": "1",
    "retention_days": "30",
    "recording_fps": "8",
    "viewer_feed_access": "1",
    "camera_source": "",
    "clip_storage": "local",
    "detection_hold_seconds": "12",
}
LAST_CLIP_CLEANUP = 0.0

HUMAN_LABELS = {"Person", "Man", "Woman", "Boy", "Girl", "Human body"}

ANIMAL_LABELS = {
    "Animal",
    "Bat (Animal)",
    "Bear",
    "Bird",
    "Brown bear",
    "Cat",
    "Caterpillar",
    "Cattle",
    "Cheetah",
    "Chicken",
    "Deer",
    "Dog",
    "Duck",
    "Elephant",
    "Fish",
    "Fox",
    "Goat",
    "Goldfish",
    "Horse",
    "Jaguar (Animal)",
    "Jellyfish",
    "Leopard",
    "Lion",
    "Monkey",
    "Pig",
    "Polar bear",
    "Rabbit",
    "Reptile",
    "Sea lion",
    "Seahorse",
    "Sheep",
    "Shellfish",
    "Snake",
    "Squirrel",
    "Starfish",
    "Tiger",
}


def display_label(raw_label):
    if raw_label in HUMAN_LABELS:
        return "Human"
    if raw_label in ANIMAL_LABELS:
        return raw_label
    return None


def color_for(label):
    if label == "Human":
        return (0, 255, 0)
    if label == "Bird":
        return (0, 200, 255)
    return (255, 100, 0)


def run_command(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_secret_key():
    configured = os.environ.get("DETECTFIELD_SECRET_KEY")
    if configured:
        return configured
    if os.path.exists(SESSION_SECRET_PATH):
        with open(SESSION_SECRET_PATH, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    secret = secrets.token_hex(32)
    with open(SESSION_SECRET_PATH, "w", encoding="utf-8") as handle:
        handle.write(secret)
    os.chmod(SESSION_SECRET_PATH, 0o600)
    return secret


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 180000)
    return f"{salt}${base64.b64encode(digest).decode()}"


def verify_password(password, stored):
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, salt), stored)


def init_db():
    with db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'operator', 'viewer')),
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS detection_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                labels TEXT NOT NULL DEFAULT '[]',
                peak_confidence INTEGER,
                event_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                event_type TEXT NOT NULL,
                label TEXT,
                confidence INTEGER,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES detection_sessions(id)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(detection_sessions)")}
        if "clip_path" not in columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN clip_path TEXT")
        if "clip_created_at" not in columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN clip_created_at TEXT")
        if "clip_deleted_at" not in columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN clip_deleted_at TEXT")
        if "clip_url" not in columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN clip_url TEXT")
        if "storage_provider" not in columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN storage_provider TEXT")
        if "upload_status" not in columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN upload_status TEXT")
        if "clip_remote_path" not in columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN clip_remote_path TEXT")
        existing = conn.execute("SELECT id FROM users LIMIT 1").fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, active, created_at) VALUES (?, ?, 'admin', 1, ?)",
                (DEFAULT_ADMIN_USER, hash_password(DEFAULT_ADMIN_PASSWORD), utc_now()),
            )
        for key, value in SETTINGS_DEFAULTS.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )


def load_settings():
    settings = dict(SETTINGS_DEFAULTS)
    with db_connect() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    settings.update({row["key"]: row["value"] for row in rows})
    return settings


def bool_setting(settings, key):
    return str(settings.get(key, "0")).lower() in {"1", "true", "yes", "on"}


def int_setting(settings, key, minimum, maximum):
    try:
        value = int(settings.get(key, SETTINGS_DEFAULTS[key]))
    except (TypeError, ValueError):
        value = int(SETTINGS_DEFAULTS[key])
    return max(minimum, min(maximum, value))


def public_settings(settings=None):
    settings = settings or load_settings()
    return {
        "notificationsEnabled": bool_setting(settings, "notifications_enabled"),
        "recordingEnabled": bool_setting(settings, "recording_enabled"),
        "retentionDays": int_setting(settings, "retention_days", 7, 365),
        "recordingFps": int_setting(settings, "recording_fps", 4, 24),
        "viewerFeedAccess": bool_setting(settings, "viewer_feed_access"),
        "cameraSource": settings.get("camera_source", ""),
        "clipStorage": settings.get("clip_storage", "local"),
        "detectionHoldSeconds": int_setting(settings, "detection_hold_seconds", 5, 30),
    }


def firebase_bucket():
    bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET", "").strip()
    if not bucket_name:
        return None
    service_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if service_json:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".firebase-service-account.json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(service_json)
            os.chmod(path, 0o600)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    try:
        from google.cloud import storage
    except Exception:
        return None
    try:
        return storage.Client().bucket(bucket_name)
    except Exception:
        return None


def upload_clip_to_firebase(local_path, session_id):
    bucket = firebase_bucket()
    if bucket is None:
        return None
    blob_name = f"detectfield/clips/session-{session_id}-{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type="video/mp4")
    return {
        "remote_path": blob_name,
        "url": blob.generate_signed_url(expiration=datetime.now(timezone.utc) + timedelta(days=7)),
    }


def firebase_signed_url(remote_path):
    bucket = firebase_bucket()
    if bucket is None or not remote_path:
        return None
    try:
        return bucket.blob(remote_path).generate_signed_url(
            expiration=datetime.now(timezone.utc) + timedelta(days=7)
        )
    except Exception:
        return None


def delete_firebase_clip(remote_path):
    bucket = firebase_bucket()
    if bucket is None or not remote_path:
        return
    try:
        bucket.blob(remote_path).delete()
    except Exception:
        pass


def cleanup_old_clips(settings=None, force=False):
    global LAST_CLIP_CLEANUP
    if not force and time.time() - LAST_CLIP_CLEANUP < 3600:
        return
    LAST_CLIP_CLEANUP = time.time()
    settings = settings or load_settings()
    retention_days = int_setting(settings, "retention_days", 7, 365)
    cutoff = time.time() - retention_days * 86400
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, clip_path, clip_remote_path, clip_created_at
            FROM detection_sessions
            WHERE (clip_path IS NOT NULL OR clip_remote_path IS NOT NULL) AND clip_deleted_at IS NULL
            """
        ).fetchall()
        for row in rows:
            path = row["clip_path"]
            remote_path = row["clip_remote_path"]
            if path and os.path.exists(path):
                should_delete = os.path.getmtime(path) < cutoff
            else:
                created = row["clip_created_at"] or ""
                try:
                    created_ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    created_ts = time.time()
                should_delete = bool(remote_path) and created_ts < cutoff
            if should_delete:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                    delete_firebase_clip(remote_path)
                except OSError:
                    continue
                conn.execute(
                    "UPDATE detection_sessions SET clip_deleted_at = ?, clip_path = NULL, clip_url = NULL, clip_remote_path = NULL WHERE id = ?",
                    (utc_now(), row["id"]),
                )


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    with db_connect() as conn:
        row = conn.execute(
            "SELECT id, username, role, active FROM users WHERE id = ? AND active = 1",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def require_role(*roles):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = current_user()
            if not user:
                return jsonify({"error": "login_required"}), 401
            if roles and user["role"] not in roles:
                return jsonify({"error": "forbidden"}), 403
            return func(*args, **kwargs)

        return wrapper

    return decorator


def send_whatsapp_alert(message):
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    sender = os.environ.get("TWILIO_WHATSAPP_FROM", "").strip()
    recipient = os.environ.get("ALERT_WHATSAPP_TO", "").strip()
    if not all([sid, token, sender, recipient]):
        return

    def worker():
        data = parse.urlencode({"From": sender, "To": recipient, "Body": message}).encode()
        req = urlrequest.Request(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            data=data,
            method="POST",
        )
        auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
        req.add_header("Authorization", f"Basic {auth}")
        try:
            urlrequest.urlopen(req, timeout=8).read()
        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()


class SurveillanceEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.capture_thread = None
        self.detection_thread = None
        self.stop_event = threading.Event()
        self.model = None
        self.capture = None
        self.camera_active = False
        self.camera_source = ""
        self.surveillance_on = False
        self.manual_live_feed_on = False
        self.alert_live_feed_on = False
        self.was_detecting = False
        self.last_event_id = 0
        self.last_frame = None
        self.latest_raw_frame = None
        self.last_jpeg = None
        self.last_boxes = []
        self.last_detection_labels = set()
        self.last_detection_run = 0.0
        self.last_seen_time = 0.0
        self.active_session_id = None
        self.session_labels = set()
        self.session_peak_confidence = None
        self.last_presence_event_time = 0.0
        self.settings = load_settings()
        self.recording_writer = None
        self.recording_path = None
        self.recording_session_id = None
        self.last_record_write = 0.0
        self.notifications = deque(maxlen=50)
        self.error = ""

    def refresh_settings(self):
        with self.lock:
            self.settings = load_settings()

    def detection_clear_seconds(self):
        return int_setting(self.settings, "detection_hold_seconds", 5, 30)

    def ensure_model(self):
        if self.model is None:
            self.model = YOLO(MODEL_PATH)

    def start_surveillance(self):
        with self.lock:
            self.surveillance_on = True
            self.was_detecting = False
            self.alert_live_feed_on = False
            self.error = ""
            self._ensure_thread()

    def stop_surveillance(self):
        with self.lock:
            self.surveillance_on = False
            self.last_detection_labels = set()
            self.was_detecting = False
            self.alert_live_feed_on = False
            self._close_active_session()
            if not self.manual_live_feed_on:
                self.error = ""
                self._stop_thread_locked()

    def start_live_feed(self):
        with self.lock:
            self.manual_live_feed_on = True
            self.error = ""
            self._ensure_thread()

    def stop_live_feed(self):
        with self.lock:
            self.manual_live_feed_on = False
            if not self.surveillance_on:
                self.error = ""
                self._stop_thread_locked()

    def _ensure_thread(self):
        if self.capture_thread and self.capture_thread.is_alive():
            if self.detection_thread and self.detection_thread.is_alive():
                return
        self.stop_event.clear()
        if not self.capture_thread or not self.capture_thread.is_alive():
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
        if not self.detection_thread or not self.detection_thread.is_alive():
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()

    def _stop_thread_locked(self):
        self.stop_event.set()
        self._stop_recording()
        self.camera_active = False
        self.last_jpeg = None
        self.last_frame = None
        self.latest_raw_frame = None
        self.last_boxes = []
        self.alert_live_feed_on = False

    def _candidate_sources(self):
        configured = os.environ.get("AEGISFIELD_CAMERA", "").strip()
        if configured:
            if configured.isdigit():
                return [int(configured)]
            return [configured]

        configured = self.settings.get("camera_source", "").strip()
        if configured:
            if configured.isdigit():
                return [int(configured)]
            return [configured]

        return list(range(6))

    def _open_camera(self):
        errors = []
        for source in self._candidate_sources():
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                cap.release()
                errors.append(str(source))
                continue

            ok, frame = cap.read()
            if ok and frame is not None:
                self.camera_source = str(source)
                return cap

            cap.release()
            errors.append(str(source))

        self.camera_source = ""
        checked = ", ".join(errors[:10])
        raise RuntimeError(f"No usable camera found. Checked: {checked}")

    def _capture_loop(self):
        try:
            self.capture = self._open_camera()
            with self.lock:
                self.camera_active = True
                self.error = ""

            while not self.stop_event.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    with self.lock:
                        self.error = "Camera frame read failed"
                    time.sleep(0.2)
                    continue

                with self.lock:
                    self.latest_raw_frame = frame.copy()
                    boxes = list(self.last_boxes)

                display = frame.copy()
                self.draw_boxes(display, boxes)
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    display,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )
                if ok:
                    with self.lock:
                        self.last_frame = display
                        self.last_jpeg = jpeg.tobytes()
                self._record_clip_frame(display)

                time.sleep(0.001)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.camera_active = False
        finally:
            if self.capture is not None:
                self.capture.release()
            self._stop_recording()
            with self.lock:
                self.capture = None
                self.camera_active = False

    def _detection_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                should_detect = self.surveillance_on and self.latest_raw_frame is not None
                if should_detect:
                    frame = self.latest_raw_frame.copy()
                else:
                    frame = None

            now = time.time()
            if frame is None or now - self.last_detection_run < DETECT_EVERY_SECONDS:
                time.sleep(0.05)
                continue

            boxes = self.detect(frame)
            self.last_detection_run = time.time()
            self._record_detection_state(boxes)

            with self.lock:
                absence_seconds = time.time() - self.last_seen_time
                if boxes or absence_seconds >= self.detection_clear_seconds():
                    self.last_boxes = boxes

    def detect(self, frame):
        self.ensure_model()
        results = self.model(frame, verbose=False, conf=CONFIDENCE)
        boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_label = self.model.names[cls_id]
                label = display_label(raw_label)
                if label is None:
                    continue
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2, label, confidence))
        return boxes

    def draw_boxes(self, frame, boxes):
        for x1, y1, x2, y2, label, confidence in boxes:
            color = color_for(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.0%}",
                (x1, max(22, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

    def _record_detection_state(self, boxes):
        current = {label for *_, label, _ in boxes}
        new_labels = sorted(current - self.last_detection_labels)
        timestamp = datetime.now().strftime("%H:%M:%S")
        now = time.time()

        if current:
            with self.lock:
                self.alert_live_feed_on = True
                self.last_seen_time = now
            self._ensure_active_session(boxes)

        if new_labels:
            for label in new_labels:
                best = max(conf for *_, box_label, conf in boxes if box_label == label)
                self._push_event(
                    event_type="detected",
                    timestamp=timestamp,
                    label=label,
                    confidence=round(best * 100),
                    message=f"DETECTED: {label}",
                )

        if current and not new_labels and now - self.last_presence_event_time >= PRESENCE_EVENT_SECONDS:
            best_label, best_conf = self._best_detection(boxes)
            self.last_presence_event_time = now
            self._push_event(
                event_type="presence",
                timestamp=timestamp,
                label=best_label,
                confidence=round(best_conf * 100),
                message=f"Presence: {', '.join(sorted(current))}",
            )

        enough_time_clear = now - self.last_seen_time >= self.detection_clear_seconds()
        if self.was_detecting and not current and enough_time_clear:
            self.alert_live_feed_on = False
            self._push_event(
                event_type="left",
                timestamp=timestamp,
                label="",
                confidence=None,
                message="Left feed",
            )
            self._close_active_session()

        if current or enough_time_clear:
            self.was_detecting = bool(current)
            self.last_detection_labels = current

    def _best_detection(self, boxes):
        best = max(boxes, key=lambda box: box[5])
        return best[4], best[5]

    def _ensure_active_session(self, boxes):
        labels = {label for *_, label, _ in boxes}
        best_confidence = round(max(conf for *_, conf in boxes) * 100)
        now = utc_now()
        if self.active_session_id is None:
            with db_connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO detection_sessions (started_at, labels, peak_confidence) VALUES (?, ?, ?)",
                    (now, json.dumps(sorted(labels)), best_confidence),
                )
                self.active_session_id = cursor.lastrowid
            self.session_labels = labels
            self.session_peak_confidence = best_confidence
            self.last_presence_event_time = time.time()
            return

        self.session_labels.update(labels)
        self.session_peak_confidence = max(self.session_peak_confidence or 0, best_confidence)
        with db_connect() as conn:
            conn.execute(
                "UPDATE detection_sessions SET labels = ?, peak_confidence = ? WHERE id = ?",
                (json.dumps(sorted(self.session_labels)), self.session_peak_confidence, self.active_session_id),
            )

    def _close_active_session(self):
        if self.active_session_id is None:
            return
        self._stop_recording()
        with db_connect() as conn:
            conn.execute(
                "UPDATE detection_sessions SET ended_at = ? WHERE id = ?",
                (utc_now(), self.active_session_id),
            )
        self.active_session_id = None
        self.session_labels = set()
        self.session_peak_confidence = None
        self.last_presence_event_time = 0.0

    def _record_clip_frame(self, frame):
        with self.lock:
            session_id = self.active_session_id
            settings = dict(self.settings)
        if not session_id or not bool_setting(settings, "recording_enabled"):
            self._stop_recording()
            return

        fps = int_setting(settings, "recording_fps", 4, 24)
        now = time.time()
        if now - self.last_record_write < 1 / fps:
            return
        self.last_record_write = now

        if self.recording_writer is None or self.recording_session_id != session_id:
            self._start_recording(session_id, frame, fps)
        if self.recording_writer is not None:
            self.recording_writer.write(frame)

    def _start_recording(self, session_id, frame, fps):
        self._stop_recording()
        os.makedirs(CLIP_DIR, exist_ok=True)
        filename = f"detectfield-session-{session_id}-{int(time.time())}.avi"
        path = os.path.join(CLIP_DIR, filename)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            with db_connect() as conn:
                conn.execute(
                    "UPDATE detection_sessions SET upload_status = ? WHERE id = ?",
                    ("recording_failed", session_id),
                )
            return
        self.recording_writer = writer
        self.recording_path = path
        self.recording_session_id = session_id
        with db_connect() as conn:
            conn.execute(
                "UPDATE detection_sessions SET clip_path = ?, clip_created_at = ?, storage_provider = ?, upload_status = ? WHERE id = ?",
                (path, utc_now(), self.settings.get("clip_storage", "firebase"), "recording", session_id),
            )

    def _stop_recording(self):
        path = self.recording_path
        session_id = self.recording_session_id
        if self.recording_writer is not None:
            self.recording_writer.release()
        self.recording_writer = None
        self.recording_path = None
        self.recording_session_id = None
        self.last_record_write = 0.0
        if path and session_id:
            threading.Thread(target=self._finalize_clip, args=(path, session_id), daemon=True).start()

    def _finalize_clip(self, path, session_id):
        storage_provider = self.settings.get("clip_storage", "firebase")
        mp4_path = os.path.splitext(path)[0] + ".mp4"
        status = "processing"
        clip_url = None
        remote_path = None
        with db_connect() as conn:
            conn.execute(
                "UPDATE detection_sessions SET upload_status = ? WHERE id = ?",
                (status, session_id),
            )

        converted = self._convert_clip_for_browser(path, mp4_path)
        final_path = mp4_path if converted else path

        if storage_provider == "firebase":
            uploaded = upload_clip_to_firebase(final_path, session_id)
            if not uploaded:
                status = "upload_failed_local_ready"
                storage_provider = "local"
            else:
                clip_url = uploaded["url"]
                remote_path = uploaded["remote_path"]
                status = "uploaded"
                try:
                    if os.path.exists(final_path):
                        os.remove(final_path)
                except OSError:
                    pass
        else:
            status = "stored"

        try:
            if converted and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

        with db_connect() as conn:
            conn.execute(
                """
                UPDATE detection_sessions
                SET clip_path = ?, clip_url = ?, clip_remote_path = ?, storage_provider = ?, upload_status = ?
                WHERE id = ?
                """,
                (None if clip_url else final_path, clip_url, remote_path, storage_provider, status, session_id),
            )

    def _convert_clip_for_browser(self, source_path, output_path):
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            source_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        try:
            subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception:
            return False

    def _push_event(self, event_type, timestamp, label, confidence, message):
        self.last_event_id += 1
        event = {
            "id": self.last_event_id,
            "type": event_type,
            "time": timestamp,
            "label": label,
            "confidence": confidence,
            "message": message,
        }
        self.notifications.appendleft(event)
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO detection_events (session_id, event_type, label, confidence, message, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (self.active_session_id, event_type, label, confidence, message, utc_now()),
            )
            if self.active_session_id is not None:
                conn.execute(
                    "UPDATE detection_sessions SET event_count = event_count + 1 WHERE id = ?",
                    (self.active_session_id,),
                )
        if event_type == "detected" and bool_setting(self.settings, "notifications_enabled"):
            send_whatsapp_alert(f"DetectField alert: {message} at {timestamp}.")

    def stream(self):
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with self.lock:
                live = self.manual_live_feed_on or self.alert_live_feed_on
                jpeg = self.last_jpeg
            if not live:
                break
            if jpeg:
                yield boundary + jpeg + b"\r\n"
            time.sleep(1 / STREAM_FPS)

    def snapshot(self):
        with self.lock:
            return {
                "piOnline": True,
                "surveillanceOn": self.surveillance_on,
                "liveFeedOn": self.manual_live_feed_on or self.alert_live_feed_on,
                "manualLiveFeedOn": self.manual_live_feed_on,
                "alertLiveFeedOn": self.alert_live_feed_on,
                "cameraActive": self.camera_active,
                "cameraSource": self.camera_source,
                "detections": list(self.notifications),
                "lastBoxes": [
                    {"label": b[4], "confidence": round(b[5] * 100)} for b in self.last_boxes
                ],
                "error": self.error,
            }


app = Flask(__name__)
app.secret_key = load_secret_key()
init_db()
engine = SurveillanceEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    with db_connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row or not row["active"] or not verify_password(password, row["password_hash"]):
        return jsonify({"error": "invalid_login"}), 401
    session.clear()
    session["user_id"] = row["id"]
    return jsonify({"user": {"id": row["id"], "username": row["username"], "role": row["role"]}})


@app.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/auth/me")
def auth_me():
    user = current_user()
    if not user:
        return jsonify({"user": None}), 401
    return jsonify({"user": user})


@app.get("/api/users")
@require_role("admin")
def list_users():
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT id, username, role, active, created_at FROM users ORDER BY username"
        ).fetchall()
    return jsonify({"users": [dict(row) for row in rows]})


@app.post("/api/users")
@require_role("admin")
def create_user():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    role = str(payload.get("role", "viewer")).strip()
    if not username or not password or role not in {"admin", "operator", "viewer"}:
        return jsonify({"error": "invalid_user"}), 400
    try:
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, active, created_at) VALUES (?, ?, ?, 1, ?)",
                (username, hash_password(password), role, utc_now()),
            )
    except sqlite3.IntegrityError:
        return jsonify({"error": "username_exists"}), 409
    return jsonify({"ok": True})


@app.patch("/api/users/<int:user_id>")
@require_role("admin")
def update_user(user_id):
    payload = request.get_json(silent=True) or {}
    fields = []
    values = []
    if "role" in payload:
        role = str(payload["role"]).strip()
        if role not in {"admin", "operator", "viewer"}:
            return jsonify({"error": "invalid_role"}), 400
        fields.append("role = ?")
        values.append(role)
    if "active" in payload:
        fields.append("active = ?")
        values.append(1 if payload["active"] else 0)
    if "password" in payload and payload["password"]:
        fields.append("password_hash = ?")
        values.append(hash_password(str(payload["password"])))
    if not fields:
        return jsonify({"ok": True})
    values.append(user_id)
    with db_connect() as conn:
        conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", values)
    return jsonify({"ok": True})


@app.post("/api/account/password")
@require_role("admin", "operator", "viewer")
def change_password():
    payload = request.get_json(silent=True) or {}
    current_password = str(payload.get("currentPassword", ""))
    new_password = str(payload.get("newPassword", ""))
    if len(new_password) < 8:
        return jsonify({"error": "weak_password"}), 400
    user = current_user()
    with db_connect() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()
        if not row or not verify_password(current_password, row["password_hash"]):
            return jsonify({"error": "invalid_password"}), 401
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(new_password), user["id"]),
        )
    return jsonify({"ok": True})


@app.get("/api/settings")
@require_role("admin", "operator", "viewer")
def get_settings():
    return jsonify({"settings": public_settings()})


@app.patch("/api/settings")
@require_role("admin")
def update_settings():
    payload = request.get_json(silent=True) or {}
    updates = {}
    if "notificationsEnabled" in payload:
        updates["notifications_enabled"] = "1" if payload["notificationsEnabled"] else "0"
    if "recordingEnabled" in payload:
        updates["recording_enabled"] = "1" if payload["recordingEnabled"] else "0"
    if "viewerFeedAccess" in payload:
        updates["viewer_feed_access"] = "1" if payload["viewerFeedAccess"] else "0"
    if "retentionDays" in payload:
        try:
            retention = int(payload["retentionDays"])
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_retention"}), 400
        if retention not in {7, 14, 30, 60, 90}:
            return jsonify({"error": "invalid_retention"}), 400
        updates["retention_days"] = str(retention)
    if "recordingFps" in payload:
        try:
            updates["recording_fps"] = str(max(4, min(24, int(payload["recordingFps"]))))
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_recording_fps"}), 400
    if "cameraSource" in payload:
        updates["camera_source"] = str(payload["cameraSource"]).strip()
    if "detectionHoldSeconds" in payload:
        try:
            hold_seconds = int(payload["detectionHoldSeconds"])
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_detection_hold"}), 400
        if hold_seconds not in {5, 10, 12, 15, 20, 30}:
            return jsonify({"error": "invalid_detection_hold"}), 400
        updates["detection_hold_seconds"] = str(hold_seconds)
    if "clipStorage" in payload:
        clip_storage = str(payload["clipStorage"]).strip()
        if clip_storage not in {"local", "firebase"}:
            return jsonify({"error": "invalid_clip_storage"}), 400
        updates["clip_storage"] = clip_storage
    with db_connect() as conn:
        for key, value in updates.items():
            conn.execute("UPDATE settings SET value = ? WHERE key = ?", (value, key))
    engine.refresh_settings()
    cleanup_old_clips(engine.settings, force=True)
    return jsonify({"settings": public_settings(engine.settings)})


def user_can_access_feed():
    user = current_user()
    if not user:
        return False
    if user["role"] in {"admin", "operator"}:
        return True
    settings = load_settings()
    return bool_setting(settings, "viewer_feed_access")


@app.post("/api/surveillance/start")
@require_role("admin", "operator")
def start_surveillance():
    engine.start_surveillance()
    return jsonify(engine.snapshot())


@app.post("/api/surveillance/stop")
@require_role("admin", "operator")
def stop_surveillance():
    engine.stop_surveillance()
    return jsonify(engine.snapshot())


@app.post("/api/feed/start")
@require_role("admin", "operator", "viewer")
def start_feed():
    if not user_can_access_feed():
        return jsonify({"error": "feed_forbidden"}), 403
    engine.start_live_feed()
    return jsonify(engine.snapshot())


@app.post("/api/feed/stop")
@require_role("admin", "operator", "viewer")
def stop_feed():
    if not user_can_access_feed():
        return jsonify({"error": "feed_forbidden"}), 403
    engine.stop_live_feed()
    return jsonify(engine.snapshot())


@app.get("/api/status")
@require_role("admin", "operator", "viewer")
def status():
    cleanup_old_clips()
    payload = engine.snapshot()
    payload["power"] = power_status()
    payload["user"] = current_user()
    payload["settings"] = public_settings()
    return jsonify(payload)


@app.get("/api/events")
@require_role("admin", "operator", "viewer")
def events():
    with db_connect() as conn:
        event_rows = conn.execute(
            """
            SELECT id, session_id, event_type, label, confidence, message, created_at
            FROM detection_events
            ORDER BY id DESC
            LIMIT 80
            """
        ).fetchall()
        session_rows = conn.execute(
            """
            SELECT id, started_at, ended_at, labels, peak_confidence, event_count
            , clip_path, clip_url, clip_created_at, clip_deleted_at, upload_status, storage_provider
            FROM detection_sessions
            ORDER BY id DESC
            LIMIT 30
            """
        ).fetchall()
    return jsonify(
        {
            "events": [dict(row) for row in event_rows],
            "sessions": [dict(row) for row in session_rows],
        }
    )


@app.get("/api/clips")
@require_role("admin", "operator", "viewer")
def clips():
    cleanup_old_clips()
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, started_at, ended_at, labels, peak_confidence, event_count, clip_path, clip_url,
                   clip_remote_path, clip_created_at, upload_status, storage_provider
            FROM detection_sessions
            WHERE (clip_path IS NOT NULL OR clip_url IS NOT NULL OR upload_status IN ('recording', 'processing'))
              AND clip_deleted_at IS NULL
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()
    payload = []
    for row in rows:
        item = dict(row)
        path = item.pop("clip_path")
        remote_path = item.pop("clip_remote_path")
        item["sizeBytes"] = os.path.getsize(path) if path and os.path.exists(path) else 0
        item["url"] = item.get("clip_url") or f"/api/clips/{item['id']}/file"
        if item.get("storage_provider") == "firebase" and remote_path:
            item["url"] = firebase_signed_url(remote_path) or item["url"]
        payload.append(item)
    return jsonify({"clips": payload})


@app.get("/api/clips/<int:session_id>/file")
@require_role("admin", "operator", "viewer")
def clip_file(session_id):
    with db_connect() as conn:
        row = conn.execute(
            "SELECT clip_path, clip_url FROM detection_sessions WHERE id = ? AND clip_deleted_at IS NULL",
            (session_id,),
        ).fetchone()
    if row and row["clip_url"]:
        return redirect(row["clip_url"])
    if not row or not row["clip_path"] or not os.path.exists(row["clip_path"]):
        return jsonify({"error": "not_found"}), 404
    return send_file(row["clip_path"], mimetype="video/mp4", as_attachment=False)


@app.delete("/api/clips/<int:session_id>")
@require_role("admin", "operator")
def delete_clip(session_id):
    if session_id == engine.active_session_id:
        return jsonify({"error": "clip_active"}), 409
    with db_connect() as conn:
        row = conn.execute(
            "SELECT clip_path, clip_remote_path FROM detection_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return jsonify({"error": "not_found"}), 404
        path = row["clip_path"]
        if path and os.path.exists(path):
            os.remove(path)
        delete_firebase_clip(row["clip_remote_path"])
        conn.execute(
            "UPDATE detection_sessions SET clip_path = NULL, clip_url = NULL, clip_remote_path = NULL, clip_deleted_at = ? WHERE id = ?",
            (utc_now(), session_id),
        )
    return jsonify({"ok": True})


@app.get("/stream")
@require_role("admin", "operator", "viewer")
def stream():
    if not user_can_access_feed():
        return jsonify({"error": "feed_forbidden"}), 403
    return Response(engine.stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


def power_status():
    temp = run_command(["vcgencmd", "measure_temp"]).replace("temp=", "")
    volts = run_command(["vcgencmd", "measure_volts", "core"]).replace("volt=", "")
    throttled = run_command(["vcgencmd", "get_throttled"]).replace("throttled=", "")
    uptime_seconds = int(time.time() - psutil.boot_time())
    return {
        "cpuPercent": psutil.cpu_percent(interval=None),
        "memoryPercent": psutil.virtual_memory().percent,
        "temperature": temp or "unavailable",
        "coreVoltage": volts or "unavailable",
        "throttled": throttled or "unavailable",
        "uptimeSeconds": uptime_seconds,
    }


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app.run(host=APP_HOST, port=APP_PORT, threaded=True)
