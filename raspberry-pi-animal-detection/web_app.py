import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import psutil
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO


APP_HOST = "0.0.0.0"
APP_PORT = 8080
MODEL_PATH = "yolov8m-oiv7.pt"
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
CONFIDENCE = 0.30
DETECT_EVERY_SECONDS = 0.75
DETECTION_CLEAR_SECONDS = 2.0
STREAM_FPS = 60
JPEG_QUALITY = 72

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
        self.notifications = deque(maxlen=50)
        self.error = ""

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

                time.sleep(0.001)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.camera_active = False
        finally:
            if self.capture is not None:
                self.capture.release()
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

        if current:
            with self.lock:
                self.alert_live_feed_on = True
                self.last_seen_time = time.time()

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

        enough_time_clear = time.time() - self.last_seen_time >= DETECTION_CLEAR_SECONDS
        if self.was_detecting and not current and enough_time_clear:
            self.alert_live_feed_on = False
            self._push_event(
                event_type="left",
                timestamp=timestamp,
                label="",
                confidence=None,
                message="Left feed",
            )

        if current or enough_time_clear:
            self.was_detecting = bool(current)
            self.last_detection_labels = current

    def _push_event(self, event_type, timestamp, label, confidence, message):
        self.last_event_id += 1
        self.notifications.appendleft(
            {
                "id": self.last_event_id,
                "type": event_type,
                "time": timestamp,
                "label": label,
                "confidence": confidence,
                "message": message,
            }
        )

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


engine = SurveillanceEngine()
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/surveillance/start")
def start_surveillance():
    engine.start_surveillance()
    return jsonify(engine.snapshot())


@app.post("/api/surveillance/stop")
def stop_surveillance():
    engine.stop_surveillance()
    return jsonify(engine.snapshot())


@app.post("/api/feed/start")
def start_feed():
    engine.start_live_feed()
    return jsonify(engine.snapshot())


@app.post("/api/feed/stop")
def stop_feed():
    engine.stop_live_feed()
    return jsonify(engine.snapshot())


@app.get("/api/status")
def status():
    payload = engine.snapshot()
    payload["power"] = power_status()
    return jsonify(payload)


@app.get("/stream")
def stream():
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
