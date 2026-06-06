print("Starting detection system...")

import argparse
from datetime import datetime
import os
import threading
import time

import cv2
from ultralytics import YOLO


MODEL_PATH = "yolov8m-oiv7.pt"
DEMO_MODE = False
DEMO_SOURCE = "investor_demo/investor_demo_source.mp4"
SLIDESHOW_MODE = False
SLIDES_DIR = "investor_demo/slides"
CONFIDENCE = 0.30
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
SNAPSHOT_PATH = "latest_detection.jpg"
WINDOW_NAME = "YOLO Detection"

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

COLORS = {
    "Human": (0, 255, 0),
    "Animal": (255, 100, 0),
    "Bird": (0, 200, 255),
}

latest_boxes = []
latest_frame = None
lock = threading.Lock()
running = True


def display_label(raw_label):
    if raw_label in HUMAN_LABELS:
        return "Human"
    if raw_label in ANIMAL_LABELS:
        return raw_label
    return None


def color_for(label):
    if label == "Human":
        return COLORS["Human"]
    if label == "Bird":
        return COLORS["Bird"]
    return COLORS["Animal"]


def yolo_worker(model):
    global latest_boxes, running
    last_detections = set()

    while running:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        results = model(frame, verbose=False, conf=CONFIDENCE)
        boxes = []
        current_detections = set()

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_label = model.names[cls_id]
                label = display_label(raw_label)
                if label is None:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2, label, confidence))
                current_detections.add(label)

        for (x1, y1, x2, y2, label, confidence) in boxes:
            color = color_for(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.0%}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        if boxes:
            cv2.imwrite(SNAPSHOT_PATH, frame)

        with lock:
            latest_boxes = boxes

        for label in sorted(current_detections - last_detections):
            timestamp = datetime.now().strftime("%H:%M:%S")
            best_confidence = max(conf for *_, box_label, conf in boxes if box_label == label)
            print(f"[{timestamp}] DETECTED: {label} ({best_confidence:.0%})", flush=True)

        last_detections = current_detections


def detect_boxes(model, frame):
    results = model(frame, verbose=False, conf=CONFIDENCE)
    boxes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_label = model.names[cls_id]
            label = display_label(raw_label)
            if label is None:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, label, confidence))

    return boxes


def draw_boxes(frame, boxes):
    for (x1, y1, x2, y2, label, confidence) in boxes:
        color = color_for(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.0%}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def alert_new_detections(boxes, last_detections):
    current_detections = {label for *_, label, _ in boxes}
    for label in sorted(current_detections - last_detections):
        timestamp = datetime.now().strftime("%H:%M:%S")
        best_confidence = max(conf for *_, box_label, conf in boxes if box_label == label)
        print(f"[{timestamp}] DETECTED: {label} ({best_confidence:.0%})", flush=True)
    return current_detections


def load_slides(slides_dir):
    slide_paths = sorted(
        path
        for path in os.listdir(slides_dir)
        if path.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    return [os.path.join(slides_dir, path) for path in slide_paths]


def render_slide(model, path):
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Could not read slide: {path}")

    boxes = detect_boxes(model, frame)
    draw_boxes(frame, boxes)
    return frame, boxes


def run_slideshow(slides_dir=SLIDES_DIR):
    print("Loading YOLO Open Images animal model...", flush=True)
    model = YOLO(MODEL_PATH)
    print("YOLO loaded!\n", flush=True)

    slides = load_slides(slides_dir)
    if not slides:
        print(f"ERROR: No slides found in {slides_dir}", flush=True)
        return

    if not os.environ.get("DISPLAY"):
        print("ERROR: No DISPLAY found, so the slideshow GUI cannot open.", flush=True)
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.moveWindow(WINDOW_NAME, 80, 80)

    print(f"Slideshow loaded: {len(slides)} slides", flush=True)
    print("Right arrow: next | Left arrow: previous | Q: quit\n", flush=True)

    cache = {}
    index = 0
    last_detections = set()

    while True:
        if index not in cache:
            cache[index] = render_slide(model, slides[index])

        frame, boxes = cache[index]
        last_detections = alert_new_detections(boxes, last_detections)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKeyEx(0)
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (83, 65363, 2555904):
            index = min(index + 1, len(slides) - 1)
        elif key in (81, 65361, 2424832):
            index = max(index - 1, 0)

    cv2.destroyAllWindows()


def open_capture(source):
    if source:
        cap = cv2.VideoCapture(source)
        source_name = source
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        source_name = f"webcam /dev/video{CAMERA_INDEX}"
    return cap, source_name


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO animal detector for webcam or demo video.")
    parser.add_argument(
        "--source",
        default=DEMO_SOURCE if DEMO_MODE else "",
        help="Optional video file path. Leave empty to use the webcam.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=DEMO_MODE,
        help="Loop the source video. Ignored for webcam.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        default=DEMO_MODE,
        help="Run detection on the exact frame being shown. Best for demo videos.",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use the webcam instead of the investor slideshow.",
    )
    parser.add_argument(
        "--slides",
        default=SLIDES_DIR,
        help="Folder of slideshow images.",
    )
    return parser.parse_args()


def run_detection(source="", loop=False, sync=False):
    global latest_frame, running

    print("Loading YOLO Open Images animal model...", flush=True)
    model = YOLO(MODEL_PATH)
    print("YOLO loaded!\n", flush=True)

    cap, source_name = open_capture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open source: {source_name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    show_window = True
    if not os.environ.get("DISPLAY"):
        show_window = False
        print("WARNING: No DISPLAY found, so the GUI window cannot open in this session.", flush=True)

    if show_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
        cv2.moveWindow(WINDOW_NAME, 80, 80)

    print(f"Source loaded: {source_name} ({fps:.0f}fps)", flush=True)
    print("Monitoring -- press Q to quit", flush=True)
    if source:
        if sync:
            print("Running synchronized demo detection.\n", flush=True)
        else:
            print("Running investor demo reel through the live detector.\n", flush=True)
    else:
        print("Show animal photos on your phone to the webcam.\n", flush=True)

    thread = None
    if not sync:
        thread = threading.Thread(target=yolo_worker, args=(model,), daemon=True)
        thread.start()

    try:
        last_detections = set()
        while True:
            ret, frame = cap.read()
            if not ret:
                if source and loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print(f"Finished source: {source_name}")
                break

            if sync:
                boxes = detect_boxes(model, frame)
                draw_boxes(frame, boxes)
                if boxes:
                    cv2.imwrite(SNAPSHOT_PATH, frame)
                last_detections = alert_new_detections(boxes, last_detections)
            else:
                with lock:
                    latest_frame = frame.copy()
                    boxes = latest_boxes.copy()
                draw_boxes(frame, boxes)

            if show_window:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(1 / max(fps, 1))
    finally:
        running = False
        cap.release()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    try:
        if SLIDESHOW_MODE and not args.camera:
            run_slideshow(args.slides)
        else:
            source = "" if args.camera else args.source
            run_detection(source=source, loop=args.loop, sync=args.sync)
    except KeyboardInterrupt:
        running = False
        print("\nStopped by user.")
    except Exception as e:
        running = False
        print(f"FATAL ERROR: {e}")
