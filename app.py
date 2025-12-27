import cv2
import os
import re
import sys
import time
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from ultralytics import YOLO
import easyocr

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.8
FRAME_SKIP_RATE = 1

DEVICE = "0" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {DEVICE}")

# ===============================
# LOAD MODELS
# ===============================
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    sys.exit(1)

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# ===============================
# OCR HELPERS
# ===============================
def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def license_complies_format(text):
    if len(text) != 10:
        return False

    return (
        text[0:2].isalpha() and
        text[2:4].isdigit() and
        text[4:6].isalpha() and
        text[6:10].isdigit()
    )


def read_license_plate(crop_rgb):
    detections = reader.readtext(crop_rgb)

    if not detections:
        return None

    texts = [clean_text(t[1]) for t in detections]
    texts = [t for t in texts if len(t) >= 8]

    if not texts:
        return None

    best = max(texts, key=len)

    if license_complies_format(best):
        return best

    return None


# ===============================
# FILE PICKER
# ===============================
def pick_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select image or video",
        filetypes=[
            ("Media", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return path


# ===============================
# IMAGE DETECTION
# ===============================
def detect_image(path):
    print(f"🖼️ Processing image: {path}")

    image = Image.open(path).convert("RGB")
    image_np = np.array(image)

    results = model.predict(
        image_np,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )

    annotated_rgb = results[0].plot()

    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            h, w = image_np.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            crop = image_np[y1:y2, x1:x2]
            plate = read_license_plate(crop)

            if plate:
                print(f"[IMAGE] Plate detected: {plate}")
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    annotated_bgr,
                    plate,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    cv2.imshow("Image Detection (Press any key)", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===============================
# VIDEO DETECTION
# ===============================
def detect_video(path):
    print(f"🎬 Processing video: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps_src) if fps_src > 0 else 33

    frame_id = 0
    latest_annotated = None
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id == 1 or frame_id % FRAME_SKIP_RATE == 0:
            results = model.predict(
                frame,
                conf=CONF_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

            annotated_rgb = results[0].plot()
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            if results[0].boxes is not None:
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    h, w = frame.shape[:2]
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)

                    crop_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                    plate = read_license_plate(crop_rgb)

                    if plate:
                        print(f"[VIDEO] Frame {frame_id} → {plate}")
                        cv2.putText(
                            annotated_bgr,
                            plate,
                            (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )

            latest_annotated = annotated_bgr

        if latest_annotated is not None:
            display = latest_annotated
        else:
            display = frame

        fps = 1 / max(1e-6, (time.time() - start))
        start = time.time()

        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Video Detection (Press Q to quit)", display)

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video finished")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    path = pick_file()

    if not path:
        print("❌ No file selected")
        sys.exit(0)

    ext = path.lower().split(".")[-1]

    if ext in ["jpg", "jpeg", "png"]:
        detect_image(path)
    elif ext in ["mp4", "avi", "mov"]:
        detect_video(path)
    else:
        print("❌ Unsupported file type")
