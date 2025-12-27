# app.py
import cv2
import sys
import torch
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import util
from util import draw_box   

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.4
OCR_EVERY_N_FRAMES = 10   
FRAME_SKIP = 1         

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {DEVICE}")

# ===============================
# LOAD MODEL
# ===============================
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("❌ Failed to load model:", e)
    sys.exit(1)

reader = util.reader 

# ===============================
# DRAW FUNCTION
# ===============================
# def draw_box(frame, box, conf, plate_text):
#     x1, y1, x2, y2 = map(int, box)
#     color = (0, 255, 0) # Green

#     # 1. Bounding Box
#     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#     # 2. Label Background
#     label = f"{plate_text}"
#     (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#     cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w, y1), color, -1)
    
#     # 3. Put Text
#     cv2.putText(frame, label, (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# ===============================
# IMAGE MODE
# ===============================
def run_image(path):
    img = cv2.imread(path)
    if img is None: return

    results = model(img, conf=CONF_THRESHOLD, device=DEVICE, verbose=False, imgsz=320)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_bi = cv2.bilateralFilter(gray, 11, 17, 17)

        plate_text, _ = util.read_license_plate(gray_bi, reader)
        
        # --- MODIFIED: ONLY DRAW IF PLATE IS READ ---
        if plate_text:
            print(f" Found: {plate_text}")
            draw_box(img, (x1, y1, x2, y2), conf, plate_text)

    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===============================
# VIDEO MODE
# ===============================
def run_video(path):
    cap = cv2.VideoCapture(path)
    frame_count = 0
    active_detections = {} 

    cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Detection", 960, 720)
    # cv2.setWindowProperty(
    #     "Video Detection",
    #     cv2.WND_PROP_FULLSCREEN,
    #     cv2.WINDOW_FULLSCREEN
    # )

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            cv2.imshow("Video Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        results = model(frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)

        # Create a set of IDs seen in THIS frame to clear old active_detections
        current_frame_ids = []

        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            if frame_count % OCR_EVERY_N_FRAMES == 0:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    gray_bi = cv2.bilateralFilter(gray, 11, 17, 17)
                    plate_text, _ = util.read_license_plate(gray_bi, reader)
                    
                    if plate_text:
                        active_detections[i] = plate_text
                    else:
                        # If OCR fails now but worked before, clear it
                        if i in active_detections:
                            del active_detections[i]
            
            # --- MODIFIED: ONLY DRAW IF PLATE EXISTS IN ACTIVE DETECTIONS ---
            if i in active_detections:
                draw_box(frame, (x1, y1, x2, y2), conf, active_detections[i])

        util.draw_quit_hint(frame, "Quit(Q)")

        cv2.imshow("Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def pick_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Select Media", filetypes=[("Media", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")])
    root.destroy()
    return path

if __name__ == "__main__":
    path = pick_file()
    if path:
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            run_image(path)
        else:
            run_video(path)