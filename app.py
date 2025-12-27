import cv2
import sys
import torch
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import util
from util import draw_box   


# CONFIG
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.4
OCR_EVERY_N_FRAMES = 10         

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {DEVICE}")

# LOAD MODEL

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("Failed to load model:", e)
    sys.exit(1)

reader = util.reader 

# IMAGE MODE

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
        
        if plate_text:
            print(f" Found: {plate_text}")
            draw_box(img, (x1, y1, x2, y2), conf, plate_text)

    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# VIDEO MODE
def run_video(path):
    cap = cv2.VideoCapture(path)
    frame_count = 0
    active_detections = {} 

    cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Detection", 960, 720)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: break

        frame_count += 1

        results = model(frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)

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
                        if i in active_detections:
                            del active_detections[i]

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