import cv2
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
import time 
import sys 
import tkinter as tk
from tkinter import filedialog

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.8

# FIX: Ensure DEVICE is a string
DEVICE = str(0) if torch.cuda.is_available() else "cpu"

# Optimization: Skip frames for faster detection processing
# The fewer frames we process (higher FRAME_SKIP_RATE), the faster the detection FPS.
FRAME_SKIP_RATE = 1

print(f"🔥 Using device: {DEVICE.upper()}")

# -------------------------------
# LOAD MODEL
# -------------------------------
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}. Check if '{MODEL_PATH}' exists.")
    sys.exit(1)

# -------------------------------
# FILE PICKER (Tkinter GUI)
# -------------------------------
def pick_file():
    """Opens a GUI file dialog to select an image or video file."""
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        title="Select image or video file for detection",
        filetypes=[
            ("Media files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

# -------------------------------
# IMAGE DETECTION (Static)
# -------------------------------
def detect_image(image_path):
    """Performs detection on a single image and displays the result."""
    try:
        print(f"🖼️ Processing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        results = model.predict(
            image_np,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        annotated = results[0].plot()

        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        cv2.imshow(
            "YOLO Detection - Image (Press any key to close)",
            annotated_bgr
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"❌ Error processing image: {e}")

# -------------------------------
# VIDEO DETECTION (True Real-Time Playback)
# -------------------------------
def detect_video(video_path):
    """Plays video in real-time with detection results layered on top."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video file: {video_path}")
        return

    # --- 1. Get Video Properties & Calculate Playback Delay ---
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the required delay in milliseconds to achieve original FPS
    # T_frame = (1 / FPS) * 1000. We cast to int for cv2.waitKey()
    if original_fps > 0:
        delay_ms = int(275 / original_fps)
    else:
        # Default to 30 FPS if information is missing
        delay_ms = 33
    
    print(f"🎬 Playing video: {video_path}")
    print(f"   Original FPS: {original_fps:.2f} (Delay: {delay_ms}ms). Detection Skip Rate: 1/{FRAME_SKIP_RATE}")

    # --- Loop variables ---
    frame_counter = 0
    latest_annotated_frame = None 
    start_time = time.time()
    fps = 0.0 # Initialized
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # ----------------------------------------------------
        # OPTIMIZATION: FRAME SKIPPING FOR DETECTION SPEED
        # ----------------------------------------------------
        # Only run the heavy detection model every Nth frame
        if frame_counter == 1 or frame_counter % FRAME_SKIP_RATE == 0:
            # RUN DETECTION
            results = model.predict(
                frame,
                conf=CONF_THRESHOLD,
                device=DEVICE,
                verbose=False
            )
            # Store the latest annotated frame (RGB)
            latest_annotated_frame = results[0].plot()
        
        # ----------------------------------------------------
        # LAYER DETECTION RESULTS ON FRAME
        # ----------------------------------------------------
        # If we have a recent annotated frame, use it; otherwise, use the raw frame
        if latest_annotated_frame is not None:
            # CRITICAL: Convert RGB (YOLO output) to BGR (OpenCV format)
            frame_to_display = cv2.cvtColor(latest_annotated_frame, cv2.COLOR_RGB2BGR)
        else:
            frame_to_display = frame 

        # ----------------------------------------------------
        # FPS Calculation and Info Display
        # ----------------------------------------------------
        # Calculate detection/processing FPS
        if frame_counter % 30 == 0:
            # We calculate FPS based on how fast the loop runs to show *processing* speed
            fps = 30 / (time.time() - start_time) 
            start_time = time.time()

        fps_text = f"Proc FPS: {fps:.1f} | Device: {DEVICE.upper()} | Skip: 1/{FRAME_SKIP_RATE}"
        cv2.putText(frame_to_display, fps_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ----------------------------------------------------
        # DISPLAY AND DELAY FOR REAL-TIME PLAYBACK
        # ----------------------------------------------------
        cv2.imshow(
            f"YOLO Detection - Real-Time Playback (Press Q to quit)",
            frame_to_display
        )

        # This waits for exactly 'delay_ms' milliseconds to enforce original video speed.
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video playback finished.")

# -------------------------------
# MAIN EXECUTION (using Tkinter)
# -------------------------------
if __name__ == "__main__":
    
    file_path = pick_file()

    if not file_path:
        print("❌ No file selected. Exiting.")
        sys.exit(0)

    if not os.path.exists(file_path):
        print(f"\n❌ ERROR: Selected file not found at path: {file_path}")
        sys.exit(1)

    ext = file_path.lower().split(".")[-1]

    if ext in ["jpg", "jpeg", "png"]:
        detect_image(file_path)
    elif ext in ["mp4", "avi", "mov"]:
        detect_video(file_path)
    else:
        print(f"❌ Unsupported file type: {ext}")