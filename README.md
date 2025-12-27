# 🚘 License Plate Recognition (YOLO + EasyOCR)

## 📌 Overview

This project implements a **computer vision–based license plate recognition system** that detects vehicle number plates and extracts the plate number from images and videos.

The system is designed specifically for **Indian-style long rectangular number plates**, where characters are arranged horizontally.
It was developed as a **learning-focused project** to understand object detection, OCR, and model fine-tuning workflows.

---

## ✨ Features

* 🚗 **License plate detection** using YOLO
* 🔤 **Text extraction** from detected plates using EasyOCR
* 🖼️ Works on **images and videos**
* 🧠 Multiple YOLO models trained and evaluated
* 🖥️ Simple **GUI using Tkinter** for uploading media
* ⚡ Supports **CPU and GPU execution**

---

## 🧠 Model Details

### Object Detection

* **Models trained**:

  * YOLOv8n
  * YOLOv8s
  * YOLOv10n
  * YOLOv10s
* **Final model used**: **YOLOv8s**

  * Chosen due to **slightly better accuracy** compared to others

### Training

* **Training type**: Fine-tuning
* **Dataset source**: Roboflow
* **Dataset size**: ~10,000 images
* **Task**: License plate detection

---

## 🔍 OCR Pipeline

1. Detect license plate using YOLO
2. Crop the detected bounding box
3. Convert cropped image to **grayscale**
4. Apply **bilinear filtering**
5. Extract text using **EasyOCR**

Other techniques such as Gaussian filtering and thresholding were tested, but grayscale + bilinear filtering produced relatively better OCR results (though still imperfect).

---

## 🧾 Output

* Displays:

  * Bounding box around the license plate
  * Extracted plate number **above the bounding box**
* ❌ Output is **not saved to disk**
* ❌ No logging or database storage

---

## 🖥️ User Interface

* Built using **Tkinter**
* User can:

  * Upload an **image** or **video**
  * See detection and OCR results visually
* No command-line interaction required

---

## ⚠️ Limitations

This project has **known and explicit limitations**:

* Works **only on Indian-style long rectangular plates**
* Plates must be **flat and horizontally aligned**
* OCR accuracy is **not reliable in all cases**
* Performs poorly on:

  * Angled plates
  * Low-resolution images
  * Motion blur
  * Night or low-light conditions
* **Very low FPS on videos**, making it unsuitable for real-time use
* Detection may succeed even when OCR output is inaccurate

These limitations are acknowledged and documented as part of the learning process.

---

## 🧑‍💻 Tech Stack

* **Language**: Python
* **Object Detection**: YOLO (Ultralytics)
* **OCR**: EasyOCR
* **GUI**: Tkinter
* **Image Processing**: OpenCV

---

## 📁 Repository Structure

```
├── best.pt
├── app.py
├── eda.ipynb
├── model_training.ipynb
├── ocr.ipynb
```

> Note: Repository structure is minimal and reflects an experimental / learning workflow.

---

## ▶️ How to Run

1. Install dependencies (manually or via `pip`)
2. Place the trained model (`best.pt`) in the project directory
3. Run:

```bash
python app.py
```

4. Upload an image or video via the GUI

---

## 🎯 Project Purpose

This project was built as a **learning exercise** to:

* Understand YOLO fine-tuning
* Explore OCR challenges in real-world images
* Experiment with preprocessing techniques
* Gain hands-on experience in computer vision pipelines

It is **not intended for production use**.

---

## 👤 Author

**GitHub**: Vinayak Marar
