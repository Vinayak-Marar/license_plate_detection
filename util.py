import string
import easyocr
import re
import numpy as np
import cv2
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=DEVICE)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'Z': '7'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '7': 'Z'}

def license_complies_format(text):
    if len(text) != 10:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
       (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()) and \
       (text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[8] in dict_char_to_int.keys()) and \
       (text[9] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[9] in dict_char_to_int.keys()) :
        return True
    else:
        return False


def format_license(text):

    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int,6: dict_char_to_int, 7: dict_char_to_int,8: dict_char_to_int, 9: dict_char_to_int,}
    for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop, reader):

    detections = reader.readtext(license_plate_crop)

    def clean_text(t):
        return re.sub(r'[^A-Z0-9]', '', t.upper())
            
    texts = [t[1] for t in detections]
    texts = [clean_text(t) for t in texts]

    for text in texts:
    
        if len(text) < 10:
            continue
    
        if license_complies_format(text):
            return format_license(text), None

    return None, None

def read_image(image, model, reader=reader):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image_rgb, conf=0.6, device=DEVICE, verbose=False)

    # 1. Start the loop for detected boxes
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2] 

        # 2. Pre-process the crop IMMEDIATELY inside the loop
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray_bi = cv2.bilateralFilter(gray, 11, 17, 17)

        # 3. Call your specialized OCR function
        # This function already handles the 10-char check and formatting
        text, _ = read_license_plate(gray_bi, reader=reader)

        # 4. If a valid 10-char plate is found, return it immediately
        if text:
            return text ,(x1, y1, x2, y2)

    # 5. If the loop finishes and no valid plate was found
    return None, None

def draw_quit_hint(frame, text="Quit (Q)"):
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = w - tw - 15   # right padding
    y = th + 15       # top padding

    # Background box (optional but looks professional)
    cv2.rectangle(
        frame,
        (x - 10, y - th - 10),
        (x + tw + 10, y + 5),
        (0, 0, 0),
        -1
    )

    # Text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )

def draw_box(frame, box, conf, plate_text):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0) # Green

    # 1. Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 2. Label Background
    label = f"{plate_text}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w, y1), color, -1)
    
    # 3. Put Text
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)