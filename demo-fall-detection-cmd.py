import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import json
import time
import yaml
import cv2
from ultralytics import YOLO

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(JsonEncoder, self).default(obj)

def preprocess_frame(frame):
    frame_np = np.array(frame)
    frame_resized = cv2.resize(frame_np, (640, 640))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    return frame_rgb

def postprocess_results(results):
    fall_detected = False
    confidence = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  #here class 0 is 'fall'
                fall_detected = True
                confidence = box.conf.item()
                break
        if fall_detected:
            break
    return fall_detected, confidence

def Fall_prediction(img1, img2, img3=None):
    model = YOLO(r'C:\Users\User\Desktop\fall-detection\yolov8n-pose.pt')

    frame1 = preprocess_frame(img1)
    frame2 = preprocess_frame(img2)
    results1 = model(frame1)
    results2 = model(frame2)
    
    fall1, conf1 = postprocess_results(results1)
    fall2, conf2 = postprocess_results(results2)
    
    if img3 is not None:
        frame3 = preprocess_frame(img3)
        results3 = model(frame3)
        fall3, conf3 = postprocess_results(results3)
    else:
        fall3, conf3 = False, 0

    fall_detected = fall1 or fall2 or fall3
    max_confidence = max(conf1, conf2, conf3)

    if fall_detected:
        return {
            'category': 'Fall',
            'confidence': max_confidence,
            'angle': None, 
            'keypoint_corr': None  
        }
    else:
        return None

parser = argparse.ArgumentParser()
parser.add_argument("--image_1", type=Path, help='Path to the First Image', required=True)
parser.add_argument("--image_2", type=Path, help='Path to the Second Image', required=True)
parser.add_argument("--image_3", type=Path, help='Path to the Third Image')

p = parser.parse_args()
img1 = Image.open(p.image_1)
img2 = Image.open(p.image_2)
img3 = Image.open(p.image_3) if p.image_3 else None

response = Fall_prediction(img1, img2, img3)

if response:
    print("There is", response['category'])
    print("Confidence:", response['confidence'])
    print("Angle: Not available with YOLOv8")
    print("Keypoint_corr: Not available with YOLOv8")

    time_str = time.strftime("%Y%m%d-%H%M%S")
    json_str = json.dumps(response, cls=JsonEncoder)

    with open(f"tmp/{time_str}.yaml", "w") as file:
        yaml.dump(json.loads(json_str), file)
else:
    print("There is no fall detection...")
