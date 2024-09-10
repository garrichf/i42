import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

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
            if box.cls == 0:  # here the class 0 is 'fall'
                fall_detected = True
                confidence = box.conf.item()
                break
        if fall_detected:
            break
    return fall_detected, confidence

def Fall_prediction(img1, img2, img3):
    model = YOLO(r'C:\Users\User\Desktop\fall-detection\yolov8n-pose.pt')

    frame1 = preprocess_frame(img1)
    frame2 = preprocess_frame(img2)
    frame3 = preprocess_frame(img3)
    results1 = model(frame1)
    results2 = model(frame2)
    results3 = model(frame3)

    fall1, conf1 = postprocess_results(results1)
    fall2, conf2 = postprocess_results(results2)
    fall3, conf3 = postprocess_results(results3)

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

if __name__ == "__main__":
    img1 = Image.open("Images/fall_img_1.png")
    img2 = Image.open("Images/fall_img_2.png")
    img3 = Image.open("Images/fall_img_3.png")

    response = Fall_prediction(img1, img2, img3)

    if response:
        print("There is", response['category'])
        print("Confidence:", response['confidence'])
        print("Angle: Not available with YOLOv8")
        print("Keypoint_corr: Not available with YOLOv8")
    else:
        print("There is no fall detection...")
