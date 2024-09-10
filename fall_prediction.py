import os
import time
from ultralytics import YOLO
import cv2
import numpy as np

class FallDetector:
    def __init__(self, model_path, confidence_threshold=0.6):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.min_time_between_frames = 0.1  

    def process_sample(self, image):
        img = self._preprocess_image(image)
        
        results = self.model(img)
        
        fall_detected, confidence = self._postprocess_results(results)
        
        inference_result = {
            'label': 'Fall' if fall_detected else 'No Fall',
            'confidence': confidence,
            'leaning_angle': None,  
            'keypoint_corr': None   
        }
        
        return [{'inference_result': [inference_result]}]

    def _preprocess_image(self, image):
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        return cv2.resize(image, (640, 640))

    def _postprocess_results(self, results):
        fall_detected = False
        confidence = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0 and box.conf > self.confidence_threshold:  
                    fall_detected = True
                    confidence = box.conf.item()
                    break
            if fall_detected:
                break
        return fall_detected, confidence

def _fall_detect_config():
    _dir = os.path.dirname(os.path.abspath(__file__))
    _yolov8_model = os.path.join(_dir, 'ai_models/yolov8_fall_detection.pt')
    config = {
        'model_path': _yolov8_model,
        'confidence_threshold': 0.6,
    }
    return config

def Fall_prediction(img_1, img_2, img_3=None):
    config = _fall_detect_config()
    fall_detector = FallDetector(**config)
    result = None

    def process_response(response):
        nonlocal result
        for res in response:
            result = res['inference_result']

    process_response(fall_detector.process_sample(image=img_1))
    time.sleep(fall_detector.min_time_between_frames)
    process_response(fall_detector.process_sample(image=img_2))

    if result and len(result) == 1:
        return {
            "category": result[0]['label'],
            "confidence": result[0]['confidence'],
            "angle": result[0]['leaning_angle'],
            "keypoint_corr": result[0]['keypoint_corr']
        }
    elif img_3:
        time.sleep(fall_detector.min_time_between_frames)
        process_response(fall_detector.process_sample(image=img_3))
        if result and len(result) == 1:
            return {
                "category": result[0]['label'],
                "confidence": result[0]['confidence'],
                "angle": result[0]['leaning_angle'],
                "keypoint_corr": result[0]['keypoint_corr']
            }
    
    return None
