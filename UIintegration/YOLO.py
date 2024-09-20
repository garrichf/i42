import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

class YOLOModel:
    def __init__(self):
        self.pose_model = YOLO('yolov8n-pose.pt')
        self.fall_detection_model = load_model("falldetect_16092024_1307_alldatacombined_reg_reduceplateau.keras")

    def process_frame(self, frame, confidence_threshold):
        results = self.pose_model(frame)
        keypoints = results[0].keypoints.xy[0] if len(results[0].keypoints.xy) > 0 else []

        # Check if keypoints are detected
        if len(keypoints) == 0:
            print("Warning: No keypoints detected.")
            # Create a placeholder for 115 keypoints (x,y) pairs
            keypoints = np.zeros((115, 2))  # Assuming that each keypoint has x and y coordinates

        # Ensure the correct number of keypoints for prediction
        if len(keypoints) < 115:
            print(f"Warning: Expected 115 keypoints, but got {len(keypoints)}. Padding with zeros.")
            padded_keypoints = np.zeros((115, 2))
            padded_keypoints[:len(keypoints)] = keypoints
            keypoints = padded_keypoints

        fall_detected = self.predict_fall(keypoints, confidence_threshold)
        frame = self.annotate_frame(frame, keypoints, fall_detected)
        return frame, fall_detected

    def predict_fall(self, keypoints, confidence_threshold):
        features = np.array(keypoints).flatten().reshape(1, -1)
        prediction = self.fall_detection_model.predict(features)
        return prediction[0][1] > confidence_threshold

    def annotate_frame(self, frame, keypoints, fall_detected):
        color = (0, 255, 0) if not fall_detected else (0, 0, 255)
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 5, color, -1)
        return frame