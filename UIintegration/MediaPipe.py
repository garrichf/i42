import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

class MediaPipeModel:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.fall_detection_model = load_model("falldetect_16092024_1307_alldatacombined_reg_reduceplateau.keras")

    def process_frame(self, frame, confidence_threshold):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        keypoints = []

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])])

        # Handle no detections case
        if len(keypoints) == 0:
            print("Warning: No pose landmarks detected.")
            keypoints = np.zeros((115, 2))  # Placeholder for 115 keypoints

        # Ensure we have exactly 115 keypoints
        while len(keypoints) < 115:
            print(f"Warning: Expected 115 keypoints but got {len(keypoints)}. Padding with zeros.")
            keypoints.append([0, 0]) 

        fall_detected = self.predict_fall(keypoints[:115], confidence_threshold)
        frame = self.annotate_frame(frame, keypoints[:115], fall_detected)
        
        return frame, fall_detected

    def predict_fall(self, keypoints, confidence_threshold):
        features = np.array(keypoints).flatten().reshape(1, -1)
        prediction = self.fall_detection_model.predict(features)
        return prediction[0][1] > confidence_threshold

    def annotate_frame(self, frame, keypoints, fall_detected):
        color = (0, 255, 0) if not fall_detected else (0, 0, 255)

        for kp in keypoints:
            cv2.circle(frame, (kp[0], kp[1]), 5, color, -1)

        return frame