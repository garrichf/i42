import os
import cv2
import numpy as np
import torch
from tensorflow.keras.models import load_model
import mediapipe as mp
from ultralytics import YOLO  # Import YOLO from the Ultralytics package

# Load fall detection model
fall_detection_model = load_model("falldetect_16092024_1307_alldatacombined_reg_reduceplateau.keras")

# Load YOLOv8 model directly using Ultralytics
model_path = r"C:\Users\User\Desktop\UIintegration\yolov8n-pose.pt"
yolo_model = YOLO(model_path)  # Load your custom model

# Initialize MediaPipe Pose if selected
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Function to compute angles between keypoints
def compute_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

# Function to add angles to keypoints
def add_angles(keypoints):
    angles = {}
    if len(keypoints) >= 17:
        angles['neck_angle'] = compute_angle(keypoints[5], keypoints[1], keypoints[2])
        angles['spine_angle'] = compute_angle(keypoints[1], keypoints[8], keypoints[11])
        angles['left_knee_angle'] = compute_angle(keypoints[11], keypoints[13], keypoints[15])
        angles['right_knee_angle'] = compute_angle(keypoints[12], keypoints[14], keypoints[16])
        angles['left_elbow_angle'] = compute_angle(keypoints[5], keypoints[7], keypoints[9])
        angles['right_elbow_angle'] = compute_angle(keypoints[6], keypoints[8], keypoints[10])
    return angles

# Function to predict fall based on features
def predict_fall(features):
    expected_length = 115
    if len(features) < expected_length:
        features += [0] * (expected_length - len(features))  # Pad with zeros
    elif len(features) > expected_length:
        features = features[:expected_length]  # Truncate to expected length
    
    features = np.array(features).reshape(1, 1, -1)  # Reshape for model input
    prediction = fall_detection_model.predict(features)
    return prediction[0][1] > 0.3

# Function to draw annotations on the frame
def draw_annotations(frame, keypoints, fall_detected):
    color_box = (0, 255, 0) if not fall_detected else (0, 0, 255)
    
    # Draw bounding box (example coordinates)
    cv2.rectangle(frame, (10, 10), (300, 100), color_box, 2)

    # Draw keypoint dots
    for keypoint in keypoints:
        x, y = int(keypoint[0]), int(keypoint[1])
        color_dot = (0, 255, 0) if not fall_detected else (0, 0, 255)
        cv2.circle(frame, (x, y), 5, color_dot, -1)

    return frame

# Function to process each frame of the video using YOLOv8 for pose estimation
def process_frame(frame):
    results = yolo_model(frame[..., ::-1])  # Convert BGR to RGB for YOLOv8
    
    keypoints = []
    
    # Extracting pose landmarks from YOLOv8 results
    if isinstance(results, list) and len(results) > 0:
        detections = results[0]  # Get the first result (the detections)
        
        # Check if there are any detections
        if detections is not None and len(detections) > 0:
            for result in detections:
                # Each result should be a tensor or a list with bounding box info
                if isinstance(result, (list, tuple)) and len(result) >= 6:
                    x1, y1, x2, y2, conf, cls = result[:6]  # Extract first six elements
                    
                    # Calculate center coordinates
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    
                    keypoints.append([x_center, y_center])
                else:
                    print("Unexpected detection format:", result)

    angles = add_angles(keypoints)
    
    return keypoints, angles

# Main function to process dataset and create output video
def process_dataset(dataset_dir):
    output_video_path = "output_video.avi"
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = None
    
    for video_file in os.listdir(dataset_dir):
        video_path = os.path.join(dataset_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame and get keypoints and angles
            keypoints, angles = process_frame(frame)

            # Prepare features for fall detection
            features = [coord for kp in keypoints for coord in kp] + list(angles.values())
            fall_detected = predict_fall(features)

            # Draw annotations on the frame
            annotated_frame = draw_annotations(frame.copy(), keypoints, fall_detected)

            # Initialize video writer if not already done
            if video_writer is None:
                video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0,
                                                (frame.shape[1], frame.shape[0]))

            video_writer.write(annotated_frame)

        cap.release()

    if video_writer is not None:
        video_writer.release()

# Run the processing function with the specified directory
dataset_dir = r"C:\Users\User\Desktop\UIintegration\Videos"
process_dataset(dataset_dir)

print("Processing complete. Output saved to:", "output_video.avi")