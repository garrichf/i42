import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# I'm loading the YOLO model for pose detection and the fall detection model here.
pose_model = YOLO('yolov8n-pose.pt')  # Adjust the path if needed
fall_detection_model = load_model("falldetect_test.h5")  # Adjust the path if needed

# This function calculates the angle between three points.
def compute_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

# This function calculates the velocity and acceleration of keypoints.
def calculate_velocity_acceleration(df, keypoint_columns):
    df = df.sort_values(by='Frame')  # Sorting by frame number
    for col in keypoint_columns:
        df[f'{col}_velocity'] = df.groupby('Video')[col].diff()  # Calculate velocity
        df[f'{col}_acceleration'] = df.groupby('Video')[f'{col}_velocity'].diff()  # Calculate acceleration
    return df

# Here’s how I process each frame to extract pose keypoints and bounding boxes.
def process_frame(frame, frame_index, video_name):
    results = pose_model(frame)
    keypoints = results[0].keypoints.xy[0] if len(results[0].keypoints.xy) > 0 else None
    boxes = results[0].boxes.xyxy if len(results[0].boxes.xyxy) > 0 else None
    
    # Convert keypoints to a DataFrame for further processing.
    if keypoints is not None:
        keypoints = keypoints.flatten()
        df_keypoints = pd.DataFrame([keypoints], columns=[f'keypoint_{i}' for i in range(len(keypoints))])
        df_keypoints['Frame'] = frame_index
        df_keypoints['Video'] = video_name
        return df_keypoints, boxes
    return None, boxes

# I need to reshape the keypoints to match the expected size of the model.
def reshape_keypoints(keypoints, expected_size=34):
    keypoints = keypoints[:expected_size]  # Trim if necessary
    if len(keypoints) < expected_size:
        keypoints = np.pad(keypoints, (0, expected_size - len(keypoints)), 'constant')  # Pad if needed
    return keypoints

# This function predicts if a fall has occurred based on the keypoints.
def predict_fall(features):
    # Reshape the keypoints to fit the model’s expected input size.
    features = reshape_keypoints(features)
    
    # Convert features to a NumPy array and ensure the data type is float32.
    features = np.array(features, dtype=np.float32)
    
    # Reshape to match the model’s input shape (1, 1, 34) for sequences.
    features = features.reshape(1, 1, -1)
    
    # Predict if a fall happened using the fall detection model.
    prediction = fall_detection_model.predict(features)
    
    # Return True if a fall is detected, otherwise False.
    return prediction[0][0] > 0.5

# This function draws keypoints and bounding boxes on the frame.
def draw_annotations(frame, keypoints, boxes, fall_detected):
    color = (0, 255, 0) if not fall_detected else (0, 0, 255)
    if keypoints is not None:
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 5, color, -1)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame

# This is the main function where I capture video and process each frame.
def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    all_keypoints = pd.DataFrame()
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        df_keypoints, boxes = process_frame(frame, frame_index, video_name="video_1")
        
        if df_keypoints is not None:
            flattened_keypoints = df_keypoints.iloc[0].drop(['Frame', 'Video'])
            fall_detected = predict_fall(flattened_keypoints)
            frame = draw_annotations(frame, flattened_keypoints.values.reshape(-1, 2), boxes, fall_detected)
            all_keypoints = pd.concat([all_keypoints, df_keypoints], ignore_index=True)
        
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_index += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Now I calculate the velocity and acceleration of keypoints.
    keypoint_columns = [col for col in all_keypoints.columns if 'keypoint_' in col]
    all_keypoints_with_velocity = calculate_velocity_acceleration(all_keypoints, keypoint_columns)
    all_keypoints_with_velocity.to_csv("processed_keypoints.csv", index=False)

if __name__ == "__main__":
    main()
