import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load the YOLO model for pose detection and the fall detection model.
pose_model = YOLO('yolov8n-pose.pt')
fall_detection_model = load_model("falldetect_16092024_1307_alldatacombined_reg_reduceplateau.keras")

input_shape = fall_detection_model.input_shape[2] 

def compute_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def calculate_velocity_acceleration(df, keypoint_columns):
    df = df.sort_values(by='Frame')
    for col in keypoint_columns:
        df[f'{col}_velocity'] = df.groupby('Video')[col].diff()
        df[f'{col}_acceleration'] = df.groupby('Video')[f'{col}_velocity'].diff()
    return df

def process_frame(frame, frame_index, video_name):
    results = pose_model(frame)
    keypoints = results[0].keypoints.xy[0] if len(results[0].keypoints.xy) > 0 else None
    boxes = results[0].boxes.xyxy if len(results[0].boxes.xyxy) > 0 else None
    
    if keypoints is not None:
        keypoints = keypoints.flatten()
        df_keypoints = pd.DataFrame([keypoints], columns=[f'keypoint_{i}' for i in range(len(keypoints))])
        df_keypoints['Frame'] = frame_index
        df_keypoints['Video'] = video_name
        return df_keypoints, boxes
    return None, boxes

def reshape_keypoints(keypoints, expected_size=input_shape):
    keypoints = keypoints[:expected_size]
    if len(keypoints) < expected_size:
        keypoints = np.pad(keypoints, (0, expected_size - len(keypoints)), 'constant')
    return keypoints

def create_feature_vector(keypoints, angle, velocity, acceleration):
    return np.concatenate([keypoints, [angle], [velocity], [acceleration]])

def predict_fall(df_keypoints, previous_keypoints):
    if df_keypoints is None or len(df_keypoints.columns) < 6:
        return False

    # Calculate angle
    shoulder = (df_keypoints['keypoint_1'].values[0], df_keypoints['keypoint_2'].values[0])
    hip = (df_keypoints['keypoint_3'].values[0], df_keypoints['keypoint_4'].values[0])
    knee = (df_keypoints['keypoint_5'].values[0], df_keypoints['keypoint_6'].values[0])
    angle = compute_angle(shoulder, hip, knee)

    # Calculate velocity and acceleration
    keypoint_columns = [col for col in df_keypoints.columns if 'keypoint_' in col]
    df_with_va = calculate_velocity_acceleration(df_keypoints, keypoint_columns)

    # Extract velocity and acceleration for the latest frame
    velocity = df_with_va[f'{keypoint_columns[-1]}_velocity'].iloc[-1]
    acceleration = df_with_va[f'{keypoint_columns[-1]}_acceleration'].iloc[-1]

    # Create feature vector
    flattened_keypoints = df_keypoints.iloc[0].drop(['Frame', 'Video'])
    features = create_feature_vector(flattened_keypoints.values, angle, velocity, acceleration)

    # Reshape and predict
    features = reshape_keypoints(features)
    features = np.array(features, dtype=np.float32)
    features = features.reshape(1, 1, -1) 
    prediction = fall_detection_model.predict(features)
    return prediction[0][1] > 0.5

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

def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    all_keypoints = pd.DataFrame()
    frame_index = 0
    previous_keypoints = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        df_keypoints, boxes = process_frame(frame, frame_index, video_name="video_1")
        
        if df_keypoints is not None:
            fall_detected = predict_fall(df_keypoints, previous_keypoints)

            flattened_keypoints = df_keypoints.iloc[0].drop(['Frame', 'Video'])
            frame = draw_annotations(frame, flattened_keypoints.values.reshape(-1, 2), boxes, fall_detected)
            all_keypoints = pd.concat([all_keypoints, df_keypoints], ignore_index=True)
        
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        previous_keypoints = df_keypoints
        frame_index += 1
    
    cap.release()
    cv2.destroyAllWindows()

    keypoint_columns = [col for col in all_keypoints.columns if 'keypoint_' in col]
    all_keypoints_with_features = calculate_velocity_acceleration(all_keypoints, keypoint_columns)
    all_keypoints_with_features.to_csv("processed_keypoints_with_features.csv", index=False)

if __name__ == "__main__":
    main()