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

def predict_fall(features):
    features = reshape_keypoints(features)
    features = np.array(features, dtype=np.float32)
    features = features.reshape(1, 1, -1)  # Reshape for the model
    prediction = fall_detection_model.predict(features)
    return prediction[0][0] > 0.5

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
    angles_list = []
    velocity_list = []
    acceleration_list = []

    previous_keypoints = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        df_keypoints, boxes = process_frame(frame, frame_index, video_name="video_1")
        
        if df_keypoints is not None:
            if len(df_keypoints.columns) >= 6:
                shoulder = (df_keypoints['keypoint_1'].values[0], df_keypoints['keypoint_2'].values[0])
                hip = (df_keypoints['keypoint_3'].values[0], df_keypoints['keypoint_4'].values[0])
                knee = (df_keypoints['keypoint_5'].values[0], df_keypoints['keypoint_6'].values[0])
                
                angle = compute_angle(shoulder, hip, knee)
                angles_list.append(angle)

            flattened_keypoints = df_keypoints.iloc[0].drop(['Frame', 'Video'])

            if previous_keypoints is not None:
                velocity = np.linalg.norm(flattened_keypoints.values - previous_keypoints)
                velocity_list.append(velocity)

                if len(velocity_list) > 1:
                    acceleration = velocity_list[-1] - velocity_list[-2]
                    acceleration_list.append(acceleration)
                else:
                    acceleration = 0
                    acceleration_list.append(acceleration)
            else:
                velocity = 0
                acceleration = 0
                velocity_list.append(velocity)
                acceleration_list.append(acceleration)

            features = create_feature_vector(flattened_keypoints.values, angle, velocity, acceleration)
            fall_detected = predict_fall(features)

            frame = draw_annotations(frame, flattened_keypoints.values.reshape(-1, 2), boxes, fall_detected)
            all_keypoints = pd.concat([all_keypoints, df_keypoints], ignore_index=True)
        
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        previous_keypoints = flattened_keypoints.values
        frame_index += 1
    
    cap.release()
    cv2.destroyAllWindows()

    keypoint_columns = [col for col in all_keypoints.columns if 'keypoint_' in col]
    all_keypoints_with_velocity = calculate_velocity_acceleration(all_keypoints, keypoint_columns)
    
    all_keypoints_with_velocity['Angle'] = pd.Series(angles_list)
    all_keypoints_with_velocity['Velocity'] = pd.Series(velocity_list)
    all_keypoints_with_velocity['Acceleration'] = pd.Series(acceleration_list)
    all_keypoints_with_velocity.to_csv("processed_keypoints_with_features.csv", index=False)

if __name__ == "__main__":
    main()
