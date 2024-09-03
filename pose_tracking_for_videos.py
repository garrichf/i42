from ultralytics import YOLO
from tensorflow.keras.models import load_model
import cv2
import pandas as pd
import csv
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load YOLO model for pose estimation
model = YOLO('yolov8n-pose.pt')

# Load the fall detection model
fall_detection_model = load_model('falldetection_version_220820241432.keras')

# Define max_length for padding, it should match the value used during training
max_length = 17  # Adjust this based on your dataset (number of keypoints per frame)

# Read CSV file
df = pd.read_csv('found_files.csv', encoding='latin-1')
# Output CSV file
output_csv_file = 'output_with_predictions.csv'

# Check if output CSV file exists, if not, create it with header
if not os.path.isfile(output_csv_file):
    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Folder Name', 'File Name', 'FallType', 'Prediction',
            'Nose_X', 'Nose_Y',
            'Left Eye_X', 'Left Eye_Y',
            'Right Eye_X', 'Right Eye_Y',
            'Left Ear_X', 'Left Ear_Y',
            'Right Ear_X', 'Right Ear_Y',
            'Left Shoulder_X', 'Left Shoulder_Y',
            'Right Shoulder_X', 'Right Shoulder_Y',
            'Left Elbow_X', 'Left Elbow_Y',
            'Right Elbow_X', 'Right Elbow_Y',
            'Left Wrist_X', 'Left Wrist_Y',
            'Right Wrist_X', 'Right Wrist_Y',
            'Left Hip_X', 'Left Hip_Y',
            'Right Hip_X', 'Right Hip_Y',
            'Left Knee_X', 'Left Knee_Y',
            'Right Knee_X', 'Right Knee_Y',
            'Left Ankle_X', 'Left Ankle_Y',
            'Right Ankle_X', 'Right Ankle_Y',
            'Head_Tilt_Angle', 'Shoulder_Angle', 'Left_Torso_Incline_Angle',
            'Right_Torso_Incline_Angle', 'Left_Elbow_Angle', 'Right_Elbow_Angle',
            'Left_Hip_Knee_Angle', 'Right_Hip_Knee_Angle', 'Left_Knee_Ankle_Angle',
            'Right_Knee_Ankle_Angle', 'Leg_Spread_Angle', 'Head_to_Shoulders_Angle',
            'Head_to_Hips_Angle'
        ])

def get_fall_type(file_path):
    folder_name = os.path.basename(os.path.dirname(file_path))
    if folder_name.startswith("nfall"):
        return 0
    elif folder_name.startswith("fall"):
        return 1
    else:
        return None

def preprocess_keypoints_for_model(keypoints, max_length):
    if len(keypoints) > 1:
        keypoints = keypoints[0]
    
    keypoints_array = np.array(keypoints, dtype='float32')

    if keypoints_array.size != 34:
        print(f"Unexpected number of keypoints: {keypoints_array.size}. Skipping this frame.")
        return None

    keypoints_flattened = keypoints_array.flatten()
    keypoints_flattened = keypoints_flattened.reshape(1, 1, 34)

    return keypoints_flattened

def compute_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def add_angles_to_df(df):
    def calculate_angle(row, p1_name, p2_name, p3_name):
        p1 = (row[f'{p1_name}_X'], row[f'{p1_name}_Y'])
        p2 = (row[f'{p2_name}_X'], row[f'{p2_name}_Y'])
        p3 = (row[f'{p3_name}_X'], row[f'{p3_name}_Y'])
        if all(coord != 0 for coord in p1 + p2 + p3):
            return compute_angle(p1, p2, p3)
        else:
            return np.nan

    df['Head_Tilt_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Eye', 'Nose', 'Right Eye'), axis=1)
    df['Shoulder_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Shoulder', 'Right Shoulder', 'Left Hip'), axis=1)
    df['Left_Torso_Incline_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Hip', 'Left Shoulder', 'Left Elbow'), axis=1)
    df['Right_Torso_Incline_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Hip', 'Right Shoulder', 'Right Elbow'), axis=1)
    df['Left_Elbow_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Shoulder', 'Left Elbow', 'Left Wrist'), axis=1)
    df['Right_Elbow_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Shoulder', 'Right Elbow', 'Right Wrist'), axis=1)
    df['Left_Hip_Knee_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Hip', 'Left Knee', 'Left Ankle'), axis=1)
    df['Right_Hip_Knee_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Hip', 'Right Knee', 'Right Ankle'), axis=1)
    df['Left_Knee_Ankle_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Knee', 'Left Ankle', 'Left Hip'), axis=1)
    df['Right_Knee_Ankle_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Knee', 'Right Ankle', 'Right Hip'), axis=1)
    df['Leg_Spread_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Hip', 'Right Hip', 'Left Knee'), axis=1)
    df['Head_to_Shoulders_Angle'] = df.apply(lambda row: calculate_angle(row, 'Nose', 'Left Shoulder', 'Right Shoulder'), axis=1)
    df['Head_to_Hips_Angle'] = df.apply(lambda row: calculate_angle(row, 'Nose', 'Left Hip', 'Right Hip'), axis=1)

    return df

def process_frame(frame, folder_name, file_name, fall_type):
    kpt_table = {}
    results = model.predict(frame, conf=0.3)

    if results[0].keypoints is None or results[0].keypoints.xyn.numel() == 0:
        print(f"No keypoints detected for frame {file_name}. Skipping.")
        return frame

    keypoints = results[0].keypoints.xyn
    processed_keypoints = preprocess_keypoints_for_model(keypoints, max_length)

    if processed_keypoints is None:
        return frame

    fall_prediction = fall_detection_model.predict(processed_keypoints)
    predicted_class = np.argmax(fall_prediction, axis=1)[0]
    label = "Fall" if predicted_class == 1 else "No Fall"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    keypoint_xy = results[0].keypoints.xy
    for i, point in enumerate(keypoint_xy[0]):
        x, y = int(point[0].item()), int(point[1].item())
        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(frame, str(i), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame

for index, row in df.iterrows():
    file_path = row['File Path']
    folder_name = row['Folder Name']
    file_name = row['File Name']
    file_name_without_extension = os.path.splitext(file_name)[0]
    fall_type = get_fall_type(file_path)

    if fall_type is None:
        print(f"Invalid fall type for file: {file_path}. Skipping.")
        continue

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        continue

    cap = cv2.VideoCapture(file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, folder_name, file_name, fall_type)
        output_frame_path = f'processed_frames/{file_name_without_extension}_processed.avi'

        if not os.path.isfile(output_frame_path):
            os.makedirs(os.path.dirname(output_frame_path), exist_ok=True)

        out = cv2.VideoWriter(output_frame_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (frame.shape[1], frame.shape[0]))
        out.write(processed_frame)
        out.release()

    cap.release()

    df = add_angles_to_df(df)

df.to_csv(output_csv_file, index=False)
