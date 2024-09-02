from ultralytics import YOLO
import sys
import cv2
import pandas as pd
import csv
import os
import time
import numpy as np
from sklearn.impute import KNNImputer
from tensorflow.keras.models import load_model
# Load YOLO model
pose_model = YOLO('yolov8n-pose.pt')
fall_detection_model = load_model("falldetect_30082024_0320.keras")
test_path = "test_video/test.mp4"

# Output CSV file
output_csv_file = 'current_runtime.csv'

keypoint_columns = [   'Nose_X', 'Nose_Y',
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
            'Right Ankle_X', 'Right Ankle_Y']

# Check if output CSV file exists, if not, create it with header
if not os.path.isfile(output_csv_file):
    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
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
            'Right Ankle_X', 'Right Ankle_Y'
        ])
# Function to calculate acceleration
def calculate_acceleration(df, columns):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected input to be a DataFrame")
    
    if not all(col in df.columns for col in columns):
        raise ValueError("Some columns are missing from the DataFrame")
    # df = df.sort_values(by='File Name')  # Sort by frame number
    for col in columns:
        df[f'{col}_velocity'] = df[col].diff()  # Calculate the rate of change (velocity)
        df[f'{col}_acceleration'] = df[f'{col}_velocity'].diff()  # Calculate the rate of change of velocity (acceleration)
    
    # Set velocity and acceleration to 0 for the first frame
    for col in columns:
        df.loc[0, f'{col}_velocity'] = 0
        df.loc[0, f'{col}_acceleration'] = 0

    return df

def add_angles(row):
    def get_point(name):
    # Extract scalar values from the Series
        return (row[f'{name}_X'], row[f'{name}_Y'])

    def compute_angle(p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle
    
    def safe_calculate_angle(p1_name, p2_name, p3_name):
        p1 = get_point(p1_name)
        p2 = get_point(p2_name)
        p3 = get_point(p3_name)
        print(f"p1: {p1}, p2: {p2}, p3: {p3}")  # Debug print
        if all(coord != 0 for coord in p1 + p2 + p3):
            return compute_angle(p1, p2, p3)
        else:
            return np.nan
    
    angles = {
        'Head_Tilt_Angle': safe_calculate_angle('Left Eye', 'Nose', 'Right Eye'),
        'Shoulder_Angle': safe_calculate_angle('Left Shoulder', 'Right Shoulder', 'Left Hip'),
        'Left_Torso_Incline_Angle': safe_calculate_angle('Left Hip', 'Left Shoulder', 'Left Elbow'),
        'Right_Torso_Incline_Angle': safe_calculate_angle('Right Hip', 'Right Shoulder', 'Right Elbow'),
        'Left_Elbow_Angle': safe_calculate_angle('Left Shoulder', 'Left Elbow', 'Left Wrist'),
        'Right_Elbow_Angle': safe_calculate_angle('Right Shoulder', 'Right Elbow', 'Right Wrist'),
        'Left_Hip_Knee_Angle': safe_calculate_angle('Left Hip', 'Left Knee', 'Left Ankle'),
        'Right_Hip_Knee_Angle': safe_calculate_angle('Right Hip', 'Right Knee', 'Right Ankle'),
        'Left_Knee_Ankle_Angle': safe_calculate_angle('Left Knee', 'Left Ankle', 'Left Hip'),
        'Right_Knee_Ankle_Angle': safe_calculate_angle('Right Knee', 'Right Ankle', 'Right Hip'),
        'Leg_Spread_Angle': safe_calculate_angle('Left Hip', 'Right Hip', 'Left Knee'),
        'Head_to_Shoulders_Angle': safe_calculate_angle('Nose', 'Left Shoulder', 'Right Shoulder'),
        'Head_to_Hips_Angle': safe_calculate_angle('Nose', 'Left Hip', 'Right Hip')
    }
    
    for key, value in angles.items():
        row[key] = value
    
    return row


def process_data(data_path, index):
    df = pd.read_csv(data_path)

    # Ensure index is valid
    if index < 0 or index >= len(df):
        raise IndexError("Index is out of range")
    
    # Step 1: Calculate angles for the row
    df = add_angles(df)
    
    # Function to calculate acceleration (make sure this is applicable to entire DataFrame)
    df = calculate_acceleration(df, keypoint_columns)
    
    # Return the processed row
    processed_row = df.iloc[index]
    
    # Optionally save the processed DataFrame
    df.to_csv("processed_" + data_path, index=False)
    
    return processed_row


def process_frame(frame):
    kpt_table = {}
    results = pose_model.predict(frame, conf=0.3)
    keypoints = results[0].keypoints.xyn  # Normalized xy values for model training
    keypoint_xy = results[0].keypoints.xy  # For display

    # print("Keypoints extracted:", keypoints)  # Debugging line
    # print("Keypoints XY:", keypoint_xy)  # Debugging line

    # Retrieving keypoint data
    for i, point in enumerate(keypoints[0]):
        xn = point[0].item()
        yn = point[1].item()
        kpt_table[i] = [xn, yn]

    # print("Keypoints table:", kpt_table)  # Debugging line

    # Check if kpt_table contains expected keys
    for i in range(17):  # Assuming 17 keypoints based in the dataset
        if i not in kpt_table:
            print(f"Key {i} is missing in kpt_table")

    # Write keypoints to CSV
    with open(output_csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
                        kpt_table.get(0, [None, None])[0], kpt_table.get(0, [None, None])[1]
                        , kpt_table.get(1, [None, None])[0], kpt_table.get(1, [None, None])[1]
                        , kpt_table.get(2, [None, None])[0], kpt_table.get(2, [None, None])[1]
                        , kpt_table.get(3, [None, None])[0], kpt_table.get(3, [None, None])[1]
                        , kpt_table.get(4, [None, None])[0], kpt_table.get(4, [None, None])[1]
                        , kpt_table.get(5, [None, None])[0], kpt_table.get(5, [None, None])[1]
                        , kpt_table.get(6, [None, None])[0], kpt_table.get(6, [None, None])[1]
                        , kpt_table.get(7, [None, None])[0], kpt_table.get(7, [None, None])[1]
                        , kpt_table.get(8, [None, None])[0], kpt_table.get(8, [None, None])[1]
                        , kpt_table.get(9, [None, None])[0], kpt_table.get(9, [None, None])[1]
                        , kpt_table.get(10, [None, None])[0], kpt_table.get(10, [None, None])[1]
                        , kpt_table.get(11, [None, None])[0], kpt_table.get(11, [None, None])[1]
                        , kpt_table.get(12, [None, None])[0], kpt_table.get(12, [None, None])[1]
                        , kpt_table.get(13, [None, None])[0], kpt_table.get(13, [None, None])[1]
                        , kpt_table.get(14, [None, None])[0], kpt_table.get(14, [None, None])[1]
                        , kpt_table.get(15, [None, None])[0], kpt_table.get(15, [None, None])[1]
                        , kpt_table.get(16, [None, None])[0], kpt_table.get(16, [None, None])[1]
                        ])
    return frame

cap = cv2.VideoCapture(test_path)
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    process_frame(frame)
    features = process_data(output_csv_file, frame_num)
    print("features")
    print(features.shape)
    results = fall_detection_model.predict(features)
        
    score0 = round(results[0][0],2)
    score1 = round(results[0][1],2)
    print("Class 0: " + str(score0))
    print("Class 1: " + str(score1))

    frame_num += 1

cap.release()
# Handle key press to exit
cv2.destroyAllWindows()

print("Processing complete. All windows closed.")
