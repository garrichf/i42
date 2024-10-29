from ultralytics import YOLO
import pandas as pd
import numpy as np

#Load model
model = YOLO('yolov8n-pose.pt')
keypoint_names = [
    'Nose_X', 'Nose_Y',
    'Left Eye_X', 'Left Eye_Y',
    'Right Eye_X', 'Right Eye_Y',
    'Left Ear_X','Left Ear_Y',
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
]

def process_frame(frame):
    df = pd.DataFrame(columns=keypoint_names)
    kpt_data = {}
    results = model.predict(frame, conf=0.3)
    keypoints = results[0].keypoints.xyn  # Normalized xy values for model training
    # print(keypoints)
    # Retrieving keypoint data
    for i, point in enumerate(keypoints[0]):
        xn = point[0].item()
        yn = point[1].item()
        kpt_data[keypoint_names[2 * i]] = xn  # Assign X to the corresponding column
        kpt_data[keypoint_names[2 * i + 1]] = yn  # Assign Y to the corresponding column
    # Create a new DataFrame from the dictionary and concatenate it with the main DataFrame
    row_df = pd.DataFrame([kpt_data])
    df = pd.concat([df, row_df], ignore_index=True)
    print(df)

    return df

def YOLO_pose(frame):
    print("YOLO is Running")
    keypoints = process_frame(frame)
    return keypoints