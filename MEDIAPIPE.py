import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

POSE_CONFIDENCE_THRESHOLD = 0.25

mp_pose = mp.solutions.pose
model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Keypoint names based on the MediaPipe model
keypoint_names = [
    'Nose', 'Left Eye Inner', 'Left Eye', 'Left Eye Outer', 'Right Eye Inner', 'Right Eye', 'Right Eye Outer',
    'Left Ear', 'Right Ear', 'Mouth Left', 'Mouth Right', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 
    'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Pinky', 'Right Pinky', 'Left Index', 'Right Index', 
    'Left Thumb', 'Right Thumb', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
    'Left Heel', 'Right Heel', 'Left Foot Index', 'Right Foot Index'
]

# Columns to keep
keypoint_columns = [
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
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    # Perform pose estimation
    results = model.process(frame_rgb)
    
    # Initialize a dictionary to store keypoints for the current frame
    frame_keypoints = {}
    
    # print("Results Landmark")
    # print(results.pose_landmarks)
    if results.pose_landmarks:
            average_confidence = np.mean([landmark.visibility for landmark in results.pose_landmarks.landmark])
            # print("AVERAGE CONFIDENCE: " + str(average_confidence))

            if average_confidence >= POSE_CONFIDENCE_THRESHOLD:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    frame_keypoints[f'{keypoint_names[idx]}_X'] = landmark.x
                    frame_keypoints[f'{keypoint_names[idx]}_Y'] = landmark.y
                    # print(f"Keypoint: {keypoint_names[idx]}, X: {landmark.x}, Y: {landmark.y}")
    else:
        # Set all keypoint values to NaN if no landmarks are detected
        frame_keypoints = {f'{keypoint_name}_X': np.nan for keypoint_name in keypoint_names}
        frame_keypoints.update({f'{keypoint_name}_Y': np.nan for keypoint_name in keypoint_names})


    # If no keypoints are found, return an empty DataFrame
    if not frame_keypoints:
        return pd.DataFrame(columns=keypoint_columns)
    # Convert the keypoints dictionary into a DataFrame (with one row)
    row_df = pd.DataFrame([frame_keypoints])
    df = pd.concat([df, row_df], ignore_index=True)
    # Retain only the specified keypoint columns
    df = df[keypoint_columns]
    return df

def MEDIAPIPE_pose(frame):
    print("MEDIAPIPE is Running")
    keypoints = process_frame(frame)
    return keypoints
