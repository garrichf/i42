import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

def MediaPipe_detect_pose_sequence(vid_path):
    """
    Detects pose keypoints from a video using the MediaPipe model.
    
    Parameters:
    vid_path (str): Path to the input video file.
    
    Returns:
    pd.DataFrame: DataFrame containing keypoints for each frame.
    """
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("MediaPipe Pose Estimation Model loaded successfully.")
    except Exception as error:
        print(f"Failed to load the model: {error}")
        return None

    POSE_CONFIDENCE_THRESHOLD = 0.25
    vid = cv2.VideoCapture(vid_path)
    if not vid.isOpened():
        print("Error: Could not open video.")
        return None

    all_keypoints = []
    keypoint_names = [
        'Nose', 'Left Eye Inner', 'Left Eye', 'Left Eye Outer', 'Right Eye Inner', 'Right Eye', 'Right Eye Outer',
        'Left Ear', 'Right Ear', 'Mouth Left', 'Mouth Right', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 
        'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Pinky', 'Right Pinky', 'Left Index', 'Right Index', 
        'Left Thumb', 'Right Thumb', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
        'Left Heel', 'Right Heel', 'Left Foot Index', 'Right Foot Index'
    ]

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame_keypoints = {}

        if results.pose_landmarks:
            average_confidence = np.mean([landmark.visibility for landmark in results.pose_landmarks.landmark])

            if average_confidence >= POSE_CONFIDENCE_THRESHOLD:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    frame_keypoints[f'{keypoint_names[idx]}_X'] = landmark.x
                    frame_keypoints[f'{keypoint_names[idx]}_Y'] = landmark.y

        all_keypoints.append(frame_keypoints)

    vid.release()
    df = pd.DataFrame(all_keypoints)
    return df

def compute_angle(p1, p2, p3):
    """
    Computes the angle between three points in a 2D plane.
    
    Parameters:
    p1, p2, p3 (tuple): Coordinates of the points (x, y).
    
    Returns:
    float: The angle in degrees between the points.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def add_angles(df):
    """
    Adds calculated angles to the DataFrame based on key body points.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing body point coordinates.
    
    Returns:
    pd.DataFrame: DataFrame with added angle columns.
    """
    def calculate_angle(row, p1_name, p2_name, p3_name):
        p1 = (row[f'{p1_name}_X'], row[f'{p1_name}_Y'])
        p2 = (row[f'{p2_name}_X'], row[f'{p2_name}_Y'])
        p3 = (row[f'{p3_name}_X'], row[f'{p3_name}_Y'])
        if all(coord != 0 for coord in p1 + p2 + p3):
            return compute_angle(p1, p2, p3)
        else:
            return np.nan

    # Calculate all angles similar to the MoveNet version
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

def calculate_acceleration(df, columns):
    """
    Calculate velocity and acceleration for each keypoint in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing keypoint coordinates.
    columns (list): List of column names for which to calculate velocity and acceleration.
    
    Returns:
    pd.DataFrame: DataFrame with added velocity and acceleration columns.
    """
    for col in columns:
        df[f'{col}_velocity'] = df[col].diff()
        df[f'{col}_acceleration'] = df[f'{col}_velocity'].diff()
    
    # Set the first frame's velocity and acceleration to NaN
    for col in columns:
        df.loc[0, f'{col}_velocity'] = np.nan
        df.loc[0, f'{col}_acceleration'] = np.nan

    return df

# Example usage:
vid_path = "ADL.mp4"
df_keypoints = MediaPipe_detect_pose_sequence(vid_path)
if df_keypoints is not None:
    df_with_angles = add_angles(df_keypoints)
    keypoints_columns = [
        'Nose_X', 'Nose_Y', 'Left Eye_X', 'Left Eye_Y', 'Right Eye_X', 'Right Eye_Y',
        'Left Shoulder_X', 'Left Shoulder_Y', 'Right Shoulder_X', 'Right Shoulder_Y',
        'Left Elbow_X', 'Left Elbow_Y', 'Right Elbow_X', 'Right Elbow_Y', 'Left Wrist_X', 'Left Wrist_Y',
        'Right Wrist_X', 'Right Wrist_Y', 'Left Hip_X', 'Left Hip_Y', 'Right Hip_X', 'Right Hip_Y',
        'Left Knee_X', 'Left Knee_Y', 'Right Knee_X', 'Right Knee_Y', 'Left Ankle_X', 'Left Ankle_Y',
        'Right Ankle_X', 'Right Ankle_Y'
    ]
    df_with_acceleration = calculate_acceleration(df_with_angles, keypoints_columns)
    print(df_with_acceleration)
