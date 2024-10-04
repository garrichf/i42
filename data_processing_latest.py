# Data Processing version 2.0
# Description: This file contains data processing functions to generate additional features required by the Fall Detection Model from the original keypoints data.
# Implemented by Nick Bui, adapting Garrich's codebased main_dataprocess.ipynb and previous data_processing.py

import pandas as pd
import numpy as np

def compute_angle(p1, p2, p3):
    """
    Computes the angle (in degrees) between three points (p1, p2, p3) in a 2D plane.

    The angle is calculated at point p2, with p1 and p3 forming the arms of the angle.

    Parameters:
    p1 (tuple): A tuple representing the coordinates (x, y) of the first point.
    p2 (tuple): A tuple representing the coordinates (x, y) of the second point (vertex of the angle).
    p3 (tuple): A tuple representing the coordinates (x, y) of the third point.

    Returns:
    float: The angle in degrees between the three points.
    """
    # Compute the vectors between the points
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Compute the angle between the vectors
    dot_product = np.dot(v1, v2)
    # Compute the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    # Compute the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    # Compute the angle in degrees
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    return angle

def add_angles(df):
    """
    Adds various angles to the DataFrame based on key body points.

    Parameters:
    df (pandas.DataFrame): DataFrame containing body point coordinates with columns named in the format '{point_name}_X' and '{point_name}_Y'.

    Returns:
    pandas.DataFrame: DataFrame with additional columns for each calculated angle.

    Calculated Angles:
    - Head_Tilt_Angle: Angle of the nose relative to the left and right eyes.
    - Shoulder_Angle: Angle of the shoulders relative to the spine.
    - Left_Torso_Incline_Angle: Incline angle of the left torso (left hip, left shoulder, left elbow).
    - Right_Torso_Incline_Angle: Incline angle of the right torso (right hip, right shoulder, right elbow).
    - Left_Elbow_Angle: Angle of the left elbow (left shoulder, left elbow, left wrist).
    - Right_Elbow_Angle: Angle of the right elbow (right shoulder, right elbow, right wrist).
    - Left_Hip_Knee_Angle: Angle of the left hip to knee (left hip, left knee, left ankle).
    - Right_Hip_Knee_Angle: Angle of the right hip to knee (right hip, right knee, right ankle).
    - Left_Knee_Ankle_Angle: Angle of the left knee to ankle (left knee, left ankle, left hip).
    - Right_Knee_Ankle_Angle: Angle of the right knee to ankle (right knee, right ankle, right hip).
    - Leg_Spread_Angle: Spread angle of the hips relative to each other (left hip, right hip, left knee).
    - Head_to_Shoulders_Angle: Angle of the head relative to the shoulders (nose, left shoulder, right shoulder).
    - Head_to_Hips_Angle: Angle of the head relative to the hips (nose, left hip, right hip).
    """
    # Function to calculate the angle between three points
    def calculate_angle(row, p1_name, p2_name, p3_name):
        p1 = (row[f'{p1_name}_X'], row[f'{p1_name}_Y'])
        p2 = (row[f'{p2_name}_X'], row[f'{p2_name}_Y'])
        p3 = (row[f'{p3_name}_X'], row[f'{p3_name}_Y'])
        if all(coord != 0 for coord in p1 + p2 + p3):
            return compute_angle(p1, p2, p3)
        else:
            return np.nan

    # Shoulder Angle (e.g., Shoulder angles with spine)
    df['Shoulder_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Shoulder', 'Right Shoulder', 'Left Hip'), axis=1)
    
    # Torso Incline Angles (e.g., Hips relative to shoulders)
    df['Left_Torso_Incline_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Hip', 'Left Shoulder', 'Left Elbow'), axis=1)
    df['Right_Torso_Incline_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Hip', 'Right Shoulder', 'Right Elbow'), axis=1)
    
    # Elbow Angles (e.g., Shoulder-Elbow-Wrist)
    df['Left_Elbow_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Shoulder', 'Left Elbow', 'Left Wrist'), axis=1)
    df['Right_Elbow_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Shoulder', 'Right Elbow', 'Right Wrist'), axis=1)
    
    # Hip-Knee Angles (e.g., Hip-Knee-Ankle)
    df['Left_Hip_Knee_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Hip', 'Left Knee', 'Left Ankle'), axis=1)
    df['Right_Hip_Knee_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Hip', 'Right Knee', 'Right Ankle'), axis=1)
    
    # Knee-Ankle Angles (e.g., Knee-Ankle-Foot, if you have foot points)
    df['Left_Knee_Ankle_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Knee', 'Left Ankle', 'Left Hip'), axis=1)
    df['Right_Knee_Ankle_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Knee', 'Right Ankle', 'Right Hip'), axis=1)
    
    # Head to Shoulders Angle
    df['Head_to_Shoulders_Angle'] = df.apply(lambda row: calculate_angle(row, 'Nose', 'Left Shoulder', 'Right Shoulder'), axis=1)
    
    # Head to Hips Angle
    df['Head_to_Hips_Angle'] = df.apply(lambda row: calculate_angle(row, 'Nose', 'Left Hip', 'Right Hip'), axis=1)

    return df

# Example calculation output
# angles = add_angles(detect_pose_sequence("ADL.mp4"))
# print(result)


def calculate_acceleration(df):
    """
    Calculate the velocity and acceleration for specified columns in a DataFrame.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    columns (list of str): List of column names for which to calculate velocity and acceleration.
    Returns:
    pd.DataFrame: The DataFrame with additional columns for velocity and acceleration for each specified column.
    Raises:
    TypeError: If the input df is not a pandas DataFrame.
    ValueError: If any of the specified columns are missing from the DataFrame.
    Notes:
    - Velocity is calculated as the first-order difference of the specified columns.
    - Acceleration is calculated as the first-order difference of the velocity.
    - The first frame's velocity and acceleration are set to None.
    """
    # Keypoints columns
    keypoints_columns = [
                        'Nose_X', 'Nose_Y',
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
    # Check if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected input to be a DataFrame")

    # Check if all specified columns are present in the DataFrame
    if not all(col in df.columns for col in keypoints_columns):
        raise ValueError("Some columns are missing from the DataFrame")

    # Calculate velocity and acceleration for each specified column
    for col in keypoints_columns:
        df[f'{col}_velocity'] = df[col].diff()  # Calculate the first-order difference for velocity
        df[f'{col}_acceleration'] = df[f'{col}_velocity'].diff()  # Calculate the rate of change of velocity (acceleration)

    # Set velocity and acceleration to None for the first frame and frames with frame_idx 0
    for col in keypoints_columns:
        df.loc[df['frame_idx'] == 0, f'{col}_velocity'] = -99
        df.loc[df['frame_idx'] == 0, f'{col}_acceleration'] = -99

    return df

def remove_velocity_columns(df):
    """
    Remove all velocity columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with velocity columns removed.
    """
    velocity_columns = [col for col in df.columns if '_velocity' in col]
    df.drop(columns=velocity_columns, inplace=True)
    return df