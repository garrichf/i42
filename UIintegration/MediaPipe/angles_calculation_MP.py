import numpy as np
import pandas as pd
from MediaPipe_demo import MediaPipe_detect_pose_sequence

def compute_angle(p1, p2, p3):
    """
    I’m calculating the angle (in degrees) between three points in a 2D plane.
    The angle is formed at the middle point (p2), with the other two points (p1, p3) as the arms of the angle.
    
    Parameters:
    - p1: The coordinates (x, y) of the first point.
    - p2: The coordinates (x, y) of the second point (the vertex where the angle is formed).
    - p3: The coordinates (x, y) of the third point.

    Returns:
    - The angle in degrees between these three points.
    """
    # I’ll create vectors between the points first
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Now, I’ll calculate the angle between these vectors using the dot product
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)  # The magnitude (length) of the first vector
    magnitude_v2 = np.linalg.norm(v2)  # The magnitude (length) of the second vector
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)  # Using the cosine formula

    # Finally, I’ll convert the result from radians to degrees and return it
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return angle

def add_angles(df):
    """
    I’m adding new angles to the DataFrame based on specific body keypoints. 
    These angles help us understand the posture by calculating the angles between key body parts.
    
    Parameters:
    - df: A DataFrame containing the keypoint coordinates for body parts (with columns like '{point_name}_X' and '{point_name}_Y').

    Returns:
    - The updated DataFrame with the additional angles.
    """
    # Here’s a helper function to calculate an angle between three points in the DataFrame
    def calculate_angle(row, p1_name, p2_name, p3_name):
        p1 = (row[f'{p1_name}_X'], row[f'{p1_name}_Y'])
        p2 = (row[f'{p2_name}_X'], row[f'{p2_name}_Y'])
        p3 = (row[f'{p3_name}_X'], row[f'{p3_name}_Y'])
        
        # I only calculate the angle if all coordinates are available (not missing)
        if all(pd.notna(coord) for coord in p1 + p2 + p3):
            return compute_angle(p1, p2, p3)
        else:
            return np.nan

    # Let's calculate various body angles based on the body keypoints
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

# Here's a list of all keypoints I'm working with
keypoints_columns = [
    'Nose_X', 'Nose_Y',
    'Left Eye Inner_X', 'Left Eye Inner_Y',
    'Left Eye_X', 'Left Eye_Y',
    'Left Eye Outer_X', 'Left Eye Outer_Y',
    'Right Eye Inner_X', 'Right Eye Inner_Y',
    'Right Eye_X', 'Right Eye_Y',
    'Right Eye Outer_X', 'Right Eye Outer_Y',
    'Left Ear_X', 'Left Ear_Y',
    'Right Ear_X', 'Right Ear_Y',
    'Mouth Left_X', 'Mouth Left_Y',
    'Mouth Right_X', 'Mouth Right_Y',
    'Left Shoulder_X', 'Left Shoulder_Y',
    'Right Shoulder_X', 'Right Shoulder_Y',
    'Left Elbow_X', 'Left Elbow_Y',
    'Right Elbow_X', 'Right Elbow_Y',
    'Left Wrist_X', 'Left Wrist_Y',
    'Right Wrist_X', 'Right Wrist_Y',
    'Left Pinky_X', 'Left Pinky_Y',
    'Right Pinky_X', 'Right Pinky_Y',
    'Left Index_X', 'Left Index_Y',
    'Right Index_X', 'Right Index_Y',
    'Left Thumb_X', 'Left Thumb_Y',
    'Right Thumb_X', 'Right Thumb_Y',
    'Left Hip_X', 'Left Hip_Y',
    'Right Hip_X', 'Right Hip_Y',
    'Left Knee_X', 'Left Knee_Y',
    'Right Knee_X', 'Right Knee_Y',
    'Left Ankle_X', 'Left Ankle_Y',
    'Right Ankle_X', 'Right Ankle_Y',
    'Left Heel_X', 'Left Heel_Y',
    'Right Heel_X', 'Right Heel_Y',
    'Left Foot Index_X', 'Left Foot Index_Y',
    'Right Foot Index_X', 'Right Foot Index_Y'
]

# I’m calculating acceleration for the specified columns here
def calculate_acceleration(df, columns):
    """
    For each keypoint, I’m calculating its velocity and acceleration.
    First, I compute the velocity (change in position), and then I get acceleration by finding the rate of change of velocity.

    Parameters:
    - df: The DataFrame that contains the keypoints.
    - columns: The list of keypoints to calculate velocity and acceleration for.

    Returns:
    - The DataFrame, now with extra columns for velocity and acceleration.
    """
    # Check if the input is valid
    if not isinstance(df, pd.DataFrame):
        raise TypeError("I expected a pandas DataFrame")
    
    # Check that all required columns exist
    if not all(col in df.columns for col in columns):
        raise ValueError("Some columns are missing in the DataFrame")

    # For each column, calculate velocity and acceleration
    for col in columns:
        df[f'{col}_velocity'] = df[col].diff()  # Velocity is just the difference between subsequent frames
        df[f'{col}_acceleration'] = df[f'{col}_velocity'].diff()  # Acceleration is the rate of change of velocity

    # For the first frame, there's no velocity or acceleration, so let's set those to None
    for col in columns:
        df.loc[0, f'{col}_velocity'] = None
        df.loc[0, f'{col}_acceleration'] = None

    return df

# Here’s an example of how we'd use this script
video_path = "ADL.mp4"
keypoints = MediaPipe_detect_pose_sequence(video_path)  # Detect keypoints from the video
angles = add_angles(keypoints)  # Add the angle calculations
result = calculate_acceleration(angles, keypoints_columns)  # Compute velocity and acceleration for each keypoint
print(result)
