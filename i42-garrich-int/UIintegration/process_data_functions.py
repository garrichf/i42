import numpy as np
import pandas as pd
def compute_angle(p1, p2, p3):
    """
    Compute the angle formed by three points (p1, p2, p3) where p2 is the vertex.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def add_angles(df):
    def calculate_angle(row, p1_name, p2_name, p3_name):
        p1 = (row[f'{p1_name}_X'], row[f'{p1_name}_Y'])
        p2 = (row[f'{p2_name}_X'], row[f'{p2_name}_Y'])
        p3 = (row[f'{p3_name}_X'], row[f'{p3_name}_Y'])
        if not any(np.isnan([*p1, *p2, *p3])) and all(coord != 0 for coord in p1 + p2 + p3):
            return compute_angle(p1, p2, p3)
        else:
            return 0

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

# List of keypoint position columns
keypoint_columns = [
    'Nose_X', 'Nose_Y', 
    'Left Shoulder_X', 'Left Shoulder_Y', 'Right Shoulder_X', 'Right Shoulder_Y',
    'Left Elbow_X', 'Left Elbow_Y', 'Right Elbow_X', 'Right Elbow_Y',
    'Left Wrist_X', 'Left Wrist_Y', 'Right Wrist_X', 'Right Wrist_Y',
    'Left Hip_X', 'Left Hip_Y', 'Right Hip_X','Right Hip_Y',
    'Left Knee_X', 'Left Knee_Y', 'Right Knee_X', 'Right Knee_Y',
    'Left Ankle_X', 'Left Ankle_Y', 'Right Ankle_X', 'Right Ankle_Y'
]

# Function to calculate acceleration
def calculate_acceleration(current_df, history_df, columns, current_index):
    for col in columns:
        # Initialize the columns for velocity and acceleration
        current_df[f'{col}_velocity'] = np.nan
        current_df[f'{col}_acceleration'] = np.nan

        if current_index > 0:
           # Update the velocity calculation without chained assignment warning
            current_df.loc[0, f'{col}_velocity'] = current_df.loc[0, col] - history_df.loc[history_df.index[-1], col]

            # Calculate acceleration based on the calculated velocity
            current_df.loc[0, f'{col}_acceleration'] = current_df.loc[0, f'{col}_velocity'] - history_df.iloc[-1][f'{col}_velocity']
            # print(current_df.loc[0, f'{col}_acceleration'])
            # Replace NaN in velocity and acceleration with -1
            current_df[f'{col}_velocity'] = current_df[f'{col}_velocity'].fillna(0)
            current_df[f'{col}_acceleration'] = current_df[f'{col}_acceleration'].fillna(0)
        else:
            current_df[f'{col}_velocity'].iloc[0] = 0
            current_df[f'{col}_acceleration'].iloc[0] = 0
    return current_df



def remove_outliers_with_fixed_iqr(df, column, fixed_bounds):
    if column not in fixed_bounds:
        # If the fixed bounds do not exist for this column, return the dataframe as-is
        return df

    # Get the fixed lower and upper bounds
    lower_bound, upper_bound = fixed_bounds[column]

    # Cap outliers based on the precomputed bounds
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

    return df


# Function to apply Min-Max Scaling with the fixed range [0, 180]
def min_max_scale_fixed_range(column, min_val, max_val):
    # print(f"Before scaling {column.name}: Min = {column.min()}, Max = {column.max()}")
    
    scaled_column = (column - min_val) / (max_val - min_val)  # Scale to [0, 1]
    
    # Print out the min/max after scaling for debugging
    # print(f"After scaling {column.name}: Min = {scaled_column.min()}, Max = {scaled_column.max()}")
    
    return scaled_column

def z_score_normalize(frame, precomputed_mean, precomputed_std):
    normalized_frame = (frame - precomputed_mean) / precomputed_std
    return normalized_frame