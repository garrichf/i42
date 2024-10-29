# Stores function that are needed to run in main and also process_data
import numpy as np
import pandas as pd
import cv2
import os
import csv
from tensorflow.keras import backend as K

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
    """
    Adds various angle calculations to the given DataFrame.
    This function calculates several angles based on the coordinates of specific body points
    and adds these angles as new columns to the DataFrame. The angles calculated include:
    - Shoulder Angle: Angle between the left shoulder, right shoulder, and left hip.
    - Left Torso Incline Angle: Angle between the left hip, left shoulder, and left elbow.
    - Right Torso Incline Angle: Angle between the right hip, right shoulder, and right elbow.
    - Left Elbow Angle: Angle between the left shoulder, left elbow, and left wrist.
    - Right Elbow Angle: Angle between the right shoulder, right elbow, and right wrist.
    - Left Hip-Knee Angle: Angle between the left hip, left knee, and left ankle.
    - Right Hip-Knee Angle: Angle between the right hip, right knee, and right ankle.
    - Left Knee-Ankle Angle: Angle between the left knee, left ankle, and left hip.
    - Right Knee-Ankle Angle: Angle between the right knee, right ankle, and right hip.
    - Head to Shoulders Angle: Angle between the nose, left shoulder, and right shoulder.
    - Head to Hips Angle: Angle between the nose, left hip, and right hip.
    Parameters:
    df (pandas.DataFrame): DataFrame containing the coordinates of body points. The coordinates
                           should be in columns named in the format '{point_name}_X' and '{point_name}_Y'.
    Returns:
    pandas.DataFrame: The original DataFrame with additional columns for each calculated angle.
    """
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
    """
    Calculate the velocity and acceleration for specified columns in the current DataFrame
    based on the historical DataFrame.

    Parameters:
    current_df (pd.DataFrame): The current DataFrame containing the latest data.
    history_df (pd.DataFrame): The historical DataFrame containing previous data.
    columns (list of str): List of column names for which velocity and acceleration need to be calculated.
    current_index (int): The index of the current row in the current DataFrame.

    Returns:
    pd.DataFrame: The updated current DataFrame with new columns for velocity and acceleration.
    """
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
    """
    Remove outliers from a specified column in a DataFrame using precomputed fixed bounds.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    column (str): The name of the column from which to remove outliers.
    fixed_bounds (dict): A dictionary where keys are column names and values are tuples 
                         containing the lower and upper bounds for outlier removal.

    Returns:
    pandas.DataFrame: The DataFrame with outliers capped based on the fixed bounds.
    """
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
    """
    Scales a pandas Series to a fixed range [0, 1] based on provided minimum and maximum values.
    Parameters:
    column (pandas.Series): The data column to be scaled.
    min_val (float): The minimum value for scaling.
    max_val (float): The maximum value for scaling.
    Returns:
    pandas.Series: The scaled data column with values in the range [0, 1].
    """
    # print(f"Before scaling {column.name}: Min = {column.min()}, Max = {column.max()}")
    
    scaled_column = (column - min_val) / (max_val - min_val)  # Scale to [0, 1]
    
    # Print out the min/max after scaling for debugging
    # print(f"After scaling {column.name}: Min = {scaled_column.min()}, Max = {scaled_column.max()}")
    
    return scaled_column

def z_score_normalize(frame, precomputed_mean, precomputed_std):
    """
    Normalize the given DataFrame using z-score normalization.

    Parameters:
    frame (pd.DataFrame): The DataFrame to be normalized.
    precomputed_mean (pd.Series or float): The mean value(s) used for normalization.
    precomputed_std (pd.Series or float): The standard deviation value(s) used for normalization.

    Returns:
    pd.DataFrame: The normalized DataFrame.
    """
    normalized_frame = (frame - precomputed_mean) / precomputed_std
    return normalized_frame

# for drawing bounding box
def find_min_max_coordinates(df):
    """
    Finds the minimum and maximum X and Y coordinates from a DataFrame containing body part coordinates.
    Args:
        df (pd.DataFrame): A DataFrame containing columns for X and Y coordinates of various body parts.
                           The DataFrame is assumed to contain a single row of data.
    Returns:
        tuple: A tuple containing four values:
            - min_x (float): The minimum X coordinate value.
            - min_y (float): The minimum Y coordinate value.
            - max_x (float): The maximum X coordinate value.
            - max_y (float): The maximum Y coordinate value.
    Note:
        The function replaces any -1 values in the DataFrame with NaN before calculating the min and max values.
        This is to ignore any invalid or missing coordinates represented by -1.
    """
    # List of X and Y coordinate columns
    x_columns = ['Nose_X', 'Left Shoulder_X', 'Right Shoulder_X', 'Left Elbow_X', 'Right Elbow_X',
                 'Left Wrist_X', 'Right Wrist_X', 'Left Hip_X', 'Right Hip_X', 'Left Knee_X',
                 'Right Knee_X', 'Left Ankle_X', 'Right Ankle_X']
                 
    y_columns = ['Nose_Y', 'Left Shoulder_Y', 'Right Shoulder_Y', 'Left Elbow_Y', 'Right Elbow_Y',
                 'Left Wrist_Y', 'Right Wrist_Y', 'Left Hip_Y', 'Right Hip_Y', 'Left Knee_Y',
                 'Right Knee_Y', 'Left Ankle_Y', 'Right Ankle_Y']
    
  # Assuming df contains a single row
    df_x_filtered = df[x_columns].replace(-1, np.nan)
    df_y_filtered = df[y_columns].replace(-1, np.nan)

    # Assuming df contains a single row
    min_x = df_x_filtered.iloc[0].min()  # Minimum X value ignoring zeros
    max_x = df_x_filtered.iloc[0].max()  # Maximum X value ignoring zeros

    min_y = df_y_filtered.iloc[0].min()  # Minimum Y value ignoring zeros
    max_y = df_y_filtered.iloc[0].max()  # Maximum Y value ignoring zeros

    print(f"Min X: {min_x}, Max X: {max_x}, Min Y: {min_y}, Max Y: {max_y}")
    
    return min_x, min_y, max_x, max_y

def draw_keypoints_on_frame(df, frame):
    """
    Draws keypoints on a given video frame based on coordinates from a dataframe.
    Parameters:
    df (pandas.DataFrame): DataFrame containing normalized keypoint coordinates.
                           The DataFrame should have columns for each keypoint in the format:
                           'Keypoint_X' and 'Keypoint_Y' (e.g., 'Nose_X', 'Nose_Y').
    frame (numpy.ndarray): The video frame on which keypoints will be drawn. 
                           This should be a 3-channel image (e.g., from OpenCV).
    Returns:
    numpy.ndarray: The video frame with keypoints drawn on it.
    Notes:
    - Keypoints are drawn as red circles with a radius proportional to the frame height.
    - Keypoint names are drawn as white text above each keypoint.
    - Coordinates (0, 0) are ignored as they may represent missing keypoints.
    """
    # Define the keypoint columns
    keypoint_columns = {
        'Nose': ('Nose_X', 'Nose_Y'),
        'Left Shoulder': ('Left Shoulder_X', 'Left Shoulder_Y'),
        'Right Shoulder': ('Right Shoulder_X', 'Right Shoulder_Y'),
        'Left Elbow': ('Left Elbow_X', 'Left Elbow_Y'),
        'Right Elbow': ('Right Elbow_X', 'Right Elbow_Y'),
        'Left Wrist': ('Left Wrist_X', 'Left Wrist_Y'),
        'Right Wrist': ('Right Wrist_X', 'Right Wrist_Y'),
        'Left Hip': ('Left Hip_X', 'Left Hip_Y'),
        'Right Hip': ('Right Hip_X', 'Right Hip_Y'),
        'Left Knee': ('Left Knee_X', 'Left Knee_Y'),
        'Right Knee': ('Right Knee_X', 'Right Knee_Y'),
        'Left Ankle': ('Left Ankle_X', 'Left Ankle_Y'),
        'Right Ankle': ('Right Ankle_X', 'Right Ankle_Y'),
    }

    # Get frame dimensions (width and height)
    frame_height, frame_width = frame.shape[:2]
    
    # Define circle size and font size relative to frame dimensions
    circle_radius = int(frame_height * 0.01)  # Circle radius as 1% of frame height
    font_scale = frame_height * 0.001  # Font scale relative to frame height
    font_thickness = max(1, int(frame_height * 0.002))  # Font thickness relative to frame height

    # Iterate through each keypoint
    for keypoint_name, (x_col, y_col) in keypoint_columns.items():
        # Extract the normalized X and Y coordinates from the dataframe
        normalized_x = df[x_col].iloc[0]
        normalized_y = df[y_col].iloc[0]
        
        # Ignore (0, 0) coordinates (which may represent missing keypoints)
        if normalized_x != 0 and normalized_y != 0:
            # Scale the normalized coordinates to the frame dimensions
            x = int(normalized_x * frame_width)
            y = int(normalized_y * frame_height)

            # Draw a circle for each keypoint with relative size
            cv2.circle(frame, (x, y), circle_radius, (0, 0, 255), -1)  # Red circles
            
            # Draw the text (keypoint name) with relative size
            cv2.putText(frame, keypoint_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return frame

def history_csv(row, history_output_file):
    """
    Appends the last row of a DataFrame to a CSV file.

    Parameters:
    row (pandas.DataFrame): The DataFrame containing the row to be appended.
    history_output_file (str): The file path of the CSV file to append the row to.

    Returns:
    None
    """
    with open(history_output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row.iloc[-1])
        
def initialize_log_output(history_csv_file, processed_output_csv):
    """
    Initializes log output by creating necessary CSV files if they do not exist.
    This function creates a 'logs' directory if it does not already exist. It then checks for the existence of two CSV files:
    `history_csv_file` and `processed_output_csv`. If these files do not exist, it creates them and writes the appropriate headers.
    Parameters:
    history_csv_file (str): The file path for the history CSV file.
    processed_output_csv (str): The file path for the processed output CSV file.
    Returns:
    None
    """
    
    # Specify the path of the folder you want to create
    folder_path = 'logs'

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    if not os.path.isfile(history_csv_file):
        with open(history_csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
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
            ])
            
    if not os.path.isfile(processed_output_csv):
        with open(processed_output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Nose_X', 'Nose_Y', 'Left Shoulder_X', 'Left Shoulder_Y',
        'Right Shoulder_X', 'Right Shoulder_Y', 'Left Elbow_X', 'Left Elbow_Y',
        'Right Elbow_X', 'Right Elbow_Y', 'Left Wrist_X', 'Left Wrist_Y',
        'Right Wrist_X', 'Right Wrist_Y', 'Left Hip_X', 'Left Hip_Y',
        'Right Hip_X', 'Right Hip_Y', 'Left Knee_X', 'Left Knee_Y',
        'Right Knee_X', 'Right Knee_Y', 'Left Ankle_X', 'Left Ankle_Y',
        'Right Ankle_X', 'Right Ankle_Y', 'Shoulder_Angle',
        'Left_Torso_Incline_Angle', 'Right_Torso_Incline_Angle',
        'Left_Elbow_Angle', 'Right_Elbow_Angle', 'Left_Hip_Knee_Angle',
        'Right_Hip_Knee_Angle', 'Left_Knee_Ankle_Angle',
        'Right_Knee_Ankle_Angle', 'Head_to_Shoulders_Angle',
        'Head_to_Hips_Angle', 'Nose_X_acceleration', 'Nose_Y_acceleration',
        'Left Shoulder_X_acceleration', 'Left Shoulder_Y_acceleration',
        'Right Shoulder_X_acceleration', 'Right Shoulder_Y_acceleration',
        'Left Elbow_X_acceleration', 'Left Elbow_Y_acceleration',
        'Right Elbow_X_acceleration', 'Right Elbow_Y_acceleration',
        'Left Wrist_X_acceleration', 'Left Wrist_Y_acceleration',
        'Right Wrist_X_acceleration', 'Right Wrist_Y_acceleration',
        'Left Hip_X_acceleration', 'Left Hip_Y_acceleration',
        'Right Hip_X_acceleration', 'Right Hip_Y_acceleration',
        'Left Knee_X_acceleration', 'Left Knee_Y_acceleration',
        'Right Knee_X_acceleration', 'Right Knee_Y_acceleration',
        'Left Ankle_X_acceleration', 'Left Ankle_Y_acceleration',
        'Right Ankle_X_acceleration', 'Right Ankle_Y_acceleration'
            ])
            
def f1_score(y_true, y_pred):
    """
    Computes the F1 score, which is the harmonic mean of precision and recall.
    Args:
        y_true (tensor): Ground truth binary labels.
        y_pred (tensor): Predicted binary labels.
    Returns:
        tensor: F1 score.
    Raises:
        AssertionError: If y_true and y_pred do not have the same shape.
    Notes:
        - The function assumes that y_true and y_pred are tensors.
        - The function clips the values of y_true and y_pred to be between 0 and 1.
        - The function uses Keras backend functions to perform the calculations.
    """
    # Ensure y_true and y_pred are of the same shape
    print(f'y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}')  # Debugging line
    y_pred = K.squeeze(y_pred, axis=-1)  # Remove the last dimension if it's 1
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    
    # Convert to float32
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(y_pred, K.floatx())

    # Calculate True Positives, False Positives, and False Negatives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))      # TP + FP
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))        # TP + FN

    # Calculate precision and recall
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    # Calculate F1 Score
    return 2 * (precision * recall) / (precision + recall + K.epsilon())