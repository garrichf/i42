import pandas as pd
import numpy as np
import process_data_functions
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import cv2

columns_to_remove = [
'Left Eye_Y', 'Left Eye_X', 
'Right Eye_Y', 'Right Eye_X', 
'Left Ear_Y', 'Left Ear_X', 
'Right Ear_Y', 'Right Ear_X'
]

keypoint_columns = [
    'Nose_X', 'Nose_Y', 
    'Left Shoulder_X', 'Left Shoulder_Y', 'Right Shoulder_X', 'Right Shoulder_Y',
    'Left Elbow_X', 'Left Elbow_Y', 'Right Elbow_X', 'Right Elbow_Y',
    'Left Wrist_X', 'Left Wrist_Y', 'Right Wrist_X', 'Right Wrist_Y',
    'Left Hip_X', 'Left Hip_Y', 'Right Hip_X','Right Hip_Y',
    'Left Knee_X', 'Left Knee_Y', 'Right Knee_X', 'Right Knee_Y',
    'Left Ankle_X', 'Left Ankle_Y', 'Right Ankle_X', 'Right Ankle_Y'
]

min_max_columns = [
    'Shoulder_Angle', 'Left_Torso_Incline_Angle', 
    'Right_Torso_Incline_Angle', 'Left_Elbow_Angle', 'Right_Elbow_Angle', 
    'Left_Hip_Knee_Angle', 'Right_Hip_Knee_Angle', 'Left_Knee_Ankle_Angle', 
    'Right_Knee_Ankle_Angle', 'Head_to_Shoulders_Angle', 
    'Head_to_Hips_Angle'
]

# Columns for Z-Score normalization
z_score_columns = [
    'Nose_X_acceleration', 'Nose_Y_acceleration', 
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
]

min_val = 0
max_val = 180

loaded_mean = np.load('precomputed_mean.npy')
loaded_std = np.load('precomputed_std.npy')
# iqr_minmax = 

# for drawing bounding box
def find_min_max_coordinates(df):
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

            # Draw a circle for each keypoint
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red circles
            cv2.putText(frame, keypoint_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return frame

def process_data(keypoints_df, index, history_csv):
    history_df = pd.read_csv(history_csv)
    if index > 0:
        keypoints_df = keypoints_df.drop(columns=columns_to_remove)
        df = process_data_functions.add_angles(keypoints_df)
        df = process_data_functions.calculate_acceleration(df, history_df, keypoint_columns, current_index=index)
        # print("CURRENT DF")
        # print(df)
        # for col in min_max_columns:
        #     df = process_data_functions.remove_outliers_with_fixed_iqr(df, col, lower_bound_minmax, upper_bound_minmax)
        # for col in z_score_columns:
        #     df = process_data_functions.remove_outliers_with_fixed_iqr(df, col, lower_bound_minmax, upper_bound_minmax)
        # # Append the new row (last row) to the CSV without writing the header again
        for col in min_max_columns:
            df[col] = process_data_functions.min_max_scale_fixed_range(df[col], min_val, max_val)

        # print("Before")
        # print(df[z_score_columns])
        # for col, mean, std in zip(z_score_columns, loaded_mean, loaded_std):
        #     # Add epsilon to prevent division by zero if std is zero
        #     epsilon = 1e-10
        #     non_zero_mask = df[col] != 0
        #     df.loc[non_zero_mask, col] = (df.loc[non_zero_mask, col] - mean) / (std + epsilon)

        # print("After")
        # print(df[z_score_columns])

        df.iloc[-1:].to_csv(history_csv, mode='a', header=False, index=False)
  
    else:
        keypoints_df = keypoints_df.drop(columns=columns_to_remove)
        df = process_data_functions.add_angles(keypoints_df)
        df = process_data_functions.calculate_acceleration(df, history_df, keypoint_columns, current_index=index)
        # for col in min_max_columns:
        #     df = process_data_functions.remove_outliers_with_fixed_iqr(df, col, lower_bound_zscore, upper_bound_zscore)
        # for col in z_score_columns:
        #     df = process_data_functions.remove_outliers_with_fixed_iqr(df, col, lower_bound_zscore, upper_bound_zscore)
        # # Apply the custom scaling for each column in min_max_columns
        for col in min_max_columns:
            df[col] = process_data_functions.min_max_scale_fixed_range(df[col], min_val, max_val)
            
        # assert loaded_mean.shape[0] == len(z_score_columns), "Shape mismatch: mean and z-score columns length do not match"
        # assert loaded_std.shape[0] == len(z_score_columns), "Shape mismatch: std and z-score columns length do not match"

        # print("Before")
        # print(df[z_score_columns])
        # for col, mean, std in zip(z_score_columns, loaded_mean, loaded_std):
        #     # Add epsilon to prevent division by zero if std is zero
        #     epsilon = 1e-10
        #    # Apply z-score normalization only to non-zero elements
        #     non_zero_mask = df[col] != 0
        #     df.loc[non_zero_mask, col] = (df.loc[non_zero_mask, col] - mean) / (std + epsilon)

        # print("After")
        # print(df[z_score_columns])
        df.to_csv(history_csv, index=False)
        
    columns_to_drop = [col for col in df.columns if 'velocity' in col]
    # Drop the identified columns from the DataFrame
    df.drop(columns=columns_to_drop, inplace=True)
    return df