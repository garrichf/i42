import pandas as pd
import numpy as np
import process_data_functions
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import cv2


# Columns to remove from the DataFrame
columns_to_remove = [
'Left Eye_Y', 'Left Eye_X', 
'Right Eye_Y', 'Right Eye_X', 
'Left Ear_Y', 'Left Ear_X', 
'Right Ear_Y', 'Right Ear_X'
]

# Columns for Keypoints
keypoint_columns = [
    'Nose_X', 'Nose_Y', 
    'Left Shoulder_X', 'Left Shoulder_Y', 'Right Shoulder_X', 'Right Shoulder_Y',
    'Left Elbow_X', 'Left Elbow_Y', 'Right Elbow_X', 'Right Elbow_Y',
    'Left Wrist_X', 'Left Wrist_Y', 'Right Wrist_X', 'Right Wrist_Y',
    'Left Hip_X', 'Left Hip_Y', 'Right Hip_X','Right Hip_Y',
    'Left Knee_X', 'Left Knee_Y', 'Right Knee_X', 'Right Knee_Y',
    'Left Ankle_X', 'Left Ankle_Y', 'Right Ankle_X', 'Right Ankle_Y'
]

# Columns for Min-Max normalization
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

loaded_mean = np.load('constants/precomputed_mean.npy')
loaded_std = np.load('constants/precomputed_std.npy')

def process_data(keypoints_df, index, log_csv_filepath):
    """
    Processes the keypoints DataFrame by adding angles, calculating acceleration, 
    and applying scaling transformations. The processed data is then appended to 
    or saved in a CSV file.
    Args:
        keypoints_df (pd.DataFrame): DataFrame containing keypoints data.
        index (int): Index to determine the processing flow. If greater than 0, 
                     the processed data is appended to the CSV file; otherwise, 
                     it is saved as a new CSV file.
        log_csv_filepath (str): Filepath to the CSV file where the processed data 
                                will be saved or appended.
    Returns:
        pd.DataFrame: The processed DataFrame with angles, acceleration, and 
                      scaling transformations applied.
    """
    log_df = pd.read_csv(log_csv_filepath)
    if index > 0:
        keypoints_df = keypoints_df.drop(columns=columns_to_remove)
        df = process_data_functions.add_angles(keypoints_df)
        df = process_data_functions.calculate_acceleration(df, log_df, keypoint_columns, current_index=index)
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

        df.iloc[-1:].to_csv(log_csv_filepath, mode='a', header=False, index=False)
  
    else:
        keypoints_df = keypoints_df.drop(columns=columns_to_remove)
        df = process_data_functions.add_angles(keypoints_df)
        df = process_data_functions.calculate_acceleration(df, log_df, keypoint_columns, current_index=index)
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
        df.to_csv(log_csv_filepath, index=False)
        
    columns_to_drop = [col for col in df.columns if 'velocity' in col]
    # Drop the identified columns from the DataFrame
    df.drop(columns=columns_to_drop, inplace=True)
    return df