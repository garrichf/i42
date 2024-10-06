import cv2
import pandas as pd
import numpy as np
import YOLO
import os
import csv
from datetime import datetime
import process_data
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def f1_score(y_true, y_pred):
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

file_path = "ADL.mp4"
history_csv_file = f"{datetime.now().strftime('%d%m%Y_%H%M')}_LOG.csv"
processed_output_csv = f"{datetime.now().strftime('%d%m%Y_%H%M')}_OUTPUT.csv"

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
        
def history_csv(row, history_output_file):
    with open(history_output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row.iloc[-1])
        
index = 0
confidence_threshold = 0.5
# Open the video file
cap = cv2.VideoCapture(file_path)
model = load_model('falldetect_main.keras', custom_objects={'f1_score': f1_score})

if not cap.isOpened():
    print(f"Error reading video file {file_path}")
    exit()  # Exit if the video can't be opened

# Buffer to hold the last 30 frames of processed data
frame_buffer = []
sequence_length = 30  # Number of frames to collect before prediction
predictions_class = 0 # No Fall by default
fall_detected_buffer = 99 
fall_counter = 0
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
box_color = (255,255,255)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print(f"Failed to read frame or end of video reached for {file_path}")
        break
    # if index > 80:
    #     break
    keypoints = YOLO.YOLO_pose(frame)
    # Retain a history of running and the data
    # history_csv(keypoints, history_csv_file)
    processed_df = process_data.process_data(keypoints, index, history_csv_file)
    history_csv(processed_df, processed_output_csv)
    # Replace 0 and NaN values with -1
    processed_df = processed_df.replace(0, -1).fillna(-1)
    # Scale the normalized coordinates back to frame size
    frame_with_keypoints = process_data.draw_keypoints_on_frame(processed_df, frame)
    min_x, min_y, max_x, max_y = process_data.find_min_max_coordinates(processed_df)
    # print(processed_df)
    frame_buffer.append(processed_df)
    # Assuming your input data for prediction is a numpy array or similar
    # If fall is detected, increment fall_detected_buffer and continue
    if (predictions_class) and fall_detected_buffer < 30:
        fall_detected_buffer += 1
        frame_buffer.pop(0)
        # Draw text on the frame to indicate fall has been detected
        text = "Fall Detected (Buffering)"
        color = (0, 0, 255)  # Red color to indicate fall detected
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    else:
        # Make a prediction once enough frames are collected
        if len(frame_buffer) == sequence_length:
            data_array = np.vstack(frame_buffer).astype(np.float32).reshape(1, sequence_length, 63)
            predictions = model.predict(data_array)
            fall_probability = predictions[0][0]  # Get the probability of a fall
            print(fall_probability)
            predictions_class = int(fall_probability > confidence_threshold)  # Convert to binary (0 or 1)
            print(predictions_class)
            print("Frame Number: " + str(index))
            if(predictions_class):
                print("\033[91mFall DETECTED\033[0m")  # Red text
            # Reset the fall detection buffer counter if a fall is detected
            if predictions_class:
                fall_detected_buffer = 0
            # Slide the buffer by removing the oldest frame
            frame_buffer.pop(0)
            
            # Draw text on the frame
            text = "Fall" if predictions_class else "No Fall"
            box_color = (0,0,255) if predictions_class else (0,255,0)
            if(predictions_class):
                fall_detected_buffer = 0
                fall_counter += 1
            color = (0, 0, 255) if predictions_class else (255, 255, 255)  # Red for Fall, White for No Fall
            cv2.putText(frame_with_keypoints, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame_with_keypoints, str(predictions_class), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_with_keypoints, "Starting Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Display the current frame
    # Draw the bounding box around the detected keypoints
    
    if not (np.isnan(min_x) or np.isnan(min_y) or np.isnan(max_x) or np.isnan(max_y)):
        # Scale the min and max values to the frame dimensions
        min_x_scaled = int(min_x * frame_width)
        max_x_scaled = int(max_x * frame_width)
        min_y_scaled = int(min_y * frame_height)
        max_y_scaled = int(max_y * frame_height)

        # Draw the bounding box
        cv2.rectangle(frame, (min_x_scaled, min_y_scaled), (max_x_scaled, max_y_scaled), (0, 255, 0), 2)
        print("Drawing bounding box")
    else:
        # If any of the values are NaN, skip drawing the bounding box
        print("All values are NaN, not drawing bounding box.")
    cv2.putText(frame_with_keypoints, "Frame: " + str(index), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Display Window", frame_with_keypoints)
    
    # Handle key press and close if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    index += 1

# Release the video capture object
cap.release()
print("Number of Fall Detected: " + str(fall_counter))
# Close all OpenCV windows
cv2.destroyAllWindows()
