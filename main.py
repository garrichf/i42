import cv2
import pandas as pd
import numpy as np
import YOLO
import MEDIAPIPE
import MOVENET
import SETTINGS
import os
import csv
import process_data
from tensorflow.keras.models import load_model


log_csv_filepath = SETTINGS.LOG_FILEPATH
processed_output_csv = SETTINGS.PROCESSED_OUTPUT_CSV
# Initialize Files for Log Files
process_data.process_data_functions.initialize_log_output(log_csv_filepath, processed_output_csv)


file_path = "video/Footage6_CAUCAFDD_Subject_1_Fall_1.mp4"
index = 0
is_demo_mode = SETTINGS.DEMO_MODE
confidence_threshold = SETTINGS.CONFIDENCE_THRESHOLD
pose_model_used = SETTINGS.POSE_MODEL_USED
frame_buffer = []
sequence_length = 30  # Number of frames to collect before prediction
predictions_class = 0 # initializing No Fall by default
fall_detected_buffer = 99  # initializing a buffers
fall_counter = 0 # counts the number of times fall is detected within a runtime
box_color = (255,255,255) # Initialize bounding box colour to white during set up phase

if is_demo_mode:
    # Open the video file
    cap = cv2.VideoCapture(file_path)
else:
    # Turns on Local Webcam
    cap = cv2.VideoCapture(0)
    
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

model = load_model('falldetect_main.keras', custom_objects={'f1_score': process_data.process_data_functions.f1_score})

if not cap.isOpened():
    print(f"Error reading video file {file_path}")
    exit()  # Exit if the video can't be opened

# Buffer to hold the last 30 frames of processed data
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print(f"Failed to read frame or end of video reached for {file_path}")
        break

    print("Frame Number: " + str(index))
    if pose_model_used == "YOLO":
        keypoints = YOLO.YOLO_pose(frame)
    if pose_model_used == "MEDIAPIPE":
        keypoints = MEDIAPIPE.MEDIAPIPE_pose(frame)
    if pose_model_used == "MOVENET":
        keypoints = MOVENET.MOVENET_pose(frame)
        
    print(keypoints)
    print(keypoints.shape)
    print("Exit Keypoint Dataframe")
    # Retain a history of running and the data
    processed_df = process_data.process_data(keypoints, index, log_csv_filepath)
    # process_data.process_data_functions.history_csv(processed_df, processed_output_csv)
    
    # Replace 0 and NaN values with -1
    processed_df = processed_df.replace(0, -1).fillna(-1)
    
    # Scale the normalized coordinates back to frame size
    frame_with_keypoints = process_data.process_data_functions.draw_keypoints_on_frame(processed_df, frame)
    min_x, min_y, max_x, max_y = process_data.process_data_functions.find_min_max_coordinates(processed_df)
    frame_buffer.append(processed_df)

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
            predictions_class = int(fall_probability > confidence_threshold)  # Convert to binary (0 or 1)
            
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