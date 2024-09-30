from ultralytics import YOLO
import sys
import cv2
import pandas as pd
import csv
import os
import time
# Load YOLO model
model = YOLO('yolov8x-pose.pt')

# Read CSV file
df = pd.read_csv('found_files_testing.csv')

# Output CSV file
output_csv_file = 'output_test_set.csv'



# Check if output CSV file exists, if not, create it with header
if not os.path.isfile(output_csv_file):
    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Folder Name', 'File Name','FallType', 
            'Nose_X', 'Nose_Y',
            'Left Eye_X', 'Left Eye_Y',
            'Right Eye_X', 'Right Eye_Y',
            'Left Ear_X', 'Left Ear_Y',
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
        ])

# Check if output CSV file exists and load processed data if it does
if os.path.isfile(output_csv_file):
    processed_df = pd.read_csv(output_csv_file)
    processed_files = set(zip(processed_df['Folder Name'], processed_df['File Name']))
else:
    processed_files = set()

# def get_fall_type(file_path):
#     # Extract the folder name from the file path
#     folder_name = os.path.basename(os.path.dirname(file_path))
    
#     # Check if the folder name contains "nfall" or "fall"
#     if folder_name.startswith("nfall"):
#         return 0
#     elif folder_name.startswith("fall"):
#         return 1
#     else:
#         return None  # Or some other default value or action

def estimate_time_to_completion(start_time, current_row, total_rows):
    elapsed_time = time.time() - start_time
    average_time_per_row = elapsed_time / current_row
    remaining_rows = total_rows - current_row
    estimated_time_remaining = average_time_per_row * remaining_rows
    remaining_minutes, remaining_seconds = divmod(int(estimated_time_remaining), 60)
    return remaining_minutes, remaining_seconds

def process_frame(frame, folder_name, file_name, fall_type):
    kpt_table = {}
    results = model.predict(frame, conf=0.3)
    keypoints = results[0].keypoints.xyn  # Normalized xy values for model training
    keypoint_xy = results[0].keypoints.xy  # For display

    # print("Keypoints extracted:", keypoints)  # Debugging line
    # print("Keypoints XY:", keypoint_xy)  # Debugging line

    # Retrieving keypoint data
    for i, point in enumerate(keypoints[0]):
        xn = point[0].item()
        yn = point[1].item()
        kpt_table[i] = [xn, yn]

    # print("Keypoints table:", kpt_table)  # Debugging line

    # Check if kpt_table contains expected keys
    for i in range(17):  # Assuming 17 keypoints based in the dataset
        if i not in kpt_table:
            print(f"Key {i} is missing in kpt_table")

    # Write keypoints to CSV
    with open(output_csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([folder_name
                        , file_name, fall_type
                        , kpt_table.get(0, [None, None])[0], kpt_table.get(0, [None, None])[1]
                        , kpt_table.get(1, [None, None])[0], kpt_table.get(1, [None, None])[1]
                        , kpt_table.get(2, [None, None])[0], kpt_table.get(2, [None, None])[1]
                        , kpt_table.get(3, [None, None])[0], kpt_table.get(3, [None, None])[1]
                        , kpt_table.get(4, [None, None])[0], kpt_table.get(4, [None, None])[1]
                        , kpt_table.get(5, [None, None])[0], kpt_table.get(5, [None, None])[1]
                        , kpt_table.get(6, [None, None])[0], kpt_table.get(6, [None, None])[1]
                        , kpt_table.get(7, [None, None])[0], kpt_table.get(7, [None, None])[1]
                        , kpt_table.get(8, [None, None])[0], kpt_table.get(8, [None, None])[1]
                        , kpt_table.get(9, [None, None])[0], kpt_table.get(9, [None, None])[1]
                        , kpt_table.get(10, [None, None])[0], kpt_table.get(10, [None, None])[1]
                        , kpt_table.get(11, [None, None])[0], kpt_table.get(11, [None, None])[1]
                        , kpt_table.get(12, [None, None])[0], kpt_table.get(12, [None, None])[1]
                        , kpt_table.get(13, [None, None])[0], kpt_table.get(13, [None, None])[1]
                        , kpt_table.get(14, [None, None])[0], kpt_table.get(14, [None, None])[1]
                        , kpt_table.get(15, [None, None])[0], kpt_table.get(15, [None, None])[1]
                        , kpt_table.get(16, [None, None])[0], kpt_table.get(16, [None, None])[1]
                        ])

    # Draw circles on keypoints
    for i, point in enumerate(keypoint_xy[0]):
        x, y = int(point[0].item()), int(point[1].item())
        label = f"{i}: {x}, {y}"
        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        cv2.putText(frame, fall_type, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    return frame

index_row = 0
dataset_size = len(df)
start_time = time.time()

for index, row in df.iterrows():
    index_row += 1
    # print(str(index_row)+" / "+ str(dataset_size))
    file_path = row['File Path']
    folder_name = row['Folder Name']
    file_name = row['File Name']
    file_name_without_extension = os.path.splitext(file_name)[0]
    fall_type = row['Category']

     # Check if the file has already been processed
    if (folder_name, file_name) in processed_files:
        print(f"Skipping already processed file: {file_name} in folder: {folder_name}")
        continue

     # Estimate and print time to completion every 10 rows
    remaining_minutes, remaining_seconds = estimate_time_to_completion(start_time, index_row, dataset_size)
    print(f"Processed {index_row}/{dataset_size} rows. Estimated Remaining Time: {remaining_minutes}m {remaining_seconds}s")

    # Process single image frame
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Error reading image file {file_path}")
            continue

        processed_frame = process_frame(frame, folder_name, file_name, fall_type)
       # Specify the folder where you want to save the file
        folder_path = "annotated_test/" + folder_name

        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Construct the full path for the output file
        output_file_path = os.path.join(folder_path, f"annotated_{file_name_without_extension}.jpg")
        cv2.imwrite(output_file_path, processed_frame)
        # This is just for live viewing
        cv2.imwrite("results.jpg",processed_frame) 
    
    # Process video file
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error reading video file {file_path}")
            continue

        counter = 1
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time_video = time.time()
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print(f"Failed to read frame or end of video reached for {file_path}")
                break
            # Estimate and print time to completion every 10 rows
            remaining_minutes_video, remaining_seconds_video = estimate_time_to_completion(start_time_video, counter, frame_count)
            print(f"{folder_name} : Processed {counter}/{frame_count} Frames. Estimated Remaining Time: {remaining_minutes_video}m {remaining_seconds_video}s")
            frame_name = f"annotated_Frame{counter}.jpg"
            processed_frame = process_frame(frame, folder_name, frame_name, fall_type)
            cv2.imshow("Image with Keypoints", processed_frame)
            # Specify the folder where you want to save the file
            folder_path = "annotated_test/" + folder_name

            # Ensure the folder exists
            os.makedirs(folder_path, exist_ok=True)

            # Construct the full path for the output file
            output_file_path = os.path.join(folder_path, frame_name)
            cv2.imwrite(output_file_path, processed_frame)
            counter += 1

        cap.release()

    else:
        print(f"Unsupported file type: {file_path}")

    # Handle key press to exit
    cv2.destroyAllWindows()

print("Processing complete. All windows closed.")
