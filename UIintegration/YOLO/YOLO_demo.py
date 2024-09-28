import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Set the path to your input video
vid_path = r"C:\Users\User\Desktop\m\ADL.mp4"  # Use raw string to handle backslashes

# Function to perform pose estimation on a video
def YOLO_detect_pose_sequence(vid_path):
    
    # Load the YOLOv8 pose model
    try:
        model = YOLO('yolov8n-pose.pt')  # Load the YOLOv8 pose model
        print("YOLOv8 Pose Estimation Model loaded successfully.")
    except Exception as error:
        print(f"Failed to load the model: {error}")

    # Load the video
    vid = cv2.VideoCapture(vid_path)
    frames = []
    
    # Read frames from the video
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frames.append(frame)
    
    # Initialize an empty list to store keypoints for each frame
    all_keypoints = []
    
    # Iterate through each frame
    for frame in frames:
        # Perform inference
        results = model(frame)  # Predict keypoints on the current frame
        
        # Process results for keypoints
        if results:  # Check if results are not empty
            for result in results:  # Iterate through each result (if multiple persons detected)
                if hasattr(result, 'keypoints') and result.keypoints is not None:  # Check if keypoints exist
                    
                    keypoints = result.keypoints.numpy()  # Get keypoints for detected persons
                    
                    # Create a dictionary to store keypoints for the current frame
                    frame_keypoints = {}
                    
                    for idx, keypoint in enumerate(keypoints):
                        if len(keypoint) == 3:  # Ensure we have x, y, and confidence values
                            x, y, conf = keypoint  # x, y coordinates and confidence score
                            frame_keypoints[f'Keypoint_{idx}_X'] = x
                            frame_keypoints[f'Keypoint_{idx}_Y'] = y
                            frame_keypoints[f'Keypoint_{idx}_Confidence'] = conf
                        else:
                            print(f"Unexpected keypoint format: {keypoint}")
                    
                    all_keypoints.append(frame_keypoints)
                else:
                    print("No keypoints detected.")
                    all_keypoints.append({f'Keypoint_{i}_X': None for i in range(17)})
                    all_keypoints.append({f'Keypoint_{i}_Y': None for i in range(17)})
                    all_keypoints.append({f'Keypoint_{i}_Confidence': None for i in range(17)})
        else:
            print("No results returned from model.")
            all_keypoints.append({f'Keypoint_{i}_X': None for i in range(17)})
            all_keypoints.append({f'Keypoint_{i}_Y': None for i in range(17)})
            all_keypoints.append({f'Keypoint_{i}_Confidence': None for i in range(17)})

    df = pd.DataFrame(all_keypoints)
    return df

# Perform pose estimation on the input video
result = YOLO_detect_pose_sequence(vid_path)

# Optionally, print or save the result DataFrame for inspection.
print(result.head())  # Display the first few rows of the DataFrame.