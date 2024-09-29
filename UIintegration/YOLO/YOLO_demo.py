import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def YOLOv8_detect_and_display_pose(vid_path):
    # Load the YOLOv8 model
    try:
        model = YOLO('yolov8n-pose.pt')
        print("YOLOv8 Pose Estimation Model loaded successfully.")
    except Exception as error:
        print(f"Failed to load the model: {error}")
        return None

    # Set a threshold for average confidence
    POSE_CONFIDENCE_THRESHOLD = 0.25
    
    # Load the video
    vid = cv2.VideoCapture(vid_path)
    
    if not vid.isOpened():
        print("Error: Could not open video.")
        return None

    # Get video properties
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # Initialize an empty list to store keypoints for each frame
    all_keypoints = []

    # Keypoint names based on the YOLOv8 model
    keypoint_names = [
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
    ]

    # Define the pairs of keypoints to connect with edges
    keypoint_edges = [
        ('Nose', 'Left Eye'), ('Nose', 'Right Eye'), ('Left Eye', 'Left Ear'), ('Right Eye', 'Right Ear'),
        ('Nose', 'Left Shoulder'), ('Nose', 'Right Shoulder'),
        ('Left Shoulder', 'Right Shoulder'), ('Left Shoulder', 'Left Elbow'), ('Right Shoulder', 'Right Elbow'),
        ('Left Elbow', 'Left Wrist'), ('Right Elbow', 'Right Wrist'), ('Left Shoulder', 'Left Hip'),
        ('Right Shoulder', 'Right Hip'), ('Left Hip', 'Right Hip'), ('Left Hip', 'Left Knee'),
        ('Right Hip', 'Right Knee'), ('Left Knee', 'Left Ankle'), ('Right Knee', 'Right Ankle')
    ]

    while True:
        ret, frame = vid.read()
        if not ret:
            break


        # Perform inference
        results = model(frame)[0]
        
        # Initialize a dictionary to store keypoints for the current frame
        frame_keypoints = {}

        # Extract keypoints if a person is detected
        if len(results.keypoints) > 0:
            keypoints = results.keypoints[0].data[0]  # Get keypoints of the first person

            average_confidence = keypoints[:, 2].mean().item()

            if average_confidence >= POSE_CONFIDENCE_THRESHOLD:
                for idx, keypoint in enumerate(keypoints):
                    x, y, conf = keypoint
                    if conf >= POSE_CONFIDENCE_THRESHOLD:
                        frame_keypoints[f'{keypoint_names[idx]}_X'] = x.item() / frame.shape[1]  # Normalize to 0-1
                        frame_keypoints[f'{keypoint_names[idx]}_Y'] = y.item() / frame.shape[0]  # Normalize to 0-1
                        # Draw keypoints on the frame
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)


                # Draw edges connecting the keypoints
                for edge in keypoint_edges:
                    if (f'{edge[0]}_X' in frame_keypoints and f'{edge[0]}_Y' in frame_keypoints and
                        f'{edge[1]}_X' in frame_keypoints and f'{edge[1]}_Y' in frame_keypoints):
                        pt1 = (int(frame_keypoints[f'{edge[0]}_X'] * frame.shape[1]), int(frame_keypoints[f'{edge[0]}_Y'] * frame.shape[0]))
                        pt2 = (int(frame_keypoints[f'{edge[1]}_X'] * frame.shape[1]), int(frame_keypoints[f'{edge[1]}_Y'] * frame.shape[0]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Append the frame keypoints to the all_keypoints list
        all_keypoints.append(frame_keypoints)

        # Display the frame
        cv2.imshow('YOLOv8 Pose Estimation', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the video capture object and close all OpenCV windows
    vid.release()
    cv2.destroyAllWindows()

    # Convert  keypoints to a DataFrame
    df = pd.DataFrame(all_keypoints)
    return df

# Example usage:
result = YOLOv8_detect_and_display_pose("ADL.mp4")
print(result)  # This will be print the DataFrame with all the keypoints