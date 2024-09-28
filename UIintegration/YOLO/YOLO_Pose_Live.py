import torch
import numpy as np
import cv2
from ultralytics import YOLO

def YOLO_detect_pose_live():
    
    # I’ll start by trying to load the YOLO pose estimation model.
    try:
        model = YOLO('yolov8n-pose.pt')
        print("YOLO Pose Estimation Model loaded successfully.")
    except Exception as error:
        # If loading the model fails, I’ll print the error and exit the function.
        print(f"Failed to load the model: {error}")
        return

    POSE_CONFIDENCE_THRESHOLD = 0.25  # I set a threshold for pose confidence.
    vid = cv2.VideoCapture(0)  # I’m opening the video stream from the default camera.
    
    if not vid.isOpened():
        # If I can't open the video stream, I'll print an error message and exit.
        print("Error: Could not open video stream.")
        return
    
    # Here are the names of the keypoints I'm interested in for pose estimation.
    keypoint_names = [
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
    ]
    
    # These are the connections (edges) between keypoints that I want to visualize.
    keypoint_edges = [
        ('Nose', 'Left Eye'), ('Nose', 'Right Eye'), ('Left Eye', 'Left Ear'), 
        ('Right Eye', 'Right Ear'), ('Nose', 'Left Shoulder'), ('Nose', 'Right Shoulder'),
        ('Left Shoulder', 'Right Shoulder'), ('Left Shoulder', 'Left Elbow'), 
        ('Right Shoulder', 'Right Elbow'), ('Left Elbow', 'Left Wrist'), 
        ('Right Elbow', 'Right Wrist'), ('Left Shoulder', 'Left Hip'),
        ('Right Shoulder', 'Right Hip'), ('Left Hip', 'Right Hip'), 
        ('Left Hip', 'Left Knee'), ('Right Hip', 'Right Knee'), 
        ('Left Knee', 'Left Ankle'), ('Right Knee', 'Right Ankle')
    ]
    
    # This function checks if two keypoints are valid based on their positions.
    def is_valid_keypoint(kp1, kp2, axis='y', threshold=0.1):
        if kp1 in frame_keypoints and kp2 in frame_keypoints:
            if axis == 'y':
                return abs(frame_keypoints[f'{kp1}_Y'] - frame_keypoints[f'{kp2}_Y']) > threshold
            elif axis == 'x':
                return abs(frame_keypoints[f'{kp1}_X'] - frame_keypoints[f'{kp2}_X']) > threshold
        return True

    # This function validates all keypoints and removes any that are invalid.
    def validate_all_keypoints():
        invalid_keypoints = []
        keypoint_pairs = [
            ('Left Ankle', 'Left Knee'), ('Right Ankle', 'Right Knee'),
            ('Left Knee', 'Left Hip'), ('Right Knee', 'Right Hip'),
            ('Left Hip', 'Right Hip'), ('Left Shoulder', 'Right Shoulder'),
            ('Left Elbow', 'Left Shoulder'), ('Right Elbow', 'Right Shoulder'),
            ('Left Wrist', 'Left Elbow'), ('Right Wrist', 'Right Elbow')
        ]
        
        for kp1, kp2 in keypoint_pairs:
            if not is_valid_keypoint(kp1, kp2):
                invalid_keypoints.append(kp1)
                invalid_keypoints.append(kp2)
        
        # I’ll remove invalid keypoints from frame_keypoints.
        for kp in set(invalid_keypoints):
            frame_keypoints.pop(f'{kp}_X', None)
            frame_keypoints.pop(f'{kp}_Y', None)

    while True:
        ret, frame = vid.read()  # I read a frame from the video stream.
        
        if not ret:
            # If capturing the image fails, I’ll print an error message and break the loop.
            print("Error: Failed to capture image")
            break
        
        results = model(frame)  # I run the YOLO model on the captured frame.

        if hasattr(results[0], 'keypoints'):
            keypoints = results[0].keypoints.cpu().numpy()  # Extracting keypoints from results.
            if len(keypoints) < 17:  # I want to make sure there are enough keypoints detected.
                print(f"Warning: Expected at least 17 keypoints but got {len(keypoints)}")
                continue
        else:
            print("No keypoints found in results.")  # If no keypoints were found, I’ll skip this iteration.
            continue
        
        frame_keypoints = {}  # This will store the valid keypoint coordinates for the current frame.
        
        for idx in range(min(len(keypoints), 17)):  # Process up to 17 keypoints safely.
            x, y, confidence = keypoints[idx][0], keypoints[idx][1], keypoints[idx][2]
            if confidence > POSE_CONFIDENCE_THRESHOLD:  # Only consider high-confidence keypoints.
                frame_keypoints[f'{keypoint_names[idx]}_X'] = x / frame.shape[1]  # Normalizing X coordinate.
                frame_keypoints[f'{keypoint_names[idx]}_Y'] = y / frame.shape[0]  # Normalizing Y coordinate.
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Drawing a circle around each valid keypoint.

        validate_all_keypoints()  # I validate all detected keypoints.

        for edge in keypoint_edges:
            if (f'{edge[0]}_X' in frame_keypoints and f'{edge[0]}_Y' in frame_keypoints and
                f'{edge[1]}_X' in frame_keypoints and f'{edge[1]}_Y' in frame_keypoints):
                pt1 = (int(frame_keypoints[f'{edge[0]}_X'] * frame.shape[1]), int(frame_keypoints[f'{edge[0]}_Y'] * frame.shape[0]))
                pt2 = (int(frame_keypoints[f'{edge[1]}_X'] * frame.shape[1]), int(frame_keypoints[f'{edge[1]}_Y'] * frame.shape[0]))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Drawing lines between connected key points.

        print("Keypoints coordinates:")  # Printing out the coordinates of each detected keypoint.
        for key, value in frame_keypoints.items():
            print(f"{key}: {value:.2f}")

        cv2.imshow('Pose Estimation', frame)  # Displaying the annotated video feed with pose estimation.
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # If I press ‘q’, I'll exit the loop and stop capturing.

    vid.release()  # Releasing the video capture object when done.
    cv2.destroyAllWindows()  # Closing all OpenCV windows.

# Finally, I'm calling my function to start pose detection live!
YOLO_detect_pose_live()