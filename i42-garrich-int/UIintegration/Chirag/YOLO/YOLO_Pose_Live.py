import cv2
import numpy as np
from ultralytics import YOLO

def YOLOv8_detect_pose_live():
    # Load the YOLOv8 model
    try:
        model = YOLO('yolov8n-pose.pt')
        print("YOLOv8 Pose Estimation Model loaded successfully.")
    except Exception as error:
        print(f"Failed to load the model: {error}")
        return

    # Set a threshold for average confidence
    POSE_CONFIDENCE_THRESHOLD = 0.25

    # Initialize the video capture object to capture from the default camera
    vid = cv2.VideoCapture(0)

    if not vid.isOpened():
        print("Error: Could not open video stream.")
        return

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

    # Validate keypoints based on their relative positions
    def is_valid_keypoint(kp1, kp2, axis='y', threshold=0.1):
        if kp1 in frame_keypoints and kp2 in frame_keypoints:
            if axis == 'y':
                return abs(frame_keypoints[f'{kp1}_Y'] - frame_keypoints[f'{kp2}_Y']) > threshold
            elif axis == 'x':
                return abs(frame_keypoints[f'{kp1}_X'] - frame_keypoints[f'{kp2}_X']) > threshold
        return True

    # Function to validate all keypoints
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
        for kp in set(invalid_keypoints):
            frame_keypoints.pop(f'{kp}_X', None)
            frame_keypoints.pop(f'{kp}_Y', None)

    while True:
        ret, frame = vid.read()
        if not ret:
            print("Error: Failed to capture image")
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

                # Validate all keypoints
                validate_all_keypoints()

                # Draw edges connecting the keypoints
                for edge in keypoint_edges:
                    if (f'{edge[0]}_X' in frame_keypoints and f'{edge[0]}_Y' in frame_keypoints and
                        f'{edge[1]}_X' in frame_keypoints and f'{edge[1]}_Y' in frame_keypoints):
                        pt1 = (int(frame_keypoints[f'{edge[0]}_X'] * frame.shape[1]), int(frame_keypoints[f'{edge[0]}_Y'] * frame.shape[0]))
                        pt2 = (int(frame_keypoints[f'{edge[1]}_X'] * frame.shape[1]), int(frame_keypoints[f'{edge[1]}_Y'] * frame.shape[0]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                # Print the keypoints coordinates
                print("Keypoints coordinates:")
                for key, value in frame_keypoints.items():
                    print(f"{key}: {value:.2f}")

        # Display the frame
        cv2.imshow('Pose Estimation', frame)

        # Press 'q' to exit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    vid.release()
    cv2.destroyAllWindows()

# Run the live pose estimation
YOLOv8_detect_pose_live()