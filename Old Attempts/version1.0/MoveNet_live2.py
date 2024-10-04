import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import pandas as pd

# Function to perform live pose estimation on a video feed
def MoveNet_detect_pose_live():
    
    # Load the MoveNet model from TensorFlow Hub
    try:
        movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        print("MoveNet Pose Estimation Model loaded successfully.")
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
    
    # Keypoint names based on the MoveNet model
    keypoint_names = [
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
    ]
    
    # Define the pairs of keypoints to connect with edges
    keypoint_edges = [
        ('Nose', 'Left Eye'), ('Nose', 'Right Eye'), ('Left Eye', 'Left Ear'), ('Right Eye', 'Right Ear'), ('Nose', 'Left Shoulder'), ('Nose', 'Right Shoulder'),
        ('Left Shoulder', 'Right Shoulder'), ('Left Shoulder', 'Left Elbow'), ('Right Shoulder', 'Right Elbow'),
        ('Left Elbow', 'Left Wrist'), ('Right Elbow', 'Right Wrist'), ('Left Shoulder', 'Left Hip'),
        ('Right Shoulder', 'Right Hip'), ('Left Hip', 'Right Hip'), ('Left Hip', 'Left Knee'),
        ('Right Hip', 'Right Knee'), ('Left Knee', 'Left Ankle'), ('Right Knee', 'Right Ankle')
    ]
    
    # Validate keypoints based on their relative positions
    def is_valid_keypoint(kp1, kp2, axis='y', threshold=0.1):
        """
        Determines if the difference between two keypoints along a specified axis exceeds a given threshold.

        Args:
            kp1 (str): The first keypoint identifier.
            kp2 (str): The second keypoint identifier.
            axis (str, optional): The axis to compare ('x' or 'y'). Defaults to 'y'.
            threshold (float, optional): The threshold value for the difference. Defaults to 0.1.

        Returns:
            bool: True if the difference exceeds the threshold; False if either keypoint is missing or the difference is below the threshold.
        """
        # Check if both keypoints are present in the frame
        if kp1 in frame_keypoints and kp2 in frame_keypoints:
            if axis == 'y': # Check the y-axis difference
                return abs(frame_keypoints[f'{kp1}_Y'] - frame_keypoints[f'{kp2}_Y']) > threshold
            elif axis == 'x': # Check the x-axis difference
                return abs(frame_keypoints[f'{kp1}_X'] - frame_keypoints[f'{kp2}_X']) > threshold
        return True # If either keypoint is missing, return False

    # Function to validate all keypoints
    def validate_all_keypoints():
        """
        Validates all keypoints by checking pairs of keypoints for validity.

        This function iterates through a predefined list of keypoint pairs and 
        checks if each pair is valid using the `is_valid_keypoint` function. 
        If a pair is found to be invalid, both keypoints in the pair are added 
        to a list of invalid keypoints. Finally, it removes the X and Y 
        coordinates of each invalid keypoint from the `frame_keypoints` dictionary.

        Keypoint pairs checked:
            - ('Left Ankle', 'Left Knee')
            - ('Right Ankle', 'Right Knee')
            - ('Left Knee', 'Left Hip')
            - ('Right Knee', 'Right Hip')
            - ('Left Hip', 'Right Hip')
            - ('Left Shoulder', 'Right Shoulder')
            - ('Left Elbow', 'Left Shoulder')
            - ('Right Elbow', 'Right Shoulder')
            - ('Left Wrist', 'Left Elbow')
            - ('Right Wrist', 'Right Elbow')

        Note:
            The function assumes the existence of the `is_valid_keypoint` function 
            and the `frame_keypoints` dictionary in the same scope.

        Returns:
            None
        """
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
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize frame to the expected input size of MoveNet
        frame_resized = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 256, 256) # 256 for thunder
        # Convert the resized frame tensor to a NumPy array with dtype uint8
        frame_np = frame_resized.numpy().astype(np.int32)
        # Perform inference
        outputs = movenet.signatures["serving_default"](tf.constant(frame_np))
        # Extract the keypoints
        keypoints = outputs['output_0'].numpy()
        
        # Initialize a dictionary to store keypoints for the current frame
        frame_keypoints = {}
        
        average_confidence = np.mean(keypoints[:, :, :, 2]) # Calculate average confidence
        # Check if the average confidence is above the threshold
        if average_confidence < POSE_CONFIDENCE_THRESHOLD:
            keypoints = np.zeros_like(keypoints)  # Discard detection
        else:
            for idx, keypoint in enumerate(keypoints[0][0]):    # Iterate through the keypoints
                x, y, confidence = keypoint[1], keypoint[0], keypoint[2]    # Extract the x, y, and confidence
                if confidence > POSE_CONFIDENCE_THRESHOLD:
                    frame_keypoints[f'{keypoint_names[idx]}_X'] = x   # Store the x coordinate
                    frame_keypoints[f'{keypoint_names[idx]}_Y'] = y   # Store the y coordinate
                    # Draw keypoints on the frame
                    cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)

            # Validate all keypoints
            validate_all_keypoints()

            # Draw edges connecting the keypoints
            for edge in keypoint_edges:
                if (f'{edge[0]}_X' in frame_keypoints and f'{edge[0]}_Y' in frame_keypoints and
                    f'{edge[1]}_X' in frame_keypoints and f'{edge[1]}_Y' in frame_keypoints):
                    pt1 = (int(frame_keypoints[f'{edge[0]}_X'] * frame.shape[1]), int(frame_keypoints[f'{edge[0]}_Y'] * frame.shape[0]))
                    pt2 = (int(frame_keypoints[f'{edge[1]}_X'] * frame.shape[1]), int(frame_keypoints[f'{edge[1]}_Y'] * frame.shape[0]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                else:
                    print(f"Missing keypoints for edge: {edge}")

            # Print the keypoints coordinates
            print("Keypoints coordinates:")
            for key, value in frame_keypoints.items():  # Iterate through the keypoint coordinates
                print(f"{key}: {value:.2f}")            # Print the keypoint coordinates
        
        # Display the frame
        cv2.imshow('Pose Estimation', frame)
        
        # Press 'q' to exit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all OpenCV windows
    vid.release()
    cv2.destroyAllWindows()

# Run the live pose estimation
MoveNet_detect_pose_live()