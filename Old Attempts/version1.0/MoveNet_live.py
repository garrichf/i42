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
        ('Nose', 'Left Eye'), ('Nose', 'Right Eye'), ('Left Eye', 'Left Ear'), ('Right Eye', 'Right Ear'),
        ('Left Shoulder', 'Right Shoulder'), ('Left Shoulder', 'Left Elbow'), ('Right Shoulder', 'Right Elbow'),
        ('Left Elbow', 'Left Wrist'), ('Right Elbow', 'Right Wrist'), ('Left Shoulder', 'Left Hip'),
        ('Right Shoulder', 'Right Hip'), ('Left Hip', 'Right Hip'), ('Left Hip', 'Left Knee'),
        ('Right Hip', 'Right Knee'), ('Left Knee', 'Left Ankle'), ('Right Knee', 'Right Ankle')
    ]
    
    
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
                frame_keypoints[f'{keypoint_names[idx]}_X'] = x   # Store the x coordinate
                frame_keypoints[f'{keypoint_names[idx]}_Y'] = y   # Store the y coordinate
                if confidence > POSE_CONFIDENCE_THRESHOLD:
                    # Draw keypoints on the frame
                    cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)

            # Draw edges connecting the keypoints
            for edge in keypoint_edges:
                pt1 = (int(frame_keypoints[f'{edge[0]}_X'] * frame.shape[1]), int(frame_keypoints[f'{edge[0]}_Y'] * frame.shape[0]))
                pt2 = (int(frame_keypoints[f'{edge[1]}_X'] * frame.shape[1]), int(frame_keypoints[f'{edge[1]}_Y'] * frame.shape[0]))
                if frame_keypoints[f'{edge[0]}_X'] != 0 and frame_keypoints[f'{edge[0]}_Y'] != 0 and frame_keypoints[f'{edge[1]}_X'] != 0 and frame_keypoints[f'{edge[1]}_Y'] != 0:
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

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