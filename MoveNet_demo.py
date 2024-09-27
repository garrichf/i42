import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import pandas as pd

# vid_path = "ADL.mp4"  # Replace with the path to your input video

# Function to perform pose estimation on a video
def MoveNet_detect_pose_sequence(vid_path):
    
    # Load the MoveNet model from TensorFlow Hub
    try:
        movenet = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4")
        print("MoveNet Pose Estimation Model loaded successfully.")
    except Exception as error:
        print(f"Failed to load the model: {error}")

    # Set a threshold for average confidence
    POSE_CONFIDENCE_THRESHOLD = 0.25
    
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
    
    # Keypoint names based on the MoveNet model
    keypoint_names = [
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
    ]
    
    # Iterate through each frame
    for frame in frames:
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

        if average_confidence < POSE_CONFIDENCE_THRESHOLD:
            keypoints = np.zeros_like(keypoints)  # Discard detection
        else: 
            for idx, keypoint in enumerate(keypoints[0][0]):
                x, y = keypoint[1], keypoint[0]
                frame_keypoints[f'{keypoint_names[idx]}_X'] = x
                frame_keypoints[f'{keypoint_names[idx]}_Y'] = y

        # Append the frame keypoints to the all_keypoints list
        all_keypoints.append(frame_keypoints)

        print(f"Processed frame {len(all_keypoints)}")
    
    df = pd.DataFrame(all_keypoints)
    # Return keypoints for all frames
    return df

# Perform pose estimation on the input video
# result = detect_pose_sequence(vid_path)