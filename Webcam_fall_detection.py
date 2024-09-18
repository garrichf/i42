import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# I start by loading the YOLO model for pose detection and the fall detection model.
pose_model = YOLO('yolov8n-pose.pt')
fall_detection_model = load_model("falldetect_test.h5")

# I need a function to compute the angle between three keypoints, like shoulder, hip, and knee.
def compute_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])  # Vector from p2 to p1
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])  # Vector from p2 to p3
    dot_product = np.dot(v1, v2)  # Dot product of the vectors
    magnitude_v1 = np.linalg.norm(v1)  # Magnitude of vector v1
    magnitude_v2 = np.linalg.norm(v2)  # Magnitude of vector v2
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)  # Cosine of the angle
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))  # Convert to degrees
    return angle

# I need to calculate velocity and acceleration of keypoints across frames.
def calculate_velocity_acceleration(df, keypoint_columns):
    df = df.sort_values(by='Frame')  # Sort frames in order
    for col in keypoint_columns:
        df[f'{col}_velocity'] = df.groupby('Video')[col].diff()  # Calculate velocity
        df[f'{col}_acceleration'] = df.groupby('Video')[f'{col}_velocity'].diff()  # Calculate acceleration
    return df

# I’ll process each video frame to extract keypoints and bounding boxes.
def process_frame(frame, frame_index, video_name):
    results = pose_model(frame)  # Get pose detection results from YOLO model
    keypoints = results[0].keypoints.xy[0] if len(results[0].keypoints.xy) > 0 else None  # Extract keypoints
    boxes = results[0].boxes.xyxy if len(results[0].boxes.xyxy) > 0 else None  # Extract bounding boxes
    
    if keypoints is not None:
        keypoints = keypoints.flatten()
        df_keypoints = pd.DataFrame([keypoints], columns=[f'keypoint_{i}' for i in range(len(keypoints))])
        df_keypoints['Frame'] = frame_index  # Add frame number for tracking
        df_keypoints['Video'] = video_name  # Add video name for tracking
        return df_keypoints, boxes
    return None, boxes

# I need to reshape keypoints to fit the model's expected input size.
def reshape_keypoints(keypoints, expected_size=34):
    keypoints = keypoints[:expected_size]  # Trim if necessary
    if len(keypoints) < expected_size:
        keypoints = np.pad(keypoints, (0, expected_size - len(keypoints)), 'constant')  # Pad if needed
    return keypoints

# I will combine keypoints, angles, and velocity/acceleration into a single feature vector.
def create_feature_vector(keypoints, angle, velocity, acceleration):
    return np.concatenate([keypoints, [angle], [velocity], [acceleration]])

# I will predict if a fall has occurred based on the features.
def predict_fall(features):
    features = reshape_keypoints(features)  # Ensure correct input size
    features = np.array(features, dtype=np.float32)  # Convert to NumPy array
    features = features.reshape(1, 1, -1)  # Reshape for the model (1, 1, 34 + extra features)
    prediction = fall_detection_model.predict(features)  # Get prediction from the model
    return prediction[0][0] > 0.5  # If the prediction is above 0.5, I consider it a fall

# I will draw keypoints and bounding boxes on the video frame, and highlight if a fall is detected.
def draw_annotations(frame, keypoints, boxes, fall_detected):
    color = (0, 255, 0) if not fall_detected else (0, 0, 255)  # Green for normal, red for fall
    if keypoints is not None:
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 5, color, -1)  # Draw filled circle at each keypoint
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw rectangle for each bounding box
    return frame

# I will now write the main function to handle video processing and fall detection.
def main(video_source=0):
    cap = cv2.VideoCapture(video_source)  # Open the video source (0 for webcam)
    all_keypoints = pd.DataFrame()  # DataFrame to store keypoints for later calculations
    frame_index = 0  # Counter for frames
    angles_list = []  # List to store angles
    velocity_list = []  # List to store velocities
    acceleration_list = []  # List to store accelerations

    previous_keypoints = None  # Variable to keep track of previous keypoints for calculations

    while True:
        ret, frame = cap.read()  # Capture a frame from the video
        if not ret:
            break  # Exit loop if no frame is captured
        
        # Process the frame to extract keypoints and bounding boxes
        df_keypoints, boxes = process_frame(frame, frame_index, video_name="video_1")
        
        if df_keypoints is not None:
            # I need at least 6 keypoints to compute the angle (e.g., shoulder-hip-knee)
            if len(df_keypoints.columns) >= 6:
                shoulder = (df_keypoints['keypoint_1'].values[0], df_keypoints['keypoint_2'].values[0])
                hip = (df_keypoints['keypoint_3'].values[0], df_keypoints['keypoint_4'].values[0])
                knee = (df_keypoints['keypoint_5'].values[0], df_keypoints['keypoint_6'].values[0])
                
                # Compute the angle for these keypoints
                angle = compute_angle(shoulder, hip, knee)
                angles_list.append(angle)  # Store the angle for this frame

            # Flatten keypoints for fall prediction
            flattened_keypoints = df_keypoints.iloc[0].drop(['Frame', 'Video'])

            # If I have previous keypoints, I will calculate velocity and acceleration
            if previous_keypoints is not None:
                velocity = np.linalg.norm(flattened_keypoints.values - previous_keypoints)  # Velocity as distance between frames
                velocity_list.append(velocity)  # Store velocity for this frame

                if len(velocity_list) > 1:  # Need at least two velocities to compute acceleration
                    acceleration = velocity_list[-1] - velocity_list[-2]  # Acceleration as difference in velocities
                    acceleration_list.append(acceleration)  # Store acceleration
                else:
                    acceleration = 0
                    acceleration_list.append(acceleration)
            else:
                velocity = 0  # No velocity for the first frame
                acceleration = 0  # No acceleration for the first frame
                velocity_list.append(velocity)
                acceleration_list.append(acceleration)

            # Create the feature vector with keypoints, angle, velocity, and acceleration
            features = create_feature_vector(flattened_keypoints.values, angle, velocity, acceleration)
            
            # Predict if a fall has occurred using the feature vector
            fall_detected = predict_fall(features)

            # Draw annotations on the frame
            frame = draw_annotations(frame, flattened_keypoints.values.reshape(-1, 2), boxes, fall_detected)
            
            # Add the keypoints to the DataFrame for future velocity/acceleration calculations
            all_keypoints = pd.concat([all_keypoints, df_keypoints], ignore_index=True)
        
        # Display the annotated frame
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit loop if 'q' is pressed

        previous_keypoints = flattened_keypoints.values  # Update previous keypoints
        frame_index += 1  # Move to the next frame
    
    cap.release()  # Release video capture
    cv2.destroyAllWindows()  # Close OpenCV windows
    
    # After processing the video, calculate velocity and acceleration for all keypoints
    keypoint_columns = [col for col in all_keypoints.columns if 'keypoint_' in col]
    all_keypoints_with_velocity = calculate_velocity_acceleration(all_keypoints, keypoint_columns)
    
    # Add the angles, velocity, and acceleration to the DataFrame and save to CSV
    all_keypoints_with_velocity['Angle'] = pd.Series(angles_list)
    all_keypoints_with_velocity['Velocity'] = pd.Series(velocity_list)
    all_keypoints_with_velocity['Acceleration'] = pd.Series(acceleration_list)
    all_keypoints_with_velocity.to_csv("processed_keypoints_with_features.csv", index=False)

if __name__ == "__main__":
    main()  # Run the fall detection system
