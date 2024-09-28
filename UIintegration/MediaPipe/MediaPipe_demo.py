import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Function to perform pose estimation on a video
def MediaPipe_detect_pose_sequence(vid_path):
    # Load the MediaPipe pose model
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("MediaPipe Pose Estimation Model loaded successfully.")
    except Exception as error:
        print(f"Failed to load the model: {error}")
        return None

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

    # Keypoint names based on the MediaPipe model
    keypoint_names = [
        'Nose', 'Left Eye Inner', 'Left Eye', 'Left Eye Outer', 'Right Eye Inner', 'Right Eye', 'Right Eye Outer',
        'Left Ear', 'Right Ear', 'Mouth Left', 'Mouth Right', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 
        'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Pinky', 'Right Pinky', 'Left Index', 'Right Index', 
        'Left Thumb', 'Right Thumb', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
        'Left Heel', 'Right Heel', 'Left Foot Index', 'Right Foot Index'
    ]

    # Iterate through each frame
    for frame in frames:
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform pose estimation
        results = pose.process(frame_rgb)

        # Initialize a dictionary to store keypoints for the current frame
        frame_keypoints = {}

        if results.pose_landmarks:
            average_confidence = np.mean([landmark.visibility for landmark in results.pose_landmarks.landmark])

            if average_confidence >= POSE_CONFIDENCE_THRESHOLD:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    frame_keypoints[f'{keypoint_names[idx]}_X'] = landmark.x
                    frame_keypoints[f'{keypoint_names[idx]}_Y'] = landmark.y

        # Append the frame keypoints to the all_keypoints list
        all_keypoints.append(frame_keypoints)

    vid.release()
    df = pd.DataFrame(all_keypoints)
    # Return keypoints for all frames
    return df

def process_and_display_video(vid_path):
    # Load the MediaPipe pose model
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open the video file
    cap = cv2.VideoCapture(vid_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get the pose landmarks
        results = pose.process(frame_rgb)

        # Draw the pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Processed Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video processing complete. Output saved as 'output_video.mp4'")

# Main execution
if __name__ == "__main__":
    video_path = "ADL.mp4"
    
    # Process the video and get keypoints
    keypoints_df = MediaPipe_detect_pose_sequence(video_path)
    
    # Save keypoints to CSV
    keypoints_df.to_csv("pose_keypoints.csv", index=False)
    print("Keypoints saved to 'pose_keypoints.csv'")

    # Process and display the video with keypoints
    process_and_display_video(video_path)