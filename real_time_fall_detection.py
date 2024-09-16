# I start by importing the necessary libraries for computer vision, file operations, and numerical computations.
import cv2
import os
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# I load the pre-trained YOLO model for pose detection and the fall detection model.
pose_model = YOLO('yolov8n-pose.pt')  # I adjust the path as needed
fall_detection_model = load_model("falldetect_test.h5")  # I adjust the path as needed

# I define a function to process each frame and extract pose keypoints using the YOLO model.
def process_frame(frame):
    # I use the YOLO model to detect poses in the frame.
    results = pose_model(frame)
    
    # I extract the keypoints from the results.
    keypoints = results[0].keypoints.xy[0]
    
    # I flatten the keypoints and return them.
    return keypoints.flatten() if len(keypoints) > 0 else None

# I define a function to predict if a fall occurred based on the extracted features.
def predict_fall(features):
    # I reshape the features for model input.
    features = np.array(features).reshape(1, 1, -1)
    
    # I use the fall detection model to predict if a fall occurred.
    prediction = fall_detection_model.predict(features)
    
    # I return True if a fall is detected, False otherwise.
    return prediction[0][0] > 0.5

# I define the main function to capture video and process frames.
def main(video_source=0):
    # I open a video capture object (using 0 for the webcam or providing a video file path).
    cap = cv2.VideoCapture(video_source)
    
    while True:
        # I read a frame from the video capture.
        ret, frame = cap.read()
        
        # If no frame is read, I break out of the loop.
        if not ret:
            break
        
        # I process the frame to extract pose keypoints.
        keypoints = process_frame(frame)
        
        # If keypoints are extracted,
        if keypoints is not None:
            # I use the fall detection model to predict if a fall occurred.
            fall_detected = predict_fall(keypoints)
            
            # I set the label based on the prediction.
            label = "Fall Detected" if fall_detected else "No Fall"
            
            # I draw the label on the frame.
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # I display the processed frame.
        cv2.imshow("Video Feed", frame)
        
        # I check if the 'q' key is pressed to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # I release the video capture object and close all windows.
    cap.release()
    cv2.destroyAllWindows()

# I run the main function if this script is executed directly.
if __name__ == "__main__":
    main()