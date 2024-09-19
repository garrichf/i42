import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

# First, I load the YOLO model for pose detection and the fall detection model.
pose_model = YOLO('yolov8n-pose.pt')  # I might need to update this path if necessary.
fall_detection_model = load_model("falldetect_16092024_1307_alldatacombined_reg_reduceplateau.keras")  # Similarly, I should check this path.

def compute_angle(p1, p2, p3):
    """
    Here, I compute the angle between three keypoints.
    This is useful for understanding body posture.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))  # I convert the angle from radians to degrees.
    return angle

def add_angles(keypoints):
    """
    In this function, I calculate and add angles based on the detected keypoints.
    This helps in analyzing the person's posture.
    """
    angles = {}
    if len(keypoints) >= 3:  # I check if there are enough keypoints to calculate an angle.
        angles['Head_Tilt_Angle'] = compute_angle(keypoints[1], keypoints[0], keypoints[2])  # Example angle calculation.
        # I can add more angles here if needed.
    return angles

def draw_annotations(frame, keypoints, results):
    """
    Here, I draw keypoints and bounding boxes on the video frame for visualization.
    This makes it easier to see what the model is detecting.
    """
    # I draw keypoints as green dots on the frame.
    for kp in keypoints:
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
    
    # If there are bounding boxes available from detection results, I draw them as well.
    boxes = results[0].boxes.xyxy  # I extract bounding boxes from the results.
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # I use a blue color for bounding boxes.

    return frame

def process_frame(frame):
    """
    In this function, I process a single video frame: detect poses,
    calculate keypoints and angles, and then annotate the frame with this information.
    """
    results = pose_model(frame)  # I use the pose model to get detection results for the frame.
    keypoints = results[0].keypoints.xy[0]  # I extract keypoints from the results.
    
    keypoints_list = keypoints.tolist()  # I convert keypoints to a list format for easier handling.
    
    angles = add_angles(keypoints)  # I calculate angles based on the detected keypoints.
    
    annotated_frame = draw_annotations(frame, keypoints_list, results)  # I annotate the frame with visual data.
    
    return annotated_frame, keypoints_list, angles
def predict_fall(features):
    """
    Predict whether a fall has occurred based on extracted features from video frames.
    The input features vector is padded or truncated to match the LSTM model input.
    """
    expected_length = 115  # This should match the LSTM's input size

    # Adjust features vector to match the expected input size of the LSTM model
    if len(features) < expected_length:
        # Padding with zeros if features are shorter than expected
        features = features + [0] * (expected_length - len(features))
    elif len(features) > expected_length:
        # Truncating if features are longer than expected
        features = features[:expected_length]

    features = np.array(features).reshape(1, 1, -1)  # Reshaping to match the LSTM input requirements

    prediction = fall_detection_model.predict(features)  # Get predictions from the model

    return prediction[0][1] > 0.5  # Return True if the prediction is greater than 0.5 (fall detected)


def calculate_velocity_acceleration(df):
    """
    In this function, I calculate velocity and acceleration for each detected keypoint across frames.
    This helps in understanding movement dynamics over time.
    """
    df = df.sort_values(by='Frame')  # First, I sort frames by their number to maintain order.
    
    for col in df.columns:
        if 'keypoint' in col:  # I'm looking for columns that contain 'keypoint' in their names.
            df[f'{col}_velocity'] = df.groupby('Video')[col].diff()  # I calculate velocity by taking differences between frames.
            df[f'{col}_acceleration'] = df.groupby('Video')[f'{col}_velocity'].diff()  # Acceleration is calculated similarly.

    return df

def process_dataset(dataset_dir):
    """
    Process the dataset: detect falls, collect data, and save to a CSV file.
    """
    true_labels = []
    predicted_labels = []
    
    frames_data = []
    valid_frames = []  # List for valid frames to create video
    
    for label_dir in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_dir)
        if not os.path.isdir(label_path):
            continue
        true_label = 1 if label_dir.lower() == 'fall' else 0
        
        for subdir, _, files in os.walk(label_path):
            for file in sorted(files):
                img_path = os.path.join(subdir, file)
                frame = cv2.imread(img_path)
                
                if frame is not None:
                    processed_frame, keypoints, angles = process_frame(frame)

                    if keypoints:
                        # Flattening keypoints to create feature vector
                        features_vector = [coord for kp in keypoints for coord in kp]
                        features_vector += list(angles.values())  # Add angles to features vector
                        
                        # Add velocity and acceleration features
                        velocities = [0] * len(keypoints)  
                        accelerations = [0] * len(keypoints)  
                        
                        features_vector += velocities  # X velocities
                        features_vector += velocities  # Y velocities
                        features_vector += accelerations  # X accelerations
                        features_vector += accelerations  # Y accelerations

                        # Ensure we have the expected number of features
                        if len(features_vector) > 34:
                            features_vector = features_vector[:34]  # Truncate if too long
                        elif len(features_vector) < 34:
                            print("Warning: Features vector has less than 34 features.")
                            continue  # Skip this frame if not enough features

                        fall_detected = predict_fall(features_vector)
                        predicted_labels.append(1 if fall_detected else 0)
                        true_labels.append(true_label)

                        frame_data = {
                            'Frame': file,
                            'Label': true_label,
                            'Prediction': predicted_labels[-1]
                        }
                        frame_data.update(angles)
                        frames_data.append(frame_data)

                        valid_frames.append(processed_frame)  # Collect valid frames
                else:
                    print(f"Failed to load image: {img_path}")
    
    df = pd.DataFrame(frames_data)

    # Calculate velocity and acceleration after collecting all data points
    df = calculate_velocity_acceleration(df)

    # Save processed data to CSV
    df.to_csv("processed_data.csv", index=False)

    return true_labels, predicted_labels, valid_frames

def evaluate_model(true_labels, predicted_labels):
   """
   In this function, I'm evaluating my model's performance using precision,
   recall and F1-score metrics. This helps me understand how well my model is doing overall.
   """
   precision = precision_score(true_labels, predicted_labels)
   recall = recall_score(true_labels, predicted_labels)
   f1 = f1_score(true_labels, predicted_labels)

   print("\nEvaluation Results:")
   print(f"Precision: {precision:.4f}") 
   print(f"Recall: {recall:.4f}") 
   print(f"F1-score: {f1:.4f}") 

def create_and_play_video(frames, output_path="output_video.avi", fps=10):
   """
   Here I'm creating a video from all valid frames I've collected,
   and then I'll play it back so that I can visually inspect what happened during processing.
   """
   if not frames:
       print("No valid frames to create video.")
       return
   
   height, width, layers = frames[0].shape
   fourcc = cv2.VideoWriter_fourcc(*'XVID')
   out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
   
   for frame in frames:
       out.write(frame)
   
   out.release()
   cap = cv2.VideoCapture(output_path)
   
   while cap.isOpened():
       ret, frame = cap.read()
       if ret:
           cv2.imshow('Video Playback', frame)
           if cv2.waitKey(25) & 0xFF == ord('q'):
               break
       else:
           break
   
   cap.release()
   cv2.destroyAllWindows()

def main():
   """
   This is my main function where everything comes together:
   processing the dataset of videos,
   evaluating my model's performance,
   and creating a video playback of what I've processed so far.
   """
   dataset_path = r"C:\Users\User\Desktop\Garrich_i42\i42\SmallerDataset"
   true_labels, predicted_labels, valid_frames = process_dataset(dataset_path)
   evaluate_model(true_labels, predicted_labels)
   create_and_play_video(valid_frames)

if __name__ == "__main__":
   main()