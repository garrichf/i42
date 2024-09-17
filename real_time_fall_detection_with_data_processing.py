import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score
from playsound import playsound  # For notification sound

# Load YOLO model for pose detection and fall detection model
pose_model = YOLO('yolov8n-pose.pt')  # Update path if needed
fall_detection_model = load_model("falldetect_test.h5")  # Update path if needed

def compute_angle(p1, p2, p3):
    """
    Compute the angle between three keypoints.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def add_angles(keypoints):
    """
    Calculate and add angles based on keypoints.
    """
    angles = {}
    if len(keypoints) >= 3:  # Check if there are enough keypoints
        angles['Head_Tilt_Angle'] = compute_angle(keypoints[1], keypoints[0], keypoints[2])  # Example
        # Add more angles if needed
    return angles

def draw_annotations(frame, keypoints, results):
    """
    Draw keypoints and bounding boxes on the frame.
    """
    # Draw keypoints as green dots
    for kp in keypoints:
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
    
    # Draw bounding boxes (if available)
    boxes = results[0].boxes.xyxy  # Get bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box

    return frame

def process_frame(frame):
    """
    Process a single video frame: detect poses, calculate keypoints and angles, then annotate the frame.
    """
    results = pose_model(frame)
    keypoints = results[0].keypoints.xy[0]
    
    keypoints_list = keypoints.tolist()
    keypoints_flattened = [coord for kp in keypoints_list for coord in kp]
    
    angles = add_angles(keypoints)
    annotated_frame = draw_annotations(frame, keypoints, results)
    return annotated_frame, keypoints_flattened, angles

def predict_fall(features):
    """
    Predict if a fall has occurred based on extracted features.
    """
    features = np.array(features).reshape(1, 1, -1)
    prediction = fall_detection_model.predict(features)
    return prediction[0][0] > 0.5

def calculate_velocity_acceleration(df, keypoint_columns):
    """
    Calculate velocity and acceleration for keypoints.
    """
    df = df.sort_values(by='Frame')  # Sort frames by number
    for col in keypoint_columns:
        df[f'{col}_velocity'] = df.groupby('Video')[col].diff()
        df[f'{col}_acceleration'] = df.groupby('Video')[f'{col}_velocity'].diff()
    return df

def process_dataset(dataset_dir):
    """
    Process the dataset: detect falls, collect data, and save to a CSV file.
    """
    true_labels = []
    predicted_labels = []
    frames_data = []
    keypoint_columns = []  # Define keypoint columns based on your data
    
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
                        fall_detected = predict_fall(keypoints)
                        predicted_labels.append(1 if fall_detected else 0)
                        true_labels.append(true_label)
                        
                        frame_data = {'Frame': file, 'Label': true_label, 'Prediction': predicted_labels[-1]}
                        frame_data.update(angles)
                        frames_data.append(frame_data)
                        
                        valid_frames.append(processed_frame)  # Collect valid frames
                else:
                    print(f"Failed to load image: {img_path}")
    
    df = pd.DataFrame(frames_data)
    df = calculate_velocity_acceleration(df, keypoint_columns)
    
    # Save processed data to CSV
    df.to_csv("processed_data.csv", index=False)
    
    return true_labels, predicted_labels, valid_frames

def evaluate_model(true_labels, predicted_labels):
    """
    Evaluate the performance of the model using precision, recall, and F1-score.
    """
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

def create_and_play_video(frames, output_path="output_video.avi", fps=10):
    """
    Create a video from the frames and play it.
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
    Main function to process dataset, evaluate model, and create video.
    """
    dataset_path = r"C:\Users\User\Desktop\Garrich_i42\i42\SmallerDataset"
    true_labels, predicted_labels, valid_frames = process_dataset(dataset_path)
    evaluate_model(true_labels, predicted_labels)
    create_and_play_video(valid_frames)

if __name__ == "__main__":
    main()
