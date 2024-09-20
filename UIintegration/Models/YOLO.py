import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

# Load YOLO model for pose detection and fall detection model
pose_model = YOLO('yolov8n-pose.pt')
fall_detection_model = load_model("falldetect_16092024_1307_alldatacombined_reg_reduceplateau.keras")

def compute_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def add_angles(keypoints):
    angles = {}
    if len(keypoints) >= 17:  
        # Neck angle
        angles['neck_angle'] = compute_angle(keypoints[5], keypoints[1], keypoints[2])
        # Spine angle
        angles['spine_angle'] = compute_angle(keypoints[1], keypoints[8], keypoints[11])
        # Left knee angle
        angles['left_knee_angle'] = compute_angle(keypoints[11], keypoints[13], keypoints[15])
        # Right knee angle
        angles['right_knee_angle'] = compute_angle(keypoints[12], keypoints[14], keypoints[16])
        # Left elbow angle
        angles['left_elbow_angle'] = compute_angle(keypoints[5], keypoints[7], keypoints[9])
        # Right elbow angle
        angles['right_elbow_angle'] = compute_angle(keypoints[6], keypoints[8], keypoints[10])
    return angles

def draw_annotations(frame, keypoints, results, fall_detected):
    # Define colors
    green = (0, 255, 0)  # Green color for no fall
    red = (0, 0, 255)    # Red color for fall detected

    # Choose color based on fall detection
    color = red if fall_detected else green

    # Ensure that only one bounding box is drawn, select the most confident one
    if len(results[0].boxes.xyxy) > 0:
        # Sort boxes by confidence if needed (optional, based on your use case)
        best_box = results[0].boxes.xyxy[0]  # Choose the first detected person
        x1, y1, x2, y2 = map(int, best_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw keypoint dots
    for keypoint in keypoints:
        x, y = map(int, keypoint)
        cv2.circle(frame, (x, y), 5, color, -1)  # -1 fills the circle

    return frame


def process_frame(frame, fall_detected):
    if frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    
    results = pose_model(frame)
    keypoints = results[0].keypoints.xy[0]
    keypoints_list = keypoints.tolist()
    angles = add_angles(keypoints_list)
    annotated_frame = draw_annotations(frame, keypoints_list, results, fall_detected)
    return annotated_frame, keypoints_list, angles

def predict_fall(df_row, keypoints, angles):
    features = [coord for kp in keypoints for coord in kp]
    features += list(angles.values())
    
    for col in df_row.columns:
        if '_velocity' in col or '_acceleration' in col:
            features.append(df_row[col].values[0])
    
    expected_length = 115  
    if len(features) < expected_length:
        features = features + [0] * (expected_length - len(features))
    elif len(features) > expected_length:
        features = features[:expected_length]
    
    features = np.array(features).reshape(1, 1, -1)
    prediction = fall_detection_model.predict(features)
    return prediction[0][1] > 0.3

def calculate_velocity_acceleration(df):
    df = df.sort_values(by='Frame')
    for col in df.columns:
        if 'keypoint' in col:
            df[f'{col}_velocity'] = df.groupby('Video')[col].diff()
            df[f'{col}_acceleration'] = df.groupby('Video')[f'{col}_velocity'].diff()
    return df

def process_dataset(dataset_dir):
    true_labels = []
    predicted_labels = []
    frames_data = []
    
    current_video = None
    video_frames = []
    
    cv2.namedWindow('Fall Detection', cv2.WINDOW_NORMAL)
    
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
                    if current_video != subdir:
                        if video_frames:
                            process_video_frames(video_frames, true_label, true_labels, predicted_labels, frames_data)
                        current_video = subdir
                        video_frames = []
                    
                    processed_frame, keypoints, angles = process_frame(frame, False)
                    if keypoints:
                        video_frames.append((frame, keypoints, angles, file))
                        
                        # Display the processed frame
                        cv2.imshow('Fall Detection', processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            return true_labels, predicted_labels, frames_data
                else:
                    print(f"Failed to load image: {img_path}")
    
    if video_frames:
        process_video_frames(video_frames, true_label, true_labels, predicted_labels, frames_data)
    
    cv2.destroyAllWindows()
    
    df = pd.DataFrame(frames_data)
    df.to_csv("processed_data.csv", index=False)
    
    return true_labels, predicted_labels, frames_data

def process_video_frames(video_frames, true_label, true_labels, predicted_labels, frames_data):
    print(f"Processing video frames, true label: {true_label}")  # Add this print

    df = pd.DataFrame([
        {
            'Frame': file,
            'Video': os.path.dirname(file),
            **{f'keypoint_{i}_{coord}': value for i, kp in enumerate(keypoints) for coord, value in zip(['x', 'y'], kp)},
            **angles
        }
        for _, keypoints, angles, file in video_frames
    ])

    df = calculate_velocity_acceleration(df)

    for i, (frame, keypoints, angles, file) in enumerate(video_frames):
        fall_detected = predict_fall(df.iloc[i:i+1], keypoints, angles)
        predicted_labels.append(1 if fall_detected else 0)
        true_labels.append(true_label)  # Ensure this is correctly populated

        processed_frame, _, _ = process_frame(frame, fall_detected)
        
        cv2.imshow('Fall Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

        frame_data = {
            'Frame': file,
            'Label': true_label,
            'Prediction': predicted_labels[-1],
            **angles
        }
        frames_data.append(frame_data)


def evaluate_model(true_labels, predicted_labels):
    # Debugging prints
    print(f"True Labels: {true_labels}")
    print(f"Predicted Labels: {predicted_labels}")

    if len(true_labels) == 0 or len(predicted_labels) == 0:
        print("No predictions or true labels to evaluate.")
        return

    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")



def main():
    dataset_path = r"C:\Users\User\Desktop\i42\SmallerDataset"
    true_labels, predicted_labels, frames_data = process_dataset(dataset_path)
    evaluate_model(true_labels, predicted_labels)

if __name__ == "__main__":
    main()

