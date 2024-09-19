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
    if len(keypoints) >= 17:  # Assuming COCO keypoint format
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
    for kp in keypoints:
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
    
    boxes = results[0].boxes.xyxy
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if fall_detected else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    return frame

def process_frame(frame, fall_detected):
    results = pose_model(frame)
    keypoints = results[0].keypoints.xy[0]
    keypoints_list = keypoints.tolist()
    angles = add_angles(keypoints_list)
    annotated_frame = draw_annotations(frame, keypoints_list, results, fall_detected)
    return annotated_frame, keypoints_list, angles

def predict_fall(keypoints, angles):
    features = [coord for kp in keypoints for coord in kp]
    features += list(angles.values())
    
    expected_length = 115
    if len(features) < expected_length:
        features = features + [0] * (expected_length - len(features))
    elif len(features) > expected_length:
        features = features[:expected_length]
    
    features = np.array(features).reshape(1, 1, -1)
    prediction = fall_detection_model.predict(features)
    return prediction[0][1] > 0.5

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
    valid_frames = []
    
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
                    processed_frame, keypoints, angles = process_frame(frame, False)
                    if keypoints:
                        fall_detected = predict_fall(keypoints, angles)
                        predicted_labels.append(1 if fall_detected else 0)
                        true_labels.append(true_label)

                        # Reprocess the frame with the updated fall detection status for bounding box color
                        processed_frame, keypoints, angles = process_frame(frame, fall_detected)

                        frame_data = {
                            'Frame': file,
                            'Label': true_label,
                            'Prediction': predicted_labels[-1]
                        }
                        frame_data.update(angles)
                        frames_data.append(frame_data)

                        valid_frames.append(processed_frame)
                else:
                    print(f"Failed to load image: {img_path}")
    
    df = pd.DataFrame(frames_data)
    df = calculate_velocity_acceleration(df)
    df.to_csv("processed_data.csv", index=False)
    
    return true_labels, predicted_labels, valid_frames


def evaluate_model(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

def create_and_play_video(frames, output_path="output_video.avi", fps=10):
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
    dataset_path = r"C:\Users\User\Desktop\Garrich_i42\i42\SmallerDataset"
    true_labels, predicted_labels, valid_frames = process_dataset(dataset_path)
    evaluate_model(true_labels, predicted_labels)
    create_and_play_video(valid_frames)

if __name__ == "__main__":
    main()
