import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

# Load models
pose_model = tf.lite.Interpreter(model_path='posenet_resnet_50_416_288_16_quant_cpu_decoder.tflite')
pose_model.allocate_tensors()
fall_detection_model = tf.keras.models.load_model('falldetect_16092024_1307_alldatacombined_reg_reduceplateau.keras')

# Define functions
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
        angles['neck_angle'] = compute_angle(keypoints[5], keypoints[1], keypoints[2])
        angles['spine_angle'] = compute_angle(keypoints[1], keypoints[8], keypoints[11])
        angles['left_knee_angle'] = compute_angle(keypoints[11], keypoints[13], keypoints[15])
        angles['right_knee_angle'] = compute_angle(keypoints[12], keypoints[14], keypoints[16])
        angles['left_elbow_angle'] = compute_angle(keypoints[5], keypoints[7], keypoints[9])
        angles['right_elbow_angle'] = compute_angle(keypoints[6], keypoints[8], keypoints[10])
    return angles

def calculate_velocity_acceleration(df):
    df = df.sort_values(by='Frame')
    for col in df.columns:
        if 'keypoint' in col:
            df[f'{col}_velocity'] = df.groupby('Video')[col].diff()
            df[f'{col}_acceleration'] = df.groupby('Video')[f'{col}_velocity'].diff()
    return df

def draw_annotations(frame, keypoints, fall_detected):
    color_box = (0, 255, 0) if not fall_detected else (0, 0, 255)
    
    # Draw bounding box
    cv2.rectangle(frame, (10, 10), (300, 100), color_box, 2)

    # Draw keypoint dots
    for keypoint in keypoints:
        x, y = int(keypoint[0]), int(keypoint[1])
        color_dot = (0, 255, 0) if not fall_detected else (0, 0, 255)
        cv2.circle(frame, (x, y), 5, color_dot, -1)

    return frame

def process_frame(frame):
    input_details = pose_model.get_input_details()
    output_details = pose_model.get_output_details()
    
    # Preprocess the frame for pose estimation
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
    
    pose_model.set_tensor(input_details[0]['index'], frame_resized[np.newaxis, ...])
    pose_model.invoke()
    
    # Get the results from the model
    landmarks_output = pose_model.get_tensor(output_details[0]['index'])[0]
    
    keypoints = []
    for landmark in landmarks_output:
        x, y = landmark[:2]
        keypoints.append([x * frame.shape[1], y * frame.shape[0]])

    angles = add_angles(keypoints)
    
    return keypoints, angles

def predict_fall(features):
    expected_length = 115
    if len(features) < expected_length:
        features += [0] * (expected_length - len(features))  # Pad with zeros
    elif len(features) > expected_length:
        features = features[:expected_length]  # Truncate to expected length
    
    features = np.array(features).reshape(1, 1, -1)  # Reshape for model input
    prediction = fall_detection_model.predict(features)
    
    return prediction[0][1] > 0.3

def process_dataset(dataset_dir):
    true_labels = []
    predicted_labels = []
    
    video_writer = None
    
    for label_dir in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_dir)
        true_label = 1 if label_dir.lower() == 'fall' else 0
        
        for subdir, _, files in os.walk(label_path):
            for file in sorted(files):
                img_path = os.path.join(subdir, file)
                frame = cv2.imread(img_path)
                if frame is not None:
                    keypoints, angles = process_frame(frame)

                    # Prepare features for fall detection
                    features = [coord for kp in keypoints for coord in kp] + list(angles.values())
                    fall_detected = predict_fall(features)

                    # Draw annotations on the frame
                    annotated_frame = draw_annotations(frame.copy(), keypoints, fall_detected)

                    # Initialize video writer
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter('output_video.avi', fourcc, 30.0,
                                                        (frame.shape[1], frame.shape[0]))

                    video_writer.write(annotated_frame)

                    true_labels.append(true_label)
                    predicted_labels.append(1 if fall_detected else 0)

                else:
                    print(f"Failed to load image: {img_path}")
    
    if video_writer is not None:
        video_writer.release()

    return true_labels, predicted_labels

def evaluate_model(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Main execution
dataset_directory = r"C:\Users\User\Desktop\i42\SmallerDataset"
true_labels, predicted_labels = process_dataset(dataset_directory)
evaluate_model(true_labels, predicted_labels)