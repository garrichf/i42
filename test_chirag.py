import os
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time

# Paths
dataset_path = r"C:\Users\User\Desktop\Garrich_i42\i42\SmallerDataset"
yolo_model_path = r"C:\Users\User\Desktop\Garrich_i42\i42\yolov8n-pose.pt"
fall_detect_model_path = r"C:\Users\User\Desktop\Garrich_i42\falldetect_test.h5"

# Load models
yolo_model = YOLO(yolo_model_path)
fall_detect_model = load_model(fall_detect_model_path)

def extract_pose_features(frame_path):
    try:
        results = yolo_model(frame_path)
        keypoints = results[0].keypoints
        if len(keypoints) == 0:
            return None
        xy = keypoints[0].xy[0]
        flattened_keypoints = xy.flatten().tolist()
        if len(flattened_keypoints) < 34:
            flattened_keypoints.extend([0] * (34 - len(flattened_keypoints)))
        elif len(flattened_keypoints) > 34:
            flattened_keypoints = flattened_keypoints[:34]
        return flattened_keypoints
    except Exception as e:
        print(f"Error processing image {frame_path}: {str(e)}")
        return None

def predict_fall(features):
    features = np.array(features).reshape(1, 1, 34)
    prediction = fall_detect_model.predict(features)
    class_prediction = int(prediction[0][0] > 0.5)
    confidence = prediction[0][0] if class_prediction == 1 else 1 - prediction[0][0]
    return class_prediction, confidence

def process_dataset(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.startswith('._') or not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            file_path = os.path.join(root, file)
            actual_label = 'Fall' if 'fall' in root.lower() else 'No Fall'
            
            pose_features = extract_pose_features(file_path)
            if pose_features is not None:
                class_prediction, confidence = predict_fall(pose_features)
                predicted_label = 'No Fall' if class_prediction == 1 else 'Fall'
                
                # Display image with predictions
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Error reading image: {file_path}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.title(f"File: {file}")
                plt.axis('off')
                
                text = f"Actual: {actual_label}\nPredicted: {predicted_label}\nConfidence: {confidence:.2f}"
                if actual_label == predicted_label:
                    text += "\nCorrect Prediction!"
                else:
                    text += "\nIncorrect Prediction"
                
                plt.text(10, 30, text, fontsize=12, color='white', backgroundcolor='black')
                plt.show(block=False)
                plt.pause(5)  # 5-second delay
                plt.close()
                
                print(f"Image: {file}, Actual: {actual_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}")
            else:
                print(f"Image: {file}, No person detected or error in processing")

def main():
    process_dataset(dataset_path)

if __name__ == "__main__":
    main()