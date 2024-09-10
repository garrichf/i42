import cv2
import numpy as np
from ultralytics import YOLO
import os

model = YOLO("C:/Users/User/Desktop/fall-detection/yolov8n-pose.pt")

def detect_fall(keypoints):
    if keypoints.xy.shape[1] < 13:
        return False

    left_shoulder = keypoints.xy[0][5][:2]
    right_shoulder = keypoints.xy[0][6][:2]
    left_hip = keypoints.xy[0][11][:2]
    right_hip = keypoints.xy[0][12][:2]

    torso_vector = [(left_shoulder[0] + right_shoulder[0]) / 2 - (left_hip[0] + right_hip[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2 - (left_hip[1] + right_hip[1]) / 2]
    vertical_vector = [0, -1]

    angle = np.arccos(np.dot(torso_vector, vertical_vector) /
                      (np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector)))
    angle_degrees = np.degrees(angle)

    return angle_degrees > 45

def process_images(dataset_path):
    fall_path = os.path.join(dataset_path, 'fall')
    no_fall_path = os.path.join(dataset_path, 'no_fall')

    test_data = []
    for path, label in [(fall_path, 'fall'), (no_fall_path, 'no_fall')]:
        if os.path.exists(path):
            for img in os.listdir(path):
                test_data.append((os.path.join(path, img), label))

    correct = 0
    total = len(test_data)

    for img_path, true_label in test_data:
        frame = cv2.imread(img_path)
        results = model(frame, verbose=False)
        pred_label = 'no_fall'

        for result in results:
            if result.keypoints is None:
                continue

            boxes = result.boxes.cpu().numpy()
            keypoints = result.keypoints

            for box, kpts in zip(boxes, keypoints):
                if kpts.xy.shape[1] >= 13:
                    is_fall = detect_fall(kpts)
                    pred_label = 'fall' if is_fall else 'no_fall'
                else:
                    print(f"Insufficient keypoints for image: {img_path}")

        if pred_label == true_label:
            correct += 1
        print(f"Image: {os.path.basename(img_path)}, True: {true_label}, Predicted: {pred_label}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    dataset_path = r"C:\Users\User\Desktop\fall-detection\fall_dataset"
    process_images(dataset_path)
