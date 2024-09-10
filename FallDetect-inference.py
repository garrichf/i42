import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import random

model = YOLO('yolov8n-pose.pt')

image_path = 'fall_dataset/'
fall_path = os.path.join(image_path, 'fall')
no_fall_path = os.path.join(image_path, 'no_fall')

def predict_fall(image_path):
    results = model(image_path)
    for r in results:
        for box in r.boxes:
            if box.cls == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                aspect_ratio = (y2 - y1) / (x2 - x1)
                if aspect_ratio < 1:
                    return "fall"
    return "no_fall"

def evaluate_model(test_data):
    correct = 0
    total = len(test_data)
    for img_path, true_label in test_data:
        pred_label = predict_fall(img_path)
        if pred_label == true_label:
            correct += 1
    return correct / total

validation_data = []
for img in os.listdir(fall_path):
    validation_data.append((os.path.join(fall_path, img), 'fall'))
for img in os.listdir(no_fall_path):
    validation_data.append((os.path.join(no_fall_path, img), 'no_fall'))

random.shuffle(validation_data)

accuracy = evaluate_model(validation_data)
print(f'Accuracy: {accuracy:.2f}')

def get_label_color(pred_label, true_label):
    return 'black' if pred_label == true_label else 'red'

plt.figure(figsize=(20, 30))
for i, (image_path, true_label) in enumerate(validation_data[:30]):
    ax = plt.subplot(6, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    image = Image.open(image_path)
    plt.imshow(image)
    
    pred_label = predict_fall(image_path)
    color = get_label_color(pred_label, true_label)
    ax.xaxis.label.set_color(color)
    plt.xlabel(f'Pred: {pred_label}\nTrue: {true_label}')

plt.tight_layout()
plt.show()

print(f'Accuracy: {accuracy:.2f}')