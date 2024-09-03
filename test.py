import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from pose_tracking import preprocess_keypoints_for_model, compute_angle

# Load the fall detection model
model = load_model('falldetection_version_220820241432.keras')

# Define max_length for padding
max_length = 17  # Adjust this based on your dataset (number of keypoints per frame)

# Read the test dataset
df = pd.read_csv('test_data_with_features.csv')

# Extract features and labels
X = df[['Nose_X', 'Nose_Y', 'Left Eye_X', 'Left Eye_Y', 'Right Eye_X', 'Right Eye_Y',
        'Left Ear_X', 'Left Ear_Y', 'Right Ear_X', 'Right Ear_Y',
        'Left Shoulder_X', 'Left Shoulder_Y', 'Right Shoulder_X', 'Right Shoulder_Y',
        'Left Elbow_X', 'Left Elbow_Y', 'Right Elbow_X', 'Right Elbow_Y',
        'Left Wrist_X', 'Left Wrist_Y', 'Right Wrist_X', 'Right Wrist_Y',
        'Left Hip_X', 'Left Hip_Y', 'Right Hip_X', 'Right Hip_Y',
        'Left Knee_X', 'Left Knee_Y', 'Right Knee_X', 'Right Knee_Y',
        'Left Ankle_X', 'Left Ankle_Y', 'Right Ankle_X', 'Right Ankle_Y']].values
y_true = df['FallType'].values

X = pad_sequences(X, maxlen=max_length, dtype='float32')

# Predict
y_pred_proba = model.predict(X)
y_pred = np.argmax(y_pred_proba, axis=1)

# Evaluate
print(classification_report(y_true, y_pred))
