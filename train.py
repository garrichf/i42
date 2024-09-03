import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define constants
max_length = 17  # Adjust this based on your dataset (number of keypoints per frame)

# Load dataset
df = pd.read_csv('training_data_with_features.csv')

# Extract features and labels
X = df[['Nose_X', 'Nose_Y', 'Left Eye_X', 'Left Eye_Y', 'Right Eye_X', 'Right Eye_Y',
        'Left Ear_X', 'Left Ear_Y', 'Right Ear_X', 'Right Ear_Y',
        'Left Shoulder_X', 'Left Shoulder_Y', 'Right Shoulder_X', 'Right Shoulder_Y',
        'Left Elbow_X', 'Left Elbow_Y', 'Right Elbow_X', 'Right Elbow_Y',
        'Left Wrist_X', 'Left Wrist_Y', 'Right Wrist_X', 'Right Wrist_Y',
        'Left Hip_X', 'Left Hip_Y', 'Right Hip_X', 'Right Hip_Y',
        'Left Knee_X', 'Left Knee_Y', 'Right Knee_X', 'Right Knee_Y',
        'Left Ankle_X', 'Left Ankle_Y', 'Right Ankle_X', 'Right Ankle_Y']].values
y = df['FallType'].values

X = pad_sequences(X, maxlen=max_length, dtype='float32')

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(max_length, 34), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))  # Assuming binary classification (fall or no fall)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save model
model.save('falldetection_version_220820241432.keras')
