from ultralytics import YOLO
import numpy as np
import tensorflow as tf

# Define constants
SEQUENCE_LENGTH = 10  # Length of keypoint sequences for LSTM
NUM_KEYPOINTS = 17    # Adjust based on the model you're using (e.g., COCO keypoints)
EPOCHS = 50           # Number of training epochs
BATCH_SIZE = 16       # Batch size for training

def extract_keypoints(model, image):
    results = model(image)
    keypoints = results.keypoints  # Extract keypoints from results
    return keypoints

def prepare_sequences(keypoints, sequence_length=SEQUENCE_LENGTH):
    sequences = []
    for i in range(len(keypoints) - sequence_length + 1):
        sequences.append(keypoints[i:i + sequence_length])
    return np.array(sequences)

def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_path, model_path, epochs=EPOCHS, batch=BATCH_SIZE):
    # Load the YOLO model
    yolo_model = YOLO(model_path)

    # Load your dataset here (e.g., images and corresponding labels)
    # For demonstration, let's assume you have a list of image paths and labels
    image_paths = []  # Populate this with paths to your images
    labels = []       # Populate this with corresponding labels (0 for no fall, 1 for fall)

    all_sequences = []
    all_labels = []

    # Extract keypoints and prepare sequences
    for img_path, label in zip(image_paths, labels):
        keypoints = extract_keypoints(yolo_model, img_path)
        sequences = prepare_sequences(keypoints)
        all_sequences.extend(sequences)
        all_labels.extend([label] * len(sequences))  # Duplicate label for each sequence

    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)

    # Create LSTM model
    lstm_model = create_lstm_model(input_shape=(SEQUENCE_LENGTH, NUM_KEYPOINTS, 2))  # Assuming 2D keypoints

    # Train the model
    lstm_model.fit(all_sequences, all_labels, epochs=epochs, batch_size=batch)

    # Print training results
    print("Training completed.")

if __name__ == "__main__":
    # Specify the path to your dataset and model
    dataset_path = r"C:\Users\User\Desktop\fall-detection\fall_dataset.yaml"  # Path to your dataset YAML file
    model_path = r"C:\Users\User\Desktop\fall-detection\yolov8n-pose.pt"  # Path to your pre-trained model

    # Train the model
    train_model(data_path=dataset_path, model_path=model_path, epochs=EPOCHS, batch=BATCH_SIZE)