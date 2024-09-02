import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from datetime import datetime
import os


# Load the CSV file
file_path = 'df_impute_acceleration.csv'
data = pd.read_csv(file_path)

# Grouping the data by 'Folder Name'
grouped = data.groupby(['Folder Name'])


'''
Need to edit the way sequences are made, im pretty sure its wrong right now.
'''
# Prepare sequences and labels
sequence_length = 30  # Define the desired sequence length
sequences = []
labels = []

for name, group in grouped:
    # Drop non-feature columns
    feature_data = group.drop(columns=['Folder Name', 'File Name', 'FallType'])
    
    # Handle missing values (interpolation or filling with zero)
    feature_data = feature_data.interpolate().fillna(0)
    
    # Convert to numpy array
    data = feature_data.values
    
    # Create sequences of the specified length
    for i in range(0, len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
        labels.append(group['FallType'].values[0])  # Assuming all frames in the sequence have the same label

# Convert to numpy arrays
sequences = np.array(sequences, dtype=np.float32)  # Keep as object array to handle varying sequence lengths
labels = np.array(labels)

# Convert labels to categorical
'''
Might need to change the method the target variable is identified, not sure
'''
categorical_labels = to_categorical(labels, num_classes=len(np.unique(labels)))

# Shuffling the sequences randomly, to prevent the model from memorizing the sequence and using that to make predictions
sequences, categorical_labels = shuffle(sequences, categorical_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, categorical_labels, test_size=0.2, random_state=42)


# Now, padded_sequences can be used as input to the LSTM model.
# Define the LSTM model
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(None, X_train.shape[2])))  # Input shape: (timesteps, features)
model.add(LSTM(256, return_sequences=False))  # LSTM layer with 256 units
model.add(Dropout(0.2))  # Dropout layer
# model.add(LSTM(64, return_sequences=False))  # No need to return sequences for the final LSTM layer
# model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(Dense(64, activation='relu'))  # Dense layer with 64 units
model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(len(np.unique(labels)), activation='softmax'))  # Output layer with softmax activation


# Define the learning rate schedule
initial_learning_rate = 0.001
# this lr_schedule, dynamically adjusts the learning rate for better model training
lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

# Create the Adam optimizer with the learning rate schedule
optimizer = Adam(learning_rate=lr_schedule)

# Compile the model
'''
Loss functions can be experimented with
'''
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

class_weights = {0: 0.55, 1: 1.82} # Class distribution [0:64.572265, 1:35.427735]

# Train the model
history = model.fit(X_train,y_train, batch_size=32, class_weight=class_weights, epochs=300, validation_split=0.2, callbacks=[early_stopping])

# Save the trained model
formatted_filename_modelname = f"falldetect_{datetime.now().strftime('%d%m%Y_%H%M')}.keras"
model.save(formatted_filename_modelname)

'''
Model testing
'''
# Predict on the test set (assuming you have a separate test set)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert categorical_labels to class indices for test set
y_true = np.argmax(y_test, axis=1)

# Generate classification report
report = classification_report(y_true, y_pred_classes, target_names=[str(cls) for cls in np.unique(labels)])
print(report)

start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.4f} seconds")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Assume X_test contains image data in a format like (num_samples, height, width, channels)

# Set up a grid to display images
fig, axes = plt.subplots(8, 8, figsize=(10, 10))  # 3x3 grid, adjust as needed

# Loop through a few images and their predictions
for i, ax in enumerate(axes.flat):
    if i >= len(X_test):
        break
    
    # Display the image
    ax.imshow(X_test[i], cmap='gray')  # Change 'gray' if images are in color
    
    # Set title with true and predicted labels
    ax.set_title(f"True: {y_true[i]}, Pred: {y_pred_classes[i]}")
    
    # Remove axis ticks
    ax.axis('off')

# Adjust layout
plt.tight_layout()
plt.show()
