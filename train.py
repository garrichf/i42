import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, f1_score
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
file_path = 'output.csv'
data = pd.read_csv(file_path)

# Grouping the data by 'Folder Name' and 'File Name'
grouped = data.groupby(['Folder Name', 'File Name'])


'''
Need to edit the way sequences are made, im pretty sure its wrong right now.
'''
# Prepare sequences and labels
sequences = []
labels = []

for name, group in grouped:
    # Drop non-feature columns
    feature_data = group.drop(columns=['Folder Name', 'File Name', 'FallType'])
    
    # Handle missing values (interpolation or filling with zero)
    '''
    Need to edit the preprocessing mechanism, whether here or somewhere else to remove the frames with no subject inside
    '''
    feature_data = feature_data.interpolate().fillna(0)
    
    sequences.append(feature_data.values)
    labels.append(group['FallType'].values[0])  # Assuming 'FallType' is the label

# Convert to numpy arrays
sequences = np.array(sequences, dtype=object)  # Keep as object array to handle varying sequence lengths
labels = np.array(labels)

# Check the shape of a few sequences and their corresponding labels
# print([seq.shape for seq in sequences[:5]], labels[5:])

# Padding sequences to the maximum length
# Ensure right padding ('post), left padding is ('pre')
padded_sequences = pad_sequences(sequences, padding='post', value=0.0, dtype='float32')
print(padded_sequences[0])

# Convert labels to categorical
'''
Might need to change the method the target variable is identified, not sure
'''
categorical_labels = to_categorical(labels, num_classes=len(np.unique(labels)))

# Shuffling the sequences randomly, to prevent the model from memorizing the sequence and using that to make predictions
padded_sequences, categorical_labels = shuffle(padded_sequences, categorical_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, categorical_labels, test_size=0.2, random_state=42)


# Now, padded_sequences can be used as input to the LSTM model.
# Define the LSTM model
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(None, X_train.shape[2])))  # Input shape: (timesteps, features)
model.add(LSTM(256, return_sequences=False, use_bias=False))  # LSTM layer with 256 units
model.add(Dropout(0.2))  # Dropout layer
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
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train,y_train, batch_size=32, epochs=300, validation_split=0.2, callbacks=[early_stopping])

# Save the trained model
# current_datetime = datetime.now()
# formatted_filename_modelname = current_datetime('falldetection_version_%d%m%Y%H%M.keras')
formatted_filename_modelname = "falldetect_test.h5"
model.save(formatted_filename_modelname)



'''
Model testing
'''
# Predict on the test set (assuming you have a separate test set)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert categorical_labels to class indices for test set
y_true = np.argmax(y_test, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred_classes)
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

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

# Print classification report
print(classification_report(y_true, y_pred_classes, target_names=[str(cls) for cls in np.unique(labels)]))

import matplotlib.pyplot as plt

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
