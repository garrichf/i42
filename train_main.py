import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, BatchNormalization
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
import tensorflow as tf
import os


# Load the training CSV file
train_file_path = 'Experiment_df.csv'
train_data = pd.read_csv(train_file_path)

# Load the validation CSV file
# val_file_path = 'combined_norm_removeBMLV.csv'  # Adjust the path to your validation data
# val_data = pd.read_csv(val_file_path)

# Function to prepare sequences and labels
def prepare_sequences(data, sequence_length):
    sequences = []
    labels = []
    grouped = data.groupby(['Folder Name'])  # Grouping the data by 'Folder Name'

    for name, group in grouped:
        feature_data = group.drop(columns=['Folder Name', 'File Name', 'FallType'])
        feature_data = feature_data.interpolate().fillna(0)  # Handle missing values
        data_array = feature_data.values
        
        # Create sequences
        for i in range(0, len(data_array) - sequence_length + 1):
            sequence = data_array[i:i + sequence_length]
            sequences.append(sequence)
            labels.append(group['FallType'].values[0])  # Assuming all frames in the sequence have the same label
            
    return np.array(sequences, dtype=np.float32), np.array(labels)

# Prepare training sequences and labels
sequence_length = 30
X_train, y_train = prepare_sequences(train_data, sequence_length)

#Make a consistent mask, identified an issue where there is two 0 values. 0 and 0.0 so we have to make it the same value for a consistent mask
X_train_masked = np.where((X_train == 0) | (X_train == 0.0), -1.0, X_train)

# print(X_train_masked)

# Prepare validation sequences and labels
# X_val, y_val = prepare_sequences(val_data, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_masked, y_train, test_size=0.20, random_state=42)


from tensorflow.keras import regularizers

# Now, padded_sequences can be used as input to the LSTM model.
# Define the LSTM model
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(None, X_train.shape[2])))
model.add(LSTM(128, return_sequences=False, kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Single output for binary classification

from sklearn.metrics import precision_score, recall_score, f1_score
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super(MetricsCallback, self).__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(self.X_val) > 0.5).astype("int32")  # Binary predictions
        precision = precision_score(self.y_val, y_pred)
        recall = recall_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)

        print(f"\nEpoch {epoch + 1}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Use this callback during model training
# metrics_callback = MetricsCallback(X_val, y_val)  # Assuming X_val and y_val are your validation data

optimizer = Adam(learning_rate=0.001)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',   # The metric to monitor (could be 'accuracy', etc.)
    factor=0.2,           # Factor by which the learning rate will be reduced
    patience=4,           # Number of epochs with no improvement after which learning rate is reduced
    min_lr=1e-6           # Minimum learning rate that ReduceLROnPlateau can reduce to
)

# Compile the model
'''
Loss functions can be experimented with
'''
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Recall, Precision
from keras.saving import register_keras_serializable

@register_keras_serializable()
def f1_score(y_true, y_pred):
    # Ensure y_true and y_pred are of the same shape
    print(f'y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}')  # Debugging line
    y_pred = K.squeeze(y_pred, axis=-1)  # Remove the last dimension if it's 1
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    
    # Convert to float32
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(y_pred, K.floatx())

    # Calculate True Positives, False Positives, and False Negatives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))      # TP + FP
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))        # TP + FN

    # Calculate precision and recall
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    # Calculate F1 Score
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision()])

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# from sklearn.utils.class_weight import compute_class_weight

# class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weights = dict(enumerate(class_weights))
# print(class_weights)

# Manually set class weights
class_weights = {
    0: 0.52,  # Weight for class 0 (non-fall)
    1: 9.527   # Weight for class 1 (fall), give higher weight to the minority class
}

# Train the model
# history = model.fit(X_train,y_train, batch_size=32, class_weight=class_weights, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr, metrics_callback])
history = model.fit(X_train,y_train, batch_size=32, class_weight=class_weights, epochs=25, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# Save the trained model
formatted_filename_modelname = f"falldetect_{datetime.now().strftime('%d%m%Y_%H%M')}.keras"
model.save(formatted_filename_modelname)

'''
Model testing
'''
# Predict on the test set (assuming you have a separate test set)
y_pred = model.predict(X_test)
# Convert sigmoid predictions to binary predictions
y_pred_classes = (y_pred > 0.5).astype(int)


# Generate classification report
report = classification_report(y_test, y_pred_classes, target_names=[str(cls) for cls in np.unique(y_train)])
print(report)

start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.4f} seconds")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
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
    ax.set_title(f"True: {y_test[i]}, Pred: {y_pred_classes[i]}")
    
    # Remove axis ticks
    ax.axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
