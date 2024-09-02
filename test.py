from tensorflow.keras.models import load_model
import pandas
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import tensorflow as tf

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Load the CSV file
file_path = 'df_impute_acceleration.csv'
data = pd.read_csv(file_path)

# Grouping the data by 'Folder Name' and 'File Name'
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

# Load the pre-trained model
model = load_model('falldetect_30082024_0320.keras')

# Manually set 'use_cudnn' to False for each LSTM layer after loading
# for layer in model.layers:
#     if isinstance(layer, LSTM):
#         layer._could_use_gpu_kernel = False

# Evaluate the model on the test data



# test_loss, test_accuracy = model.predict(X_test)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Select a single sequence to test (for example, the first sequence)
test_sequences = sequences[:30]  # Select the first sequence and keep it in the correct shape

# Predict on the test set
start_time = time.time()
y_pred = model.predict(test_sequences)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.4f} seconds")

score0 = round(y_pred[0][0],2)
score1 = round(y_pred[0][1],2)
print("Class 0: " + str(score0))
print("Class 1: " + str(score1))
y_pred_classes = np.argmax(y_pred, axis=1)
# print(y_pred_classes)

# Convert categorical_labels to class indices for test set
y_true = np.argmax(y_test, axis=1)

# Calculate metrics
# accuracy = accuracy_score(y_true, y_pred_classes)
# recall = recall_score(y_true, y_pred_classes, average='weighted')
# f1 = f1_score(y_true, y_pred_classes, average='weighted')

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")

# # Compute confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred_classes)

# # Plot confusion matrix using seaborn
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=np.unique(labels), yticklabels=np.unique(labels))
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

# # Print classification report
# print(classification_report(y_true, y_pred_classes, target_names=[str(cls) for cls in np.unique(labels)]))
