from tensorflow.keras.models import load_model
import pandas
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import tensorflow as tf

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Load the CSV file
# file_path = 'df_impute_acceleration_test.csv'
# file_path = 'Experiment_df.csv'
# file_path = 'Experiment_df_balanced_removedBMLV_sorted.csv'
file_path = 'Fall.csv'
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
#Make a consistent mask, identified an issue where there is two 0 values. 0 and 0.0 so we have to make it the same value for a consistent mask
X_train_masked = np.where((sequences == 0) | (sequences == 0.0), -1.0, sequences)
labels = np.array(labels)

# Convert labels to categorical
'''
Might need to change the method the target variable is identified, not sure
'''
# categorical_labels = to_categorical(labels, num_classes=len(np.unique(labels)))

# Shuffling the sequences randomly, to prevent the model from memorizing the sequence and using that to make predictions
# X_test, y_test = shuffle(sequences, categorical_labels)

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(sequences, categorical_labels, test_size=0.2, random_state=42)

# Load the pre-trained model
# model = load_model('falldetect_30082024_0320.keras')
# model = load_model('falldetect_finetuned_full_40percent_08092024_1739.keras')
# model = load_model('falldetect_finetuned_full_40percent_5epoch_08092024_1758.keras')
from tensorflow.keras import backend as K
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



model = load_model('falldetect_30092024_1747.keras', custom_objects={'f1_score': f1_score})
print(model.metrics)  # Check what metrics are defined
# Evaluate the model
results = model.evaluate(X_train_masked, labels)
# test_loss, test_accuracy, additional_metric = results  # Adjust based on the number of metrics

# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")

# Select a single sequence to test (for example, the first sequence)
test_sequences = X_train_masked[:1]  # Select the first sequence and keep it in the correct shape
# Predict on the test set
start_time = time.time()
y_pred = model.predict(test_sequences)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.4f} seconds")

# score0 = round(y_pred[0][0],2)
# score1 = round(y_pred[0][1],2)
# print("Class 0: " + str(score0))
# print("Class 1: " + str(score1))
y_pred_classes = (y_pred > 0.8).astype(int)
print(y_pred_classes[0])

'''
Classification report
'''
# Predict on the test set (assuming you have a separate test set)
y_pred = model.predict(X_train_masked)
y_pred_classes = (y_pred > 0.8).astype(int)

# Generate classification report
report = classification_report(labels, y_pred_classes, target_names=["0","1"])
print(report)

# Compute confusion matrix
conf_matrix = confusion_matrix(labels, y_pred_classes)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_curve, auc


# Calculate FPR and TPR
fpr, tpr, thresholds = roc_curve(labels, y_pred_classes)

# Calculate AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

print(f"AUC: {roc_auc:.4f}")

# Plotting ROC curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line (no discrimination)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.show()
