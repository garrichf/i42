import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import pandas as pd


try:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
except Exception as error:
    print(f"Failed to load the model: {error}")


KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}
# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.35

# Size of the square crop region around the person's center.
INPUT_SIZE = 256

def movenet(input_image):
    """
    Perform pose estimation using the MoveNet model.
    Args:
        input_image (tf.Tensor): A tensor representing the input image. The image should be preprocessed 
                                 to match the input requirements of the MoveNet model.
    Returns:
        np.ndarray: A numpy array containing the keypoints and their corresponding scores. The output 
                    shape is [1, 1, 17, 3], where 17 represents the number of keypoints and 3 represents 
                    the (y, x, score) for each keypoint.
    Raises:
        Exception: If there is an error loading the MoveNet model from TensorFlow Hub.
    """


    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    
    return keypoints_with_scores 

def init_crop_region(image_height, image_width):
  """Defines the default crop region.

  The function provides the initial crop region (pads the full image from both
  sides to make it a square image) when the algorithm cannot reliably determine
  the crop region from the previous frame.
  """
  if image_width > image_height:
    box_height = image_width / image_height
    box_width = 1.0
    y_min = (image_height / 2 - image_width / 2) / image_height
    x_min = 0.0
  else:
    box_height = 1.0
    box_width = image_height / image_width
    y_min = 0.0
    x_min = (image_width / 2 - image_height / 2) / image_width

  return {
    'y_min': y_min,
    'x_min': x_min,
    'y_max': y_min + box_height,
    'x_max': x_min + box_width,
    'height': box_height,
    'width': box_width
  }

def torso_visible(keypoints):
  """Checks whether there are enough torso keypoints.

  This function checks whether the model is confident at predicting one of the
  shoulders/hips which is required to determine a good crop region.
  """
  return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE) and
          (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE))

def determine_torso_and_body_range(
    keypoints, target_keypoints, center_y, center_x):
  """Calculates the maximum distance from each keypoints to the center location.

  The function returns the maximum distances from the two sets of keypoints:
  full 17 keypoints and 4 torso keypoints. The returned information will be
  used to determine the crop size. See determineCropRegion for more detail.
  """
  torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
  max_torso_yrange = 0.0
  max_torso_xrange = 0.0
  for joint in torso_joints:
    dist_y = abs(center_y - target_keypoints[joint][0])
    dist_x = abs(center_x - target_keypoints[joint][1])
    if dist_y > max_torso_yrange:
      max_torso_yrange = dist_y
    if dist_x > max_torso_xrange:
      max_torso_xrange = dist_x

  max_body_yrange = 0.0
  max_body_xrange = 0.0
  for joint in KEYPOINT_DICT.keys():
    if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
      continue
    dist_y = abs(center_y - target_keypoints[joint][0]);
    dist_x = abs(center_x - target_keypoints[joint][1]);
    if dist_y > max_body_yrange:
      max_body_yrange = dist_y

    if dist_x > max_body_xrange:
      max_body_xrange = dist_x

  return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(
      keypoints, image_height,
      image_width):
  """Determines the region to crop the image for the model to run inference on.

  The algorithm uses the detected joints from the previous frame to estimate
  the square region that encloses the full body of the target person and
  centers at the midpoint of two hip joints. The crop size is determined by
  the distances between each joints and the center point.
  When the model is not confident with the four torso joint predictions, the
  function returns a default crop which is the full image padded to square.
  """
  target_keypoints = {}
  for joint in KEYPOINT_DICT.keys():
    target_keypoints[joint] = [
      keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
      keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
    ]

  if torso_visible(keypoints):
    center_y = (target_keypoints['left_hip'][0] +
                target_keypoints['right_hip'][0]) / 2;
    center_x = (target_keypoints['left_hip'][1] +
                target_keypoints['right_hip'][1]) / 2;

    (max_torso_yrange, max_torso_xrange,
      max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
          keypoints, target_keypoints, center_y, center_x)

    crop_length_half = np.amax(
        [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
          max_body_yrange * 1.2, max_body_xrange * 1.2])

    tmp = np.array(
        [center_x, image_width - center_x, center_y, image_height - center_y])
    crop_length_half = np.amin(
        [crop_length_half, np.amax(tmp)]);

    crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

    if crop_length_half > max(image_width, image_height) / 2:
      return init_crop_region(image_height, image_width)
    else:
      crop_length = crop_length_half * 2;
      return {
        'y_min': crop_corner[0] / image_height,
        'x_min': crop_corner[1] / image_width,
        'y_max': (crop_corner[0] + crop_length) / image_height,
        'x_max': (crop_corner[1] + crop_length) / image_width,
        'height': (crop_corner[0] + crop_length) / image_height -
            crop_corner[0] / image_height,
        'width': (crop_corner[1] + crop_length) / image_width -
            crop_corner[1] / image_width
      }
  else:
    return init_crop_region(image_height, image_width)
  
def crop_and_resize(image, crop_region, crop_size):
  """Crops and resize the image to prepare for the model input."""
  boxes=[[crop_region['y_min'], crop_region['x_min'],
          crop_region['y_max'], crop_region['x_max']]]
  output_image = tf.image.crop_and_resize(
      image, box_indices=[0], boxes=boxes, crop_size=crop_size)
  return output_image

def run_inference(movenet, image, crop_region, crop_size):
  """Runs model inference on the cropped region.

  The function runs the model inference on the cropped region and updates the
  model output to the original image coordinate system.
  """
  image_height, image_width, _ = image.shape
  input_image = crop_and_resize(
    tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
  # Run model inference.
  keypoints_with_scores = movenet(input_image)
  # Update the coordinates.
  for idx in range(17):
    keypoints_with_scores[0, 0, idx, 0] = (
        crop_region['y_min'] * image_height +
        crop_region['height'] * image_height *
        keypoints_with_scores[0, 0, idx, 0]) / image_height
    keypoints_with_scores[0, 0, idx, 1] = (
        crop_region['x_min'] * image_width +
        crop_region['width'] * image_width *
        keypoints_with_scores[0, 0, idx, 1]) / image_width
  return keypoints_with_scores

def keypoints_to_dataframe(keypoints_with_scores):
  """
  Converts keypoints with scores to a pandas DataFrame, reorganizes the columns, and removes eye and ear columns.
  
  Args:
    keypoints_with_scores (numpy.ndarray): A numpy array of shape 
    (1, 1, 17, 3) containing keypoints and their scores. The first 
    dimension is the batch size, the second dimension is the number 
    of instances, the third dimension is the number of keypoints 
    (17 for MoveNet), and the fourth dimension contains the 
    coordinates (x, y) and the score.
    frame_idx (int): The index of the frame to be added as a column in the DataFrame.
  
  Returns:
    pandas.DataFrame: A DataFrame containing the keypoints' coordinates 
    with columns named after the keypoint names followed by '_X' 
    and '_Y' for the x and y coordinates respectively, reorganized 
    such that x-coordinates come before y-coordinates for each keypoint, 
    and with eye and ear columns removed.
  """
  keypoints = keypoints_with_scores[0, 0, :, :2]  # Extract keypoints
  keypoint_names = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear', 'Left Shoulder', 'Right Shoulder', 
    'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
  ]
  
  # Create column names
  columns = []
  for name in keypoint_names:
    columns.append(f'{name}_Y')
    columns.append(f'{name}_X')

  # Flatten the keypoints array and create a DataFrame
  keypoints_flat = keypoints.flatten()
  df = pd.DataFrame([keypoints_flat], columns=columns)
  
  # Reorganize columns so that x comes before y
  x_columns = [col for col in columns if '_X' in col]
  y_columns = [col for col in columns if '_Y' in col]
  reorganized_columns = []

  for x_col, y_col in zip(x_columns, y_columns):
    reorganized_columns.append(x_col)
    reorganized_columns.append(y_col)
  df = df[reorganized_columns]

  return df

def load_stream(stream_path):
    """
    Load a video stream from the specified path.

    Args:
        stream_path (str): The path to the video stream.

    Returns:
        cv2.VideoCapture: The video stream object.
    """
    stream = cv2.VideoCapture(stream_path)
    if not stream.isOpened():
        print("Error: Could not load stream.")
        return None

    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            print("Reached the end of the stream or could not read the frame.")
            break
            
        yield frame

    stream.release()
    cv2.destroyAllWindows()

def frame_inference(frame, movenet, input_size, init_crop_region, run_inference, determine_crop_region):
    
    # Get the frame dimensions
    image_height, image_width, _ = frame.shape

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize the crop region
    crop_region = init_crop_region(image_height, image_width)

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a tensor
    frame_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
    # Add batch dimension
    frame_tensor = tf.expand_dims(frame_tensor, axis=0)

    # Run inference
    keypoints_with_scores = run_inference(
        movenet, frame_tensor[0], crop_region,
        crop_size=[input_size, input_size])
    
    # Update crop region
    crop_region = determine_crop_region(
        keypoints_with_scores, image_height, image_width)

    # Set coordinates with low confidence to 0
    keypoints_with_scores[0, 0, keypoints_with_scores[0, 0, :, 2] < MIN_CROP_KEYPOINT_SCORE, :2] = 0

    # Convert keypoints to DataFrame
    df = keypoints_to_dataframe(keypoints_with_scores)
    
    return df

def MOVENET_pose(frame):
    
    print("MoveNet is Running")
    # Load the MoveNet model to process frame.
    df = frame_inference(frame, movenet, INPUT_SIZE, init_crop_region, run_inference, determine_crop_region)

    return df