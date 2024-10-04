# MoveNet Pose Estimation version 2.0
# Description: This file contains helper functions and the stream inference functions for the MoveNet model.
# Implemented by Nick Bui adapting MoveNet's codebased https://www.tensorflow.org/hub/tutorials/movenet.

# Comment:
# This model saw the basic functions of the MoveNet implementation with successful laod of the model, performing the pose estimation, and drawing the keypoints on the image.
# There are main 2 functions the user can use to perform the pose estimation on the loaded video or live video stream.
# Latest Update:
# - Added the function to convert keypoints with scores to a pandas DataFrame, reorganize the columns, and remove eye and ear columns.
# - Added the function to draw the bounding box around the detected keypoints. 
# - Added the function to calculate the acceleration for specified columns in a DataFrame.
# - Separated the function for loading the video file.
# - Separated the function for display the processed frame.
# - Attempt batch processing of frames for pose estimation to improve performance but it is not working as expected.

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import pandas as pd
import time

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches



# Dictionary that maps from joint names to keypoint indices.
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

# Mapping from keypoint indices to human-readable names.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.25):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image(image, keypoints_with_scores, crop_region=None, close_figure=False, output_image_height=None):
    
    """
    Draws keypoints and edges on an image and returns the resulting image.
    Args:
        image (np.ndarray): The input image on which to draw.
        keypoints_with_scores (np.ndarray): Array containing keypoints and their scores.
        crop_region (dict, optional): Dictionary specifying the crop region with keys 'x_min', 'y_min', 'x_max', and 'y_max'. Defaults to None.
        close_figure (bool, optional): Whether to close the figure after drawing. Defaults to False.
        output_image_height (int, optional): Desired height of the output image. If specified, the output image will be resized to this height while maintaining the aspect ratio. Defaults to None.
    Returns:
        np.ndarray: The image with keypoints and edges drawn on it.
    """

    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin,ymin),rec_width,rec_height,
            linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC)
    return image_from_plot

# Define the input size for the MoveNet model
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

    # Load the MoveNet model from TensorFlow Hub
    try:
        module = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4")
    except Exception as error:
        print(f"Failed to load the model: {error}")

    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    
    return keypoints_with_scores 

# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.2

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

def keypoints_to_dataframe(keypoints_with_scores, frame_index):
    """
    Converts keypoints with scores to a pandas DataFrame, reorganizes the columns, and removes eye and ear columns.
    
    Args:
      keypoints_with_scores (numpy.ndarray): A numpy array of shape 
      (1, 1, 17, 3) containing keypoints and their scores. The first 
      dimension is the batch size, the second dimension is the number 
      of instances, the third dimension is the number of keypoints 
      (17 for MoveNet), and the fourth dimension contains the 
      coordinates (x, y) and the score.
      frame_index (int): The index of the frame.
    
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
    
    # # Add frame index column
    # df['frame_idx'] = frame_index
    
    # # Reorder columns to have frame_idx first
    # df = df[['frame_idx'] + reorganized_columns]
    
    # Remove eye and ear columns
    columns_to_remove = [
        'Left Eye_Y', 'Left Eye_X', 
        'Right Eye_Y', 'Right Eye_X', 
        'Left Ear_Y', 'Left Ear_X', 
        'Right Ear_Y', 'Right Ear_X'
    ]
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
    
    return df

def add_bounding_box(image, keypoints_with_scores, threshold=0.4, margin=0.2):
    """
    Add a bounding box to the image based on the keypoints with an additional margin.

    Args:
        image (numpy.ndarray): The image with keypoints.
        keypoints_with_scores (numpy.ndarray): The keypoints with scores.
        threshold (float): The confidence threshold to consider a keypoint.
        margin (float): The margin to add around the bounding box as a percentage of the box dimensions.

    Returns:
        numpy.ndarray: The image with the bounding box.
    """
    # Extract keypoints
    keypoints = keypoints_with_scores[0, 0, :, :2]
    scores = keypoints_with_scores[0, 0, :, 2]

    # Filter keypoints based on the confidence threshold
    valid_keypoints = keypoints[scores > threshold]

    if valid_keypoints.size == 0:
        return image

    # Calculate the bounding box coordinates
    x_min = np.min(valid_keypoints[:, 1])
    y_min = np.min(valid_keypoints[:, 0])
    x_max = np.max(valid_keypoints[:, 1])
    y_max = np.max(valid_keypoints[:, 0])

    # Convert to integer coordinates
    x_min = int(x_min * image.shape[1])
    y_min = int(y_min * image.shape[0])
    x_max = int(x_max * image.shape[1])
    y_max = int(y_max * image.shape[0])

    # Add margin to the bounding box
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_min = max(0, x_min - int(margin * box_width))
    y_min = max(0, y_min - int(margin * box_height))
    x_max = min(image.shape[1], x_max + int(margin * box_width))
    y_max = min(image.shape[0], y_max + int(margin * box_height))

    # Draw the bounding box on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return image

def load_video(video_path):
    stream = cv2.VideoCapture(video_path)
    
    if not stream.isOpened():
        print("Error: Could not load video.")
        return None
    
    stream.release()
    return stream

def display_processed_frame(frame):
    """
    Displays the processed frame as fast as possible, maintaining real-time performance.
    
    Args:
        frame (np.ndarray): The frame to display.
    
    Returns:
        bool: Whether to continue displaying (False if 'q' is pressed).
    """
    cv2.imshow('Processed Frame', frame)

    # Check if the user wants to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False  # Signal to stop the display
    return True  # Continue the display

def infer_from_video(video_path, movenet, input_size, init_crop_region, run_inference, draw_prediction_on_image, determine_crop_region, confidence_threshold=0.28, batch_size=10):
    stream = load_video(video_path)
    if stream is None:
        return
    
    # Get video properties
    num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    image_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = stream.get(cv2.CAP_PROP_FPS)
    delay_per_frame = 1000 / fps  # Time per frame in milliseconds for real-time synchronization
    
    crop_region = init_crop_region(image_height, image_width)
    
    keypoints_list = []
    frames_batch = []
    frame_idx = 0
    batch_idx = 0
    keep_displaying = True
    
    while True:
        start_time = time.time()  # Track when processing of the frame starts

        ret, frame = stream.read()
        if not ret:
            break
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_batch.append(frame_rgb)
        frame_idx += 1
        
        # Process batch if we've collected enough frames
        if len(frames_batch) == batch_size or frame_idx == num_frames:
            # Convert frames batch to a tensor
            frames_np = np.array(frames_batch)
            frames_tensor = tf.convert_to_tensor(frames_np, dtype=tf.uint8)
            
            batch_keypoints_list = []
            # Process each frame in the batch
            for i, frame_rgb in enumerate(frames_batch):
                # Run inference on the frame
                keypoints_with_scores = run_inference(
                    movenet, frames_tensor[i], crop_region, crop_size=[input_size, input_size])
                
                # Set coordinates with low confidence to -1
                keypoints_with_scores[0, 0, keypoints_with_scores[0, 0, :, 2] < confidence_threshold, :2] = None

                # Convert keypoints to DataFrame
                keypoints_df = keypoints_to_dataframe(keypoints_with_scores, frame_idx - len(frames_batch) + i)
                keypoints_df.insert(0, 'frame_idx', frame_idx - len(frames_batch) + i)
                keypoints_df.insert(1, 'batch_idx', batch_idx)
                batch_keypoints_list.append(keypoints_df)

                # # Draw predictions and add bounding box
                # output_image = draw_prediction_on_image(frame_rgb, keypoints_with_scores)
                # output_image_with_bbox = add_bounding_box(output_image, keypoints_with_scores)

                # # Convert frame back to BGR for display
                # output_image_with_bbox_bgr = cv2.cvtColor(output_image_with_bbox, cv2.COLOR_RGB2BGR)

                # # Display the frame using the separate display function
                # keep_displaying = display_processed_frame(output_image_with_bbox_bgr)

                # if not keep_displaying:
                #     break

                # # Ensure real-time display by accounting for processing time
                # processing_time = (time.time() - start_time) * 1000  # Processing time in milliseconds
                # if delay_per_frame > processing_time:
                #     time.sleep((delay_per_frame - processing_time) / 1000)  # Wait if ahead of schedule
            
            # Clear the batch for the next set of frames
            frames_batch = []
            batch_idx += 1

            # Concatenate keypoints from the current batch
            batch_keypoints_df = pd.concat(batch_keypoints_list, ignore_index=True)

            # Output the DataFrame for the batch
            yield batch_keypoints_df  # Return the DataFrame for this batch for further processing

        if not keep_displaying:
            break
    
    # Release the video capture object
    # stream.release()
    # cv2.destroyAllWindows()

def infer_from_camera(movenet, input_size, init_crop_region, run_inference, draw_prediction_on_image, determine_crop_region):
    """
    Capture live stream from the system camera and run inference on the captured frames.

    Args:
        movenet: The pose estimation model.
        input_size: The input size for the model.
        init_crop_region: Function to initialize the crop region.
        run_inference: Function to run inference on a frame.
        draw_prediction_on_image: Function to draw predictions on a frame.
        determine_crop_region: Function to determine the crop region based on keypoints.

    Returns:
        None
    """
    # Open the system camera
    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize crop region
    ret, frame = stream.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame_rgb.shape
    crop_region = init_crop_region(image_height, image_width)

    frame_idx = 0  # Initialize frame index

    while True:
        ret, frame = stream.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
        frame_tensor = tf.expand_dims(frame_tensor, axis=0)  # Add batch dimension

        # Run inference
        keypoints_with_scores = run_inference(
            movenet, frame_tensor[0], crop_region,
            crop_size=[input_size, input_size])

        # Convert keypoints to DataFrame
        keypoints_df = keypoints_to_dataframe(keypoints_with_scores)

        # Add frame index to DataFrame
        keypoints_df['frame_idx'] = frame_idx

        # Increment frame index
        frame_idx += 1

        # Draw predictions on the frame
        output_frame = draw_prediction_on_image(
            frame_rgb, keypoints_with_scores, crop_region=None,
            close_figure=True, output_image_height=300)
        
        # Update crop region
        crop_region = determine_crop_region(
            keypoints_with_scores, image_height, image_width)
        
        # Add bounding box to the frame
        output_frame = add_bounding_box(output_frame, keypoints_with_scores)

        # Display the live stream with predictions
        cv2.imshow('Live Stream', cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()

def compute_angle(p1, p2, p3):
    """
    Computes the angle (in degrees) between three points (p1, p2, p3) in a 2D plane.

    The angle is calculated at point p2, with p1 and p3 forming the arms of the angle.

    Parameters:
    p1 (tuple): A tuple representing the coordinates (x, y) of the first point.
    p2 (tuple): A tuple representing the coordinates (x, y) of the second point (vertex of the angle).
    p3 (tuple): A tuple representing the coordinates (x, y) of the third point.

    Returns:
    float: The angle in degrees between the three points.
    """
    # Compute the vectors between the points
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Compute the angle between the vectors
    dot_product = np.dot(v1, v2)
    # Compute the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    # Compute the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    # Compute the angle in degrees
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    return angle

def add_angles(df):
    """
    Adds various angles to the DataFrame based on key body points.

    Parameters:
    df (pandas.DataFrame): DataFrame containing body point coordinates with columns named in the format '{point_name}_X' and '{point_name}_Y'.

    Returns:
    pandas.DataFrame: DataFrame with additional columns for each calculated angle.

    Calculated Angles:
    - Head_Tilt_Angle: Angle of the nose relative to the left and right eyes.
    - Shoulder_Angle: Angle of the shoulders relative to the spine.
    - Left_Torso_Incline_Angle: Incline angle of the left torso (left hip, left shoulder, left elbow).
    - Right_Torso_Incline_Angle: Incline angle of the right torso (right hip, right shoulder, right elbow).
    - Left_Elbow_Angle: Angle of the left elbow (left shoulder, left elbow, left wrist).
    - Right_Elbow_Angle: Angle of the right elbow (right shoulder, right elbow, right wrist).
    - Left_Hip_Knee_Angle: Angle of the left hip to knee (left hip, left knee, left ankle).
    - Right_Hip_Knee_Angle: Angle of the right hip to knee (right hip, right knee, right ankle).
    - Left_Knee_Ankle_Angle: Angle of the left knee to ankle (left knee, left ankle, left hip).
    - Right_Knee_Ankle_Angle: Angle of the right knee to ankle (right knee, right ankle, right hip).
    - Leg_Spread_Angle: Spread angle of the hips relative to each other (left hip, right hip, left knee).
    - Head_to_Shoulders_Angle: Angle of the head relative to the shoulders (nose, left shoulder, right shoulder).
    - Head_to_Hips_Angle: Angle of the head relative to the hips (nose, left hip, right hip).
    """
    # Function to calculate the angle between three points
    def calculate_angle(row, p1_name, p2_name, p3_name):
        p1 = (row[f'{p1_name}_X'], row[f'{p1_name}_Y'])
        p2 = (row[f'{p2_name}_X'], row[f'{p2_name}_Y'])
        p3 = (row[f'{p3_name}_X'], row[f'{p3_name}_Y'])
        if all(coord != 0 for coord in p1 + p2 + p3):
            return compute_angle(p1, p2, p3)
        else:
            return np.nan

    # Shoulder Angle (e.g., Shoulder angles with spine)
    df['Shoulder_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Shoulder', 'Right Shoulder', 'Left Hip'), axis=1)
    
    # Torso Incline Angles (e.g., Hips relative to shoulders)
    df['Left_Torso_Incline_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Hip', 'Left Shoulder', 'Left Elbow'), axis=1)
    df['Right_Torso_Incline_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Hip', 'Right Shoulder', 'Right Elbow'), axis=1)
    
    # Elbow Angles (e.g., Shoulder-Elbow-Wrist)
    df['Left_Elbow_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Shoulder', 'Left Elbow', 'Left Wrist'), axis=1)
    df['Right_Elbow_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Shoulder', 'Right Elbow', 'Right Wrist'), axis=1)
    
    # Hip-Knee Angles (e.g., Hip-Knee-Ankle)
    df['Left_Hip_Knee_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Hip', 'Left Knee', 'Left Ankle'), axis=1)
    df['Right_Hip_Knee_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Hip', 'Right Knee', 'Right Ankle'), axis=1)
    
    # Knee-Ankle Angles (e.g., Knee-Ankle-Foot, if you have foot points)
    df['Left_Knee_Ankle_Angle'] = df.apply(lambda row: calculate_angle(row, 'Left Knee', 'Left Ankle', 'Left Hip'), axis=1)
    df['Right_Knee_Ankle_Angle'] = df.apply(lambda row: calculate_angle(row, 'Right Knee', 'Right Ankle', 'Right Hip'), axis=1)
    
    # Head to Shoulders Angle
    df['Head_to_Shoulders_Angle'] = df.apply(lambda row: calculate_angle(row, 'Nose', 'Left Shoulder', 'Right Shoulder'), axis=1)
    
    # Head to Hips Angle
    df['Head_to_Hips_Angle'] = df.apply(lambda row: calculate_angle(row, 'Nose', 'Left Hip', 'Right Hip'), axis=1)

    return df

# Example calculation output
# angles = add_angles(detect_pose_sequence("ADL.mp4"))
# print(result)


def calculate_acceleration(df):
    """
    Calculate the velocity and acceleration for specified columns in a DataFrame.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    columns (list of str): List of column names for which to calculate velocity and acceleration.
    Returns:
    pd.DataFrame: The DataFrame with additional columns for velocity and acceleration for each specified column.
    Raises:
    TypeError: If the input df is not a pandas DataFrame.
    ValueError: If any of the specified columns are missing from the DataFrame.
    Notes:
    - Velocity is calculated as the first-order difference of the specified columns.
    - Acceleration is calculated as the first-order difference of the velocity.
    - The first frame's velocity and acceleration are set to None.
    """
    # Keypoints columns
    keypoints_columns = [
                        'Nose_X', 'Nose_Y',
                        'Left Shoulder_X', 'Left Shoulder_Y',
                        'Right Shoulder_X', 'Right Shoulder_Y',
                        'Left Elbow_X', 'Left Elbow_Y',
                        'Right Elbow_X', 'Right Elbow_Y',
                        'Left Wrist_X', 'Left Wrist_Y',
                        'Right Wrist_X', 'Right Wrist_Y',
                        'Left Hip_X', 'Left Hip_Y',
                        'Right Hip_X', 'Right Hip_Y',
                        'Left Knee_X', 'Left Knee_Y',
                        'Right Knee_X', 'Right Knee_Y',
                        'Left Ankle_X', 'Left Ankle_Y',
                        'Right Ankle_X', 'Right Ankle_Y'
                    ]
    # Check if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected input to be a DataFrame")

    # Check if all specified columns are present in the DataFrame
    if not all(col in df.columns for col in keypoints_columns):
        raise ValueError("Some columns are missing from the DataFrame")

    # Calculate velocity and acceleration for each specified column
    for col in keypoints_columns:
        df[f'{col}_velocity'] = df[col].diff()  # Calculate the first-order difference for velocity
        df[f'{col}_acceleration'] = df[f'{col}_velocity'].diff()  # Calculate the rate of change of velocity (acceleration)

    # Set velocity and acceleration to None for the first frame and frames with frame_idx 0
    for col in keypoints_columns:
        df.loc[df['frame_idx'] == 0, f'{col}_velocity'] = -99
        df.loc[df['frame_idx'] == 0, f'{col}_acceleration'] = -99

    return df

def remove_velocity_columns(df):
    """
    Remove all velocity columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with velocity columns removed.
    """
    velocity_columns = [col for col in df.columns if '_velocity' in col]
    df.drop(columns=velocity_columns, inplace=True)
    return df

def demo():
  # Initialize the MoveNet model and other necessary components
  video_path = "ADL.mp4"
  for batch_df in infer_from_video_in_batches(video_path, movenet, INPUT_SIZE, init_crop_region, run_inference, draw_prediction_on_image, determine_crop_region):
      # Perform further calculations with batch_df
      angles = add_angles(batch_df)
      df_with_acc = calculate_acceleration(angles)
      final = remove_velocity_columns(df_with_acc)
      print(f"Processed batch with {len(batch_df)} frames")
      print(final)
      # print(final)
      # Example: You could calculate velocity/acceleration, save the batch, etc.

demo()