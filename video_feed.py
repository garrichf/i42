import tkinter as tk
from PIL import Image, ImageTk
import cv2
import pandas as pd
import numpy as np
import YOLO
import MEDIAPIPE
import MOVENET
import SETTINGS
import process_data
from tensorflow.keras.models import load_model
from tkinter import Label
import time
import os
class VideoFeed:
    def __init__(self, parent, toggle_state_var,trigger_fall_detection):
        """
        Initializes the VideoFeed class.

        Args:
            parent (tkinter.Tk or tkinter.Frame): The parent widget.
            toggle_state_var (tkinter.BooleanVar): A Tkinter variable to toggle the state.
            trigger_fall_detection (function): A callback function to trigger fall detection.

        Attributes:
            trigger_fall_detection (function): A callback function to trigger fall detection.
            log_csv_filepath (str): Path to the log CSV file.
            processed_output_csv (str): Path to the processed output CSV file.
            parent (tkinter.Tk or tkinter.Frame): The parent widget.
            toggle_state_var (tkinter.BooleanVar): A Tkinter variable to toggle the state.
            video_label (tkinter.Label): Label widget to display video.
            cap (cv2.VideoCapture or None): Video capture object.
            video_path (str): Path to the video file.
            is_live (bool): Flag to indicate if the video feed is live.
            frame_counter (int): Counter for the frames processed.
            model (tensorflow.keras.Model): Loaded fall detection model.
            pose_model_used (str): Pose model used for detection.
            confidence_threshold (float): Confidence threshold for detection.
            sequence_length (int): Length of the sequence for detection.
            frame_buffer (list): Buffer to store frames.
            predictions_class (int): Class of the predictions.
            fall_detected_buffer (int): Buffer to store fall detection results.
            fall_counter (int): Counter for the falls detected.
            box_color (tuple): Color of the bounding box.
            index (int): Index for frame processing.
            frame_width (int): Width of the video frame.
            frame_height (int): Height of the video frame.
            after_id (int or None): ID of the scheduled `after` call.
        """
        self.trigger_fall_detection = trigger_fall_detection
        self.log_csv_filepath = SETTINGS.LOG_FILEPATH
        self.processed_output_csv = SETTINGS.PROCESSED_OUTPUT_CSV
        process_data.process_data_functions.initialize_log_output(self.log_csv_filepath, self.processed_output_csv)
        self.parent = parent
        self.toggle_state_var = toggle_state_var  
        self.video_label = Label(parent, bg="black")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)  
        self.cap = None
        self.video_path = "video\Footage4_CAUCAFDD_Subject_10_Fall_4.mp4" 
        self.is_live = False  
        self.frame_counter = 0  
        self.model = load_model('falldetect_main.keras', custom_objects={'f1_score': process_data.process_data_functions.f1_score})
        self.log_csv_filepath = SETTINGS.LOG_FILEPATH
        self.processed_output_csv = SETTINGS.PROCESSED_OUTPUT_CSV
        self.pose_model_used = SETTINGS.POSE_MODEL_USED
        self.confidence_threshold = SETTINGS.CONFIDENCE_THRESHOLD
        self.sequence_length = 30
        self.frame_buffer = []
        self.predictions_class = 0
        self.fall_detected_buffer = 99
        self.fall_counter = 0
        self.box_color = (255,255,255) 
        self.index = 0
        self.frame_width = 0
        self.frame_height = 0
        self.after_id = None  # Store the `after` call ID
        self.update_video_source()

    
    def update_video_source(self):
        """
        Updates the video source based on the current state.
        If the current video source is open, it releases it first. Then, it checks the state of `toggle_state_var` to determine
        whether to use a live video feed from the default camera or a pre-recorded video from a specified path. It also updates
        the frame width and height properties based on the new video source.
        Attributes:
            self.cap (cv2.VideoCapture): The video capture object.
            self.is_live (bool): The state indicating whether to use a live video feed.
            self.frame_width (float): The width of the video frames.
            self.frame_height (float): The height of the video frames.
        """
        
        if self.cap is not None:
            self.cap.release()

        self.is_live = self.toggle_state_var.get()

        if self.is_live:
            self.cap = cv2.VideoCapture(0)
           
        else:
            self.cap = cv2.VideoCapture(self.video_path)
        self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.show_frame()

    def process_frame(self, frame, index):
        """
        Processes a video frame for pose estimation and fall detection.
        Args:
            frame (numpy.ndarray): The video frame to process.
            index (int): The index of the current frame in the video stream.
        Returns:
            numpy.ndarray: The processed video frame with keypoints and annotations.
        This method performs the following steps:
        1. Pose estimation using the specified model (YOLO, MEDIAPIPE, or MOVENET).
        2. Processes the keypoints and performs fall detection.
        3. Draws keypoints and bounding boxes on the frame.
        4. Buffers frames and makes fall predictions based on a sequence of frames.
        5. Annotates the frame with fall detection status and frame index.
        The method updates internal buffers and counters used for fall detection.
        """
         # Perform pose estimation
        
        if self.pose_model_used == "YOLO":
            keypoints = YOLO.YOLO_pose(frame)
        elif self.pose_model_used == "MEDIAPIPE":
            keypoints = MEDIAPIPE.MEDIAPIPE_pose(frame)
        elif self.pose_model_used == "MOVENET":
            keypoints = MOVENET.MOVENET_pose(frame)
            
        # Process keypoints and perform fall detection
        processed_df = process_data.process_data(keypoints, index, self.log_csv_filepath)
        processed_df = processed_df.replace(0, -1).fillna(-1)
        frame_with_keypoints = process_data.process_data_functions.draw_keypoints_on_frame(processed_df, frame)
        min_x, min_y, max_x, max_y = process_data.process_data_functions.find_min_max_coordinates(processed_df)

        self.frame_buffer.append(processed_df)
        if self.predictions_class and self.fall_detected_buffer < 30:
            self.fall_detected_buffer += 1
            self.frame_buffer.pop(0)
           # self.fall_detection_callback()
            text = "Fall Detected (Buffering)"
            color = (0, 0, 255)
            cv2.putText(frame_with_keypoints, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        else:
            if len(self.frame_buffer) == self.sequence_length:
                data_array = np.vstack(self.frame_buffer).astype(np.float32).reshape(1, self.sequence_length, 63)
                
                # Record start time for prediction
                predict_start = time.time()

                predictions = self.model.predict(data_array)
                self.fall_probability = predictions[0][0]
                self.predictions_class = int(self.fall_probability > self.confidence_threshold)
                if self.predictions_class:
                    self.fall_detected_buffer = 0
                    self.fall_counter += 1
                    self.trigger_fall_detection()    
                
                # Record and calculate the time taken for prediction
                self.predict_time = round(((time.time() - predict_start) * 1000), 2)
                
                self.frame_buffer.pop(0)
                text = "Fall" if self.predictions_class else "No Fall"
                color = (0, 0, 255) if self.predictions_class else (255, 255, 255)
                cv2.putText(frame_with_keypoints, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(frame_with_keypoints, str(self.predictions_class), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame_with_keypoints, "Starting Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        # Define a margin percentage for the bounding box
        margin_percentage = 0.15

        if not (np.isnan(min_x) or np.isnan(min_y) or np.isnan(max_x) or np.isnan(max_y)):
            min_x_scaled = int(min_x * self.frame_width)
            max_x_scaled = int(max_x * self.frame_width)
            min_y_scaled = int(min_y * self.frame_height)
            max_y_scaled = int(max_y * self.frame_height)

            # Calculate the margin in pixels
            margin_x = int((max_x_scaled - min_x_scaled) * margin_percentage)
            margin_y = int((max_y_scaled - min_y_scaled) * margin_percentage)

            # Adjust the coordinates to include the margin
            min_x_scaled = max(0, min_x_scaled - margin_x)
            max_x_scaled = min(self.frame_width, max_x_scaled + margin_x)
            min_y_scaled = max(0, min_y_scaled - margin_y)
            max_y_scaled = min(self.frame_height, max_y_scaled + margin_y)

            box_color = (0, 0, 255) if self.predictions_class else (0, 255, 0)

            # Ensure coordinates are integers
            cv2.rectangle(frame_with_keypoints, (int(min_x_scaled), int(min_y_scaled)), (int(max_x_scaled), int(max_y_scaled)), box_color, 2)           
            
        cv2.putText(frame_with_keypoints, "Frame: " + str(index), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        self.index += 1
        return frame_with_keypoints


    def show_frame(self):
        """
        Continuously captures and displays frames from a video source.
        This method checks if the video capture object is available and opened.
        If not, it updates the video label with an error message. It processes
        every second frame to reduce the processing load. The frame is processed,
        converted to RGB, resized to fit the video label's dimensions, and then
        displayed on the video label. If the video is not live and the end is 
        reached, it resets to the first frame. The method schedules itself to 
        run again after a short delay.
        Attributes:
            cap (cv2.VideoCapture): The video capture object.
            video_label (tk.Label): The label widget to display the video frames.
            frame_counter (int): Counter to keep track of frames.
            is_live (bool): Flag to indicate if the video is live.
            after_id (str): ID of the scheduled method call.
            parent (tk.Widget): The parent widget for scheduling the method.
        """
        if self.cap is None or not self.cap.isOpened():
            self.video_label.config(text="Unable to access video source")
            return

        self.frame_counter += 1
        if self.frame_counter % 2 == 0:
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self.process_frame(frame, self.index)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                container_width = self.video_label.winfo_width()
                container_height = self.video_label.winfo_height()

                if container_width > 0 and container_height > 0:
                    frame_rgb = cv2.resize(frame_rgb, (container_width, container_height), interpolation=cv2.INTER_AREA)
                image = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_label.config(image=image)
                self.video_label.image = image 
            else:
                if not self.is_live:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
        
        if self.after_id is not None:
            self.parent.after_cancel(self.after_id)
        self.after_id = self.parent.after(10, self.show_frame)

        #self.parent.after(40, self.show_frame)

    def stop_video(self):
        """
        Stops the video feed by releasing the video capture object and cancelling any scheduled updates.

        This method performs the following actions:
        1. Releases the video capture object if it is currently active.
        2. Cancels any scheduled updates to the video feed.

        Attributes:
            cap (cv2.VideoCapture or None): The video capture object.
            after_id (str or None): The identifier for the scheduled update.
            parent (tkinter.Widget): The parent widget that schedules the updates.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.after_id is not None:
            self.parent.after_cancel(self.after_id)
            self.after_id = None

    def reload_settings(self):
        """
        Reloads the settings for the video feed.

        This method performs the following actions:
        1. Stops the current video feed.
        2. Loads the fall detection model with custom objects.
        3. Resets the frame counter and index.
        4. Updates the pose model and confidence threshold from settings.
        5. Clears the frame buffer.
        6. Updates the video source.

        Raises:
            Any exceptions that `load_model` or `update_video_source` might raise.
        """
        self.stop_video() 
        self.model = load_model('falldetect_main.keras', custom_objects={'f1_score': process_data.process_data_functions.f1_score})
        self.frame_counter = 0
        self.index = 0
        self.pose_model_used = SETTINGS.POSE_MODEL_USED
        self.confidence_threshold = SETTINGS.CONFIDENCE_THRESHOLD
        self.frame_buffer.clear()
        self.update_video_source() 