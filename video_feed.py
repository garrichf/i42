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
import os
import time

class VideoFeed:
    """
    A class to handle video feed for fall detection using machine learning models.
    Attributes:
    -----------
    parent : tkinter.Tk
        The parent tkinter window.
    toggle_state_var : tkinter.BooleanVar
        A tkinter variable to toggle between live feed and video files.
    trigger_fall_detection : function
        A callback function to trigger when a fall is detected.
    log_csv_filepath : str
        Path to the log CSV file.
    processed_output_csv : str
        Path to the processed output CSV file.
    video_label : tkinter.Label
        Label widget to display the video feed.
    cap : cv2.VideoCapture
        Video capture object.
    video_folder : str
        Folder containing video files.
    video_files : list
        List of video file paths.
    current_video_index : int
        Index of the current video file being played.
    is_live : bool
        Flag to indicate if live feed is being used.
    frame_counter : int
        Counter for the number of frames processed.
    model : keras.Model
        Loaded machine learning model for fall detection.
    pose_model_used : str
        Pose estimation model used (YOLO, MEDIAPIPE, MOVENET).
    confidence_threshold : float
        Confidence threshold for fall detection.
    sequence_length : int
        Length of the sequence of frames for prediction.
    frame_buffer : list
        Buffer to store frames for prediction.
    predictions_class : int
        Class of the prediction (fall or no fall).
    fall_detected_buffer : int
        Buffer counter for fall detection.
    fall_counter : int
        Counter for the number of falls detected.
    box_color : tuple
        Color of the bounding box for detected falls.
    index : int
        Index of the current frame.
    frame_width : int
        Width of the video frame.
    frame_height : int
        Height of the video frame.
    after_id : int
        ID of the `after` call for tkinter.
    predict_time : str
        Time taken for prediction.
    Methods:
    --------
    load_video_files(folder):
        Load all video file paths from the specified folder.
    clear_frame_buffer():
        Clear the frame buffer.
    update_video_source():
        Update the video source (live feed or video file).
    stop_video():
        Stop the current video feed.
    reload_settings():
        Reload settings and reinitialize the video feed.
    process_frame(frame, index):
        Process a single frame for pose estimation and fall detection.
    show_frame():
        Display the current frame in the video feed.
    """
    def __init__(self, parent, toggle_state_var, trigger_fall_detection):
        self.trigger_fall_detection = trigger_fall_detection
        self.log_csv_filepath = SETTINGS.LOG_FILEPATH
        self.processed_output_csv = SETTINGS.PROCESSED_OUTPUT_CSV
        process_data.process_data_functions.initialize_log_output(self.log_csv_filepath, self.processed_output_csv)
        self.parent = parent
        self.toggle_state_var = toggle_state_var  
        self.video_label = Label(parent, bg="black")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)  
        self.cap = None
        self.video_folder = "video"  # Folder containing videos
        self.video_files = self.load_video_files(self.video_folder)
        self.current_video_index = 0
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
        self.predict_time = "N/A"

    def load_video_files(self, folder):
        """
        Load all video file paths from the specified folder.

        Args:
            folder (str): The path to the folder containing video files.

        Returns:
            list: A list of file paths to the video files in the specified folder.
        """
        # Load all video file paths from the specified folder
        video_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"Loaded video files: {video_files}")  # Debug statement
        return video_files

    def clear_frame_buffer(self):
        # Clear the frame buffer
        self.frame_buffer = []

    def update_video_source(self):
        """
        Updates the video source for the video feed.

        This method stops the current video, checks whether the video source should be live or from a file,
        and initializes the video capture accordingly. If the source is live, it attempts to open the live
        video stream. If the source is from a file, it loads the next video file in the list and updates the
        current video index. It also handles errors in opening the video streams.

        Attributes:
            self.is_live (bool): Indicates whether the video source is live or from a file.
            self.cap (cv2.VideoCapture): The video capture object.
            self.frame_width (float): The width of the video frames.
            self.frame_height (float): The height of the video frames.
            self.index (int): The frame index for the current video.
            self.predictions_class (int): The class of the detected fall in the previous video.

        Methods:
            self.stop_video(): Stops the current video.
            self.clear_frame_buffer(): Clears the buffer frames for every new video playback.
            self.show_frame(): Displays the next frame after a delay.

        Debug Statements:
            Prints the path of the video file being loaded.
            Prints the next video index.
            Prints error messages if the video stream cannot be opened or initialized.
        """
        self.stop_video()  # Stop the current video before loading the next one

        self.is_live = self.toggle_state_var.get()

        if self.is_live:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open live video stream.")
                self.cap = None
        else:
            if self.current_video_index >= len(self.video_files):
                self.current_video_index = 0  # Restart from the first video
            video_path = self.video_files[self.current_video_index]
            print(f"Loading video file: {video_path}")  # Debug statement
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                print(f"Error: Could not open video stream file {video_path}.")
                self.cap = None
            self.current_video_index += 1
            print(f"Next video index: {self.current_video_index}")  # Debug statement

        if self.cap is not None:
            self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.clear_frame_buffer()  # Clear the buffer frames for every new video playback
            self.index = 0  # Reset the frame index for each new video
            self.predictions_class = 0  # Clear fall detected class of the previous video

            # Add a 3-second delay before showing the next video
            self.parent.after(3000, self.show_frame)
        else:
            print("Error: Video stream is not initialized.")

    def stop_video(self):
        """
        Stops the video feed by releasing the video capture object and cancelling any scheduled updates.

        This method performs the following actions:
        1. Releases the video capture object if it is currently active.
        2. Cancels any scheduled updates using the `after_cancel` method of the parent widget.

        Attributes:
            self.cap (cv2.VideoCapture or None): The video capture object. Set to None after release.
            self.after_id (str or None): The identifier for the scheduled update. Set to None after cancellation.
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
        2. Loads a new model for fall detection with custom objects.
        3. Resets the frame counter and index.
        4. Updates the pose model and confidence threshold from the settings.
        5. Clears the frame buffer.
        6. Updates the video source.
        """
        self.stop_video() 
        self.model = load_model('falldetect_main.keras', custom_objects={'f1_score': process_data.process_data_functions.f1_score})
        self.frame_counter = 0
        self.index = 0
        self.pose_model_used = SETTINGS.POSE_MODEL_USED
        self.confidence_threshold = SETTINGS.CONFIDENCE_THRESHOLD
        self.frame_buffer.clear()
        self.update_video_source()

    def process_frame(self, frame, index):
        """
        Processes a video frame for pose estimation and fall detection.
        Args:
            frame (numpy.ndarray): The video frame to be processed.
            index (int): The index of the current frame in the video sequence.
        Returns:
            numpy.ndarray: The processed video frame with keypoints and bounding box drawn.
        The function performs the following steps:
        1. Pose estimation using the specified model (YOLO, MEDIAPIPE, or MOVENET).
        2. Processes the keypoints and performs fall detection.
        3. Draws keypoints and bounding box on the frame.
        4. Updates the frame buffer and performs fall detection based on the sequence of frames.
        5. Adds text annotations to the frame indicating fall detection status and frame index.
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
            text = "Fall Detected (Buffering)"
            color = (0, 0, 255)
            cv2.putText(frame_with_keypoints, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        else:
            if len(self.frame_buffer) == self.sequence_length:
                data_array = np.vstack(self.frame_buffer).astype(np.float32).reshape(1, self.sequence_length, 63)
                
                # Record prediction start time
                self.predict_start= time.time()
                
                predictions = self.model.predict(data_array)
                
                
                self.fall_probability = predictions[0][0]
                self.predictions_class = int(self.fall_probability > self.confidence_threshold)
                
                if self.predictions_class:
                    self.fall_detected_buffer = 0
                    self.fall_counter += 1
                    self.trigger_fall_detection()
                    
                # Record prediction time
                self.predict_time = str(round(((time.time() - self.predict_start) * 1000), 2)) + " ms"
                    
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
        Continuously captures and displays video frames.

        This method reads a frame from the video capture device, processes it,
        converts it to an RGB image, and updates the video label with the new frame.
        If the video capture is successful, it schedules the next frame capture after
        a short delay. If the video capture fails (e.g., the video ends), it updates
        the video source to load the next video.

        Returns:
            None
        """
        ret, frame = self.cap.read()
        if ret:
            self.process_frame(frame, self.index)
            self.index += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            container_width = self.video_label.winfo_width()
            container_height = self.video_label.winfo_height()

            if container_width > 0 and container_height > 0:
                frame = cv2.resize(frame, (container_width, container_height), interpolation=cv2.INTER_AREA)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.after_id = self.parent.after(10, self.show_frame)
        else:
            self.update_video_source()  # Load the next video when the current one ends

    
    def stop_video(self):
        """
        Stops the video feed by releasing the video capture object and cancelling any scheduled updates.

        This method performs the following actions:
        1. Releases the video capture object if it is currently active.
        2. Cancels any scheduled updates to the video feed.

        Attributes:
            cap (cv2.VideoCapture or None): The video capture object.
            after_id (str or None): The identifier for the scheduled update.
            parent (tkinter.Widget): The parent widget used for scheduling updates.
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
        """
        self.stop_video() 
        self.model = load_model('falldetect_main.keras', custom_objects={'f1_score': process_data.process_data_functions.f1_score})
        self.frame_counter = 0
        self.index = 0
        self.pose_model_used = SETTINGS.POSE_MODEL_USED
        self.confidence_threshold = SETTINGS.CONFIDENCE_THRESHOLD
        self.frame_buffer.clear()
        self.update_video_source() 