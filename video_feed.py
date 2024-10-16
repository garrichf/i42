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

class VideoFeed:
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
        self.video_folder = "video/"  # Folder containing video files
        self.video_files = self.get_video_files(self.video_folder)
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

    def get_video_files(self, folder):
        """Get a list of video files in the specified folder."""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')  # Add more extensions if needed
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(video_extensions)]

    def update_video_source(self):
        self.stop_video()  # Stop the current video before loading the next one

        self.is_live = self.toggle_state_var.get()

        if self.is_live:
            self.cap = cv2.VideoCapture(0)
        else:
            if self.current_video_index >= len(self.video_files):
                self.current_video_index = 0  # Restart from the first video
            self.cap = cv2.VideoCapture(self.video_files[self.current_video_index])
            self.current_video_index += 1

        self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_buffer = []  # Clear the buffer frames for every new video playback
        self.index = 0  # Reset the frame index for each new video
        self.predictions_class = 0  # Clear fall detected class of the previous video

        # Add a 3-second delay before showing the next video
        self.parent.after(3000, self.show_frame)

    def process_frame(self, frame, index):
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
                predictions = self.model.predict(data_array)
                self.fall_probability = predictions[0][0]
                self.predictions_class = int(self.fall_probability > self.confidence_threshold)
                if self.predictions_class:
                    self.fall_detected_buffer = 0
                    self.fall_counter += 1
                    self.trigger_fall_detection()
                    
                self.frame_buffer.pop(0)
                text = "Fall" if self.predictions_class else "No Fall"
                color = (0, 0, 255) if self.predictions_class else (255, 255, 255)
                cv2.putText(frame_with_keypoints, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(frame_with_keypoints, str(self.predictions_class), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame_with_keypoints, "Starting Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

       
        if not (np.isnan(min_x) or np.isnan(min_y) or np.isnan(max_x) or np.isnan(max_y)):
            min_x_scaled = int(min_x * self.frame_width)
            max_x_scaled = int(max_x * self.frame_width)
            min_y_scaled = int(min_y * self.frame_height)
            max_y_scaled = int(max_y * self.frame_height)

            box_color = (0, 0, 255) if self.predictions_class else (0, 255, 0)

            cv2.rectangle(frame_with_keypoints, (min_x_scaled, min_y_scaled), (max_x_scaled, max_y_scaled), box_color, 2)
            
        cv2.putText(frame_with_keypoints, "Frame: " + str(index), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        self.index += 1
        return frame_with_keypoints


    def show_frame(self):
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
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.after_id is not None:
            self.parent.after_cancel(self.after_id)
            self.after_id = None

    def reload_settings(self):
        self.stop_video() 
        self.model = load_model('falldetect_main.keras', custom_objects={'f1_score': process_data.process_data_functions.f1_score})
        self.frame_counter = 0
        self.index = 0
        self.pose_model_used = SETTINGS.POSE_MODEL_USED
        self.confidence_threshold = SETTINGS.CONFIDENCE_THRESHOLD
        self.frame_buffer.clear()
        self.update_video_source() 