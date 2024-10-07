import tkinter as tk
from PIL import Image, ImageTk
import cv2
import pandas as pd
import numpy as np
import YOLO
import MEDIAPIPE
import SETTINGS
import process_data
from tensorflow.keras.models import load_model
from tkinter import Label

class VideoFeed:
    def __init__(self, parent, toggle_state_var):
        self.log_csv_filepath = SETTINGS.LOG_FILEPATH
        self.processed_output_csv = SETTINGS.PROCESSED_OUTPUT_CSV
        process_data.process_data_functions.initialize_log_output(self.log_csv_filepath, self.processed_output_csv)

        self.parent = parent
        self.toggle_state_var = toggle_state_var  
        self.video_label = Label(parent, bg="black")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)  
        self.cap = None
        self.video_path = "video/Footage6_CAUCAFDD_Subject_1_Fall_1.mp4" 
        self.is_live = False  
        self.frame_counter = 0  

        #load model and set parameter:
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
        self.box_color = (255,255,255) # Initialize bounding box colour to white during set up phase
        self.index = 0
        self.frame_width = 0
        self.frame_height = 0
        self.update_video_source()
    
    def update_video_source(self):
        
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
         # Perform pose estimation
        
        if self.pose_model_used == "YOLO":
            keypoints = YOLO.YOLO_pose(frame)
        elif self.pose_model_used == "MEDIAPIPE":
            keypoints = MEDIAPIPE.MEDIAPIPE_pose(frame)
        #else
        # Process keypoints and perform fall detection
        processed_df = process_data.process_data(keypoints, index, self.log_csv_filepath)
        processed_df = processed_df.replace(0, -1).fillna(-1)
        frame_with_keypoints = process_data.process_data_functions.draw_keypoints_on_frame(processed_df, frame)
        min_x, min_y, max_x, max_y = process_data.process_data_functions.find_min_max_coordinates(processed_df)

        self.frame_buffer.append(processed_df)

        # If fall is detected, increment fall_detected_buffer and continue
        if self.predictions_class and self.fall_detected_buffer < 30:
            self.fall_detected_buffer += 1
            self.frame_buffer.pop(0)
            # Draw text on the frame to indicate fall has been detected
            text = "Fall Detected (Buffering)"
            color = (0, 0, 255)
            cv2.putText(frame_with_keypoints, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        else:
            # Make a prediction once enough frames are collected
            if len(self.frame_buffer) == self.sequence_length:
                data_array = np.vstack(self.frame_buffer).astype(np.float32).reshape(1, self.sequence_length, 63)
                predictions = self.model.predict(data_array)
                fall_probability = predictions[0][0]
                self.predictions_class = int(fall_probability > self.confidence_threshold)

                # Reset the fall detection buffer counter if a fall is detected
                if self.predictions_class:
                    self.fall_detected_buffer = 0
                    self.fall_counter += 1

                # Slide the buffer by removing the oldest frame
                self.frame_buffer.pop(0)

                # Draw text on the frame
                text = "Fall" if self.predictions_class else "No Fall"
                color = (0, 0, 255) if self.predictions_class else (255, 255, 255)
                cv2.putText(frame_with_keypoints, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(frame_with_keypoints, str(self.predictions_class), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame_with_keypoints, "Starting Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw the bounding box around the detected keypoints
        if not (np.isnan(min_x) or np.isnan(min_y) or np.isnan(max_x) or np.isnan(max_y)):
            min_x_scaled = int(min_x * self.frame_width)
            max_x_scaled = int(max_x * self.frame_width)
            min_y_scaled = int(min_y * self.frame_height)
            max_y_scaled = int(max_y * self.frame_height)
            cv2.rectangle(frame_with_keypoints, (min_x_scaled, min_y_scaled), (max_x_scaled, max_y_scaled), (0, 255, 0), 2)
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
                    # Resize the frame to match the size of the container
                    frame_rgb = cv2.resize(frame_rgb, (container_width, container_height), interpolation=cv2.INTER_AREA)
                image = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_label.config(image=image)
                self.video_label.image = image 
            else:
                if not self.is_live:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  

        self.parent.after(40, self.show_frame)

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None