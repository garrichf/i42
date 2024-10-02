import tkinter as tk
from PIL import Image, ImageTk
import cv2
from tkinter import Label

class VideoFeed:
    def __init__(self, parent, toggle_state_var):
        self.parent = parent
        self.toggle_state_var = toggle_state_var  

        self.video_label = Label(parent, bg="black")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)  
        
        self.cap = None
        self.video_path = "video/FallRightS1.avi"  
        self.is_live = False  
        self.frame_counter = 0  

        self.update_video_source()

    def update_video_source(self):
        
        if self.cap is not None:
            self.cap.release()

        self.is_live = self.toggle_state_var.get()

        if self.is_live:
            self.cap = cv2.VideoCapture(0)
           
        else:
            self.cap = cv2.VideoCapture(self.video_path)

        self.show_frame()

    def show_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.video_label.config(text="Unable to access video source")
            return

        self.frame_counter += 1
        if self.frame_counter % 2 == 0:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = ImageTk.PhotoImage(Image.fromarray(frame))
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
