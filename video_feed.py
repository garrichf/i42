import tkinter as tk
from PIL import Image, ImageTk
import cv2

class VideoFeed:
    def __init__(self, parent):
        # Create a frame for the video feed with increased padding
        self.video_frame = tk.Frame(parent, bg="black", padx=10, pady=10)  # Padding around the video feed
        self.video_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
      
        # Create a label to display the video frames
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)  # Make the video label fill the available space

        # Overlay title label on top of the video feed
        self.title_label = tk.Label(self.video_frame, text="VIDEO FEED", fg="white", font=("Arial", 14, "bold"), bg = "black")
        self.title_label.place(relx=0.5, rely=0.02, anchor="n")  # Position title at the top center of the video feed
       

        # Start video capture
        self.cap = cv2.VideoCapture(0)  # Open the default webcam

        # Set grid configuration to make the video frame responsive
       # parent.grid_columnconfigure(0, weight=3)  # Adjust weight for video feed
       # parent.grid_rowconfigure(1, weight=1)     # Allow the video feed row to expand

        # Call the function to update the frame
        self.update_frame()

    def update_frame(self):
        # Read a frame from the webcam
        ret, frame = self.cap.read()
        if ret:
            # Get the current size of the video_label
            width = max(1, self.video_label.winfo_width())   # Ensure width is at least 1
            height = max(1, self.video_label.winfo_height()) # Ensure height is at least 1

            # Resize the frame to exactly fit the video_label size
            frame = cv2.resize(frame, (width, height))
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to an ImageTk object
            image = Image.fromarray(frame)
            image_tk = ImageTk.PhotoImage(image)

            # Update the video label with the new frame
            self.video_label.configure(image=image_tk)
            self.video_label.image = image_tk

        # Schedule the next frame update
        self.video_frame.after(10, self.update_frame)

    def __del__(self):
        # Release the webcam when the object is destroyed
        if self.cap.isOpened():
            self.cap.release()
<<<<<<< Updated upstream
=======

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


>>>>>>> Stashed changes
