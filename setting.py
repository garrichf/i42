import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import SETTINGS
import video_feed

class Settings:
    def __init__(self, parent, console, toggle_state_var, video_feed):
        """
        Initializes the settings UI for the application.

        Parameters:
        parent (tk.Widget): The parent widget to which this settings frame belongs.
        console (object): The console object for logging or displaying messages.
        toggle_state_var (tk.BooleanVar): A Tkinter variable to track the toggle state.
        video_feed (object): The video feed object for displaying video.

        Attributes:
        model_choice (tk.StringVar): Variable to store the selected pose estimation model.
        confidence_threshold (tk.DoubleVar): Variable to store the confidence threshold value.
        live_state (tk.BooleanVar): Variable to store the live state of the application.
        saved_model_choice (str): The saved model choice.
        saved_confidence_value (float): The saved confidence threshold value.
        settings_file (str): The filename where settings are saved.
        settings_frame (tk.Frame): The frame containing all the settings widgets.
        """
        self.console = console
        self.toggle_state_var = toggle_state_var
        self.video_feed = video_feed
        self.model_choice = tk.StringVar(value="YOLOv8")  
        self.confidence_threshold = tk.DoubleVar(value=0.5)  
        self.live_state = tk.BooleanVar(value = False)
        self.saved_model_choice = "YOLOv8"
        self.saved_confidence_value = 0.5
        self.settings_file = "settingpara.txt"
        self.settings_frame = tk.Frame(parent, bg="#3E4A52")
        self.settings_frame.grid(row=1, column=1, sticky="nsew", padx=20, pady=10)
        settings_label = tk.Label(self.settings_frame, text="SETTING", fg="white", bg="#3E4A52", font=("Arial", 20, "bold"))
        settings_label.pack(pady=10)

        model_label = tk.Label(self.settings_frame, text="Pose Estimation Model", fg="white", bg="#3E4A52", font=("Arial", 20))
        model_label.pack(anchor="center", pady=5)
        model_dropdown = ttk.Combobox(self.settings_frame, values=["YOLOv8", "MoveNet", "MediaPipe"], textvariable=self.model_choice)
        model_dropdown.current(0)
        model_dropdown.pack(anchor="center", pady=15)

        threshold_label = tk.Label(self.settings_frame, text="Confidence Threshold", fg="white", bg="#3E4A52", font=("Arial", 20))
        threshold_label.pack(anchor="center", pady=5)
        threshold_slider = tk.Scale(self.settings_frame, from_=0, to=1, orient="horizontal", resolution=0.1, bg="#3E4A52", length=150,variable=self.confidence_threshold)
        threshold_slider.pack(anchor="center", pady=15)

        style = ttk.Style()
        style.theme_use('clam')  
        style.configure('Save.TButton', 
                        font=('Arial', 10), 
                        foreground='white', 
                        background='#27AE60',
                        borderwidth=1, 
                        relief='flat',
                        padding=6)

        style.map('Save.TButton',
                  background=[('active', '#1F8A4C'), ('pressed', '#16A085')],
                  foreground=[('active', 'white'), ('pressed', 'white')])

        style.configure('Reset.TButton', 
                        font=('Arial', 10), 
                        foreground='white', 
                        background='#2C3E50',
                        borderwidth=1, 
                        relief='flat',
                        padding=6)

        style.map('Reset.TButton',
                  background=[('active', '#34495E'), ('pressed', '#2C3E50')],
                  foreground=[('active', 'white'), ('pressed', 'white')])

        button_frame = tk.Frame(self.settings_frame, bg="#3E4A52")
        button_frame.pack(anchor="center", pady=20)

        save_button = ttk.Button(button_frame, text="✔ Save", style='Save.TButton', command=self.save_settings)
        save_button.pack(side="left", padx=5)

        reset_button = ttk.Button(button_frame, text="Reset", style='Reset.TButton', command=self.reset_settings)
        reset_button.pack(side="left")

    def write_defaults_to_file(self):
        """
        Writes the default settings to a file.

        This method opens the settings file in write mode and writes the current
        model choice and confidence threshold to the file.

        The settings are written in the following format:
        Model: <model_choice>
        Confidence Threshold: <confidence_threshold>

        Raises:
            IOError: If the file cannot be opened or written to.
        """
        with open(self.settings_file, "w") as file:
            file.write(f"Model: {self.model_choice.get()}\n")
            file.write(f"Confidence Threshold: {self.confidence_threshold.get()}\n")

    def save_settings(self):
        """
        Saves the current settings to a file and updates the application state.

        This method retrieves the current settings from the UI elements, formats them,
        and writes them to a specified settings file. It also updates the global settings
        and reloads the video feed with the new settings.

        The settings saved include:
        - Model choice
        - Confidence threshold
        - Live state (Live or Recorded)

        Additionally, a message box is displayed to inform the user that the settings
        have been successfully saved and applied.

        Attributes:
            self.live_state (bool): The current state of the live toggle.
            self.toggle_state_var (tk.BooleanVar): The variable linked to the live toggle.
            self.model_choice (tk.StringVar): The variable linked to the model choice dropdown.
            self.confidence_threshold (tk.DoubleVar): The variable linked to the confidence threshold slider.
            self.saved_model_choice (str): The saved model choice.
            self.saved_confidence_value (float): The saved confidence threshold value.
            self.settings_file (str): The path to the settings file.
            self.console (Console): The console object to add messages to.
            self.video_feed (VideoFeed): The video feed object to reload settings.

        Raises:
            IOError: If there is an error writing to the settings file.
        """
        self.live_state = self.toggle_state_var.get()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        selected_model = self.model_choice.get()
        confidence_value = self.confidence_threshold.get()
        self.saved_model_choice = selected_model
        self.saved_confidence_value = confidence_value
        if self.live_state == False:
            live_record = "Recorded"
        else:
            live_record = "Live"
        print(f"[{current_time}] Settings Saved:")
        print(f"Model: {selected_model}")
        print(f"Confidence Threshold: {confidence_value}")
        print(f"State: {live_record}")
        message = (f"[{current_time}] Settings Saved:\n"
                   f"Model: {selected_model}\n"
                   f"Confidence Threshold: {confidence_value}\n"
                   f"State: {live_record}")
        with open(self.settings_file, "w") as file:
            file.write(f"Model: {selected_model}\n")
            file.write(f"Confidence Threshold: {confidence_value}\n")

        model_text, confidence_threshold_text = SETTINGS.read_config(self.settings_file)
        SETTINGS.CONFIDENCE_THRESHOLD = confidence_threshold_text
        if model_text == "YOLOv8":
            SETTINGS.POSE_MODEL_USED = "YOLO"
        elif model_text == "MediaPipe":
            SETTINGS.POSE_MODEL_USED = "MEDIAPIPE"
        elif model_text == "MoveNet":
            SETTINGS.POSE_MODEL_USED = "MOVENET"
        messagebox.showinfo("Settings Saved", "The settings have been successfully saved and applied.")
        self.console.add_message(message)
        self.video_feed.reload_settings()

    def reset_settings(self):
        """
        Resets the settings to their default values.

        This method sets the model choice to "YOLOv8" and the confidence threshold to 0.5.
        """
        self.model_choice.set("YOLOv8")
        self.confidence_threshold.set(0.5)
        
       
