import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

class Settings:
    def __init__(self, parent, console, toggle_state_var):
        self.console = console
        self.toggle_state_var = toggle_state_var
        self.model_choice = tk.StringVar(value="YOLOv8")  # Default model choice
        self.confidence_threshold = tk.DoubleVar(value=0.5)  # Default confidence threshold
        self.live_state = tk.BooleanVar(value = False)
    
        # Create a frame for the settings pane within the parent window
        self.settings_frame = tk.Frame(parent, bg="#3E4A52")
        self.settings_frame.grid(row=1, column=1, sticky="nsew", padx=20, pady=10)
        self.settings_frame.grid_columnconfigure(0, weight=1) 

        settings_label = tk.Label(self.settings_frame, text="SETTING", fg="white", bg="#3E4A52", font=("Arial", 12, "bold"))
        settings_label.pack(pady=10)

        # Center-align the model label and dropdown with more vertical space
        model_label = tk.Label(self.settings_frame, text="Pose Estimation Model", fg="white", bg="#3E4A52", font=("Arial", 10))
        model_label.pack(anchor="center", pady=5)
        model_dropdown = ttk.Combobox(self.settings_frame, values=["YOLOv8", "fefeffe", "Mfefewfswffe"], textvariable=self.model_choice)
        model_dropdown.current(0)
        model_dropdown.pack(anchor="center", pady=15)

        # Center-align the threshold label and slider with more vertical space
        threshold_label = tk.Label(self.settings_frame, text="Confidence Threshold", fg="white", bg="#3E4A52", font=("Arial", 10))
        threshold_label.pack(anchor="center", pady=5)
        threshold_slider = tk.Scale(self.settings_frame, from_=0, to=1, orient="horizontal", resolution=0.1, bg="#3E4A52", length=150,variable=self.confidence_threshold)
        threshold_slider.pack(anchor="center", pady=15)

        # Create custom styles for buttons
        style = ttk.Style()
        style.theme_use('clam')  # Use a theme that supports color customization

        # Configure styles for Save button
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

        # Configure styles for Reset button
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

        # Create a frame for the custom-styled buttons to manage layout
        button_frame = tk.Frame(self.settings_frame, bg="#3E4A52")
        button_frame.pack(anchor="center", pady=20)

        save_button = ttk.Button(button_frame, text="✔ Save", style='Save.TButton', command=self.save_settings)
        save_button.pack(side="left", padx=5)

        reset_button = ttk.Button(button_frame, text="Reset", style='Reset.TButton', command=self.reset_settings)
        reset_button.pack(side="left")

    def save_settings(self):
        # Get the current date and time
        self.live_state = self.toggle_state_var.get()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        selected_model = self.model_choice.get()
        confidence_value = self.confidence_threshold.get()
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

        messagebox.showinfo("Settings Saved", "The settings have been successfully saved and applied.")
        self.console.add_message(message)


    def reset_settings(self):
        self.model_choice.set("YOLOv8")
        self.confidence_threshold.set(0.5)
        
        
       
