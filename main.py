import tkinter as tk
from tkinter import Toplevel, Label
from video_feed import VideoFeed
from setting import Settings
from console import Console
from history import HistoryLog
from datetime import datetime
import threading
import pygame

root = tk.Tk()
root.title("Fall Detection System")
root.geometry("1300x700")  
root.configure(bg="#2B3A42")

pygame.mixer.init()

root.grid_columnconfigure(0, weight=4) 
root.grid_columnconfigure(1, weight=1)  
root.grid_rowconfigure(0, weight=0)  
root.grid_rowconfigure(1, weight=3)  
root.grid_rowconfigure(2, weight=1)  

stop_sound_event = threading.Event()

def play_fall_sound():
    # Reset the stop event
    stop_sound_event.clear()
    # Play the sound until the stop event is set
    pygame.mixer.music.load("sound/700-hz-beeps-86815.mp3")
    pygame.mixer.music.play(-1)  # Loop indefinitely
    while not stop_sound_event.is_set():
        continue
    pygame.mixer.music.stop()

title_frame = tk.Frame(root, bg="#2B3A42")
title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
title_label = tk.Label(title_frame, text="FALL DETECTION SYSTEM", fg="white", bg="#2B3A42", font=("Arial", 30, "bold"))
title_label.pack(side="top", pady=5)  
toggle_frame = tk.Frame(title_frame, bg="#2B3A42")
toggle_frame.pack(side="right")
toggle_state = tk.BooleanVar(value=False)  
fall_detected = tk.BooleanVar(value=True)  # Boolean value to indicate if a fall has been detected

def toggle_switch():
    if toggle_state.get() == False:
        toggle_canvas.itemconfig(toggle_button, fill="#2B3A42")  
        toggle_canvas.coords(toggle_button, 22, 4, 36, 18)  
        toggle_canvas.itemconfig(toggle_background, outline="#FFFFFF")  
        toggle_label_left.config(fg="white")
        toggle_label_right.config(fg="#FFFFFF")
    else:
        toggle_canvas.itemconfig(toggle_button, fill="white")  
        toggle_canvas.coords(toggle_button, 4, 4, 18, 18)  
        toggle_canvas.itemconfig(toggle_background, outline="#FFFFFF")  
        toggle_label_left.config(fg="#FFFFFF")
        toggle_label_right.config(fg="white")
    toggle_state.set(not toggle_state.get())

def trigger_fall_detection():
    global sound_thread
    if fall_detected.get():
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        confidence_value = settings.saved_confidence_value

        # Calculate response time (10 seconds after start)
        response_time = "10 seconds"

        # Popup message
        if confidence_value is not None:
            popup = Toplevel()
            popup.title("Fall Detection Alert")
           
            popup.configure(bg="#3E4A52")
            screen_width = popup.winfo_screenwidth()
            screen_height = popup.winfo_screenheight()
            window_width = 300
            window_height = 150
            # Calculate the position to center the pop-up
            x_cordinate = int((screen_width / 2) - (window_width / 2))
            y_cordinate = int((screen_height / 2) - (window_height / 2))

            # Set the geometry and position of the pop-up
            popup.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

            # Add the details to the pop-up
            Label(popup, text="Fall Detected!", fg="white", bg="#3E4A52", font=("Arial", 14, "bold")).pack(pady=10)
            Label(popup, text=f"Date: {current_time}", fg="white", bg="#3E4A52", font=("Arial", 10)).pack(pady=5)
            Label(popup, text=f"Response Time: {response_time}", fg="white", bg="#3E4A52", font=("Arial", 10)).pack(pady=5)
            Label(popup, text=f"Current Confidence Rate: {confidence_value}", fg="white", bg="#3E4A52", font=("Arial", 10)).pack(pady=5)
            
            # Automatically close the popup after 10 seconds
            popup.after(10000, lambda: close_popup(popup))

            # Start playing the sound in a separate thread
            sound_thread = threading.Thread(target=play_fall_sound, daemon=True)
            sound_thread.start()

            # Bind the close button (X) to the close_popup function
            popup.protocol("WM_DELETE_WINDOW", lambda: close_popup(popup))

            # Add message to history log
            history_message = f"Fall detected on {current_time}"
            history_log.add_message(history_message)

def close_popup(popup):
    global sound_thread

    # Destroy the popup
    popup.destroy()

    # Stop the sound thread if it is still running
    if sound_thread is not None:
        stop_sound_event.set()  # Signal the sound thread to stop
        sound_thread.join()  # Wait for the sound thread to finish
   
def on_closing():
    stop_fall_sound()
    with open(settings.settings_file, "w") as file:
        file.write("")
    root.destroy()

def stop_fall_sound():
    stop_sound_event.set()  # Signal the sound thread to stop
    pygame.mixer.music.stop()

toggle_label_left = tk.Label(toggle_frame, text="Recorded", fg="white", bg="#2B3A42", font=("Arial", 10))
toggle_label_left.pack(side="left")

toggle_canvas = tk.Canvas(toggle_frame, width=40, height=22, bg="#2B3A42", highlightthickness=0)
toggle_canvas.pack(side="left", padx=5)

toggle_background = toggle_canvas.create_oval(2, 2, 38, 20, outline="#FFFFFF", width=2)
toggle_button = toggle_canvas.create_oval(4, 4, 18, 18, outline="#FFFFFF", fill="white", width=2)

toggle_canvas.bind("<Button-1>", lambda event: toggle_switch())

toggle_label_right = tk.Label(toggle_frame, text="Live", fg="white", bg="#2B3A42", font=("Arial", 10))
toggle_label_right.pack(side="left")
video_feed = VideoFeed(root, toggle_state)
console = Console(root)
settings = Settings(root, console, toggle_state, video_feed)
settings.write_defaults_to_file()
history_log = HistoryLog(root)
root.after(5000, trigger_fall_detection)  # 10000 ms = 10 seconds
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
