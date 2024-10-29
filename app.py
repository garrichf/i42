import tkinter as tk
from tkinter import Toplevel, Label
from video_feed import VideoFeed
from setting import Settings
from console import Console
from history import HistoryLog
from datetime import datetime
import threading
import pygame
import time

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
    """
    Plays a looping fall sound using the Pygame mixer.

    This function loads and plays a sound file in a loop. It also sets up a 
    mechanism to stop the sound when a specific event is triggered.

    The function performs the following steps:
    1. Clears the stop_sound_event to ensure the sound can play.
    2. Loads the sound file "sound/700-hz-beeps-86815.mp3".
    3. Plays the sound file in an infinite loop.
    4. Defines an inner function check_stop() that checks if the stop_sound_event 
       is set. If the event is set, it stops the sound. Otherwise, it schedules 
       another check after 100 milliseconds.
    5. Schedules the first call to check_stop() after 100 milliseconds.

    Note:
        - The sound will continue to play until stop_sound_event is set.
        - The root object must be a Tkinter root window or similar that supports 
          the after() method for scheduling.

    """
    stop_sound_event.clear()
    pygame.mixer.music.load("sound/700-hz-beeps-86815.mp3")
    pygame.mixer.music.play(-1)  
   # while not stop_sound_event.is_set():
      #  continue
   # pygame.mixer.music.stop()
    def check_stop():
        if stop_sound_event.is_set():
            pygame.mixer.music.stop()  
        else:
            root.after(100, check_stop)  
    root.after(100, check_stop)  

title_frame = tk.Frame(root, bg="#2B3A42")
title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
title_label = tk.Label(title_frame, text="FALL DETECTION SYSTEM", fg="white", bg="#2B3A42", font=("Arial", 30, "bold"))
title_label.pack(side="top", pady=5)  
toggle_frame = tk.Frame(title_frame, bg="#2B3A42")
toggle_frame.pack(side="right")
toggle_state = tk.BooleanVar(value=False)  
fall_detected_mode = True 
fall_event_active = False

def toggle_switch():
    """
    Toggles the state of a switch UI component.

    This function changes the appearance of a toggle switch based on its current state.
    When the switch is off, it changes the button color to a dark shade, adjusts the button
    position, and sets the label colors accordingly. When the switch is on, it changes the 
    button color to white, adjusts the button position, and sets the label colors accordingly.
    
    The function also updates the state of the toggle switch to reflect the new state.
    
    Returns:
        None
    """
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
    """
    Triggers the fall detection alert process.

    This function checks if a fall event is already active. If not, it sets the fall event as active and proceeds to 
    display a popup alert with fall detection details such as the time of the fall, response time, and confidence score. 
    It also starts a thread to play a fall alert sound and logs the fall detection event in the history log.

    Global Variables:
    - sound_thread: Thread object for playing the fall alert sound.
    - fall_event_active: Boolean flag indicating if a fall event is currently active.

    Popup Details:
    - Displays "Fall Detected!" message.
    - Shows the time of the fall.
    - Shows the response time.
    - Shows the confidence score of the fall detection.
    - Automatically closes after 10 seconds.

    Threads:
    - Starts a thread to play the fall alert sound.

    History Log:
    - Adds a message to the history log indicating the time of the fall detection.
    """
    global sound_thread, fall_event_active
    if fall_event_active:
        return  
    fall_event_active = True  

    if fall_detected_mode == True:
        current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        #detection_time = time.time()
        confidence_value = settings.saved_confidence_value
        fall_probality = round(float(video_feed.fall_probability), 4)
        response_time = video_feed.predict_time


        if confidence_value is not None:
            popup = Toplevel()
            popup.title("Fall Detection Alert")
            popup.configure(bg="#3E4A52")
            screen_width = popup.winfo_screenwidth()
            screen_height = popup.winfo_screenheight()
            window_width = 300
            window_height = 150
            x_cordinate = int((screen_width / 2) - (window_width / 2))
            y_cordinate = int((screen_height / 2) - (window_height / 2))
            popup.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

            Label(popup, text="Fall Detected!", fg="white", bg="#3E4A52", font=("Arial", 14, "bold")).pack(pady=10)
            Label(popup, text=f"Fall Time: {current_time}", fg="white", bg="#3E4A52", font=("Arial", 10)).pack(pady=5)
            Label(popup, text=f"Response Time: {response_time}", fg="white", bg="#3E4A52", font=("Arial", 10)).pack(pady=5)
            Label(popup, text=f"Confidence Score: {fall_probality}", fg="white", bg="#3E4A52", font=("Arial", 10)).pack(pady=5)
            #Label(popup, text=f"Confidence Rate: {confidence_value}", fg="white", bg="#3E4A52", font=("Arial", 10)).pack(pady=5)
            #Label(popup, text=f"Confidnece Score: {fall_probality}", fg="white", bg="#3E4A52", font=("Arial", 10)).pack(pady=5)
            popup.after(10000, lambda: close_popup(popup))

            sound_thread = threading.Thread(target=play_fall_sound, daemon=True)
            sound_thread.start()
            popup.protocol("WM_DELETE_WINDOW", lambda: close_popup(popup))
            history_message = f"Fall detected at {current_time}"
            history_log.add_message(history_message)

def close_popup(popup):
    """
    Closes the given popup window and stops any active sound thread.

    Args:
        popup: The popup window object to be closed.

    Globals:
        sound_thread: A global variable representing the currently running sound thread.
        stop_sound_event: A global event used to signal the sound thread to stop.
        fall_event_active: A global flag indicating whether a fall event is active.

    Behavior:
        - Destroys the popup window.
        - If a sound thread is active, signals it to stop and waits for it to terminate.
        - Sets the sound_thread to None.
        - Sets the fall_event_active flag to False.
    """
    global sound_thread, fall_event_active
    popup.destroy()
    if sound_thread is not None:
        stop_sound_event.set() 
        sound_thread.join()
        sound_thread = None
    fall_event_active = False
   
def on_closing():
    """
    Handles the closing event of the application.

    This function performs the following actions:
    1. Stops the fall sound by calling the stop_fall_sound() function.
    2. Opens the settings file in write mode and clears its contents.
    3. Destroys the root window, effectively closing the application.
    """
    stop_fall_sound()
    with open(settings.settings_file, "w") as file:
        file.write("")
    root.destroy()

def stop_fall_sound():
    """
    Stops the fall sound by setting the stop_sound_event and stopping the music playback.

    This function sets the stop_sound_event, which is presumably used to signal that the fall sound should stop.
    It then stops the music playback using pygame's mixer module.
    """
    stop_sound_event.set()  
    pygame.mixer.music.stop()

toggle_label_left = tk.Label(toggle_frame, text="DEMO", fg="white", bg="#2B3A42", font=("Arial", 20))
toggle_label_left.pack(side="left")

toggle_canvas = tk.Canvas(toggle_frame, width=40, height=22, bg="#2B3A42", highlightthickness=0)
toggle_canvas.pack(side="left", padx=5)

toggle_background = toggle_canvas.create_oval(2, 2, 38, 20, outline="#FFFFFF", width=2)
toggle_button = toggle_canvas.create_oval(4, 4, 18, 18, outline="#FFFFFF", fill="white", width=2)

toggle_canvas.bind("<Button-1>", lambda event: toggle_switch())

toggle_label_right = tk.Label(toggle_frame, text="LIVE", fg="white", bg="#2B3A42", font=("Arial", 20))
toggle_label_right.pack(side="left")
video_feed = VideoFeed(root, toggle_state, trigger_fall_detection)
console = Console(root)
settings = Settings(root, console, toggle_state, video_feed)
settings.write_defaults_to_file()
history_log = HistoryLog(root)
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
