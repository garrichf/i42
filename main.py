import tkinter as tk
from tkinter import ttk
from video_feed import VideoFeed
from setting import Settings
from console import Console
from history import HistoryLog

root = tk.Tk()
root.title("Fall Detection System")
root.geometry("1300x700")  
root.configure(bg="#2B3A42")

<<<<<<< Updated upstream
root.grid_columnconfigure(0, weight=8) 
root.grid_columnconfigure(1, weight=2)  
root.grid_rowconfigure(0, weight=0)  
root.grid_rowconfigure(1, weight=3)  
root.grid_rowconfigure(2, weight=1)  

title_frame = tk.Frame(root, bg="#2B3A42")
title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=10)

title_label = tk.Label(title_frame, text="FALL DETECTION SYSTEM", fg="white", bg="#2B3A42", font=("Arial", 16, "bold"))
=======
# Adjust the column and row weights to allocate more space to History Log and reduce Settings size
root.grid_columnconfigure(0, weight=3)  # More space for video feed
root.grid_columnconfigure(1, weight=1)  # Less space for settings and history log
root.grid_rowconfigure(1, weight=1, minsize=150)     # Video feed row weight
root.grid_rowconfigure(2, weight=3, minsize=300)     # More space for history log


title_frame = tk.Frame(root, bg="#2B3A42")
title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
title_label = tk.Label(title_frame, text="FALL DETECTION SYSTEM", fg="white", bg="#2B3A42", font=("Arial", 30, "bold"))
>>>>>>> Stashed changes
title_label.pack(side="top", pady=5)  


toggle_frame = tk.Frame(title_frame, bg="#2B3A42")
toggle_frame.pack(side="right")
toggle_state = tk.BooleanVar(value=False)  

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


toggle_label_left = tk.Label(toggle_frame, text="Recorded", fg="white", bg="#2B3A42", font=("Arial", 15))
toggle_label_left.pack(side="left")

toggle_canvas = tk.Canvas(toggle_frame, width=40, height=22, bg="#2B3A42", highlightthickness=0)
toggle_canvas.pack(side="left", padx=5)

toggle_background = toggle_canvas.create_oval(2, 2, 38, 20, outline="#FFFFFF", width=2)
toggle_button = toggle_canvas.create_oval(4, 4, 18, 18, outline="#FFFFFF", fill="white", width=2)

toggle_canvas.bind("<Button-1>", lambda event: toggle_switch())

toggle_label_right = tk.Label(toggle_frame, text="Live", fg="white", bg="#2B3A42", font=("Arial", 15))
toggle_label_right.pack(side="left")


video_feed = VideoFeed(root)

console = Console(root)
#console.console_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)

# Initialize the settings pane
#settings_frame = tk.Frame(root, width=500, bg="#3E4A52")
#settings_frame.grid(row=1, column=1, sticky="nsew", padx=20, pady=10)
settings = Settings(root, console, toggle_state)

# Initialize the history log and place it under the settings pane, next to the console
history_log = HistoryLog(root)
<<<<<<< Updated upstream
history_log.history_frame.grid(row=2, column=1, sticky="nsew", padx=20, pady=10)
root.mainloop()
=======
root.mainloop()
>>>>>>> Stashed changes
