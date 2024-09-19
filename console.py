import tkinter as tk
from datetime import datetime

class Console:
    def __init__(self, parent):
        # Create a frame for the console
        self.console_frame = tk.Frame(parent, bg="#2B3A42")
       # self.console_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
        self.console_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)

        # Create a label for the console title
        console_label = tk.Label(self.console_frame, text="CONSOLE", fg="white", bg="#2B3A42", font=("Arial", 12, "bold"))
        console_label.pack(anchor="w", padx=10)

        # Create a text widget to display console messages
        self.text_widget = tk.Text(self.console_frame, height=5, bg="#1C252B", fg="white", font=("Arial", 10))
        self.text_widget.pack(fill="both", expand=True, padx=10, pady=5)

        # Disable editing in the text widget
        self.text_widget.config(state=tk.DISABLED)

    def add_message(self, message):
        # Method to add a message to the console
        self.text_widget.config(state=tk.NORMAL)  
        self.text_widget.insert(tk.END, message + "\n")  
        self.text_widget.config(state=tk.DISABLED)  
        self.text_widget.see(tk.END)  
