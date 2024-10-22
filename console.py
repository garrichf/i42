import tkinter as tk
from datetime import datetime

class Console:
    """
    A class to create a console-like text widget in a Tkinter GUI.
    Attributes:
    -----------
    console_frame : tk.Frame
        The frame containing the console widget.
    console_label : tk.Label
        The label displaying the title "CONSOLE".
    text_widget : tk.Text
        The text widget where messages are displayed.
    Methods:
    --------
    __init__(self, parent):
        Initializes the Console with a parent widget.
    add_message(self, message):
        Adds a message to the console text widget.
    """
    def __init__(self, parent):
        self.console_frame = tk.Frame(parent, bg="#2B3A42")
        self.console_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
        self.console_label = tk.Label(self.console_frame, text="CONSOLE", fg="white", bg="#2B3A42", font=("Arial", 20, "bold"))
        self.console_label.pack(anchor="w", padx=10)
        self.text_widget = tk.Text(self.console_frame, height=10, bg="#1C252B", fg="white", font=("Arial", 10))
        self.text_widget.pack(fill="both", expand=True, padx=10, pady=5)
        self.text_widget.config(state=tk.DISABLED)

    def add_message(self, message):
        
        self.text_widget.config(state=tk.NORMAL)  
        self.text_widget.insert(tk.END, message + "\n")  
        self.text_widget.config(state=tk.DISABLED)  
        self.text_widget.see(tk.END)  
