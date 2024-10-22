import tkinter as tk

class HistoryLog:
    """
    A class to represent a history log in a Tkinter application.
    Attributes
    ----------
    history_frame : tk.Frame
        The frame widget that contains the history log components.
    history_label : tk.Label
        The label widget that displays the title "HISTORY LOG".
    text_widget : tk.Text
        The text widget that displays the log messages.
    Methods
    -------
    add_message(message):
        Adds a new message to the history log.
    """
    def __init__(self, parent):
    
        self.history_frame = tk.Frame(parent, bg="#2B3A42")
        self.history_frame.grid(row=2, column=1, sticky="nsew", padx=20, pady=10)
        self.history_label = tk.Label(self.history_frame, text="HISTORY LOG", fg="white", bg="#2B3A42", font=("Arial", 20, "bold"))
        self.history_label.pack(anchor="w", padx=10)
        self.text_widget = tk.Text(self.history_frame, height = 10, bg="#1C252B", fg="white", font=("Arial", 10))
        self.text_widget.pack(fill="both", expand=True, padx=10, pady=5)
        self.text_widget.config(state=tk.DISABLED)

    def add_message(self, message):
        self.text_widget.config(state=tk.NORMAL)  
        self.text_widget.insert(tk.END, message + "\n")  
        self.text_widget.config(state=tk.DISABLED)  
        self.text_widget.see(tk.END)  
