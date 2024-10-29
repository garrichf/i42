import tkinter as tk

class HistoryLog:
    """
    A class to represent a history log in a Tkinter application.

    Attributes
    ----------
    history_frame : tk.Frame
        A frame widget to contain the history log components.
    history_label : tk.Label
        A label widget to display the title of the history log.
    text_widget : tk.Text
        A text widget to display the history messages.

    Methods
    -------
    add_message(message):
        Adds a message to the history log.
    """
    def __init__(self, parent):
        # Create a frame for the history log
        self.history_frame = tk.Frame(parent, bg="#2B3A42")
        self.history_frame.grid(row=2, column=1, sticky="nsew", padx=20, pady=10)

        # Create a label for the history title
        self.history_label = tk.Label(self.history_frame, text="HISTORY LOG", fg="white", bg="#2B3A42", font=("Arial", 20, "bold"))
        self.history_label.pack(anchor="w", padx=10)

        # Create a text widget to display history messages
        self.text_widget = tk.Text(self.history_frame, height = 10, bg="#1C252B", fg="white", font=("Arial", 10))
        self.text_widget.pack(fill="both", expand=True, padx=10, pady=5)
        self.text_widget.config(state=tk.DISABLED)

    def add_message(self, message):
        # Method to add a message to the history log
        self.text_widget.config(state=tk.NORMAL)  # Enable editing
        self.text_widget.insert(tk.END, message + "\n")  # Insert the message at the end
        self.text_widget.config(state=tk.DISABLED)  # Disable editing
        self.text_widget.see(tk.END)  # Scroll to the end
