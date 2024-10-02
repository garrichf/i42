import tkinter as tk

class HistoryLog:
    def __init__(self, parent):
        # Create a frame for the history log
        self.history_frame = tk.Frame(parent, bg="#2B3A42")
        self.history_frame.grid(row=2, column=1, sticky="nsew", padx=20, pady=10)

        # Create a label for the history title
        history_label = tk.Label(self.history_frame, text="HISTORY LOG", fg="white", bg="#2B3A42", font=("Arial", 30, "bold"))
        history_label.pack(anchor="w", padx=10)

        # Create a text widget to display history messages
        self.text_widget = tk.Text(self.history_frame, height=5, bg="#1C252B", fg="white", font=("Arial", 25))
        self.text_widget.pack(fill="both", expand=True, padx=10, pady=5)

        # Disable editing in the text widget
        self.text_widget.config(state=tk.DISABLED)

    def add_message(self, message):
        # Method to add a message to the history log
        self.text_widget.config(state=tk.NORMAL)  # Enable editing
        self.text_widget.insert(tk.END, message + "\n")  # Insert the message at the end
        self.text_widget.config(state=tk.DISABLED)  # Disable editing
        self.text_widget.see(tk.END)  # Scroll to the end
