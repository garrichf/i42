import unittest
import tkinter as tk
from console import Console

class TestConsole(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.console = Console(self.root)

    def tearDown(self):
        self.root.destroy()

    def test_console_frame_initialization(self):
        try:
            self.assertIsInstance(self.console.console_frame, tk.Frame)
            self.assertEqual(self.console.console_frame['bg'], "#2B3A42")
            print("Test 'test_console_frame_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_console_frame_initialization': FAIL - {e}")
            raise

    def test_console_frame_placement(self):
        try:
            info = self.console.console_frame.grid_info()
            self.assertEqual(info['row'], 2)
            self.assertEqual(info['column'], 0)
            self.assertEqual(set(info['sticky']), set("nsew"))
            print("Test 'test_console_frame_placement': PASS")
        except AssertionError as e:
            print(f"Test 'test_console_frame_placement': FAIL - {e}")
            raise

    def test_console_label_initialization(self):
        try:
            label = self.console.console_frame.winfo_children()[0]
            self.assertIsInstance(label, tk.Label)
            self.assertEqual(label['text'], "CONSOLE")
            self.assertEqual(label['fg'], "white")
            self.assertEqual(label['bg'], "#2B3A42")
            self.assertEqual(label['font'], "Arial 12 bold")
            print("Test 'test_console_label_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_console_label_initialization': FAIL - {e}")
            raise

    def test_console_text_widget_initialization(self):
            try:
                text_widget = self.console.console_frame.winfo_children()[1]
                self.assertIsInstance(text_widget, tk.Text)
                self.assertEqual(text_widget['bg'], "#1C252B")  
                self.assertEqual(text_widget['fg'], "white")
                self.assertEqual(text_widget['state'], 'disabled')
                print("Test 'test_console_text_widget_initialization': PASS")
            except AssertionError as e:
                print(f"Test 'test_console_text_widget_initialization': FAIL - {e}")
                raise


    def test_console_output_text(self):
        try:
            text_widget = self.console.console_frame.winfo_children()[1]
            text_widget.configure(state='normal')
            message = "This is a test message."
            text_widget.insert('end', message)
            text_widget.configure(state='disabled')
            self.assertIn(message, text_widget.get('1.0', 'end'))
            print("Test 'test_console_output_text': PASS")
        except AssertionError as e:
            print(f"Test 'test_console_output_text': FAIL - {e}")
            raise

if __name__ == "__main__":
    unittest.main(verbosity=2)
