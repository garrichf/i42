import unittest
import tkinter as tk
from history import HistoryLog

class TestHistoryLog(unittest.TestCase):

    def setUp(self):
     
        self.root = tk.Tk()
        self.history_log = HistoryLog(self.root)

    def tearDown(self):
        
        self.root.destroy()

    def test_history_frame_initialization(self):
        try:
            
            self.assertIsInstance(self.history_log.history_frame, tk.Frame)
            self.assertEqual(self.history_log.history_frame['bg'], "#2B3A42")
            print("Test 'test_history_frame_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_history_frame_initialization': FAIL - {e}")
            raise

    def test_history_frame_placement(self):
        try:
          
            info = self.history_log.history_frame.grid_info()
            self.assertEqual(info['row'], 2)
            self.assertEqual(info['column'], 1)
           
            self.assertEqual(set(info['sticky']), set("nsew"))
            print("Test 'test_history_frame_placement': PASS")
        except AssertionError as e:
            print(f"Test 'test_history_frame_placement': FAIL - {e}")
            raise

    def test_history_label_initialization(self):
        try:
            label = self.history_log.history_frame.winfo_children()[0]
            self.assertIsInstance(label, tk.Label)
            self.assertEqual(label['text'], "HISTORY LOG")
            self.assertEqual(label['fg'], "white")
            self.assertEqual(label['bg'], "#2B3A42")
            self.assertEqual(label['font'], "Arial 12 bold")
            print("Test 'test_history_label_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_history_label_initialization': FAIL - {e}")
            raise

if __name__ == "__main__":
    unittest.main(verbosity=2)
