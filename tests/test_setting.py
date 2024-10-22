import unittest
import tkinter as tk
from setting import Settings
from unittest.mock import MagicMock

class TestSettings(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.console = MagicMock()  
        self.toggle_state_var = tk.BooleanVar(value=False) 
        self.video_feed = MagicMock()  
        self.settings = Settings(self.root, self.console, self.toggle_state_var, self.video_feed)

    def tearDown(self):
        self.root.destroy()

    def test_settings_frame_creation(self):
        try:
            self.assertIsInstance(self.settings.settings_frame, tk.Frame)
            self.assertEqual(self.settings.settings_frame['bg'], "#3E4A52")
            print("Test 'test_settings_frame_creation': PASS")
        except AssertionError as e:
            print(f"Test 'test_settings_frame_creation': FAIL - {e}")
            raise

    def test_settings_frame_placement(self):
        try:
            info = self.settings.settings_frame.grid_info()
            self.assertEqual(info['row'], 1)
            self.assertEqual(info['column'], 1)
            self.assertEqual(set(info['sticky']), set("nsew"))
            print("Test 'test_settings_frame_placement': PASS")
        except AssertionError as e:
            print(f"Test 'test_settings_frame_placement': FAIL - {e}")
            raise

    def test_model_choice_initialization(self):
        try:
            self.assertEqual(self.settings.model_choice.get(), "YOLOv8")
            print("Test 'test_model_choice_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_model_choice_initialization': FAIL - {e}")
            raise

    def test_confidence_threshold_initialization(self):
        try:
            self.assertEqual(self.settings.confidence_threshold.get(), 0.5)
            print("Test 'test_confidence_threshold_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_confidence_threshold_initialization': FAIL - {e}")
            raise

    def test_live_state_initialization(self):
        try:
            self.assertFalse(self.settings.live_state.get())
            print("Test 'test_live_state_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_live_state_initialization': FAIL - {e}")
            raise

if __name__ == "__main__":
    unittest.main(verbosity=2)
