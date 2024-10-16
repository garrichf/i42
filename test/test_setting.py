import unittest
import tkinter as tk
from setting import Settings
from unittest.mock import MagicMock

class TestSettings(unittest.TestCase):

    def setUp(self):
        # Setup root for testing
        self.root = tk.Tk()
        self.console = MagicMock()  # Mock the console component
        self.toggle_state_var = tk.BooleanVar(value=False)  # Mock toggle state
        self.video_feed = MagicMock()  # Mock the video feed component
        # Create an instance of the Settings class with mocked dependencies
        self.settings = Settings(self.root, self.console, self.toggle_state_var, self.video_feed)

    def tearDown(self):
        # Destroy root after each test
        self.root.destroy()

    def test_settings_frame_creation(self):
        try:
            # Check if the settings frame is created correctly
            self.assertIsInstance(self.settings.settings_frame, tk.Frame)
            # Update the color value to match the actual implementation
            self.assertEqual(self.settings.settings_frame['bg'], "#3E4A52")
            print("Test 'test_settings_frame_creation': PASS")
        except AssertionError as e:
            print(f"Test 'test_settings_frame_creation': FAIL - {e}")
            raise

    def test_settings_frame_placement(self):
        try:
            # Check if the settings frame is placed in the correct grid position
            info = self.settings.settings_frame.grid_info()
            # Update the row value to match the actual placement in the grid
            self.assertEqual(info['row'], 1)
            self.assertEqual(info['column'], 1)
            # Use set to compare unordered sticky values
            self.assertEqual(set(info['sticky']), set("nsew"))
            print("Test 'test_settings_frame_placement': PASS")
        except AssertionError as e:
            print(f"Test 'test_settings_frame_placement': FAIL - {e}")
            raise

    def test_model_choice_initialization(self):
        try:
            # Test if the model choice variable is correctly initialized
            self.assertEqual(self.settings.model_choice.get(), "YOLOv8")
            print("Test 'test_model_choice_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_model_choice_initialization': FAIL - {e}")
            raise

    def test_confidence_threshold_initialization(self):
        try:
            # Test if the confidence threshold is correctly initialized
            self.assertEqual(self.settings.confidence_threshold.get(), 0.5)
            print("Test 'test_confidence_threshold_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_confidence_threshold_initialization': FAIL - {e}")
            raise

    def test_live_state_initialization(self):
        try:
            # Test if the live state variable is correctly initialized
            self.assertFalse(self.settings.live_state.get())
            print("Test 'test_live_state_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_live_state_initialization': FAIL - {e}")
            raise

if __name__ == "__main__":
    unittest.main(verbosity=2)
