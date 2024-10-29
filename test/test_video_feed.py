import unittest
import tkinter as tk
import numpy as np
from unittest.mock import patch, MagicMock
from video_feed import VideoFeed

class TestVideoFeed(unittest.TestCase):

    def setUp(self):
        # Setup root for testing
        self.root = tk.Tk()
        self.toggle_state_var = tk.BooleanVar(value=False)
        self.video_feed = VideoFeed(self.root, self.toggle_state_var)

    def tearDown(self):
        # Destroy root after each test
        self.root.destroy()

    def test_video_feed_initialization(self):
        try:
            # Test if VideoFeed object initializes without errors
            self.assertIsNotNone(self.video_feed)
            self.assertEqual(self.video_feed.video_path, "video/FallRightS1.avi")
            self.assertFalse(self.video_feed.is_live)
            print("Test 'test_video_feed_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_video_feed_initialization': FAIL - {e}")
            raise

    def test_video_label_initialization(self):
        try:
            # Test if the video label is initialized correctly
            self.assertIsInstance(self.video_feed.video_label, tk.Label)
            self.assertEqual(self.video_feed.video_label['bg'], "black")
            print("Test 'test_video_label_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_video_label_initialization': FAIL - {e}")
            raise

    def test_video_label_placement(self):
        try:
            # Check if the video label is placed in the correct grid position
            info = self.video_feed.video_label.grid_info()
            self.assertEqual(info['row'], 1)
            self.assertEqual(info['column'], 0)
            # Modify the assertion to compare unordered directions
            self.assertEqual(set(info['sticky']), set("nsew"))
            print("Test 'test_video_label_placement': PASS")
        except AssertionError as e:
            print(f"Test 'test_video_label_placement': FAIL - {e}")
            raise

    @patch('cv2.VideoCapture', autospec=True)
    def test_start_video_capture(self, MockVideoCapture):
        try:
            # Mock the video capture object and set it to the instance
            mock_cap = MockVideoCapture.return_value
            mock_cap.isOpened.return_value = True
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Mock a valid frame
            mock_cap.read.return_value = (True, mock_frame)

            self.video_feed.cap = mock_cap  # Set the mock capture object

            # Use `update_video_source()` to start video capture
            self.video_feed.update_video_source()

            # Ensure video capture is started with the correct path
            MockVideoCapture.assert_called_once_with(self.video_feed.video_path)
            self.assertEqual(self.video_feed.cap, mock_cap)
            print("Test 'test_start_video_capture': PASS")
        except AssertionError as e:
            print(f"Test 'test_start_video_capture': FAIL - {e}")
            raise
        except AttributeError as e:
            print(f"Test 'test_start_video_capture': FAIL - {e}")
            raise

    @patch('cv2.VideoCapture', autospec=True)
    def test_stop_video_capture(self, MockVideoCapture):
        try:
            # Mock the video capture object and set it to the instance
            mock_cap = MockVideoCapture.return_value
            mock_cap.isOpened.return_value = True
            self.video_feed.cap = mock_cap  # Set the mock capture object

            # Call stop_video() to release the capture object
            self.video_feed.stop_video()

            # Ensure release is called on the video capture object
            mock_cap.release.assert_called()
            print("Test 'test_stop_video_capture': PASS")
        except AttributeError as e:
            print(f"Test 'test_stop_video_capture': FAIL - {e}")
            raise

if __name__ == "__main__":
    unittest.main(verbosity=2)
