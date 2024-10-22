import unittest
import tkinter as tk
import numpy as np
from unittest.mock import patch, MagicMock
from video_feed import VideoFeed

class TestVideoFeed(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.toggle_state_var = tk.BooleanVar(value=False)
        self.video_feed = VideoFeed(self.root, self.toggle_state_var)

    def tearDown(self):
        self.root.destroy()

    def test_video_feed_initialization(self):
        try:
            self.assertIsNotNone(self.video_feed)
            self.assertEqual(self.video_feed.video_path, "video/FallRightS1.avi")
            self.assertFalse(self.video_feed.is_live)
            print("Test 'test_video_feed_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_video_feed_initialization': FAIL - {e}")
            raise

    def test_video_label_initialization(self):
        try:
            self.assertIsInstance(self.video_feed.video_label, tk.Label)
            self.assertEqual(self.video_feed.video_label['bg'], "black")
            print("Test 'test_video_label_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_video_label_initialization': FAIL - {e}")
            raise

    def test_video_label_placement(self):
        try:
            info = self.video_feed.video_label.grid_info()
            self.assertEqual(info['row'], 1)
            self.assertEqual(info['column'], 0)
            self.assertEqual(set(info['sticky']), set("nsew"))
            print("Test 'test_video_label_placement': PASS")
        except AssertionError as e:
            print(f"Test 'test_video_label_placement': FAIL - {e}")
            raise

    @patch('cv2.VideoCapture', autospec=True)
    def test_start_video_capture(self, MockVideoCapture):
        try:
            mock_cap = MockVideoCapture.return_value
            mock_cap.isOpened.return_value = True
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)  
            mock_cap.read.return_value = (True, mock_frame)

            self.video_feed.cap = mock_cap  
            self.video_feed.update_video_source()

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
            mock_cap = MockVideoCapture.return_value
            mock_cap.isOpened.return_value = True
            self.video_feed.cap = mock_cap 
            self.video_feed.stop_video()
            mock_cap.release.assert_called()
            print("Test 'test_stop_video_capture': PASS")
        except AttributeError as e:
            print(f"Test 'test_stop_video_capture': FAIL - {e}")
            raise

if __name__ == "__main__":
    unittest.main(verbosity=2)
