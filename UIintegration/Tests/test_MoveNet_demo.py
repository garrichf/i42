import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import sys
import io

from YOLO_demo import YOLOv8_detect_and_display_pose

class TestYOLODemo(unittest.TestCase):

    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_YOLOv8_detect_and_display_pose(self, mock_destroy_all_windows, mock_wait_key, mock_imshow, mock_video_writer, mock_video_capture):
        # Create a mock video file path
        mock_video_path = "mock_video.mp4"
        
        # Simulate a video capture object
        mock_vid = MagicMock()
        mock_vid.isOpened.return_value = True
        
        # Simulate reading frames from a video
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Create a dummy frame
        mock_vid.read.side_effect = [(True, frame)] * 10 + [(False, None)]  # Simulate 10 frames
        
        mock_video_capture.return_value = mock_vid
        
        # Mock the VideoWriter to avoid creating an actual video file
        mock_writer = MagicMock()
        mock_video_writer.return_value = mock_writer

        # Capture the output of the function to suppress it during testing
        captured_output = io.StringIO()    # Create StringIO object to capture output
        sys.stdout = captured_output         # Redirect stdout to the StringIO object

        # Test the function
        result = YOLOv8_detect_and_display_pose(mock_video_path)

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Assertions to ensure expected behavior
        self.assertIsInstance(result, pd.DataFrame)  # Check that result is a DataFrame
        self.assertTrue(len(result) > 0)  # Ensure that the DataFrame is not empty

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestYOLODemo)
    result = unittest.TextTestRunner().run(suite)

    if result.wasSuccessful():
        print("Test: PASSED")  # Indicate test result at the end if all tests pass
    else:
        print("Test: FAILED")  # Indicate test result at the end if any test fails