import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import time

# Assuming MoveNet_detect_pose_live is defined in a module named pose_estimation
from MoveNet_live import MoveNet_detect_pose_live

class TestMoveNetLivePoseEstimation(unittest.TestCase):

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_movenet_detect_pose_live(self, mock_destroy_all_windows, mock_wait_key, mock_imshow, mock_video_capture):
        # Create a mock video capture object
        mock_vid = MagicMock()
        mock_vid.isOpened.return_value = True
        
        # Simulate reading frames for 2 seconds (assuming 30 FPS)
        frame = np.zeros((256, 256, 3), dtype=np.uint8)  # Create a dummy frame
        mock_vid.read.side_effect = [(True, frame)] * 60 + [(False, None)]  # Simulate frames for about 2 seconds
        
        mock_video_capture.return_value = mock_vid

        # Mock the model loading from TensorFlow Hub
        with patch('tensorflow_hub.load') as mock_load:
            mock_load.return_value = MagicMock()  # Simulate loaded model

            start_time = time.time()  # Start timer before calling function
            
            # Call the function to test
            MoveNet_detect_pose_live()

            elapsed_time = time.time() - start_time
            
            # Assertions to ensure expected behavior
            self.assertTrue(mock_vid.read.called)  # Ensure read was called on video capture
            self.assertTrue(mock_load.called)  # Check if model was loaded
            
            # Ensure imshow was called at least once and within the time limit
            self.assertTrue(mock_imshow.called)
            self.assertLessEqual(elapsed_time, 2.5)  # Allow slight buffer for processing time

            # Ensure cleanup methods were called after exiting loop
            self.assertTrue(mock_destroy_all_windows.called)

if __name__ == '__main__':
    unittest.main()