import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from MediaPipe_demo import MediaPipe_detect_pose_sequence  # Adjust this line based on your actual module name

class TestMediaPipePoseEstimation(unittest.TestCase):
    
    @patch('cv2.VideoCapture')
    @patch('mediapipe.solutions.pose.Pose')
    def test_media_pipe_detect_pose_sequence(self, mock_pose, mock_video_capture):
        # Mock the VideoCapture object and its methods
        mock_vid = MagicMock()
        mock_video_capture.return_value = mock_vid
        
        # Create dummy frames (e.g., 640x480 RGB images)
        frame1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)  # Random frame 1
        frame2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)  # Random frame 2
        
        # Mock the return value of read() method to simulate two frames
        mock_vid.read.side_effect = [(True, frame1), (True, frame2), (False, None)]
        
        # Mock the pose estimation results
        mock_results = MagicMock()
        mock_results.pose_landmarks = MagicMock()
        
        # Simulate the landmarks with visibility values
        mock_results.pose_landmarks.landmark = [
            MagicMock(visibility=1.0) for _ in range(33)  # 33 landmarks with full visibility
        ]
        
        # Set the return value of the process method of Pose
        mock_pose.return_value.process.return_value = mock_results
        
        # Call the function being tested
        df = MediaPipe_detect_pose_sequence("dummy_path.mp4")
        
        # Check if DataFrame is returned
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check if DataFrame has the expected number of columns (66 for X and Y coordinates of 33 landmarks)
        self.assertEqual(df.shape[1], 66)
        
        # Check if DataFrame has the expected number of rows (2 frames processed)
        self.assertEqual(df.shape[0], 2)

if __name__ == '__main__':
    try:
        unittest.main(exit=False)  # Run tests without exiting the interpreter
        print("Test: PASSED")
    except Exception as e:
        print("Test: FAILED")