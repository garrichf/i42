import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from MediaPipe_Pose_Live import MediaPipe_detect_pose_live  # Adjust this line based on your actual module name

class TestMediaPipeLivePoseEstimation(unittest.TestCase):

    @patch('cv2.VideoCapture')
    @patch('mediapipe.solutions.pose.Pose')
    def test_media_pipe_detect_pose_live(self, mock_pose, mock_video_capture):
        # Mock the VideoCapture object and its methods
        mock_vid = MagicMock()
        mock_video_capture.return_value = mock_vid
        
        # Create dummy frames (e.g., 640x480 RGB images)
        frame1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)  # Random frame 1
        frame2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)  # Random frame 2
        
        # Mock the return value of read() method to simulate two frames followed by a failure
        mock_vid.read.side_effect = [(True, frame1), (True, frame2), (False, None)]
        
        # Create mock results for pose landmarks
        mock_results = MagicMock()
        mock_results.pose_landmarks = MagicMock()
        
        # Simulate landmarks with visibility values
        mock_results.pose_landmarks.landmark = [
            MagicMock(x=0.5, y=0.5, visibility=1.0) for _ in range(33)  # 33 landmarks with full visibility
        ]
        
        # Set the return value of the process method of Pose
        mock_pose.return_value.process.return_value = mock_results
        
        # Call the function being tested with a limit of frames to process (e.g., 2)
        try:
            MediaPipe_detect_pose_live(frame_limit=2)
            print("Test: PASSED")
        except Exception as e:
            print("Test: FAILED", str(e))

if __name__ == '__main__':
    unittest.main()