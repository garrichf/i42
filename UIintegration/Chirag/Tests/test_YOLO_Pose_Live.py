import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import time
from YOLO_Pose_Live import YOLOv8_detect_pose_live  # Import the function from your script

class TestYOLOPoseLiveDuration(unittest.TestCase):
    @patch('cv2.VideoCapture')
    @patch('ultralytics.YOLO')
    def test_yolo_pose_live_duration(self, MockYOLO, MockVideoCapture):
        # Mocking the video capture object
        mock_vid = MagicMock()
        MockVideoCapture.return_value = mock_vid
        
        # Simulate the video stream being opened successfully
        mock_vid.isOpened.return_value = True
        
        # Simulate the read method to return a valid frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Create a dummy frame (black image)
        mock_vid.read.return_value = (True, mock_frame)  # Return True and the dummy frame
        
        # Simulate the model's inference results
        mock_model = MockYOLO.return_value
        mock_model.return_value[0].keypoints = [MagicMock(data=MagicMock())]
        
        # Simulate keypoints data with a confidence level above threshold
        mock_model.return_value[0].keypoints[0].data[0] = [
            [100, 150, 0.9],  # Nose
            [120, 130, 0.8],  # Left Eye
            # Add more keypoints as needed...
        ]
        
        start_time = time.time()
        
        # Call the function to test (this will run for a short duration)
        YOLOv8_detect_pose_live(max_duration=2)  # Pass max_duration explicitly
        
        elapsed_time = time.time() - start_time
        
        # Check that the elapsed time does not exceed 2.5 seconds (increased tolerance)
        self.assertLessEqual(elapsed_time, 2.5, "The live footage exceeded 2.5 seconds.")

if __name__ == '__main__':
    # Run tests and capture result
    result = unittest.main(exit=False)  # Prevents exiting Python after tests run
    
    # Print success message if all tests pass
    if result.result.wasSuccessful():
        print("Test: PASSED")  # Indicate test result at the end if all tests pass
    else:
        print("Test: FAILED")  # Indicate test result at the end if any test fails