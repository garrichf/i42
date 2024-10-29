import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import patch, MagicMock

# Import the functions to be tested
from MOVENET import (
    init_crop_region,
    torso_visible,
    determine_torso_and_body_range,
    determine_crop_region,
    keypoints_to_dataframe,
    frame_inference,
    MOVENET_pose,
    run_inference  # Add this import
)

class TestMoveNetPose(unittest.TestCase):

    def setUp(self):
        # Set up any necessary test data
        self.image_height = 480
        self.image_width = 640
        self.keypoints_with_scores = np.random.rand(1, 1, 17, 3)
        self.keypoints_with_scores[0, 0, :, 2] = 0.5  # Set all confidence scores to 0.5

    def test_init_crop_region(self):
        crop_region = init_crop_region(self.image_height, self.image_width)
        self.assertIsInstance(crop_region, dict)
        self.assertEqual(len(crop_region), 6)
        self.assertIn('y_min', crop_region)
        self.assertIn('x_min', crop_region)
        self.assertIn('y_max', crop_region)
        self.assertIn('x_max', crop_region)
        self.assertIn('height', crop_region)
        self.assertIn('width', crop_region)

    def test_torso_visible(self):
        self.keypoints_with_scores[0, 0, 5:7, 2] = 0.4  
        self.keypoints_with_scores[0, 0, 11:13, 2] = 0.4  
        self.assertTrue(torso_visible(self.keypoints_with_scores))

        # Test when torso is not visible
        self.keypoints_with_scores[0, 0, 5:7, 2] = 0.3  # Set shoulder confidence scores below threshold
        self.keypoints_with_scores[0, 0, 11:13, 2] = 0.3  # Set hip confidence scores below threshold
        self.assertFalse(torso_visible(self.keypoints_with_scores))

    def test_determine_torso_and_body_range(self):
        target_keypoints = {
            'left_shoulder': [np.random.rand(), np.random.rand()],
            'right_shoulder': [np.random.rand(), np.random.rand()],
            'left_hip': [np.random.rand(), np.random.rand()],
            'right_hip': [np.random.rand(), np.random.rand()]
        }
        center_y, center_x = 0.5, 0.5
        
        # Mock the KEYPOINT_DICT
        with patch('MOVENET.KEYPOINT_DICT', {'left_shoulder': 5, 'right_shoulder': 6, 'left_hip': 11, 'right_hip': 12}):
            ranges = determine_torso_and_body_range(self.keypoints_with_scores, target_keypoints, center_y, center_x)
        
        self.assertEqual(len(ranges), 4)
        for range_value in ranges:
            self.assertIsInstance(range_value, float)

    def test_determine_crop_region(self):
        crop_region = determine_crop_region(self.keypoints_with_scores, self.image_height, self.image_width)
        self.assertIsInstance(crop_region, dict)
        self.assertEqual(len(crop_region), 6)
        self.assertIn('y_min', crop_region)
        self.assertIn('x_min', crop_region)
        self.assertIn('y_max', crop_region)
        self.assertIn('x_max', crop_region)
        self.assertIn('height', crop_region)
        self.assertIn('width', crop_region)

    def test_keypoints_to_dataframe(self):
        df = keypoints_to_dataframe(self.keypoints_with_scores)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 1) 
        self.assertGreaterEqual(df.shape[1], 28) 

  
@patch('MOVENET.movenet')
@patch('MOVENET.cv2.cvtColor')
@patch('MOVENET.tf.convert_to_tensor')
@patch('MOVENET.tf.expand_dims')
def test_frame_inference(self, mock_expand_dims, mock_convert_to_tensor, mock_cvtColor, mock_movenet):
    mock_frame = MagicMock()
    mock_frame.shape = (self.image_height, self.image_width, 3)  
    mock_movenet.return_value = self.keypoints_with_scores
    mock_cvtColor.return_value = mock_frame
    mock_convert_to_tensor.return_value = mock_frame
    mock_expand_dims.return_value = mock_frame

    df = frame_inference(mock_frame, mock_movenet, 256, init_crop_region, run_inference, determine_crop_region)
    self.assertIsInstance(df, pd.DataFrame)
    self.assertEqual(df.shape[0], 1) 
    self.assertGreaterEqual(df.shape[1], 28)  

    @patch('MOVENET.frame_inference')
    def test_MOVENET_pose(self, mock_frame_inference):
        mock_frame = MagicMock()
        mock_df = pd.DataFrame(np.random.rand(1, 34)) 
        mock_frame_inference.return_value = mock_df

        result_df = MOVENET_pose(mock_frame)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(result_df.shape[0], 1) 
        self.assertGreaterEqual(result_df.shape[1], 28)  

if __name__ == '__main__':
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMoveNetPose)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)

    # Save test results
    with open('test_results.txt', 'w') as f:
        f.write(f"Tests run: {test_result.testsRun}\n")
        f.write(f"Tests passed: {test_result.testsRun - len(test_result.failures) - len(test_result.errors)}\n")
        f.write(f"Tests failed: {len(test_result.failures)}\n")
        f.write(f"Tests with errors: {len(test_result.errors)}\n")
        
        if test_result.wasSuccessful():
            f.write("Overall result: PASSED\n")
        else:
            f.write("Overall result: FAILED\n")

        if test_result.failures:
            f.write("\nFailures:\n")
            for failure in test_result.failures:
                f.write(f"{failure[0]}: {failure[1]}\n")

        if test_result.errors:
            f.write("\nErrors:\n")
            for error in test_result.errors:
                f.write(f"{error[0]}: {error[1]}\n")