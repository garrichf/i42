import unittest
import numpy as np
import pandas as pd
from angles_calculation_MP import MediaPipe_detect_pose_sequence, add_angles, compute_angle

class TestAngleCalMP(unittest.TestCase):
    def setUp(self):
        # Simulate a small set of keypoints (X, Y coordinates) for testing
        self.sample_keypoints = pd.DataFrame({
            'Nose_X': [0.5], 'Nose_Y': [0.1],
            'Left Eye_X': [0.4], 'Left Eye_Y': [0.1],
            'Right Eye_X': [0.6], 'Right Eye_Y': [0.1],
            'Left Shoulder_X': [0.4], 'Left Shoulder_Y': [0.5],
            'Right Shoulder_X': [0.6], 'Right Shoulder_Y': [0.5],
            'Left Elbow_X': [0.35], 'Left Elbow_Y': [0.7],
            'Right Elbow_X': [0.65], 'Right Elbow_Y': [0.7],
            'Left Wrist_X': [0.3], 'Left Wrist_Y': [0.9],
            'Right Wrist_X': [0.7], 'Right Wrist_Y': [0.9],
            'Left Hip_X': [0.4], 'Left Hip_Y': [0.9],
            'Right Hip_X': [0.6], 'Right Hip_Y': [0.9],
            'Left Knee_X': [0.4], 'Left Knee_Y': [1.0],
            'Right Knee_X': [0.6], 'Right Knee_Y': [1.0],
            'Left Ankle_X': [0.4], 'Left Ankle_Y': [1.1],
            'Right Ankle_X': [0.6], 'Right Ankle_Y': [1.1]
        })

    def test_angle_calculation(self):
        # Test angle calculation on simulated data
        result_df = add_angles(self.sample_keypoints)
        
        # Check if the calculated angles are added to the DataFrame
        self.assertIn('Head_Tilt_Angle', result_df.columns)
        self.assertIn('Shoulder_Angle', result_df.columns)
        
        # Assert the angles are not NaN and within a valid range
        self.assertFalse(np.isnan(result_df['Head_Tilt_Angle'][0]))
        self.assertFalse(np.isnan(result_df['Shoulder_Angle'][0]))
        self.assertGreaterEqual(result_df['Head_Tilt_Angle'][0], 0)
        self.assertGreaterEqual(result_df['Shoulder_Angle'][0], 0)

    def test_compute_angle(self):
        # Simple angle calculation between known points (should return 90 degrees)
        p1 = (1, 1)
        p2 = (1, 0)
        p3 = (0, 0)
        angle = compute_angle(p1, p2, p3)
        
        # Assert angle is close to 90 degrees
        self.assertAlmostEqual(angle, 90, delta=1)

if __name__ == '__main__':
    unittest.main()
