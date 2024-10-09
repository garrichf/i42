import unittest
import numpy as np
import pandas as pd
import cv2
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(r"C:\Users\User\Desktop\i42")

from MEDIAPIPE import process_frame as mediapipe_process_frame, MEDIAPIPE_pose
from process_data import process_data
from process_data_functions import (
    compute_angle, add_angles, calculate_acceleration,
    remove_outliers_with_fixed_iqr, min_max_scale_fixed_range,
    z_score_normalize, find_min_max_coordinates, draw_keypoints_on_frame
)
from YOLO import process_frame as yolo_process_frame, YOLO_pose

class TestMEDIAPIPE(unittest.TestCase):
    def setUp(self):
        self.mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @patch('MEDIAPIPE.mp_pose.Pose')
    def test_process_frame(self, mock_pose):
        mock_pose_instance = MagicMock()
        mock_pose_instance.process.return_value.pose_landmarks.landmark = [
            MagicMock(x=0.5, y=0.5, visibility=0.9) for _ in range(33)
        ]
        mock_pose.return_value.__enter__.return_value = mock_pose_instance

        result = mediapipe_process_frame(self.mock_frame)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 34) 

    @patch('MEDIAPIPE.process_frame')
    def test_MEDIAPIPE_pose(self, mock_process_frame):
        mock_process_frame.return_value = pd.DataFrame({'test': [1]})
        result = MEDIAPIPE_pose(self.mock_frame)
        self.assertIsInstance(result, pd.DataFrame)
        mock_process_frame.assert_called_once_with(self.mock_frame)


class TestProcessDataFunctions(unittest.TestCase):
    def test_compute_angle(self):
        p1, p2, p3 = (0, 0), (1, 1), (2, 0)
        angle = compute_angle(p1, p2, p3)
        self.assertAlmostEqual(angle, 90, places=5)





    def test_calculate_acceleration(self):
        current_df = pd.DataFrame({'X': [2], 'Y': [3]})
        history_df = pd.DataFrame({'X': [1], 'Y': [1], 'X_velocity': [1], 'Y_velocity': [2]})
        result = calculate_acceleration(current_df, history_df, ['X', 'Y'], 1)
        self.assertIn('X_acceleration', result.columns)
        self.assertIn('Y_acceleration', result.columns)

    def test_remove_outliers_with_fixed_iqr(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 100]})
        fixed_bounds = {'A': (0, 5)}
        result = remove_outliers_with_fixed_iqr(df, 'A', fixed_bounds)
        self.assertEqual(result['A'].max(), 5)

    def test_min_max_scale_fixed_range(self):
        column = pd.Series([0, 50, 100])
        result = min_max_scale_fixed_range(column, 0, 100)
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 1)


  
    def test_find_min_max_coordinates(self):
        df = pd.DataFrame({
            'Nose_X': [0.1], 'Nose_Y': [0.2],
            'Left Shoulder_X': [0.3], 'Left Shoulder_Y': [0.4],
            'Right Shoulder_X': [0.5], 'Right Shoulder_Y': [0.6],
            'Left Elbow_X': [0.2], 'Left Elbow_Y': [0.5],
            'Right Elbow_X': [0.6], 'Right Elbow_Y': [0.7],
            'Left Wrist_X': [0.1], 'Left Wrist_Y': [0.6],
            'Right Wrist_X': [0.7], 'Right Wrist_Y': [0.8],
            'Left Hip_X': [0.3], 'Left Hip_Y': [0.8],
            'Right Hip_X': [0.5], 'Right Hip_Y': [0.8],
            'Left Knee_X': [0.2], 'Left Knee_Y': [1.0],
            'Right Knee_X': [0.6], 'Right Knee_Y': [1.0],
            'Left Ankle_X': [0.1], 'Left Ankle_Y': [1.2],
            'Right Ankle_X': [0.7], 'Right Ankle_Y': [1.2]
        })
        min_x, min_y, max_x, max_y = find_min_max_coordinates(df)
        self.assertEqual(min_x, 0.1)
        self.assertEqual(max_x, 0.7)
        self.assertEqual(min_y, 0.2)
        self.assertEqual(max_y, 1.2)

    def test_draw_keypoints_on_frame(self):
        df = pd.DataFrame({
            'Nose_X': [0.5], 'Nose_Y': [0.5],
            'Left Shoulder_X': [0.4], 'Left Shoulder_Y': [0.6],
            'Right Shoulder_X': [0.6], 'Right Shoulder_Y': [0.6],
            'Left Elbow_X': [0.3], 'Left Elbow_Y': [0.7],
            'Right Elbow_X': [0.7], 'Right Elbow_Y': [0.7],
            'Left Wrist_X': [0.2], 'Left Wrist_Y': [0.8],
            'Right Wrist_X': [0.8], 'Right Wrist_Y': [0.8],
            'Left Hip_X': [0.45], 'Left Hip_Y': [0.9],
            'Right Hip_X': [0.55], 'Right Hip_Y': [0.9],
            'Left Knee_X': [0.4], 'Left Knee_Y': [1.1],
            'Right Knee_X': [0.6], 'Right Knee_Y': [1.1],
            'Left Ankle_X': [0.35], 'Left Ankle_Y': [1.3],
            'Right Ankle_X': [0.65], 'Right Ankle_Y': [1.3]
        })
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = draw_keypoints_on_frame(df, frame)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 100, 3))

class TestYOLO(unittest.TestCase):
    def setUp(self):
        self.mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @patch('YOLO.YOLO')
    def test_process_frame(self, mock_yolo):
        mock_results = MagicMock()
        mock_results.keypoints.xyn = [[[0.5, 0.5] for _ in range(17)]]
        mock_yolo.return_value.predict.return_value = [mock_results]

        result = yolo_process_frame(self.mock_frame)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 34)  # 17 keypoints * 2 (x and y)

    @patch('YOLO.process_frame')
    def test_YOLO_pose(self, mock_process_frame):
        mock_process_frame.return_value = pd.DataFrame({'test': [1]})
        result = YOLO_pose(self.mock_frame)
        self.assertIsInstance(result, pd.DataFrame)
        mock_process_frame.assert_called_once_with(self.mock_frame)

if __name__ == '__main__':
    unittest.main()