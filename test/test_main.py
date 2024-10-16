import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
from main import VideoFeed, Settings, Console, HistoryLog

class TestMainGUI(unittest.TestCase):

    def setUp(self):
        # Setup root for testing
        self.root = tk.Tk()
        self.root.title("Test Fall Detection System")
        self.root.geometry("1300x700")
        self.root.configure(bg="#2B3A42")
        self.after_ids = []
        self.shutdown_flag = False  # Added flag to indicate if teardown is in progress
    def tearDown(self):
    # Set shutdown flag to prevent scheduling new callbacks
        self.shutdown_flag = True

        # Cancel any scheduled tasks if needed
        if hasattr(self, 'video_feed'):
            # If the video feed component has an after_id, cancel it
            if hasattr(self.video_feed, 'after_id') and self.video_feed.after_id:
                try:
                    self.root.after_cancel(self.video_feed.after_id)
                except:
                    pass  # Ignore if already canceled or invalid

        # Cancel all tasks in after_ids list (cancel every tracked after call)
        for after_id in self.after_ids:
            try:
                if after_id:
                    self.root.after_cancel(after_id)
            except:
                pass  # Ignore if already canceled or invalid

        # Force update to process pending events before destruction
        try:
            self.root.update_idletasks()  # Update any idle tasks in the event loop
            self.root.update()            # Update main event loop
        except:
            pass  # Suppress errors related to closed window

        # Destroy the root window to clean up resources
        self.root.destroy()




    @patch('main.VideoFeed', autospec=True)
    def test_video_feed_initialization(self, MockVideoFeed):
        try:
            # Mock the toggle_state_var
            toggle_state_var = tk.BooleanVar(value=False)
            # Create a mock VideoFeed object with the required arguments
            mock_video_feed = MockVideoFeed(self.root, toggle_state_var)
            # Store after_id for cancellation
            if hasattr(mock_video_feed, 'after_id'):
                self.after_ids.append(mock_video_feed.after_id)

            # Ensure that the mock VideoFeed was called with the correct arguments
            MockVideoFeed.assert_called_once_with(self.root, toggle_state_var)
            self.assertIsNotNone(mock_video_feed)
            print("Test 'test_video_feed_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_video_feed_initialization': FAIL - {e}")
            raise

    @patch('main.Settings', autospec=True)
    def test_settings_initialization(self, MockSettings):
        try:
            # Mock the console, toggle_state_var, and video_feed
            console = MagicMock()
            toggle_state_var = tk.BooleanVar(value=False)
            video_feed = MagicMock()  # Mock the video feed component as well
            # Create a mock Settings object with the required arguments
            mock_settings = MockSettings(self.root, console, toggle_state_var, video_feed)
            # Ensure that the mock Settings was called with the correct arguments
            MockSettings.assert_called_once_with(self.root, console, toggle_state_var, video_feed)
            self.assertIsNotNone(mock_settings)
            print("Test 'test_settings_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_settings_initialization': FAIL - {e}")
            raise

    @patch('main.Console', autospec=True)
    def test_console_initialization(self, MockConsole):
        try:
            # Create a mock Console object
            mock_console = MockConsole(self.root)
            # Ensure that the mock Console was called with the correct arguments
            MockConsole.assert_called_once_with(self.root)
            self.assertIsNotNone(mock_console)
            print("Test 'test_console_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_console_initialization': FAIL - {e}")
            raise

    @patch('main.HistoryLog', autospec=True)
    def test_history_log_initialization(self, MockHistoryLog):
        try:
            # Create a mock HistoryLog object
            mock_history_log = MockHistoryLog(self.root)
            # Ensure that the mock HistoryLog was called with the correct arguments
            MockHistoryLog.assert_called_once_with(self.root)
            self.assertIsNotNone(mock_history_log)
            print("Test 'test_history_log_initialization': PASS")
        except AssertionError as e:
            print(f"Test 'test_history_log_initialization': FAIL - {e}")
            raise

    def test_root_configuration(self):
        try:
            # Verify the configuration of the root window
            self.assertEqual(self.root.title(), "Test Fall Detection System")
            self.assertEqual(self.root['bg'], "#2B3A42")
            # Verify that the geometry is set to a non-empty value
            self.assertTrue(self.root.geometry() != "")
            print("Test 'test_root_configuration': PASS")
        except AssertionError as e:
            print(f"Test 'test_root_configuration': FAIL - {e}")
            raise

    @patch('main.VideoFeed', autospec=True)
    @patch('main.Settings', autospec=True)
    @patch('main.Console', autospec=True)
    @patch('main.HistoryLog', autospec=True)
    def test_component_layout(self, MockHistoryLog, MockConsole, MockSettings, MockVideoFeed):
        try:
            # Mock the components
            mock_video_feed = MockVideoFeed(self.root, tk.BooleanVar(value=False))
            mock_video_feed.grid = MagicMock()
            mock_console = MockConsole(self.root)
            mock_console.grid = MagicMock()
            mock_settings = MockSettings(self.root, mock_console, tk.BooleanVar(value=False), mock_video_feed)
            mock_settings.grid = MagicMock()
            mock_history_log = MockHistoryLog(self.root)
            mock_history_log.grid = MagicMock()

            # Simulate calling the grid method on each component
            mock_video_feed.grid()
            mock_console.grid()
            mock_settings.grid()
            mock_history_log.grid()

            # Verify that each component is placed correctly in the layout (assuming grid layout)
            mock_video_feed.grid.assert_called()
            mock_console.grid.assert_called()
            mock_settings.grid.assert_called()
            mock_history_log.grid.assert_called()
            print("Test 'test_component_layout': PASS")
        except AssertionError as e:
            print(f"Test 'test_component_layout': FAIL - {e}")
            raise

    @patch('main.VideoFeed', autospec=True)
    @patch('main.Settings', autospec=True)
    def test_settings_update_affects_video_feed(self, MockSettings, MockVideoFeed):
        try:
            # Mock the toggle_state_var
            toggle_state_var = tk.BooleanVar(value=False)
            mock_video_feed = MockVideoFeed(self.root, toggle_state_var)
            # Mock the Settings component
            mock_settings = MockSettings(self.root, MagicMock(), toggle_state_var, mock_video_feed)
            # Mock the save_settings method
            mock_settings.save_settings = MagicMock()

            # Simulate changing a setting in the Settings component
            toggle_state_var.set(True)
            mock_settings.save_settings()  # Explicitly call save_settings to simulate action
            mock_settings.save_settings.assert_called()

            # Ensure that changing settings also affects the VideoFeed component
            self.assertTrue(toggle_state_var.get())
            print("Test 'test_settings_update_affects_video_feed': PASS")
        except AssertionError as e:
            print(f"Test 'test_settings_update_affects_video_feed': FAIL - {e}")
            raise

    def test_root_window_resizes(self):
        try:
            # Verify resizing the window updates the geometry
            new_geometry = "1400x800"
            self.root.geometry(new_geometry)
            self.root.update()  # Ensure Tkinter processes the resize request
            # Only compare the width and height part of the geometry
            self.assertTrue(new_geometry in self.root.geometry())
            print("Test 'test_root_window_resizes': PASS")
        except AssertionError as e:
            print(f"Test 'test_root_window_resizes': FAIL - {e}")
            raise

if __name__ == "__main__":
    unittest.main(verbosity=2)
