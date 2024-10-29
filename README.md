# COS40006 - Computing Technology Project B
## Industry Project i42 - Human Fall Detection using RGB Camera 📸
This is the main repository for the industry project i42 from COS40005 Computing Technology Project A and COS40006 Computing Technology Project B.

## 🌟 Overview

This project is a fall detection system that uses an RGB camera to detect falls in real-time. The system leverages machine learning models for pose estimation and fall detection, and it provides a graphical user interface (GUI) for easy interaction.

## ✨ Key Features

- **Real-time Fall Detection**: Detects falls using live video feed or pre-recorded videos.
- **Pose Estimation**: Supports multiple pose estimation models including YOLO, MediaPipe, and MoveNet.
- **Configurable Settings**: Allows users to configure model settings and confidence thresholds.
- **Logging**: Logs keypoints and processed data to CSV files for further analysis.
- **GUI**: User-friendly interface built with Tkinter.
- **Sound Alerts**: Plays a sound alert when a fall is detected.

## Installation

1. **Clone the repository**:
    ```sh
    git clone [<repository-url>](https://github.com/garrichf/i42.git)
    cd <repository-directory>
    ```

2. **Create a virtual environment**:
    ```sh
    python3.10 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements-mac.txt  # For macOS
    pip install -r requirements-window.txt  # For Windows
    ```

4. **Download the pre-trained model**:
    Place the `falldetect_main.keras` and `yolov8n-pose.pt` files in the root directory.

## Usage

1. **Run the application**:
    ```sh
    python app.py
    ```

2. **GUI Components**:
    - **Toggle Switch**: Switch between live video feed and demo mode.
    - **Settings**: Configure the model and confidence threshold.
    - **Console**: View logs and messages.
    - **History Log**: View the history of detected falls.

3. **Fall Detection**:
    - The system will start detecting falls based on the selected mode (live or demo).
    - If a fall is detected, a popup alert will be shown, and a sound alert will be played.

## Configuration

- **Settings**:
    - **Model**: Choose between YOLO, MediaPipe, and MoveNet for pose estimation.
    - **Confidence Threshold**: Set the confidence threshold for fall detection.

- **Logging**:
    - Logs are saved in the `logs/` directory.
    - Keypoints and processed data are logged in CSV files.

## Testing

- **Unit Tests**:
    - Tests are located in the `test/` directory.
    - Run tests using:
        ```sh
        python -m unittest discover -s test
        ```

## Files and Directories

- **app.py**: Main application file that initializes the GUI and components.
- **video_feed.py**: Handles video feed and fall detection logic.
- **setting.py**: Manages application settings.
- **process_data.py**: Processes keypoints and performs fall detection.
- **process_data_functions.py**: Contains helper functions for data processing.
- **SETTINGS.py**: Contains global settings and configuration.
- **logs/**: Directory for log files.
- **video/**: Directory for video files.
- **sound/**: Directory for sound files.
- **test/**: Directory for unit tests.

## Contributing

1. **Fork the repository**.
2. **Create a new branch**:
    ```sh
    git checkout -b feature-branch
    ```
3. **Make your changes**.
4. **Commit your changes**:
    ```sh
    git commit -m "Add new feature"
    ```
5. **Push to the branch**:
    ```sh
    git push origin feature-branch
    ```
6. **Create a pull request**.

## License

This project is licensed under the MIT License.

## Acknowledgements

- **COS40006 - Computing Technology Project B**
- **Industry Project i42 - Human Fall Detection using RGB Camera**

For more information, refer to the [project documentation](README.md).
