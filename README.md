# COS40006 - Computing Technology Project B
## Industry Project i42 - Human Fall Detection using RGB Camera
### Nick's MoveNet Pose Estimation Implementation

#### Version 1
##### Notes:
My first attempt at implementing MoveNet was challenging as it was not as straightforward as YOLOv8 or MediaPipe, which have dedicated packages from PyPi. Initially, I tried to load the model via a downloaded file and direct URL, but this was unsuccessful. I then learned about TensorFlow Hub and successfully loaded the model through the hub, similar to their tutorial codebase on TensorFlow Hub.

##### Implementation:
- Drafted a script to load the MoveNet Thunder 4 single pose model through a downloaded file and direct URL, which was unsuccessful.
- Successfully loaded the model through TensorFlow Hub as per the tutorial codebase (https://www.tensorflow.org/hub/tutorials/movenet).
- Adapted the YOLOv8 script structure for pose estimation and visualization.

##### Result:
- Developed a structured script that can load the MoveNet Pose Estimation Model.
- The script runs and provides keypoints, albeit with questionable accuracy and incorrect X and Y order (i.e., Y before X).

#### Version 2
##### Notes:
The second attempt at implementing MoveNet was better but still produced sub-par outputs as the visualization did not render well onto the image. Extracted keypoint coordinates were mislabeled and misplaced because MoveNet outputs the Y coordinates before the X as part of its algorithm.

##### Implementation:
- Improved upon the previous version to enhance visualization and coordinate accuracy.
- Implemented a confidence threshold to improve keypoint extraction performance.
- Adapted the entire MoveNet tutorial codebase, refactoring the helper functions and restructuring the inference functions to accept the project input (loaded video files or live video stream).
- Implemented a function to add a bounding box instead of using the existing codebase, as the given one generated a bounding box that was too large.
- Implemented a function to convert keypoints to a dataframe, which takes the inference output (keypoints) and puts them into a dataframe with columns named as specified by Garrich and the Fall Detection model.
- Implemented the Data Processing script for calculating additional angles and features (velocity, acceleration of keypoints).

##### Result:
- Achieved a working MoveNet pose estimation with accurate keypoint extractions exported to a dataframe to suit the inputs of the Fall Detection model.
- Reorganized the output dataframe with the correct coordinate order and column naming convention as specified by Garrich.
- Functional visualization with accurate keypoints drawn and edges connected well, with the bounding box of the subject within the video frame.

#### Version 3 - Latest
##### Notes:
The latest attempt aims to improve upon the previous version (adapted from the MoveNet codebase) with a well-structured and functional implementation. Garrich insisted that some columns be removed as he noticed it improved the Fall Detection model's performance, which was implemented into the code. Garrich also specified how the script should perform in the final integration, necessitating restructuring work for the integration.

##### Implementation:
- Restructured the latest codebase.
- Removed the visualization functions.
- Modified the dataframe processing function to enable the output dataframe to meet the specifications by Garrich and the Fall Detection model.
- Modified and restructured the separate (i.e., video and live inference) functions into a single function that takes a video frame and performs pose estimation.

##### Result:
- Developed a final restructured MoveNet implementation that is more efficient and has half the lines of version 2.
- Directly serves the requirements of the Fall Detection model, i.e., outputs 17 keypoint coordinates (34 total as each keypoint gives X and Y).
- Processes the frame input as part of the loop of `cv2.VideoCapture()` within the main script.
- Removed the original visualization functions as they will be used with universal visualization.
