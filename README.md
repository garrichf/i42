# COS40006 - Computing Technology Project B
## Industry Project i42 - Human Fall Detection using RGB Camera
### Nick's MoveNet Pose Estimation Implementation v.3.0

#### Version 1
##### Notes:
My first attempt to implementing MoveNet was challenging as it was not straightforward like YOLOv8 or MediaPipe with dedicated packaged from PyPi.
First, I attempted to load it via the downloaded model and direct url, which was not successful.
Then I've learnt of Tensorflow Hub and loading it through the hub as similar to their tutorial codebase on Tensorflow Hub.
##### Implementation:
- Draft a script to load the MoveNet Thunder 4 single pose model through downloaded file and direct url unsuccessful.
- Successfully loaded the model through Tensorflow Hub as with the tutorial codebase (https://www.tensorflow.org/hub/tutorials/movenet).
- Adapting YOLOv8 script structure for pose estimation and visualisation.
##### Result:
- Having a structured script that can load the MoveNet Pose Estimation Model.
- Script runs and gives keypoints abeit questionable accuracy and wrong X and Y order (i.e, Y before X).

#### Version 2
##### Notes:
The second attempt to implementing MoveNet, was better but achieved sub-par outputs as visualisation did not drawn well onto the image.
Extracted keypoints coordinates were mis-label and misplaced as MoveNet output the Y coordinates before the X as part of its algorithm.
##### Implementation:
- Work upon the previous version to improve the visualisation and accuracy of coordinates.
- Implement confidence threshold to improve keypoints extraction performance.
- Adapting the entired MoveNet tutorial codebase, refactoring the helper functions and restructuring the inference functions to accept the project input (loaded video files or live video stream).
- Implemented a function to add bounding box instead of using the existing codebase, as the given one generate a bounding box too big.
- Implementing keypoints to dataframe function, which takes the inference output (keyoints) and put them into a dataframe with columns named as specified by Garrich and Fall Detection model.
- Implementing the Data Processing script for calculating the additional angles and features (velocity, acceleration of keypoints).
##### Result:
- Working MoveNet pose estimation with accurate keypoints extractions and exported to dataframe to suit the inputs of Fall Detection model.
- Re-organised output dataframe with coordinates order and meeting column naming convention by Garrich.
- Functional visualisation with accurate keypoints drawn and edges connected well with bounding box of the subject within video frame.

#### Version 3
##### Notes:
The latest attempt is to improve upon the previous version (adapted from MoveNet codebase) with well structured and good degree of function.
Garrich insisted that some columns is to be removed as he noticed it improving the Fall Detection model performance, this was implemented into the code.
Garrich also specified how the script would perform in the final integration and thus re-structuring work is needed for the integration.
##### Implementation:
- Re-structured the latest codebase.
- Removed the visualisation functions.
- Modified the dataframe processing function to enable output dataframe to spec by Garrich and Fall Detection model.
- Modified and restructured the separated (i.e., video and live inference) into a single function that takes a video frame and perform pose estimation.
