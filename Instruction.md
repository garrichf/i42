# Instructions

## Initial Preparation
- Create a Python virtual environment with `python -m venv venv`.
- Activate the virtual environment:
  - On macOS: `source venv/bin/activate`
  - On Windows: `venv\Scripts\activate`
- To deactivate the virtual environment, use `deactivate`.
- Install the required packages with `pip install -r requirements.txt`.

## `movenet_final.py`
### Steps and Notes
- Run the script with `python movenet_final.py`.
- The script should run and perform pose estimation on the included ADL.mp4 svideo file and print the dataframe output frame by frame until the video ends.
- If you want to test live stream pose estimation, change the `video_path` variable to `0` and run the script. This will activate the device camera and perform pose estimation. To stop, press `Ctrl + C`.

## `data_processing_latest.py`
### Steps and Notes
- Use this script to perform angle, velocity, and acceleration calculations.
- It can be used in tandem with `movenet_final.py` to process the output dataframe (i.e., `data_processing_latest.py` takes input from `movenet_final.py` output and provides a dataframe with additional columns).

## All Jupyter Notebooks
### Steps and Notes
- Contains all steps and functions involved.
- Run the cells sequentially to get outputs.