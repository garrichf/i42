import cv2
import os

'''
This code is just to convert the sequence of frames into an mp4 file for easier playback. 
'''

# Path to the folder containing the subfolders with frames
base_folder = 'annotated'  # Adjust this path as needed

# Iterate over each subfolder in the base folder
for video_folder in os.listdir(base_folder):
    video_folder_path = os.path.join(base_folder, video_folder)
    
    if os.path.isdir(video_folder_path):
        # Output video file path
        output_video_file = os.path.join(base_folder, f"{video_folder}.mp4")

        # Get list of frame files in the current subfolder
        frame_files = [f for f in os.listdir(video_folder_path) if f.endswith(('.png', '.jpg'))]
        frame_files.sort()  # Ensure files are in the correct order
        print(frame_files)

        if not frame_files:
            print(f"No frames found in {video_folder_path}")
            continue

        # Read the first frame to get the frame size
        first_frame_path = os.path.join(video_folder_path, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        height, width, layers = first_frame.shape

        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        fps = 30  # Frames per second
        video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

        # Write the frames to the video file
        for frame_file in frame_files:
            frame_path = os.path.join(video_folder_path, frame_file)
            frame = cv2.imread(frame_path)
            video_writer.write(frame)

        # Release the video writer
        video_writer.release()

        print(f"Video file saved as {output_video_file}")

print("Processing complete. All videos created.")
