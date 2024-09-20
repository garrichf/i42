import cv2
from Settings import Settings
from DataProcessing import DataProcessing
from YOLO import process_frame as yolo_process
from MediaPipe import process_frame as mediapipe_process
from PoseNet import process_frame as posenet_process

# Initialize settings and data processing
settings = Settings()
data_processor = DataProcessing(settings)

# Map the model selection to the corresponding processing function
model_map = {
    "YOLOv8": yolo_process,
    "MediaPipe": mediapipe_process,
    "PoseNet": posenet_process
}

def main():
    cap = data_processor.get_video_source()  # Get the video source based on settings

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the selected model and process the frame
        model_choice = settings.get_model_choice()
        model_process = model_map[model_choice]
        processed_frame, fall_detected = data_processor.process_data(frame, model_process)

        # Display the processed video
        cv2.imshow("Fall Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
