import cv2

class DataProcessing:

    def __init__(self ,settings):
       self.settings=settings 

    def get_video_source(self):
       if self.settings.get_data_source():
           return cv2.VideoCapture(0) # Webcam

       else:
           video_path="C:/Users/User/Desktop/UIintegration/Videos/video1.mp4"
           return cv2.VideoCapture(video_path)

    def process_data(self ,frame ,model):
       # Use the selected model to process the frame 
       confidence_threshold=self.settings.get_confidence_threshold()
       processed_frame ,fall_detected=model.process_frame(frame ,confidence_threshold)

       return processed_frame ,fall_detected 