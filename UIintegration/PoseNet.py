import cv2
import numpy as np
from tensorflow.keras.models import load_model

class PoseNetModel:
    def __init__(self):
        self.fall_detection_model = load_model("falldetect_16092024_1307_alldatacombined_reg_reduceplateau.keras")
        self.net = cv2.dnn.readNetFromCaffe("pose_deploy_linevec.prototxt", "pose_iter_440000.caffemodel")

    def process_frame(self, frame, confidence_threshold):
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=True)
        self.net.setInput(blob)
        output = self.net.forward()

        keypoints = self.extract_keypoints(output)

        # Handle no detections case
        if len(keypoints) == 0:
            print("Warning: No keypoints detected.")
            keypoints = np.zeros((115, 2))  # Placeholder for 115 keypoints

        # Ensure we have exactly 115 keypoints
        while len(keypoints) < 115:
            print(f"Warning: Expected 115 keypoints but got {len(keypoints)}. Padding with zeros.")
            keypoints.append([0 ,0]) 

        fall_detected = self.predict_fall(keypoints[:115], confidence_threshold)
        
        frame = self.annotate_frame(frame,keypoints[:115],fall_detected)

        return frame ,fall_detected

    def extract_keypoints(self ,output):
       # Extracting logic remains unchanged 
       ...

    def predict_fall(self,keypoints ,confidence_threshold):
       features=np.array(keypoints).flatten().reshape(1,-1)
       prediction=self.fall_detection_model.predict(features)
       return prediction[0][1]>confidence_threshold

    def annotate_frame(self ,frame,keypoints ,fall_detected):
       color=(0 ,255 ,0) if not fall_detected else (0 ,0 ,255)

       for kp in keypoints:
           cv2.circle(frame,(kp[0],kp[1]),5,color,-1)

       return frame 