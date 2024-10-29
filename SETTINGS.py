from datetime import datetime
CONFIDENCE_THRESHOLD = 0.5
POSE_MODEL_USED = "YOLO" # or "MEDIAPIPE" or "MOVENET"
DEMO_MODE = True 
LOG_FILEPATH = f"logs/{datetime.now().strftime('%d%m%Y_%H%M')}_LOG.csv"
PROCESSED_OUTPUT_CSV = f"logs/{datetime.now().strftime('%d%m%Y_%H%M')}_OUTPUT.csv"

def read_config(file_path):
        """
        Reads the configuration file and extracts the model type and confidence threshold.

        Args:
            file_path (str): The path to the configuration file.

        Returns:
            tuple: A tuple containing:
                - model (str): The model type (default is "YOLOv8").
                - confidence_threshold (float): The confidence threshold (default is 0.5).
        """
        model = "YOLOv8"
        confidence_threshold = 0.5
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("Model:"):
                    model = line.split(":")[1].strip()
                elif line.startswith("Confidence Threshold:"):
                    confidence_threshold = float(line.split(":")[1].strip())
        return model, confidence_threshold

