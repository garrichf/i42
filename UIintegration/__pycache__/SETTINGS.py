from datetime import datetime

CONFIDENCE_THRESHOLD = 0.5
POSE_MODEL_USED = "YOLO" # or "MEDIAPIPE" or "MOVENET"
DEMO_MODE = True # demo_mode is active by default, if False, it is live mode

LOG_FILEPATH = f"logs/{datetime.now().strftime('%d%m%Y_%H%M')}_LOG.csv"
PROCESSED_OUTPUT_CSV = f"logs/{datetime.now().strftime('%d%m%Y_%H%M')}_OUTPUT.csv"