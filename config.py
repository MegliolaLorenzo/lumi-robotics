import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI — required for GPT-4o Vision detection
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Detection
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "15"))

# YOLOv8 model paths
YOLO_MODEL_PATH = "models/violence_yolov8n.pt"
YOLO_FALLBACK_MODEL = "yolov8n.pt"

# Video source: "webcam" | "robot" | "video"
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "webcam")

# Robot stream — IP of the Unitree Go2
ROBOT_IP = os.getenv("ROBOT_IP", "")
ROBOT_STREAM_URL = os.getenv(
    "ROBOT_STREAM_URL",
    "http://localhost:8080/api/video/stream"
)

# Alert output mode
# "print"   → print to terminal (default)
# "webhook" → send POST to ALERT_WEBHOOK_URL
# "file"    → append to alerts.log
ALERT_MODE = os.getenv("ALERT_MODE", "print")
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
