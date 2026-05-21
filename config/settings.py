"""
Configuration Settings for Violence Detection System
====================================================
Central configuration file for all system parameters.
Modify these settings to customize the behavior of the detection system.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
ALERTS_DIR = BASE_DIR / "alerts"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, DATA_DIR, LOGS_DIR, ALERTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# VIDEO SOURCE CONFIGURATION
# =============================================================================
class VideoConfig:
    # Video source options:
    # - Integer (0, 1, etc.) for webcam
    # - String path for local file
    # - URL string for RTSP/HTTP streams
    SOURCE = 0

    # RTSP Configuration
    RTSP_BUFFER_SIZE = 1
    RTSP_TRANSPORT = "tcp"  # "tcp" or "udp"

    # Frame processing
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS_LIMIT = 30

    # Reconnection settings for streams
    RECONNECT_DELAY = 2  # seconds
    MAX_RECONNECT_ATTEMPTS = 5

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
class ModelConfig:
    # YOLO Configuration
    # All four detector "levers" are env-overridable so you can A/B test
    # without editing code, e.g.:
    #   YOLO_MODEL=yolov8s.pt YOLO_CONFIDENCE=0.3 python run_detection.py --web ...
    # CLI flags on run_detection.py take precedence over these.
    YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")  # yolov8n/s/m.pt
    YOLO_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.5"))
    YOLO_PERSON_CLASS = 0  # COCO class ID for person

    # LSTM Configuration
    LSTM_MODEL_PATH = BASE_DIR / "lstm-model.h5"
    LSTM_SEQUENCE_LENGTH = int(os.environ.get("LSTM_SEQUENCE_LENGTH", "20"))
    LSTM_UNITS = 64
    LSTM_DROPOUT = 0.3

    # Pose landmarks
    NUM_POSE_LANDMARKS = 33
    FEATURES_PER_LANDMARK = 4  # x, y, z, visibility
    TOTAL_FEATURES = NUM_POSE_LANDMARKS * FEATURES_PER_LANDMARK  # 132

    # Detection thresholds
    VIOLENCE_THRESHOLD = float(os.environ.get("VIOLENCE_THRESHOLD", "0.6"))
    DETECTION_SMOOTHING_WINDOW = 5  # Number of predictions to average

    # Scene-level video-clip classifier (pretrained VideoMAE).
    # CLASSIFIER_MODE selects which path is active:
    #   'video_clip' — scene-level VideoMAE only (default; recommended)
    #   'pose_lstm'  — per-person LSTM only (legacy baseline)
    #   'both'       — run both and OR the violence verdict
    CLASSIFIER_MODE = os.environ.get("CLASSIFIER_MODE", "video_clip")
    VIDEO_CLASSIFIER_MODEL = os.environ.get(
        "VIDEO_CLASSIFIER_MODEL", "MCG-NJU/videomae-base-finetuned-kinetics"
    )
    VIDEO_CLIP_LENGTH = int(os.environ.get("VIDEO_CLIP_LENGTH", "16"))
    VIDEO_CLIP_STRIDE = int(os.environ.get("VIDEO_CLIP_STRIDE", "8"))

# =============================================================================
# DETECTION CONFIGURATION
# =============================================================================
class DetectionConfig:
    # Warm-up period (frames to skip at start)
    WARMUP_FRAMES = 30

    # Multi-person tracking
    MAX_PERSONS = 10
    PERSON_TRACKING_ENABLED = True

    # Temporal smoothing
    SMOOTHING_ENABLED = True
    SMOOTHING_WINDOW = 5

    # Interaction detection
    INTERACTION_DISTANCE_THRESHOLD = 100  # pixels

    # Violence types
    VIOLENCE_CLASSES = [
        "neutral",
        "pushing",
        "punching",
        "kicking",
        "fighting",
        "weapon_threat"
    ]

# =============================================================================
# ALERT CONFIGURATION
# =============================================================================
class AlertConfig:
    # Alert settings
    ENABLED = True
    COOLDOWN_SECONDS = 10  # Minimum time between alerts

    # Sound alert
    SOUND_ENABLED = True
    SOUND_FILE = BASE_DIR / "alert.wav"

    # Email alerts
    EMAIL_ENABLED = False
    EMAIL_SMTP_SERVER = "smtp.gmail.com"
    EMAIL_SMTP_PORT = 587
    EMAIL_SENDER = ""
    EMAIL_PASSWORD = ""  # Use app-specific password
    EMAIL_RECIPIENTS = []

    # Webhook alerts (Slack, Discord, etc.)
    WEBHOOK_ENABLED = False
    WEBHOOK_URL = ""

    # Screenshot capture on alert
    CAPTURE_SCREENSHOT = True
    SCREENSHOT_DIR = ALERTS_DIR / "screenshots"

    # Video clip capture
    CAPTURE_VIDEO_CLIP = True
    VIDEO_CLIP_DURATION = 10  # seconds before and after alert
    VIDEO_CLIP_DIR = ALERTS_DIR / "clips"

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
class TrainingConfig:
    # Dataset paths
    DATASET_DIR = DATA_DIR / "datasets"

    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2

    # Data augmentation
    AUGMENTATION_ENABLED = True
    AUGMENTATION_FACTOR = 3  # Multiply dataset size by this factor

    # Class weights for imbalanced data
    USE_CLASS_WEIGHTS = True

    # Early stopping
    EARLY_STOPPING_PATIENCE = 15

    # Model checkpointing
    CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# =============================================================================
# WEB DASHBOARD CONFIGURATION
# =============================================================================
class WebConfig:
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = False
    SECRET_KEY = "change-this-secret-key-in-production"

    # Video streaming
    STREAM_QUALITY = 80  # JPEG quality (1-100)
    STREAM_FPS = 15

    # CORS origins. Comma-separated list, or "*" for any origin.
    # Override in dev via env: CORS_ORIGINS=http://localhost:3100
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
class DatabaseConfig:
    # SQLite by default (no setup required).
    # Override with DATABASE_URL env var for PostgreSQL:
    #   DATABASE_URL=postgresql://user:password@localhost/violence_db
    URL = "sqlite:///violence_detection.db"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
class LogConfig:
    LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    FILE = LOGS_DIR / "violence_detection.log"
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT = 5

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================
class VisualizationConfig:
    # Bounding box colors (BGR format)
    COLOR_NEUTRAL = (0, 255, 0)  # Green
    COLOR_VIOLENT = (0, 0, 255)  # Red
    COLOR_WARNING = (0, 165, 255)  # Orange

    # Text settings
    FONT = "cv2.FONT_HERSHEY_SIMPLEX"
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2

    # Display options
    SHOW_SKELETON = True
    SHOW_BOUNDING_BOX = True
    SHOW_CONFIDENCE = True
    SHOW_FPS = True
    SHOW_PERSON_ID = True
