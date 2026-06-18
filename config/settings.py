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
    YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt
    YOLO_CONFIDENCE = 0.5
    YOLO_PERSON_CLASS = 0  # COCO class ID for person

    # LSTM Configuration
    LSTM_MODEL_PATH = BASE_DIR / "lstm-model.h5"
    LSTM_SEQUENCE_LENGTH = 20  # Number of frames for temporal analysis
    LSTM_UNITS = 64
    LSTM_DROPOUT = 0.3

    # Pose landmarks
    NUM_POSE_LANDMARKS = 33
    FEATURES_PER_LANDMARK = 4  # x, y, z, visibility
    TOTAL_FEATURES = NUM_POSE_LANDMARKS * FEATURES_PER_LANDMARK  # 132

    # Detection thresholds
    VIOLENCE_THRESHOLD = 0.6  # Confidence threshold for violence detection
    DETECTION_SMOOTHING_WINDOW = 5  # Number of predictions to average

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
# VIDEOMAE CONFIGURATION
# =============================================================================
class VideoMAEConfig:
    # Base HuggingFace model (used for initial download and as fallback)
    BASE_MODEL = os.environ.get(
        "VIDEOMAE_BASE_MODEL", "MCG-NJU/videomae-base-finetuned-kinetics"
    )
    # Fine-tuned checkpoint directory (saved by finetune_videomae.py).
    # If this directory exists it takes priority over BASE_MODEL at inference time.
    CHECKPOINT_PATH = Path(
        os.environ.get("VIDEOMAE_CHECKPOINT", str(MODELS_DIR / "violence_videomae"))
    )
    # Inference settings
    CLIP_LEN = int(os.environ.get("VIDEOMAE_CLIP_LEN", "16"))
    CLIP_STRIDE = int(os.environ.get("VIDEOMAE_CLIP_STRIDE", "2"))  # sample every N frames
    VIOLENCE_THRESHOLD = float(os.environ.get("VIDEOMAE_THRESHOLD", "0.65"))
    SMOOTH_WINDOW = int(os.environ.get("VIDEOMAE_SMOOTH_WINDOW", "5"))
    # Minimum number of persons YOLO must detect before scene violence can fire.
    # A single person cannot be in a fight — this eliminates solo-motion false positives.
    MIN_PERSONS_FOR_ALERT = int(os.environ.get("VIDEOMAE_MIN_PERSONS", "2"))
    # Training settings
    DATASET_PATH = Path(
        os.environ.get(
            "VIDEOMAE_DATASET",
            str(BASE_DIR / "violenceDetectionDataset" / "Complete Dataset"),
        )
    )
    TRAIN_EPOCHS = int(os.environ.get("VIDEOMAE_EPOCHS", "8"))
    TRAIN_BATCH_SIZE = int(os.environ.get("VIDEOMAE_BATCH_SIZE", "2"))
    TRAIN_LR = float(os.environ.get("VIDEOMAE_LR", "1e-4"))
    TRAIN_SEED = 42

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
# INTERACTION-AWARE DETECTION CONFIGURATION  (Milestone 1-3)
# =============================================================================
class InteractionConfig:
    # ── Model paths ──────────────────────────────────────────────────────────
    POSE_MODEL             = "yolov8n-pose.pt"          # auto-downloads from ultralytics
    INTERACTION_MODEL_PATH = MODELS_DIR / "violence_interaction_lstm.h5"

    # ── Sequence settings (MUST be identical in training and inference) ──────
    SEQUENCE_LENGTH = 20
    FRAME_STRIDE    = 2     # sample every Nth frame during extraction

    # ── Feature dimensions ────────────────────────────────────────────────────
    #   Isolated block (8) + scene scalars (2) = 10  → Ablation B
    #   Full (10) + interaction block (9)      = 19  → Ablation C (proposed)
    FEATURE_DIM_ISOLATED = 10
    FEATURE_DIM_FULL     = 19

    # ── Heuristic thresholds (ported from reference implementation) ──────────
    HEURISTIC_HISTORY     = 25     # keypoint frames kept per tracked person
    FIGHT_THRESHOLD       = 0.35   # combined score needed to flag a fight
    SPEED_FIGHT_THRESH    = 8.0    # avg wrist/elbow speed (px/frame) to score
    PROXIMITY_THRESH      = 250    # px — max centre-to-centre dist to score
    ANGLE_VARIANCE_THRESH = 400    # body-angle variance threshold
    OVERLAP_IOU_THRESH    = 0.05   # bbox IoU threshold
    WRIST_TO_BODY_THRESH  = 80     # px — wrist near opponent torso
    SMOOTH_WINDOW         = 8      # majority-vote window size

    # ── Dataset path ─────────────────────────────────────────────────────────
    # Supports two layouts (auto-detected at runtime):
    #   A) RLVS flat layout:       Violence/*.mp4  +  NonViolence/*.mp4
    #   B) Existing train/val layout:  train/{Fight,NonFight,...}/*.avi
    RLVS_DATASET_PATH = Path(
        os.environ.get(
            "VIOLENCE_DATASET",
            str(BASE_DIR / "violenceDetectionDataset" / "Complete Dataset"),
        )
    )
    RLVS_CACHE_DIR   = BASE_DIR / "dataset_cache" / "rlvs"
    RLVS_RESULTS_DIR = BASE_DIR / "training_results" / "interaction"
    RLVS_TRAIN_RATIO = 0.70
    RLVS_VAL_RATIO   = 0.15
    RLVS_TEST_RATIO  = 0.15
    RLVS_SEED        = 42

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
