"""
2-Stage Violence Detection Training Pipeline
=============================================
Stage 1 — YOLO     : detects and crops each person in the frame
Stage 2 — Pose     : MediaPipe runs on the YOLO crop (not full frame)
Stage 3 — Features : velocity, acceleration, angles, distances (309 total)
Stage 4 — LSTM     : Bidirectional LSTM + Attention classifies sequences

Why this is better than full-frame pose:
  - MediaPipe focuses on ONE person at a time (accurate keypoints)
  - Works correctly in multi-person scenes (fights with 2+ people)
  - Inference with run_detection.py uses the same YOLO+Pose pipeline
  - Per-person LSTM history (each person gets their own sequence buffer)

Dataset structure:
  violenceDetectionDataset/Complete Dataset/
  ├── train/
  │   ├── Fight/        -> violent  (label 1)
  │   ├── HockeyFight/  -> violent  (label 1)
  │   ├── MovieFights/  -> violent  (label 1)
  │   └── NonFight/     -> non-violent (label 0)
  └── val/  (same structure)

Output:
  models/violence_lstm_yolo_pose.h5    <- use this with detect_violence.py --yolo
  dataset_cache/yolo_train_X.npy       <- cached features (reused on re-run)
  training_results/                    <- plots and report

Usage:
  python train_violence_yolo_pose.py
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import logging
import json
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT = Path(r"C:\Users\nasih\Documents\FYP_Violence_Prevention\violenceDetectionDataset\Complete Dataset")
CACHE_DIR    = Path(r"C:\Users\nasih\Documents\FYP_Violence_Prevention\dataset_cache")
OUTPUT_DIR   = Path(r"C:\Users\nasih\Documents\FYP_Violence_Prevention\training_results")
MODEL_SAVE   = Path(r"C:\Users\nasih\Documents\FYP_Violence_Prevention\models\violence_lstm_yolo_pose.h5")

# --- YOLO settings ---
YOLO_MODEL        = "yolov8n.pt"   # yolov8n=fastest, yolov8s=balanced, yolov8m=accurate
YOLO_CONFIDENCE   = 0.4            # Lower = detect more people (catches partial occlusions)
YOLO_CROP_PADDING = 20             # Pixels of padding around each person crop

# --- Pose settings ---
MEDIAPIPE_COMPLEXITY = 1           # 0=fast, 1=balanced, 2=accurate
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE  = 0.5

# --- Sequence settings ---
FRAME_STRIDE             = 2       # Sample every Nth frame
SEQUENCE_LENGTH          = 20      # Frames per LSTM input
MAX_SEQUENCES_PER_PERSON = 6       # Per tracked person per video
MAX_PERSONS_PER_VIDEO    = 2       # Only use the 2 most prominent persons (left+right)

# --- Model & training settings ---
LSTM_UNITS        = 64             # Try 128 for more capacity
DROPOUT_RATE      = 0.4
BATCH_SIZE        = 32
EPOCHS            = 50
LEARNING_RATE     = 0.001
USE_ATTENTION     = True
USE_BIDIRECTIONAL = True
USE_TCN           = True
CLASS_WEIGHT      = True

# --- Class mapping ---
CLASS_MAP = {
    'Fight':       1,
    'HockeyFight': 1,
    'MovieFights': 1,
    'NonFight':    0,
}
CLASS_NAMES = ['NonViolent', 'Violent']

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (identical to train_violence_dataset.py and detect_violence.py)
# ─────────────────────────────────────────────────────────────────────────────

LANDMARK_INDICES = {
    'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
}

JOINT_ANGLES = [
    ('left_shoulder', 'left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow', 'right_wrist'),
    ('left_hip', 'left_shoulder', 'left_elbow'),
    ('right_hip', 'right_shoulder', 'right_elbow'),
    ('left_shoulder', 'left_hip', 'left_knee'),
    ('right_shoulder', 'right_hip', 'right_knee'),
]

VIOLENCE_DISTANCES = [
    ('left_wrist', 'nose'),
    ('right_wrist', 'nose'),
    ('left_wrist', 'right_wrist'),
    ('left_elbow', 'left_hip'),
    ('right_elbow', 'right_hip'),
    ('left_shoulder', 'right_shoulder'),
]

FEATURE_DIM = 33 * 3 * 3 + len(JOINT_ANGLES) + len(VIOLENCE_DISTANCES)  # 309


def compute_angle(a, b, c):
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def extract_features(landmarks_seq: np.ndarray) -> np.ndarray:
    """(N, 132) raw landmarks -> (N, 309) engineered features."""
    seq_len = len(landmarks_seq)
    coords  = landmarks_seq.reshape(seq_len, 33, 4)[:, :, :3]

    lsh = LANDMARK_INDICES['left_shoulder']
    rsh = LANDMARK_INDICES['right_shoulder']
    lhp = LANDMARK_INDICES['left_hip']
    rhp = LANDMARK_INDICES['right_hip']

    # Normalized coordinates
    normalized = np.zeros_like(coords)
    for i in range(seq_len):
        hip_center = (coords[i, lhp] + coords[i, rhp]) / 2.0
        sw = np.linalg.norm(coords[i, lsh] - coords[i, rsh])
        normalized[i] = (coords[i] - hip_center) / (sw + 1e-8)

    # Velocity & acceleration
    velocity = np.zeros_like(coords)
    velocity[1:] = (coords[1:] - coords[:-1]) * 10.0

    acceleration = np.zeros_like(coords)
    acceleration[2:] = velocity[2:] - velocity[1:-1]

    # Joint angles
    angles = np.zeros((seq_len, len(JOINT_ANGLES)))
    for i in range(seq_len):
        for j, (a_n, b_n, c_n) in enumerate(JOINT_ANGLES):
            angles[i, j] = compute_angle(
                coords[i, LANDMARK_INDICES[a_n]],
                coords[i, LANDMARK_INDICES[b_n]],
                coords[i, LANDMARK_INDICES[c_n]]
            ) / 180.0

    # Violence distances
    distances = np.zeros((seq_len, len(VIOLENCE_DISTANCES)))
    for i in range(seq_len):
        for j, (a_n, b_n) in enumerate(VIOLENCE_DISTANCES):
            distances[i, j] = np.linalg.norm(
                coords[i, LANDMARK_INDICES[a_n]] - coords[i, LANDMARK_INDICES[b_n]]
            )

    return np.concatenate([
        normalized.reshape(seq_len, -1),
        velocity.reshape(seq_len, -1),
        acceleration.reshape(seq_len, -1),
        angles,
        distances,
    ], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1+2: YOLO person detection → MediaPipe pose on crop
# ─────────────────────────────────────────────────────────────────────────────

def get_crop(frame: np.ndarray, bbox: Tuple, pad: int = YOLO_CROP_PADDING) -> np.ndarray:
    """Crop a person from frame with padding, clamped to frame boundaries."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame[y1:y2, x1:x2]


def video_to_sequences_yolo_pose(
    video_path: str,
    yolo_model,
    pose_extractor
) -> List[np.ndarray]:
    """
    Process one video using 2-stage YOLO+Pose pipeline.

    Returns list of feature sequences, each shape (SEQUENCE_LENGTH, FEATURE_DIM).

    Strategy:
    - Run YOLO on sampled frames to detect persons
    - Sort detections left-to-right so person-slot 0 consistently = left person
    - Run MediaPipe on each person's crop
    - Build per-person landmark buffers
    - Extract sliding-window sequences from each person's buffer
    - Return all sequences (both persons in a fight video both labeled with video class)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    # Per-person-slot landmark accumulators
    # Key = slot index (0 = leftmost person, 1 = second, etc.)
    person_landmarks: Dict[int, List[np.ndarray]] = defaultdict(list)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        # ── Stage 1: YOLO person detection ───────────────────────────────
        try:
            results = yolo_model(
                frame,
                classes=[0],          # class 0 = person in COCO
                conf=YOLO_CONFIDENCE,
                verbose=False
            )
        except Exception:
            frame_idx += 1
            continue

        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                # Area as proxy for prominence
                area = (x2 - x1) * (y2 - y1)
                detections.append((x1, y1, x2, y2, conf, area))

        if not detections:
            frame_idx += 1
            continue

        # Sort by area descending (most prominent first), then limit to MAX_PERSONS
        detections.sort(key=lambda d: d[5], reverse=True)
        detections = detections[:MAX_PERSONS_PER_VIDEO]

        # Sort left-to-right by x1 so slot 0 = leftmost person consistently
        detections.sort(key=lambda d: d[0])

        # ── Stage 2: MediaPipe on each person crop ────────────────────────
        for slot, (x1, y1, x2, y2, conf, _) in enumerate(detections):
            crop = get_crop(frame, (x1, y1, x2, y2))
            if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
                continue

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_result = pose_extractor.process(rgb_crop)

            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks.landmark
                arr = np.array(
                    [[l.x, l.y, l.z, l.visibility] for l in lm],
                    dtype=np.float32
                ).flatten()  # 132 values
                person_landmarks[slot].append(arr)

        frame_idx += 1

    cap.release()

    # ── Stage 3: Build sequences from each person's landmark buffer ───────
    all_sequences = []
    stride = max(1, SEQUENCE_LENGTH // 2)

    for slot, landmarks in person_landmarks.items():
        if len(landmarks) < SEQUENCE_LENGTH:
            continue

        landmarks_arr = np.array(landmarks)   # (N, 132)
        seq_count = 0

        for start in range(0, len(landmarks_arr) - SEQUENCE_LENGTH + 1, stride):
            seq  = landmarks_arr[start:start + SEQUENCE_LENGTH]  # (20, 132)
            feat = extract_features(seq)                          # (20, 309)
            all_sequences.append(feat)
            seq_count += 1
            if seq_count >= MAX_SEQUENCES_PER_PERSON:
                break

    return all_sequences


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING WITH CACHING
# ─────────────────────────────────────────────────────────────────────────────

def process_split(split_name: str, yolo_model, pose_extractor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process all videos in a train/ or val/ split using YOLO+Pose.
    Caches results to CACHE_DIR/yolo_{split_name}_X.npy to skip re-extraction.
    """
    cache_x = CACHE_DIR / f"yolo_{split_name}_X.npy"
    cache_y = CACHE_DIR / f"yolo_{split_name}_y.npy"

    if cache_x.exists() and cache_y.exists():
        log.info(f"[{split_name}] Loading YOLO+Pose cache from {CACHE_DIR}")
        X = np.load(cache_x)
        y = np.load(cache_y)
        log.info(f"[{split_name}] Loaded: X={X.shape} | "
                 f"Violent={int(y.sum())} NonViolent={int((y==0).sum())}")
        return X, y

    split_dir = DATASET_ROOT / split_name
    all_X, all_y = [], []

    class_dirs   = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    total_videos = sum(len(list(d.glob("*.avi"))) for d in class_dirs)
    processed    = 0
    t0           = time.time()

    log.info(f"[{split_name}] Processing {total_videos} videos (YOLO+Pose)...")

    for class_dir in class_dirs:
        class_name = class_dir.name
        label = CLASS_MAP.get(class_name)
        if label is None:
            log.warning(f"Unknown class: {class_name} — skipping")
            continue

        videos    = sorted(class_dir.glob("*.avi"))
        seq_count = 0
        no_detect = 0

        for video_path in videos:
            seqs = video_to_sequences_yolo_pose(
                str(video_path), yolo_model, pose_extractor
            )

            if seqs:
                for s in seqs:
                    all_X.append(s)
                    all_y.append(label)
                seq_count += len(seqs)
            else:
                no_detect += 1

            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                eta = (elapsed / processed) * (total_videos - processed)
                log.info(f"  [{split_name}] {processed}/{total_videos} "
                         f"({processed/total_videos*100:.0f}%) | ETA {eta/60:.1f}m")

        log.info(f"  '{class_name}' (label={label}): "
                 f"{len(videos)} videos -> {seq_count} sequences "
                 f"| {no_detect} videos with no pose detected")

    if not all_X:
        log.error(f"[{split_name}] No sequences extracted. "
                  "Check YOLO_CONFIDENCE and video quality.")
        return np.array([]), np.array([])

    X = np.array(all_X,  dtype=np.float32)
    y = np.array(all_y,  dtype=np.int32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)
    log.info(f"[{split_name}] Cached -> {CACHE_DIR}")
    log.info(f"[{split_name}] Done: X={X.shape} | "
             f"Violent={int(y.sum())} NonViolent={int((y==0).sum())}")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (identical architecture to train_violence_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(sequence_length: int, num_features: int) -> keras.Model:
    inputs = keras.Input(shape=(sequence_length, num_features), name='input')
    x = inputs

    if USE_TCN:
        x = keras.layers.Conv1D(64, 3, padding='causal', activation='relu',
                                dilation_rate=1)(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Conv1D(64, 3, padding='causal', activation='relu',
                                dilation_rate=2)(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)

    lstm1 = keras.layers.LSTM(LSTM_UNITS, return_sequences=True)
    x = keras.layers.Bidirectional(lstm1, name='bilstm_1')(x) if USE_BIDIRECTIONAL else lstm1(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)

    lstm2 = keras.layers.LSTM(LSTM_UNITS, return_sequences=True)
    x = keras.layers.Bidirectional(lstm2, name='bilstm_2')(x) if USE_BIDIRECTIONAL else lstm2(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)

    if USE_ATTENTION:
        attn = keras.layers.Dense(1, activation='tanh')(x)
        attn = keras.layers.Flatten()(attn)
        attn = keras.layers.Activation('softmax')(attn)
        lstm_dim = LSTM_UNITS * (2 if USE_BIDIRECTIONAL else 1)
        attn = keras.layers.RepeatVector(lstm_dim)(attn)
        attn = keras.layers.Permute([2, 1])(attn)
        x = keras.layers.Multiply()([x, attn])
        x = keras.layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)
    else:
        x = keras.layers.GlobalAveragePooling1D()(x)

    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(2, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='violence_yolo_pose_lstm')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_val, y_val):
    num_features = X_train.shape[2]
    log.info(f"Model input: ({SEQUENCE_LENGTH}, {num_features})")

    model = build_model(SEQUENCE_LENGTH, num_features)
    model.summary(print_fn=log.info)

    # Class weights
    cw = None
    if CLASS_WEIGHT:
        n = len(y_train)
        n_v = int(y_train.sum())
        n_nv = n - n_v
        cw = {0: n / (2.0 * n_nv), 1: n / (2.0 * n_v)}
        log.info(f"Class weights: NonViolent={cw[0]:.3f} Violent={cw[1]:.3f}")

    MODEL_SAVE.parent.mkdir(parents=True, exist_ok=True)
    best_ckpt = str(MODEL_SAVE).replace('.h5', '_best.h5')

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_ckpt, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    log.info(f"Training: {len(X_train)} train | {len(X_val)} val sequences")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )

    model.save(str(MODEL_SAVE))
    log.info(f"Model saved -> {MODEL_SAVE}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_val, y_val, history):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_acc = float(np.mean(y_pred == y_val))

    log.info(f"\n{'='*50}")
    log.info(f"  FINAL VALIDATION ACCURACY: {val_acc*100:.2f}%")
    log.info(f"{'='*50}")

    try:
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        report = classification_report(y_val, y_pred, target_names=CLASS_NAMES)
        log.info(f"\nClassification Report:\n{report}")
        (OUTPUT_DIR / 'yolo_classification_report.txt').write_text(report)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred),
                               display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
        ax.set_title(f'YOLO+Pose — Val Acc: {val_acc*100:.1f}%')
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'yolo_confusion_matrix.png', dpi=120)
        plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Val')
        ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True)
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Val')
        ax2.set_title('Loss'); ax2.legend(); ax2.grid(True)
        fig.suptitle(f'YOLO+Pose LSTM — Val Acc: {val_acc*100:.1f}%')
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'yolo_training_curves.png', dpi=120)
        plt.close(fig)

        log.info(f"Plots saved -> {OUTPUT_DIR}")
    except Exception as e:
        log.warning(f"Could not save plots: {e}")

    summary = {
        "pipeline": "YOLO + Pose Estimation (2-stage)",
        "val_accuracy": round(val_acc * 100, 2),
        "best_val_accuracy": round(max(history.history['val_accuracy']) * 100, 2),
        "epochs_trained": len(history.history['accuracy']),
        "config": {
            "YOLO_MODEL": YOLO_MODEL,
            "YOLO_CONFIDENCE": YOLO_CONFIDENCE,
            "SEQUENCE_LENGTH": SEQUENCE_LENGTH,
            "FEATURE_DIM": FEATURE_DIM,
            "FRAME_STRIDE": FRAME_STRIDE,
            "MAX_SEQUENCES_PER_PERSON": MAX_SEQUENCES_PER_PERSON,
            "MAX_PERSONS_PER_VIDEO": MAX_PERSONS_PER_VIDEO,
            "LSTM_UNITS": LSTM_UNITS,
            "DROPOUT_RATE": DROPOUT_RATE,
        }
    }
    (OUTPUT_DIR / 'yolo_training_summary.json').write_text(json.dumps(summary, indent=2))
    return val_acc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Violence Detection — YOLO + Pose 2-Stage Training Pipeline")
    log.info("=" * 60)
    log.info(f"YOLO model : {YOLO_MODEL}")
    log.info(f"Dataset    : {DATASET_ROOT}")
    log.info(f"Output     : {MODEL_SAVE}")

    for split in ['train', 'val']:
        if not (DATASET_ROOT / split).exists():
            log.error(f"Missing dataset split: {DATASET_ROOT / split}")
            sys.exit(1)

    # Initialise YOLO (shared across all videos)
    log.info(f"\nLoading YOLO model: {YOLO_MODEL}")
    yolo = YOLO(YOLO_MODEL)

    # Initialise MediaPipe Pose (shared across all frames)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MEDIAPIPE_COMPLEXITY,
        smooth_landmarks=True,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    log.info("\n[Phase 1] Extracting YOLO+Pose features from training videos...")
    X_train, y_train = process_split('train', yolo, pose)

    log.info("\n[Phase 2] Extracting YOLO+Pose features from validation videos...")
    X_val, y_val = process_split('val', yolo, pose)

    pose.close()

    if X_train.size == 0 or X_val.size == 0:
        log.error("No data extracted. Lower YOLO_CONFIDENCE or check videos.")
        sys.exit(1)

    log.info("\n[Phase 3] Training model...")
    model, history = train(X_train, y_train, X_val, y_val)

    log.info("\n[Phase 4] Evaluating...")
    val_acc = evaluate(model, X_val, y_val, history)

    log.info("\n" + "=" * 60)
    log.info("  TRAINING COMPLETE")
    log.info(f"  Validation Accuracy : {val_acc*100:.2f}%")
    log.info(f"  Model saved to      : {MODEL_SAVE}")
    log.info("\n  To run inference with YOLO+Pose:")
    log.info("    python detect_violence.py --yolo")
    log.info("    python detect_violence.py --yolo --source path/to/video.mp4")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
