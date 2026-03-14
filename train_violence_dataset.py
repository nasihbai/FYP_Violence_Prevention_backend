"""
Violence Detection Training Pipeline
=====================================
Processes the violenceDetectionDataset (AVI videos) -> extracts pose landmarks
-> applies feature engineering -> trains LSTM -> reports accuracy.

Dataset structure expected:
  violenceDetectionDataset/Complete Dataset/
  ├── train/
  │   ├── Fight/       -> violent (label 1)
  │   ├── HockeyFight/ -> violent (label 1)
  │   ├── MovieFights/ -> violent (label 1)
  │   └── NonFight/    -> non-violent (label 0)
  └── val/
      ├── Fight/
      ├── HockeyFight/
      ├── MovieFights/
      └── NonFight/

Usage:
  python train_violence_dataset.py

Outputs:
  dataset_cache/   - cached pose features (skip re-extraction on re-runs)
  models/violence_lstm_dataset.h5  - trained model
  training_results/ - accuracy plots, confusion matrix, classification report
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import json
import time
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — tune these values to experiment
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT = Path(r"C:\Users\nasih\Documents\FYP_Violence_Prevention\violenceDetectionDataset\Complete Dataset")
CACHE_DIR    = Path(r"C:\Users\nasih\Documents\FYP_Violence_Prevention\dataset_cache")
OUTPUT_DIR   = Path(r"C:\Users\nasih\Documents\FYP_Violence_Prevention\training_results")
MODEL_SAVE   = Path(r"C:\Users\nasih\Documents\FYP_Violence_Prevention\models\violence_lstm_dataset.h5")

# --- Pose extraction settings ---
FRAME_STRIDE          = 2        # Sample every Nth frame (2 = half frames, faster)
SEQUENCE_LENGTH       = 20       # Frames per LSTM sequence (matches real-time pipeline)
MAX_SEQUENCES_PER_VIDEO = 8     # Max sequences extracted per video (balance diversity)
MEDIAPIPE_COMPLEXITY  = 1        # 0=fast, 1=balanced, 2=accurate
DETECTION_CONFIDENCE  = 0.5      # MediaPipe minimum detection confidence
TRACKING_CONFIDENCE   = 0.5      # MediaPipe minimum tracking confidence

# --- Model & training settings ---
LSTM_UNITS      = 64             # LSTM layer size (try 128 for more capacity)
DROPOUT_RATE    = 0.4            # Dropout regularization (0.3-0.5)
BATCH_SIZE      = 32             # Training batch size
EPOCHS          = 50             # Max epochs (early stopping will trigger earlier)
LEARNING_RATE   = 0.001          # Adam optimizer LR (try 0.0005 for finer tuning)
USE_ATTENTION   = True           # Attention mechanism
USE_BIDIRECTIONAL = True         # Bidirectional LSTM
USE_TCN         = True           # Temporal Convolutional preprocessing
CLASS_WEIGHT    = True           # Balance class weights for imbalanced data

# --- Class mapping ---
# Map folder names to binary labels
CLASS_MAP = {
    'Fight':       1,   # violent
    'HockeyFight': 1,   # violent
    'MovieFights': 1,   # violent
    'NonFight':    0,   # non-violent
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
# STEP 1: Feature Engineering (reuses existing logic)
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


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle at joint b given three 3D points."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def extract_features_from_sequence(landmarks_seq: np.ndarray) -> np.ndarray:
    """
    Convert raw landmark sequences (N, 132) -> rich feature sequences (N, 309).

    Features:
      - Normalized coordinates (99): translation+scale invariant pose
      - Velocity (99): frame-to-frame motion, amplified x10
      - Acceleration (99): change in velocity (detects strikes)
      - Joint angles (6): arm/shoulder/body angles
      - Violence distances (6): fist-to-head, hand proximity, etc.
    """
    seq_len = len(landmarks_seq)
    lm3d = landmarks_seq.reshape(seq_len, 33, 4)   # (N, 33, 4)
    coords = lm3d[:, :, :3]                         # (N, 33, 3)

    # 1. Normalized coordinates
    lsh = LANDMARK_INDICES['left_shoulder']
    rsh = LANDMARK_INDICES['right_shoulder']
    lhp = LANDMARK_INDICES['left_hip']
    rhp = LANDMARK_INDICES['right_hip']

    normalized = np.zeros_like(coords)
    for i in range(seq_len):
        hip_center = (coords[i, lhp] + coords[i, rhp]) / 2.0
        centered = coords[i] - hip_center
        sw = np.linalg.norm(coords[i, lsh] - coords[i, rsh])
        normalized[i] = centered / (sw + 1e-8)

    # 2. Velocity
    velocity = np.zeros_like(coords)
    velocity[1:] = (coords[1:] - coords[:-1]) * 10.0

    # 3. Acceleration
    acceleration = np.zeros_like(coords)
    acceleration[2:] = velocity[2:] - velocity[1:-1]

    # 4. Joint angles
    angles = np.zeros((seq_len, len(JOINT_ANGLES)))
    for i in range(seq_len):
        for j, (a_name, b_name, c_name) in enumerate(JOINT_ANGLES):
            a = coords[i, LANDMARK_INDICES[a_name]]
            b = coords[i, LANDMARK_INDICES[b_name]]
            c = coords[i, LANDMARK_INDICES[c_name]]
            angles[i, j] = compute_angle(a, b, c) / 180.0  # normalize to [0,1]

    # 5. Violence distances
    distances = np.zeros((seq_len, len(VIOLENCE_DISTANCES)))
    for i in range(seq_len):
        for j, (a_name, b_name) in enumerate(VIOLENCE_DISTANCES):
            a = coords[i, LANDMARK_INDICES[a_name]]
            b = coords[i, LANDMARK_INDICES[b_name]]
            distances[i, j] = np.linalg.norm(a - b)

    features = np.concatenate([
        normalized.reshape(seq_len, -1),   # 99
        velocity.reshape(seq_len, -1),     # 99
        acceleration.reshape(seq_len, -1), # 99
        angles,                            # 6
        distances,                         # 6
    ], axis=1)

    return features.astype(np.float32)


FEATURE_DIM = 33 * 3 + 33 * 3 + 33 * 3 + len(JOINT_ANGLES) + len(VIOLENCE_DISTANCES)  # 309


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Video Processing
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks_from_video(video_path: str, pose_extractor) -> List[np.ndarray]:
    """
    Extract pose landmark arrays from a video file.

    Returns:
        List of 132-dim landmark arrays (one per sampled frame with valid pose).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    landmarks_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose_extractor.process(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                arr = np.array([[l.x, l.y, l.z, l.visibility] for l in lm],
                               dtype=np.float32).flatten()  # 132
                landmarks_list.append(arr)

        frame_idx += 1

    cap.release()
    return landmarks_list


def video_to_feature_sequences(video_path: str, pose_extractor) -> np.ndarray:
    """
    Convert one video -> multiple feature sequences of shape (SEQUENCE_LENGTH, FEATURE_DIM).

    Uses a sliding window with overlap to maximise training samples per video.
    Returns array of shape (num_sequences, SEQUENCE_LENGTH, FEATURE_DIM) or empty.
    """
    landmarks = extract_landmarks_from_video(video_path, pose_extractor)

    if len(landmarks) < SEQUENCE_LENGTH:
        return np.array([])

    landmarks_arr = np.array(landmarks)  # (N, 132)

    # Sliding window with stride = SEQUENCE_LENGTH // 2
    stride = max(1, SEQUENCE_LENGTH // 2)
    sequences = []
    for start in range(0, len(landmarks_arr) - SEQUENCE_LENGTH + 1, stride):
        seq = landmarks_arr[start:start + SEQUENCE_LENGTH]  # (20, 132)
        feat = extract_features_from_sequence(seq)          # (20, 309)
        sequences.append(feat)
        if len(sequences) >= MAX_SEQUENCES_PER_VIDEO:
            break

    return np.array(sequences, dtype=np.float32) if sequences else np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Dataset Loading with Caching
# ─────────────────────────────────────────────────────────────────────────────

def process_split(split_name: str, pose_extractor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process all videos in train/ or val/ split.

    Returns X (N, SEQUENCE_LENGTH, FEATURE_DIM) and y (N,) label arrays.
    Caches results to CACHE_DIR to avoid re-processing on subsequent runs.
    """
    cache_x = CACHE_DIR / f"{split_name}_X.npy"
    cache_y = CACHE_DIR / f"{split_name}_y.npy"

    if cache_x.exists() and cache_y.exists():
        log.info(f"[{split_name}] Loading cached features from {CACHE_DIR}")
        X = np.load(cache_x)
        y = np.load(cache_y)
        log.info(f"[{split_name}] Loaded: X={X.shape}, y={y.shape} | "
                 f"Violent={int(y.sum())} NonViolent={int((y==0).sum())}")
        return X, y

    split_dir = DATASET_ROOT / split_name
    all_X, all_y = [], []

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    total_videos = sum(len(list(d.glob("*.avi"))) for d in class_dirs)
    processed = 0
    t0 = time.time()

    log.info(f"[{split_name}] Processing {total_videos} videos...")

    for class_dir in class_dirs:
        class_name = class_dir.name
        label = CLASS_MAP.get(class_name)
        if label is None:
            log.warning(f"Unknown class folder: {class_name} — skipping")
            continue

        videos = sorted(class_dir.glob("*.avi"))
        seq_count = 0

        for video_path in videos:
            seqs = video_to_feature_sequences(str(video_path), pose_extractor)
            if seqs.size > 0:
                all_X.append(seqs)
                all_y.extend([label] * len(seqs))
                seq_count += len(seqs)

            processed += 1
            if processed % 50 == 0:
                elapsed = time.time() - t0
                pct = processed / total_videos * 100
                eta = (elapsed / processed) * (total_videos - processed)
                log.info(f"  [{split_name}] {processed}/{total_videos} videos "
                         f"({pct:.0f}%) | ETA: {eta/60:.1f}m")

        log.info(f"  Class '{class_name}' (label={label}): "
                 f"{len(videos)} videos -> {seq_count} sequences")

    if not all_X:
        log.error(f"[{split_name}] No sequences extracted! Check pose detection.")
        return np.array([]), np.array([])

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y, dtype=np.int32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)

    log.info(f"[{split_name}] Done: X={X.shape}, y={y.shape} | "
             f"Violent={int(y.sum())} NonViolent={int((y==0).sum())}")
    log.info(f"[{split_name}] Cached to {CACHE_DIR}")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Model Definition
# ─────────────────────────────────────────────────────────────────────────────

def build_model(sequence_length: int, num_features: int, num_classes: int = 2) -> keras.Model:
    """
    Bidirectional LSTM with optional Attention and Temporal Convolutions.
    Matches the architecture in core/lstm_model.py but standalone.
    """
    inputs = keras.Input(shape=(sequence_length, num_features), name='input')
    x = inputs

    # Temporal Convolutional preprocessing
    if USE_TCN:
        x = keras.layers.Conv1D(64, 3, padding='causal', activation='relu',
                                dilation_rate=1)(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Conv1D(64, 3, padding='causal', activation='relu',
                                dilation_rate=2)(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)

    # Bidirectional LSTM layer 1
    lstm1 = keras.layers.LSTM(LSTM_UNITS, return_sequences=True)
    if USE_BIDIRECTIONAL:
        x = keras.layers.Bidirectional(lstm1, name='bilstm_1')(x)
    else:
        x = lstm1(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)

    # Bidirectional LSTM layer 2
    lstm2 = keras.layers.LSTM(LSTM_UNITS, return_sequences=True)
    if USE_BIDIRECTIONAL:
        x = keras.layers.Bidirectional(lstm2, name='bilstm_2')(x)
    else:
        x = lstm2(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)

    # Attention or GlobalAveragePooling
    if USE_ATTENTION:
        # Simple scaled dot-product style attention
        attention = keras.layers.Dense(1, activation='tanh')(x)
        attention = keras.layers.Flatten()(attention)
        attention = keras.layers.Activation('softmax')(attention)
        attention = keras.layers.RepeatVector(LSTM_UNITS * (2 if USE_BIDIRECTIONAL else 1))(attention)
        attention = keras.layers.Permute([2, 1])(attention)
        x = keras.layers.Multiply()([x, attention])
        x = keras.layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)
    else:
        x = keras.layers.GlobalAveragePooling1D()(x)

    # Dense head
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='violence_lstm')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Training
# ─────────────────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_val, y_val):
    """Train the model and return history + trained model."""
    num_features = X_train.shape[2]
    log.info(f"Building model: input=({SEQUENCE_LENGTH}, {num_features}), "
             f"LSTM_UNITS={LSTM_UNITS}, DROPOUT={DROPOUT_RATE}, "
             f"BiLSTM={USE_BIDIRECTIONAL}, Attention={USE_ATTENTION}, TCN={USE_TCN}")

    model = build_model(SEQUENCE_LENGTH, num_features)
    model.summary(print_fn=log.info)

    # Class weights to handle imbalance
    cw = None
    if CLASS_WEIGHT:
        n_total = len(y_train)
        n_violent = int(y_train.sum())
        n_nonviolent = n_total - n_violent
        cw = {
            0: n_total / (2.0 * n_nonviolent),
            1: n_total / (2.0 * n_violent),
        }
        log.info(f"Class weights: NonViolent={cw[0]:.3f}, Violent={cw[1]:.3f}")

    # Callbacks
    MODEL_SAVE.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(MODEL_SAVE).replace('.h5', '_best.h5')

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    log.info(f"Training: {len(X_train)} train sequences | {len(X_val)} val sequences")
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
# STEP 6: Evaluation & Reporting
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_val, y_val, history):
    """Print metrics and save accuracy/loss plots + confusion matrix."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            classification_report, confusion_matrix, ConfusionMatrixDisplay
        )
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False
        log.warning("sklearn/matplotlib not installed — skipping plots")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Final accuracy on val set ---
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    val_acc = np.mean(y_pred == y_val)
    log.info(f"\n{'='*50}")
    log.info(f"  FINAL VALIDATION ACCURACY: {val_acc*100:.2f}%")
    log.info(f"{'='*50}")

    if HAS_SKLEARN:
        # Classification report
        report = classification_report(y_val, y_pred, target_names=CLASS_NAMES)
        log.info(f"\nClassification Report:\n{report}")
        report_path = OUTPUT_DIR / "classification_report.txt"
        report_path.write_text(report)

        # Confusion matrix plot
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f'Confusion Matrix (Val Acc: {val_acc*100:.1f}%)')
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=120)
        plt.close(fig)
        log.info(f"Confusion matrix saved -> {OUTPUT_DIR / 'confusion_matrix.png'}")

    # Training curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Val')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Val')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)

        fig.suptitle(f'Violence Detection LSTM — Val Acc: {val_acc*100:.1f}%')
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'training_curves.png', dpi=120)
        plt.close(fig)
        log.info(f"Training curves saved -> {OUTPUT_DIR / 'training_curves.png'}")
    except Exception as e:
        log.warning(f"Could not save training curves: {e}")

    # Save summary JSON
    summary = {
        "val_accuracy": round(val_acc * 100, 2),
        "best_val_accuracy": round(max(history.history['val_accuracy']) * 100, 2),
        "epochs_trained": len(history.history['accuracy']),
        "config": {
            "SEQUENCE_LENGTH": SEQUENCE_LENGTH,
            "FEATURE_DIM": FEATURE_DIM,
            "FRAME_STRIDE": FRAME_STRIDE,
            "MAX_SEQUENCES_PER_VIDEO": MAX_SEQUENCES_PER_VIDEO,
            "LSTM_UNITS": LSTM_UNITS,
            "DROPOUT_RATE": DROPOUT_RATE,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "USE_BIDIRECTIONAL": USE_BIDIRECTIONAL,
            "USE_ATTENTION": USE_ATTENTION,
            "USE_TCN": USE_TCN,
        }
    }
    (OUTPUT_DIR / 'training_summary.json').write_text(json.dumps(summary, indent=2))
    log.info(f"Summary -> {OUTPUT_DIR / 'training_summary.json'}")

    return val_acc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Violence Detection — Dataset Training Pipeline")
    log.info("=" * 60)
    log.info(f"Dataset : {DATASET_ROOT}")
    log.info(f"Cache   : {CACHE_DIR}")
    log.info(f"Output  : {OUTPUT_DIR}")
    log.info(f"Model   : {MODEL_SAVE}")
    log.info(f"Config  : SEQ={SEQUENCE_LENGTH}, STRIDE={FRAME_STRIDE}, "
             f"MAX_SEQ/VIDEO={MAX_SEQUENCES_PER_VIDEO}")

    # Verify dataset exists
    for split in ['train', 'val']:
        d = DATASET_ROOT / split
        if not d.exists():
            log.error(f"Dataset split not found: {d}")
            log.error("Check DATASET_ROOT in config section of this script.")
            sys.exit(1)

    # Initialise MediaPipe Pose (shared across all videos)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MEDIAPIPE_COMPLEXITY,
        smooth_landmarks=True,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    log.info("\n[Phase 1] Extracting features from training videos...")
    X_train, y_train = process_split('train', pose)

    log.info("\n[Phase 2] Extracting features from validation videos...")
    X_val, y_val = process_split('val', pose)

    pose.close()

    if X_train.size == 0 or X_val.size == 0:
        log.error("Feature extraction produced no data. "
                  "Check video paths and MediaPipe installation.")
        sys.exit(1)

    log.info("\n[Phase 3] Training model...")
    model, history = train(X_train, y_train, X_val, y_val)

    log.info("\n[Phase 4] Evaluating...")
    val_acc = evaluate(model, X_val, y_val, history)

    log.info("\n" + "=" * 60)
    log.info(f"  TRAINING COMPLETE")
    log.info(f"  Validation Accuracy : {val_acc*100:.2f}%")
    log.info(f"  Model saved to      : {MODEL_SAVE}")
    log.info(f"  Results saved to    : {OUTPUT_DIR}")
    log.info("=" * 60)
    log.info("\nTuning tips (see CONFIG block at top of script):")
    log.info("  - Lower FRAME_STRIDE (1) for denser sampling — slower but more data")
    log.info("  - Increase MAX_SEQUENCES_PER_VIDEO (10-15) for more training samples")
    log.info("  - Increase LSTM_UNITS (128) for higher model capacity")
    log.info("  - Decrease LEARNING_RATE (0.0005) if val_loss is unstable")
    log.info("  - Increase EPOCHS (100) — early stopping protects from overfitting")


if __name__ == '__main__':
    main()
