"""
Violence Detection — Real-time Inference
=========================================
Two modes:

  Single-stage (default)  — MediaPipe on full frame, one person
  2-stage      (--yolo)   — YOLO crops each person, MediaPipe on crop, per-person tracking

Usage:
  # Webcam — single stage
  python detect_violence.py

  # Webcam — 2-stage YOLO+Pose (use with model from train_violence_yolo_pose.py)
  python detect_violence.py --yolo

  # Video file
  python detect_violence.py --source path/to/video.mp4
  python detect_violence.py --source path/to/video.mp4 --yolo

  # RTSP stream
  python detect_violence.py --source rtsp://user:pass@192.168.1.100:554/stream --yolo

  # Adjust threshold
  python detect_violence.py --yolo --threshold 0.6

Controls:
  q  — quit
  r  — reset all landmark buffers
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import argparse
import time
import sys
import warnings
from collections import deque, defaultdict
from pathlib import Path

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Single-stage model (train_violence_dataset.py)
MODEL_PATH_SINGLE = Path(r"C:\Users\nasih\Documents\FYP_Violence_Prevention\models\violence_lstm_dataset.h5")
# 2-stage model (train_violence_yolo_pose.py)
MODEL_PATH_YOLO   = Path(r"C:\Users\nasih\Documents\FYP_Violence_Prevention\models\violence_lstm_yolo_pose.h5")

SEQUENCE_LENGTH      = 20     # Must match what was used in training
VIOLENCE_THRESHOLD   = 0.55   # Confidence threshold (0.5–0.7 range)
SMOOTHING_WINDOW     = 3      # Average over N predictions (lower = faster response)
WARMUP_FRAMES        = 0      # Start predicting immediately (was 30)
PREDICT_EVERY_N      = 2      # Run LSTM every N frames (1=every frame, 3=every 3rd)
SHOW_CONFIDENCE      = True   # Show confidence % on screen
SHOW_SKELETON        = True   # Draw pose skeleton

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (must exactly mirror train_violence_dataset.py)
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


def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def extract_features(landmarks_seq: np.ndarray) -> np.ndarray:
    """
    Convert (SEQUENCE_LENGTH, 132) raw landmarks -> (SEQUENCE_LENGTH, 309) features.
    Identical to the function in train_violence_dataset.py.
    """
    seq_len = len(landmarks_seq)
    lm3d   = landmarks_seq.reshape(seq_len, 33, 4)
    coords = lm3d[:, :, :3]

    lsh = LANDMARK_INDICES['left_shoulder']
    rsh = LANDMARK_INDICES['right_shoulder']
    lhp = LANDMARK_INDICES['left_hip']
    rhp = LANDMARK_INDICES['right_hip']

    # 1. Normalized coordinates
    normalized = np.zeros_like(coords)
    for i in range(seq_len):
        hip_center = (coords[i, lhp] + coords[i, rhp]) / 2.0
        centered   = coords[i] - hip_center
        sw         = np.linalg.norm(coords[i, lsh] - coords[i, rsh])
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
        for j, (a_n, b_n, c_n) in enumerate(JOINT_ANGLES):
            a = coords[i, LANDMARK_INDICES[a_n]]
            b = coords[i, LANDMARK_INDICES[b_n]]
            c = coords[i, LANDMARK_INDICES[c_n]]
            angles[i, j] = compute_angle(a, b, c) / 180.0

    # 5. Violence distances
    distances = np.zeros((seq_len, len(VIOLENCE_DISTANCES)))
    for i in range(seq_len):
        for j, (a_n, b_n) in enumerate(VIOLENCE_DISTANCES):
            a = coords[i, LANDMARK_INDICES[a_n]]
            b = coords[i, LANDMARK_INDICES[b_n]]
            distances[i, j] = np.linalg.norm(a - b)

    features = np.concatenate([
        normalized.reshape(seq_len, -1),
        velocity.reshape(seq_len, -1),
        acceleration.reshape(seq_len, -1),
        angles,
        distances,
    ], axis=1)

    return features.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_label(frame, label: str, confidence: float, is_violent: bool):
    """Overlay prediction label and confidence bar on frame."""
    h, w = frame.shape[:2]
    color = (0, 0, 220) if is_violent else (0, 200, 0)

    # Background banner
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)

    # Label text
    text = f"{label.upper()}  {confidence*100:.1f}%" if SHOW_CONFIDENCE else label.upper()
    cv2.putText(frame, text, (10, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_w = int(confidence * (w - 20))
    cv2.rectangle(frame, (10, 45), (10 + bar_w, 52), color, -1)
    cv2.rectangle(frame, (10, 45), (w - 10, 52), (80, 80, 80), 1)


def draw_buffer_progress(frame, filled: int, total: int):
    """Show how many frames have been buffered."""
    h, w = frame.shape[:2]
    text = f"Buffer: {filled}/{total}"
    cv2.putText(frame, text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def open_capture(source):
    """Open a video capture handle, with RTSP tuning if needed."""
    if isinstance(source, str) and source.startswith("rtsp://"):
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)
    return cap


def handle_eof(cap, source, buffers_to_clear):
    """Handle end-of-stream: reconnect for streams, loop for files."""
    is_stream = isinstance(source, int) or (
        isinstance(source, str) and source.startswith("rtsp://")
    )
    cap.release()
    if is_stream:
        time.sleep(1)
        cap = open_capture(source)
    else:
        cap = open_capture(source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for b in buffers_to_clear:
            b.clear()
    return cap


# ── MODE 1: Single-stage (full-frame MediaPipe) ───────────────────────────────

def run_single(source, model):
    """Detect violence using full-frame MediaPipe (single person)."""
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    cap         = open_capture(source)
    lm_buffer   = deque(maxlen=SEQUENCE_LENGTH)
    pred_hist   = deque(maxlen=SMOOTHING_WINDOW)
    frame_count = 0
    label, confidence, is_violent = "Warming up...", 0.0, False
    fps_timer, fps_display = time.time(), 0.0

    print("[INFO] Mode: Single-stage (full-frame pose)")
    print("[INFO] Press 'q' to quit | 'r' to reset buffer")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap = handle_eof(cap, source, [lm_buffer, pred_hist])
            continue

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_display = frame_count / (now - fps_timer + 1e-6)
            frame_count = 0
            fps_timer = now

        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.pose_landmarks:
            if SHOW_SKELETON:
                mp_draw.draw_landmarks(
                    frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2),
                )
            lm = result.pose_landmarks.landmark
            arr = np.array([[l.x, l.y, l.z, l.visibility] for l in lm],
                           dtype=np.float32).flatten()
            lm_buffer.append(arr)

        if len(lm_buffer) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY_N == 0:
            feat = extract_features(np.array(lm_buffer))   # (20, 309)
            probs = model.predict(np.expand_dims(feat, 0), verbose=0)[0]
            pred_hist.append(float(probs[1]))
            smoothed = float(np.mean(pred_hist))
            confidence = smoothed
            is_violent = smoothed >= VIOLENCE_THRESHOLD
            label = "VIOLENT" if is_violent else "Normal"

        draw_label(frame, label, confidence, is_violent)
        draw_buffer_progress(frame, len(lm_buffer), SEQUENCE_LENGTH)
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (frame.shape[1] - 110, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("Violence Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            lm_buffer.clear(); pred_hist.clear()
            label, confidence, is_violent = "Warming up...", 0.0, False

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


# ── MODE 2: 2-stage (YOLO crops → MediaPipe per person) ──────────────────────

def run_yolo(source, model):
    """Detect violence using YOLO person detection + per-crop MediaPipe (multi-person)."""
    from ultralytics import YOLO as UltralyticsYOLO

    yolo = UltralyticsYOLO("yolov8n.pt")
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    cap = open_capture(source)

    # Per-person buffers  {person_slot: deque}
    lm_buffers  = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    pred_hists  = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
    person_labels = {}   # slot -> (label, confidence, is_violent)

    frame_count = 0
    fps_timer, fps_display = time.time(), 0.0
    PAD = 20  # crop padding pixels

    print("[INFO] Mode: 2-stage (YOLO + per-person pose)")
    print("[INFO] Press 'q' to quit | 'r' to reset all buffers")

    while True:
        ret, frame = cap.read()
        if not ret:
            buffers = list(lm_buffers.values()) + list(pred_hists.values())
            cap = handle_eof(cap, source, buffers)
            continue

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_display = frame_count / (now - fps_timer + 1e-6)
            frame_count = 0
            fps_timer = now

        h_frame, w_frame = frame.shape[:2]

        # ── Stage 1: YOLO person detection ───────────────────────────────────
        try:
            results = yolo(frame, classes=[0], conf=0.4, verbose=False)
        except Exception:
            results = []

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                area = (x2 - x1) * (y2 - y1)
                detections.append((x1, y1, x2, y2, area))

        # Sort by area (largest first), keep top 2, then sort left-to-right
        detections.sort(key=lambda d: d[4], reverse=True)
        detections = detections[:2]
        detections.sort(key=lambda d: d[0])

        # ── Stage 2: MediaPipe on each crop ──────────────────────────────────
        for slot, (x1, y1, x2, y2, _) in enumerate(detections):
            cx1 = max(0, x1 - PAD); cy1 = max(0, y1 - PAD)
            cx2 = min(w_frame, x2 + PAD); cy2 = min(h_frame, y2 + PAD)
            crop = frame[cy1:cy2, cx1:cx2]

            if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
                continue

            pose_result = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            if pose_result.pose_landmarks:
                if SHOW_SKELETON:
                    # Different colour per person slot so they are visually distinct
                    dot_colors  = [(0, 255, 120), (0, 180, 255)]   # green, orange-blue
                    line_colors = [(255, 255, 0), (255, 100, 255)]  # yellow, pink
                    dot_c  = dot_colors[slot % len(dot_colors)]
                    line_c = line_colors[slot % len(line_colors)]
                    mp_draw.draw_landmarks(
                        crop, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=dot_c,  thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=line_c, thickness=2),
                    )
                    frame[cy1:cy2, cx1:cx2] = crop

                lm = pose_result.pose_landmarks.landmark
                arr = np.array([[l.x, l.y, l.z, l.visibility] for l in lm],
                               dtype=np.float32).flatten()
                lm_buffers[slot].append(arr)

            # ── Predict for this person ───────────────────────────────────
            buf = lm_buffers[slot]
            if len(buf) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY_N == 0:
                feat  = extract_features(np.array(buf))
                probs = model.predict(np.expand_dims(feat, 0), verbose=0)[0]
                pred_hists[slot].append(float(probs[1]))
                smoothed   = float(np.mean(pred_hists[slot]))
                is_violent = smoothed >= VIOLENCE_THRESHOLD
                label_str  = "VIOLENT" if is_violent else "Normal"
                person_labels[slot] = (label_str, smoothed, is_violent)

            # ── Draw bounding box per person ──────────────────────────────
            lbl, conf, viol = person_labels.get(slot, ("...", 0.0, False))
            box_color = (0, 0, 220) if viol else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            tag = f"ID{slot} {lbl} {conf*100:.0f}%"
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + len(tag) * 10, y1), box_color, -1)
            cv2.putText(frame, tag, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # ── Global banner: show VIOLENT if ANY person is violent ──────────
        any_violent  = any(v[2] for v in person_labels.values())
        max_conf     = max((v[1] for v in person_labels.values()), default=0.0)
        global_label = "VIOLENT" if any_violent else ("Normal" if person_labels else "Waiting...")
        draw_label(frame, global_label, max_conf, any_violent)

        min_buf = min((len(b) for b in lm_buffers.values()), default=0)
        draw_buffer_progress(frame, min_buf, SEQUENCE_LENGTH)
        cv2.putText(frame, f"FPS: {fps_display:.1f} | Persons: {len(detections)}",
                    (frame.shape[1] - 230, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("Violence Detection [YOLO+Pose]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            lm_buffers.clear(); pred_hists.clear(); person_labels.clear()
            print("[INFO] All buffers reset")

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Violence Detection')
    parser.add_argument('--source', default=0,
                        help='Video source: 0=webcam, path/to/video.mp4, rtsp://...')
    parser.add_argument('--yolo', action='store_true',
                        help='Use 2-stage YOLO+Pose pipeline (multi-person)')
    parser.add_argument('--threshold', type=float, default=VIOLENCE_THRESHOLD,
                        help=f'Violence confidence threshold (default: {VIOLENCE_THRESHOLD})')
    args = parser.parse_args()

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    VIOLENCE_THRESHOLD = args.threshold

    # Choose model path based on mode
    model_path = MODEL_PATH_YOLO if args.yolo else MODEL_PATH_SINGLE

    if not model_path.exists():
        train_cmd = "train_violence_yolo_pose.py" if args.yolo else "train_violence_dataset.py"
        print(f"[ERROR] Model not found: {model_path}")
        print(f"        Train first:  python {train_cmd}")
        sys.exit(1)

    print(f"[INFO] Loading model: {model_path}")
    # custom_objects needed because Lambda layers in the model reference 'keras'
    # by name — this makes it available in the lambda's deserialization scope
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        custom_objects={'keras': tf.keras}
    )
    print(f"[INFO] Model input shape: {model.input_shape}")
    print(f"[INFO] Source: {source}")

    if args.yolo:
        run_yolo(source, model)
    else:
        run_single(source, model)
