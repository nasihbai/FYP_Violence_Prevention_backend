"""
Thread-Safe Violence Detection Engine
======================================
Main detection engine combining YOLO, pose extraction, and LSTM classification
with proper thread synchronization.
"""

import numpy as np
import cv2
import threading
import queue
import time
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# COCO-17 skeleton pairs for drawing pose overlays
_COCO_SKELETON = [
    (0, 5), (0, 6),                          # nose → shoulders
    (5, 6),                                   # shoulder bridge
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso sides
    (11, 12),                                 # hip bridge
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]
_SKELETON_COLORS = [
    (0, 255, 128), (0, 200, 255), (255, 200, 0),
    (255, 80, 180), (180, 80, 255), (255, 130, 0),
]


@dataclass
class DetectionResult:
    """Data class for storing detection results."""
    person_id: int
    bbox: Tuple[int, int, int, int]
    is_violent: bool
    confidence: float
    class_name: str
    timestamp: float


@dataclass
class FrameResult:
    """Data class for frame processing results."""
    frame: np.ndarray
    detections: List[DetectionResult]
    fps: float
    timestamp: float
    has_violence: bool = False
    scene_violence_prob: float = 0.0  # VideoMAE scene-level probability


class ThreadSafeDetector:
    """
    Thread-safe violence detection system.

    Features:
    - Asynchronous prediction using thread pool
    - Thread-safe state management
    - Prediction smoothing
    - Alert debouncing
    """

    def __init__(
        self,
        lstm_model_path: Optional[str] = None,
        yolo_model: str = "yolov8n.pt",
        use_yolo: bool = True,
        sequence_length: int = 20,
        violence_threshold: float = 0.6,
        smoothing_window: int = 5,
        warmup_frames: int = 30,
        use_scene_classifier: bool = True,
        use_person_classifier: bool = False,
        use_pose_interaction: bool = True,
        interaction_model_path: Optional[str] = None,
    ):
        """
        Initialize detector.

        Args:
            lstm_model_path:        Path to legacy per-person LSTM model.
            yolo_model:             YOLO detection model name/path.
            use_yolo:               Use YOLO for multi-person detection.
            sequence_length:        Frames for legacy LSTM sequence.
            violence_threshold:     Confidence threshold for violence.
            smoothing_window:       Prediction smoothing window size.
            warmup_frames:          Frames to skip at startup.
            use_scene_classifier:   Enable VideoMAE scene classifier.
            use_person_classifier:  Enable per-person crop classifier.
            use_pose_interaction:   Enable the new YOLOv8-pose + interaction
                                    pipeline (M1/M3).  When True this becomes
                                    the primary violence signal.
            interaction_model_path: Path to trained interaction LSTM (.h5).
                                    When None the heuristic classifier is used
                                    (Milestone 1).
        """
        self.sequence_length = sequence_length
        self.violence_threshold = violence_threshold
        self.smoothing_window = smoothing_window
        self.warmup_frames = warmup_frames

        # Thread synchronization
        self._lock = threading.RLock()
        self._prediction_queue = queue.Queue(maxsize=10)
        self._result_cache: Dict[int, DetectionResult] = {}
        self._prediction_history: Dict[int, deque] = {}

        # State
        self._frame_count = 0
        self._running = False
        self._workers: List[threading.Thread] = []
        self._pose_cache: Dict[int, any] = {}   # person_id → last PoseLandmarks

        # Components
        self.pose_extractor = None
        self.lstm_classifier = None
        self.yolo_detector = None
        self.scene_classifier = None
        self.person_classifier = None
        self.use_yolo = use_yolo
        self._scene_violence_prob: float = 0.0

        # Interaction-aware pipeline (M1 / M3)
        self.use_pose_interaction   = use_pose_interaction
        self.pose_estimator         = None   # YOLOPoseEstimator
        self.heuristic_classifier   = None   # HeuristicFightClassifier  (M1)
        self.interaction_lstm       = None   # Keras model               (M3)
        self._feature_buffer: deque = deque(maxlen=sequence_length)
        self._prev_persons          = None   # previous frame PersonPose list
        self._last_persons          = []     # current frame PersonPose list (for draw)
        self._interaction_prob: float = 0.0
        self._per_fight:  Dict[int, bool] = {}
        self._per_fall:   Dict[int, bool] = {}

        # FPS calculation
        self._fps_history = deque(maxlen=30)
        self._last_frame_time = time.time()

        # Initialize components
        self._initialize_components(
            lstm_model_path, yolo_model,
            use_scene_classifier, use_person_classifier,
            use_pose_interaction, interaction_model_path,
        )

    def _initialize_components(
        self, lstm_model_path: Optional[str], yolo_model: str,
        use_scene_classifier: bool, use_person_classifier: bool,
        use_pose_interaction: bool = True,
        interaction_model_path: Optional[str] = None,
    ):
        """Initialize detection components."""
        # ── Interaction-aware pipeline (M1 / M3) ──────────────────────────
        if use_pose_interaction:
            try:
                from .pose_estimator import YOLOPoseEstimator
                from config.settings import InteractionConfig as _IC
                self.pose_estimator = YOLOPoseEstimator(
                    model_path=_IC.POSE_MODEL, conf=0.35
                )
                logger.info("YOLOPoseEstimator ready (%s)", _IC.POSE_MODEL)
            except Exception as exc:
                logger.warning("YOLOPoseEstimator init failed (%s); falling back to legacy pipeline", exc)
                self.use_pose_interaction = False

        if use_pose_interaction and self.pose_estimator is not None:
            # Try trained interaction LSTM (M3); fall back to heuristic (M1) if absent.
            loaded = False
            model_to_try = interaction_model_path
            if model_to_try is None:
                try:
                    from config.settings import InteractionConfig as _IC
                    model_to_try = str(_IC.INTERACTION_MODEL_PATH)
                except Exception:
                    pass
            if model_to_try and Path(model_to_try).exists():
                try:
                    import tensorflow as tf
                    self.interaction_lstm = tf.keras.models.load_model(model_to_try)
                    logger.info("Interaction LSTM loaded from %s", model_to_try)
                    loaded = True
                except Exception as exc:
                    logger.warning("Could not load interaction LSTM (%s); using heuristic", exc)

            if not loaded:
                from .heuristic_classifier import HeuristicFightClassifier
                self.heuristic_classifier = HeuristicFightClassifier()
                logger.info("Using heuristic fight classifier (Milestone 1 mode)")

        from .pose_extractor import PoseExtractor, LandmarkBuffer

        # Initialize pose extractor (legacy pipeline)
        self.pose_extractor = PoseExtractor()
        self.landmark_buffer = LandmarkBuffer(sequence_length=self.sequence_length)

        # Initialize YOLO
        if self.use_yolo:
            try:
                from .yolo_detector import YOLODetector
                self.yolo_detector = YOLODetector(
                    model_path=yolo_model,
                    enable_tracking=True
                )
                logger.info("YOLO detector initialized")
            except Exception as e:
                logger.warning(f"YOLO initialization failed: {e}")
                self.use_yolo = False

        # Initialize LSTM classifier
        if lstm_model_path and Path(lstm_model_path).exists():
            try:
                from .lstm_model import ViolenceClassifier
                self.lstm_classifier = ViolenceClassifier(
                    model_path=lstm_model_path,
                    sequence_length=self.sequence_length,
                    smoothing_window=self.smoothing_window,
                    threshold=self.violence_threshold
                )
                logger.info(f"LSTM classifier loaded from {lstm_model_path}")
            except Exception as e:
                logger.error(f"Failed to load LSTM model: {e}")

        # Initialize VideoMAE scene classifier
        if use_scene_classifier:
            try:
                from .scene_classifier import VideoMAESceneClassifier
                from config.settings import VideoMAEConfig
                self.scene_classifier = VideoMAESceneClassifier(
                    clip_len=VideoMAEConfig.CLIP_LEN,
                    clip_stride=VideoMAEConfig.CLIP_STRIDE,
                    threshold=VideoMAEConfig.VIOLENCE_THRESHOLD,
                    smooth_window=VideoMAEConfig.SMOOTH_WINDOW,
                )
                logger.info("VideoMAE scene classifier loaded (mode: %s)", self.scene_classifier._mode)
            except Exception as e:
                logger.warning("VideoMAE scene classifier unavailable: %s", e)

        # Initialize per-person crop classifier (experimental)
        if use_person_classifier:
            try:
                from .person_classifier import PersonCropClassifier
                from config.settings import VideoMAEConfig
                ckpt = str(VideoMAEConfig.CHECKPOINT_PATH)
                self.person_classifier = PersonCropClassifier(
                    checkpoint=ckpt,
                    clip_len=VideoMAEConfig.CLIP_LEN,
                    threshold=VideoMAEConfig.VIOLENCE_THRESHOLD,
                )
                logger.info("Per-person crop classifier loaded")
            except Exception as e:
                logger.warning("Per-person crop classifier unavailable: %s", e)

    def start(self, num_workers: int = 2):
        """Start prediction worker threads."""
        self._running = True

        for i in range(num_workers):
            worker = threading.Thread(
                target=self._prediction_worker,
                name=f"PredictionWorker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)

        logger.info(f"Started {num_workers} prediction workers")

    def stop(self):
        """Stop prediction workers."""
        self._running = False

        # Clear queue
        while not self._prediction_queue.empty():
            try:
                self._prediction_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for workers
        for worker in self._workers:
            worker.join(timeout=1.0)

        self._workers.clear()
        logger.info("Detection engine stopped")

    def _prediction_worker(self):
        """Worker thread for processing predictions."""
        while self._running:
            try:
                item = self._prediction_queue.get(timeout=0.1)
                if item is None:
                    continue

                person_id, sequence = item

                if self.lstm_classifier is not None:
                    # Apply feature engineering when the model expects more features
                    # than the raw landmark count (132 → 309).
                    model_features = self.lstm_classifier.model.input_shape[-1]
                    if model_features != sequence.shape[-1]:
                        from .feature_engineering import extract_features_from_sequence, DEFAULT_CONFIG
                        sequence = extract_features_from_sequence(sequence, DEFAULT_CONFIG)

                    is_violent, confidence = self.lstm_classifier.is_violent(
                        sequence, person_id
                    )

                    with self._lock:
                        if person_id not in self._result_cache:
                            self._result_cache[person_id] = DetectionResult(
                                person_id=person_id,
                                bbox=(0, 0, 0, 0),
                                is_violent=is_violent,
                                confidence=confidence,
                                class_name='violent' if is_violent else 'neutral',
                                timestamp=time.time()
                            )
                        else:
                            self._result_cache[person_id].is_violent = is_violent
                            self._result_cache[person_id].confidence = confidence
                            self._result_cache[person_id].class_name = 'violent' if is_violent else 'neutral'
                            self._result_cache[person_id].timestamp = time.time()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Prediction worker error: {e}")

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single frame for violence detection.

        Args:
            frame: Input frame (BGR)

        Returns:
            FrameResult with detection information
        """
        current_time = time.time()
        self._frame_count += 1

        # Calculate FPS
        if self._last_frame_time > 0:
            frame_time = current_time - self._last_frame_time
            if frame_time > 0:
                self._fps_history.append(1.0 / frame_time)
        self._last_frame_time = current_time

        fps = np.mean(self._fps_history) if self._fps_history else 0

        detections = []

        # Skip warmup frames
        if self._frame_count <= self.warmup_frames:
            return FrameResult(
                frame=frame,
                detections=[],
                fps=fps,
                timestamp=current_time
            )

        # ── Interaction-aware pipeline (M1 heuristic / M3 LSTM) ──────────
        if self.use_pose_interaction and self.pose_estimator is not None:
            return self._process_interaction(frame, fps, current_time)

        # Multi-person detection with YOLO
        if self.use_yolo and self.yolo_detector is not None:
            need_crops = self.person_classifier is not None
            person_detections = self.yolo_detector.detect(frame, extract_crops=need_crops)

            for det in person_detections:
                # Extract pose for this person
                pose = self.pose_extractor.extract_from_crop(
                    frame, det.bbox, det.id
                )

                if pose is not None:
                    # Cache latest pose for skeleton drawing
                    with self._lock:
                        self._pose_cache[det.id] = pose

                    # Add to buffer
                    sequence = self.landmark_buffer.add(pose)

                    # Queue for prediction if sequence ready
                    if sequence is not None:
                        try:
                            self._prediction_queue.put_nowait((det.id, sequence))
                        except queue.Full:
                            pass

                # Get cached result
                with self._lock:
                    if det.id in self._result_cache:
                        cached = self._result_cache[det.id]
                        result = DetectionResult(
                            person_id=det.id,
                            bbox=det.bbox,
                            is_violent=cached.is_violent,
                            confidence=cached.confidence,
                            class_name=cached.class_name,
                            timestamp=current_time
                        )
                    else:
                        result = DetectionResult(
                            person_id=det.id,
                            bbox=det.bbox,
                            is_violent=False,
                            confidence=0.0,
                            class_name='neutral',
                            timestamp=current_time
                        )
                    detections.append(result)

        else:
            # Single person mode
            pose = self.pose_extractor.extract(frame, person_id=0)

            if pose is not None:
                with self._lock:
                    self._pose_cache[0] = pose
                sequence = self.landmark_buffer.add(pose)

                if sequence is not None:
                    try:
                        self._prediction_queue.put_nowait((0, sequence))
                    except queue.Full:
                        pass

            with self._lock:
                if 0 in self._result_cache:
                    cached = self._result_cache[0]
                    # Create bbox from pose if available
                    h, w = frame.shape[:2]
                    detections.append(DetectionResult(
                        person_id=0,
                        bbox=(0, 0, w, h),
                        is_violent=cached.is_violent,
                        confidence=cached.confidence,
                        class_name=cached.class_name,
                        timestamp=current_time
                    ))

        # Person count gate — computed before any violence flags are applied.
        # Violence requires interaction between people; a lone person cannot fight.
        try:
            from config.settings import VideoMAEConfig as _VMC
            _min_persons = _VMC.MIN_PERSONS_FOR_ALERT
        except Exception:
            _min_persons = 2
        enough_persons = len(detections) >= _min_persons

        # Per-person crop classification.
        # Always push crops to keep the temporal buffer warm, but only apply
        # violence flags to DetectionResults when enough persons are present.
        if self.person_classifier is not None and self.use_yolo:
            self.person_classifier.push_crops(person_detections)
            if enough_persons:
                for det in detections:
                    person_score = self.person_classifier.get_score(det.person_id)
                    if person_score > 0:
                        det.confidence = person_score
                        det.is_violent = self.person_classifier.is_violent(det.person_id)
                        det.class_name = 'violent' if det.is_violent else 'neutral'

        # Scene-level classification via VideoMAE (runs on every frame, fires when clip ready)
        if self.scene_classifier is not None:
            prob = self.scene_classifier.push_frame(frame)
            if prob is not None:
                self._scene_violence_prob = prob

        logger.info("persons=%d  enough=%s  scene_prob=%.3f", len(detections), enough_persons, self._scene_violence_prob)

        has_violence = enough_persons and (
            any(d.is_violent for d in detections)
            or self._scene_violence_prob >= self.violence_threshold
        )

        return FrameResult(
            frame=frame,
            detections=detections,
            fps=fps,
            timestamp=current_time,
            has_violence=has_violence,
            scene_violence_prob=self._scene_violence_prob,
        )

    def _process_interaction(
        self,
        frame:        np.ndarray,
        fps:          float,
        current_time: float,
    ) -> FrameResult:
        """
        Interaction-aware processing path (Milestone 1 / 3).

        Uses YOLOv8-pose for full-frame multi-person pose estimation, then
        routes through either the heuristic classifier (M1) or the trained
        interaction LSTM (M3) to produce a scene-level violence decision.
        """
        from .interaction_features import frame_feature_vector, FEATURE_DIM_FULL
        from config.settings import InteractionConfig as _IC

        h, w = frame.shape[:2]

        # ── Pose estimation ────────────────────────────────────────────────
        persons = self.pose_estimator.estimate(frame)

        with self._lock:
            self._last_persons = persons

        # Build DetectionResult list from PersonPose objects
        detections: List[DetectionResult] = []
        for p in persons:
            detections.append(DetectionResult(
                person_id  = p.track_id,
                bbox       = p.bbox,
                is_violent = False,
                confidence = 0.0,
                class_name = 'neutral',
                timestamp  = current_time,
            ))

        # Minimum-person gate
        try:
            from config.settings import VideoMAEConfig as _VMC
            min_persons = _VMC.MIN_PERSONS_FOR_ALERT
        except Exception:
            min_persons = 2
        enough = len(persons) >= min_persons

        # ── Heuristic (M1) ─────────────────────────────────────────────────
        has_violence = False
        if self.heuristic_classifier is not None:
            any_fight, any_fall, scene_score, per_fight, per_fall = \
                self.heuristic_classifier.classify(persons)

            with self._lock:
                self._interaction_prob = scene_score
                self._per_fight        = per_fight
                self._per_fall         = per_fall

            for det in detections:
                tid = det.person_id
                if per_fight.get(tid) or per_fall.get(tid):
                    det.is_violent = True
                    det.confidence = scene_score
                    det.class_name = 'fight' if per_fight.get(tid) else 'fallen'

            has_violence = enough and (any_fight or any_fall)

        # ── Trained interaction LSTM (M3) ──────────────────────────────────
        elif self.interaction_lstm is not None:
            kps_list   = [p.keypoints for p in persons]
            bbox_list  = [p.bbox for p in persons]
            prev_kps   = [p.keypoints for p in self._prev_persons] \
                         if self._prev_persons else None

            feat = frame_feature_vector(
                kps_list, bbox_list,
                prev_kps=prev_kps,
                frame_h=h, frame_w=w,
                include_interaction=True,
            )
            self._feature_buffer.append(feat)

            if len(self._feature_buffer) == _IC.SEQUENCE_LENGTH:
                seq   = np.stack(list(self._feature_buffer))[np.newaxis]  # (1, T, F)
                probs = self.interaction_lstm.predict(seq, verbose=0)[0]
                prob  = float(probs[1])  # index 1 = violent class

                with self._lock:
                    self._interaction_prob = prob

                if enough and prob >= self.violence_threshold:
                    has_violence = True
                    for det in detections:
                        det.is_violent = True
                        det.confidence = prob
                        det.class_name = 'violent'

            with self._lock:
                self._per_fight = {p.track_id: has_violence for p in persons}
                self._per_fall  = {}

        self._prev_persons = persons

        return FrameResult(
            frame              = frame,
            detections         = detections,
            fps                = fps,
            timestamp          = current_time,
            has_violence       = has_violence,
            scene_violence_prob= self._interaction_prob,
        )

    def draw_results(
        self,
        frame: np.ndarray,
        result: FrameResult,
        show_skeleton: bool = False,
        show_fps: bool = True
    ) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: Input frame
            result: Detection results
            show_skeleton: Whether to show pose skeleton
            show_fps: Whether to show FPS

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # ── Interaction-mode skeleton overlay ─────────────────────────────
        if self.use_pose_interaction:
            with self._lock:
                persons_snap = list(self._last_persons)
                per_fight    = dict(self._per_fight)
                per_fall     = dict(self._per_fall)

            for p in persons_snap:
                color    = _SKELETON_COLORS[p.track_id % len(_SKELETON_COLORS)]
                is_fight = per_fight.get(p.track_id, False)
                is_fall  = per_fall.get(p.track_id, False)
                kps      = p.keypoints   # (17, 3)

                # Skeleton lines
                for (a, b) in _COCO_SKELETON:
                    if kps[a, 2] >= 0.30 and kps[b, 2] >= 0.30:
                        cv2.line(annotated,
                                 (int(kps[a, 0]), int(kps[a, 1])),
                                 (int(kps[b, 0]), int(kps[b, 1])),
                                 color, 2, cv2.LINE_AA)
                # Keypoint dots
                for i in range(17):
                    if kps[i, 2] >= 0.30:
                        cx, cy = int(kps[i, 0]), int(kps[i, 1])
                        cv2.circle(annotated, (cx, cy), 4, color,        -1, cv2.LINE_AA)
                        cv2.circle(annotated, (cx, cy), 4, (255,255,255), 1, cv2.LINE_AA)

                # Bbox + label
                x1, y1, x2, y2 = p.bbox
                if is_fight:
                    box_color = (0, 0, 220)
                    label     = f"ID:{p.track_id} FIGHT"
                elif is_fall:
                    box_color = (0, 100, 220)
                    label     = f"ID:{p.track_id} FALLEN"
                else:
                    box_color = (30, 190, 30)
                    label     = f"ID:{p.track_id}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1-th-6), (x1+tw+4, y1), box_color, -1)
                cv2.putText(annotated, label, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # Threat bar
            prob = self._interaction_prob
            if result.has_violence:
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 44), (0, 0, 180), -1)
                cv2.putText(annotated, "!!!  VIOLENCE DETECTED",
                            (14, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 1, cv2.LINE_AA)
            bar_w = 220
            bar_x = annotated.shape[1] - bar_w - 10
            bar_y = annotated.shape[0] - 26
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x+bar_w, bar_y+16), (60,60,60), -1)
            fill = int(bar_w * prob)
            fill_col = (0,0,210) if result.has_violence else (0,180,0)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x+fill, bar_y+16), fill_col, -1)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x+bar_w, bar_y+16), (200,200,200), 1)
            cv2.putText(annotated, f"Threat {prob:.0%}", (bar_x+4, bar_y+12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.40, (255,255,255), 1, cv2.LINE_AA)

        # Draw FPS
        if show_fps:
            cv2.putText(
                annotated,
                f"FPS: {result.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Draw frame count
        cv2.putText(
            annotated,
            f"Frame: {self._frame_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # Draw skeleton for each tracked person
        if show_skeleton and self.pose_extractor is not None:
            with self._lock:
                pose_snapshot = dict(self._pose_cache)
                result_snapshot = dict(self._result_cache)

            for pid, pose_lm in pose_snapshot.items():
                is_violent = result_snapshot.get(pid, DetectionResult(pid, (0,0,0,0), False, 0, 'neutral', 0)).is_violent
                dot_color = (0, 0, 220) if is_violent else (0, 220, 0)
                annotated = self.pose_extractor.draw_landmarks(annotated, pose_lm)

        # Draw detections
        scene_violent = result.scene_violence_prob >= self.violence_threshold
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox

            # Scene classifier overrides box colour when it detects violence —
            # we can't identify *which* person is fighting at scene level, so
            # all boxes go red to signal "someone in frame is fighting".
            if det.is_violent or scene_violent:
                color = (0, 0, 255)  # Red
                thickness = 3
            else:
                color = (0, 255, 0)  # Green
                thickness = 2

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label: prefer scene score when it's the active signal
            if scene_violent and not det.is_violent:
                label = f"ID:{det.person_id} fight ({result.scene_violence_prob:.2f})"
            else:
                label = f"ID:{det.person_id} {det.class_name} ({det.confidence:.2f})"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                color,
                -1
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Person count overlay — shows what the gate sees
        cv2.putText(annotated, f"Persons: {len(result.detections)}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        # Scene classifier probability overlay (always visible when classifier is active)
        if self.scene_classifier is not None:
            prob_text = f"Scene: {result.scene_violence_prob:.2f}"
            color = (0, 0, 220) if result.scene_violence_prob >= self.violence_threshold else (180, 180, 180)
            cv2.putText(annotated, prob_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Violence warning overlay
        if result.has_violence:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 60), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)

            cv2.putText(
                annotated,
                "WARNING: VIOLENCE DETECTED",
                (annotated.shape[1] // 2 - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        return annotated

    def reset(self):
        """Reset detector state."""
        with self._lock:
            self._frame_count = 0
            self._result_cache.clear()
            self._prediction_history.clear()
            self._pose_cache.clear()
            self.landmark_buffer.clear()
            self._feature_buffer.clear()
            self._prev_persons    = None
            self._last_persons    = []
            self._interaction_prob = 0.0
            self._per_fight = {}
            self._per_fall  = {}

            if self.lstm_classifier:
                self.lstm_classifier.reset_history()
            if self.heuristic_classifier:
                self.heuristic_classifier.reset()

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        with self._lock:
            return {
                'frame_count': self._frame_count,
                'active_persons': len(self._result_cache),
                'avg_fps': np.mean(self._fps_history) if self._fps_history else 0,
                'queue_size': self._prediction_queue.qsize()
            }


class VideoProcessor:
    """
    Video processing pipeline for violence detection.

    Supports:
    - Local video files
    - Webcam input
    - RTSP streams
    - HTTP streams
    """

    def __init__(
        self,
        source,
        detector: ThreadSafeDetector,
        on_violence_detected: Optional[Callable[[FrameResult], None]] = None
    ):
        """
        Initialize video processor.

        Args:
            source: Video source (int for webcam, str for file/URL)
            detector: ThreadSafeDetector instance
            on_violence_detected: Callback for violence detection
        """
        self.source = source
        self.detector = detector
        self.on_violence_detected = on_violence_detected

        self._cap = None
        self._running = False
        self._source_type = self._determine_source_type(source)

    def _determine_source_type(self, source) -> str:
        """Determine the type of video source."""
        if isinstance(source, int):
            return 'camera'
        elif isinstance(source, str):
            if source.startswith('rtsp://'):
                return 'rtsp'
            elif source.startswith(('http://', 'https://')):
                return 'http'
            else:
                return 'file'
        return 'unknown'

    def _open_source(self) -> bool:
        """Open video source."""
        try:
            if self._source_type == 'rtsp':
                import os
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self._cap = cv2.VideoCapture(self.source)

            if not self._cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False

            logger.info(f"Opened video source: {self.source} ({self._source_type})")
            return True

        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            return False

    def run(self, display: bool = True, window_name: str = "Violence Detection"):
        """
        Run the video processing loop.

        Args:
            display: Whether to display video
            window_name: Window name for display
        """
        if not self._open_source():
            return

        self._running = True
        self.detector.start()

        try:
            while self._running:
                ret, frame = self._cap.read()

                if not ret:
                    if self._source_type == 'file':
                        # Loop video file
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # Try reconnect for streams
                        logger.warning("Lost connection, attempting reconnect...")
                        time.sleep(2)
                        if not self._open_source():
                            break
                        continue

                # Process frame
                result = self.detector.process_frame(frame)

                # Violence callback
                if result.has_violence and self.on_violence_detected:
                    self.on_violence_detected(result)

                # Display
                if display:
                    annotated = self.detector.draw_results(frame, result)
                    cv2.imshow(window_name, annotated)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            self.stop()

    def stop(self):
        """Stop video processing."""
        self._running = False
        self.detector.stop()

        if self._cap is not None:
            self._cap.release()

        cv2.destroyAllWindows()
        logger.info("Video processor stopped")
