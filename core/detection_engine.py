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
        warmup_frames: int = 30
    ):
        """
        Initialize detector.

        Args:
            lstm_model_path: Path to LSTM model
            yolo_model: YOLO model name/path
            use_yolo: Whether to use YOLO for multi-person detection
            sequence_length: Number of frames for LSTM sequence
            violence_threshold: Threshold for violence detection
            smoothing_window: Window size for prediction smoothing
            warmup_frames: Frames to skip at start
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
        self.use_yolo = use_yolo

        # FPS calculation
        self._fps_history = deque(maxlen=30)
        self._last_frame_time = time.time()

        # Initialize components
        self._initialize_components(lstm_model_path, yolo_model)

    def _initialize_components(self, lstm_model_path: Optional[str], yolo_model: str):
        """Initialize detection components."""
        from .pose_extractor import PoseExtractor, LandmarkBuffer

        # Initialize pose extractor
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

        # Multi-person detection with YOLO
        if self.use_yolo and self.yolo_detector is not None:
            person_detections = self.yolo_detector.detect(frame, extract_crops=False)

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

        has_violence = any(d.is_violent for d in detections)

        return FrameResult(
            frame=frame,
            detections=detections,
            fps=fps,
            timestamp=current_time,
            has_violence=has_violence
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
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox

            # Choose color based on status
            if det.is_violent:
                color = (0, 0, 255)  # Red
                thickness = 3
            else:
                color = (0, 255, 0)  # Green
                thickness = 2

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw label
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

            if self.lstm_classifier:
                self.lstm_classifier.reset_history()

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
