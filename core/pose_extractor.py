"""
Pose Extraction Module
======================
MediaPipe-based pose landmark extraction with thread-safe buffering.
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class PoseLandmarks:
    """Data class for storing pose landmarks."""
    landmarks: np.ndarray  # Shape: (33*4,) = 132 features
    timestamp: float
    person_id: int = 0
    confidence: float = 1.0


class PoseExtractor:
    """
    MediaPipe-based pose landmark extractor.

    Features:
    - Thread-safe landmark extraction
    - Configurable detection confidence
    - Support for multiple detection modes
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize pose extractor.

        Args:
            static_image_mode: Process images independently
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Apply smoothing to landmarks
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self._lock = threading.Lock()

    def extract(self, frame: np.ndarray, person_id: int = 0) -> Optional[PoseLandmarks]:
        """
        Extract pose landmarks from a frame.

        Args:
            frame: Input frame (BGR format)
            person_id: ID of the person

        Returns:
            PoseLandmarks object or None if no pose detected
        """
        with self._lock:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks is None:
                return None

            # Extract landmarks as flat array
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

            return PoseLandmarks(
                landmarks=np.array(landmarks, dtype=np.float32),
                timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                person_id=person_id,
                confidence=np.mean([lm.visibility for lm in results.pose_landmarks.landmark])
            )

    def extract_from_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        person_id: int = 0
    ) -> Optional[PoseLandmarks]:
        """
        Extract pose from a cropped region.
        Landmarks are kept in crop-relative coordinates to match the
        feature engineering pipeline used during training.

        Args:
            frame: Full frame
            bbox: Bounding box (x1, y1, x2, y2)
            person_id: Person ID

        Returns:
            PoseLandmarks or None
        """
        x1, y1, x2, y2 = bbox

        # Add padding — matches detect_violence.py PAD=20
        h, w = frame.shape[:2]
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
            return None

        # Extract pose — landmarks stay in crop-relative coordinates
        # (same as detect_violence.py, which the model was trained on)
        return self.extract(crop, person_id)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: PoseLandmarks,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw pose landmarks on frame.

        Args:
            frame: Input frame
            landmarks: Pose landmarks
            draw_connections: Whether to draw skeleton connections

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Convert flat array back to landmark format
        points = []
        for i in range(0, len(landmarks.landmarks), 4):
            x = int(landmarks.landmarks[i] * w)
            y = int(landmarks.landmarks[i + 1] * h)
            visibility = landmarks.landmarks[i + 3]
            points.append((x, y, visibility))

        # Draw points
        for x, y, vis in points:
            if vis > 0.5:
                cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

        # Draw connections
        if draw_connections:
            connections = self.mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    start_point = points[start_idx]
                    end_point = points[end_idx]

                    if start_point[2] > 0.5 and end_point[2] > 0.5:
                        cv2.line(
                            annotated,
                            (start_point[0], start_point[1]),
                            (end_point[0], end_point[1]),
                            (0, 255, 0),
                            2
                        )

        return annotated

    def close(self):
        """Release resources."""
        self.pose.close()


class LandmarkBuffer:
    """
    Thread-safe buffer for storing landmark sequences.

    Features:
    - Per-person buffering
    - Automatic sequence extraction
    - Thread-safe operations
    """

    def __init__(self, sequence_length: int = 20, max_persons: int = 10):
        """
        Initialize buffer.

        Args:
            sequence_length: Number of frames per sequence
            max_persons: Maximum number of persons to track
        """
        self.sequence_length = sequence_length
        self.max_persons = max_persons

        self._buffers: Dict[int, deque] = {}
        self._lock = threading.Lock()

    def add(self, landmarks: PoseLandmarks) -> Optional[np.ndarray]:
        """
        Add landmarks to buffer and return the current sequence if ready.

        Uses a rolling deque — the buffer is NEVER cleared after a prediction.
        Every new frame shifts the window by one, so predictions happen on
        every frame once the buffer is full (matches detect_violence.py behaviour).

        Args:
            landmarks: Pose landmarks to add

        Returns:
            Sequence array (shape: sequence_length × 132) once buffer is full,
            None while still warming up.
        """
        with self._lock:
            person_id = landmarks.person_id

            if person_id not in self._buffers:
                if len(self._buffers) >= self.max_persons:
                    oldest = min(self._buffers.keys())
                    del self._buffers[oldest]
                self._buffers[person_id] = deque(maxlen=self.sequence_length)

            self._buffers[person_id].append(landmarks.landmarks)

            # Return rolling window once the buffer is full
            if len(self._buffers[person_id]) == self.sequence_length:
                return np.array(list(self._buffers[person_id]))

            return None

    def get_buffer_status(self, person_id: int) -> int:
        """Get current buffer fill level for a person."""
        with self._lock:
            if person_id in self._buffers:
                return len(self._buffers[person_id])
            return 0

    def clear(self, person_id: Optional[int] = None):
        """
        Clear buffer.

        Args:
            person_id: Specific person to clear (None for all)
        """
        with self._lock:
            if person_id is not None:
                self._buffers.pop(person_id, None)
            else:
                self._buffers.clear()

    def get_all_persons(self) -> List[int]:
        """Get list of all tracked person IDs."""
        with self._lock:
            return list(self._buffers.keys())


class MultiPersonPoseExtractor:
    """
    Pose extractor for multiple persons using YOLO + MediaPipe.
    """

    def __init__(
        self,
        use_yolo: bool = True,
        yolo_model: str = "yolov8n.pt",
        yolo_confidence: float = 0.5,
        sequence_length: int = 20
    ):
        """
        Initialize multi-person pose extractor.

        Args:
            use_yolo: Whether to use YOLO for person detection
            yolo_model: YOLO model path
            yolo_confidence: YOLO confidence threshold
            sequence_length: Sequence length for buffering
        """
        self.pose_extractor = PoseExtractor()
        self.landmark_buffer = LandmarkBuffer(sequence_length=sequence_length)
        self.use_yolo = use_yolo
        self.yolo_detector = None

        if use_yolo:
            try:
                from .yolo_detector import YOLODetector
                self.yolo_detector = YOLODetector(
                    model_path=yolo_model,
                    confidence=yolo_confidence,
                    enable_tracking=True
                )
            except Exception as e:
                logger.warning(f"YOLO not available: {e}. Using single-person mode.")
                self.use_yolo = False

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[List[PoseLandmarks], Dict[int, Optional[np.ndarray]]]:
        """
        Process frame and extract poses for all detected persons.

        Args:
            frame: Input frame

        Returns:
            Tuple of (landmarks_list, sequences_dict)
            - landmarks_list: List of PoseLandmarks for each person
            - sequences_dict: Dict mapping person_id to sequence (if ready)
        """
        landmarks_list = []
        sequences = {}

        if self.use_yolo and self.yolo_detector is not None:
            # Detect all persons with YOLO
            detections = self.yolo_detector.detect(frame, extract_crops=False)

            for det in detections:
                # Extract pose for each person
                pose = self.pose_extractor.extract_from_crop(
                    frame, det.bbox, det.id
                )

                if pose is not None:
                    landmarks_list.append(pose)

                    # Add to buffer
                    sequence = self.landmark_buffer.add(pose)
                    sequences[det.id] = sequence
        else:
            # Single person mode
            pose = self.pose_extractor.extract(frame, person_id=0)

            if pose is not None:
                landmarks_list.append(pose)
                sequence = self.landmark_buffer.add(pose)
                sequences[0] = sequence

        return landmarks_list, sequences

    def draw_all_poses(
        self,
        frame: np.ndarray,
        landmarks_list: List[PoseLandmarks]
    ) -> np.ndarray:
        """Draw all detected poses on frame."""
        annotated = frame.copy()

        for landmarks in landmarks_list:
            annotated = self.pose_extractor.draw_landmarks(annotated, landmarks)

        return annotated

    def close(self):
        """Release resources."""
        self.pose_extractor.close()
