"""
YOLOv8-pose wrapper.

Returns PersonPose objects with keypoints in full-frame (global) coordinates.
Uses model.track() so track IDs persist across frames.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PersonPose:
    """One detected person with COCO-17 pose keypoints in frame-pixel coords."""
    track_id:  int
    keypoints: np.ndarray          # (17, 3): x, y, conf  — full-frame pixels
    bbox:      Tuple[int, int, int, int]  # x1, y1, x2, y2


class YOLOPoseEstimator:
    """
    Thin wrapper around YOLOv8-pose.
    Uses GPU automatically when CUDA is available (ultralytics handles device selection).
    """

    def __init__(self, model_path: str = "yolov8n-pose.pt", conf: float = 0.35):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._conf  = conf
        logger.info("YOLOPoseEstimator ready: %s", model_path)

    def estimate(self, frame: np.ndarray) -> List[PersonPose]:
        """
        Run pose estimation with object tracking.

        Args:
            frame: BGR frame (numpy array)

        Returns:
            List of PersonPose, one per detected person.
        """
        results = self._model.track(
            frame, persist=True, verbose=False,
            conf=self._conf, iou=0.5,
        )

        persons: List[PersonPose] = []
        if not results or results[0].keypoints is None:
            return persons

        r       = results[0]
        kps_all = r.keypoints.data.cpu().numpy()   # (N, 17, 3)
        boxes   = r.boxes
        xyxy    = boxes.xyxy.cpu().numpy()          # (N, 4)

        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.arange(len(kps_all), dtype=int)

        for i, kps in enumerate(kps_all):
            tid = int(track_ids[i]) if i < len(track_ids) else i
            x1, y1, x2, y2 = xyxy[i]
            persons.append(PersonPose(
                track_id  = tid,
                keypoints = kps.astype(np.float32),
                bbox      = (int(x1), int(y1), int(x2), int(y2)),
            ))

        return persons
