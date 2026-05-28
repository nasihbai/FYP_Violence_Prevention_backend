"""
Per-person VideoMAE classifier using YOLO crop buffers.

Each YOLO-tracked person gets their own 16-frame crop buffer.
When the buffer fills, VideoMAE runs on that person's crop sequence
and produces an individual violence probability.

Inference is staggered across persons (round-robin by track_id slot)
so 3 people in frame = 3× fewer inferences per person, not 3× total load.

Known risk: the fine-tuned VideoMAE was trained on full-scene clips, not
tight person crops. Scores may be lower on crops than on full frames.
Test by comparing crop scores vs scene scores on a known fight clip.
"""

import logging
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Crops smaller than this in either dimension are too noisy to classify
# (distant persons, partial detections at frame edges)
_MIN_CROP_PX = 48


class PersonCropClassifier:
    """
    Runs VideoMAE inference on per-person crop sequences.

    Parameters
    ----------
    checkpoint : str
        HuggingFace checkpoint directory (the same fine-tuned model used
        by VideoMAESceneClassifier — reuses identical weights).
    clip_len : int
        Frames per clip (must match training, default 16).
    frame_size : int
        Resize each crop to this before feeding VideoMAE (224).
    threshold : float
        Violence probability above which is_violent() returns True.
    infer_every : int
        Run inference at most once every N frames per person.
        Staggered by track_id so persons don't all infer on the same frame.
    device : str | None
        "cuda" / "cpu". Auto-detected if None.
    """

    def __init__(
        self,
        checkpoint: str,
        clip_len: int = 16,
        frame_size: int = 224,
        threshold: float = 0.65,
        infer_every: int = 8,
        device: Optional[str] = None,
    ):
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        self._clip_len = clip_len
        self._frame_size = frame_size
        self._threshold = threshold
        self._infer_every = infer_every
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("PersonCropClassifier loading %s on %s", checkpoint, self._device)
        self._processor = VideoMAEImageProcessor.from_pretrained(checkpoint)
        self._model = VideoMAEForVideoClassification.from_pretrained(checkpoint)
        self._model.eval()
        self._model.to(self._device)

        num_labels = self._model.config.num_labels
        self._binary = num_labels == 2
        logger.info(
            "PersonCropClassifier ready — labels=%d  mode=%s",
            num_labels,
            "binary" if self._binary else "kinetics_fallback",
        )

        self._crop_buffers: Dict[int, deque] = {}  # track_id → RGB frame deque
        self._scores: Dict[int, float] = {}         # track_id → latest violence prob
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_crops(self, persons) -> Dict[int, float]:
        """
        Accept one frame's worth of PersonDetection objects.

        Appends each person's crop to their buffer and runs inference
        when their infer slot arrives.

        Returns a dict of {track_id: score} for persons that were
        scored this frame. All other persons retain their previous score
        (accessible via get_score / is_violent).
        """
        self._frame_count += 1
        updated: Dict[int, float] = {}

        # Clean up buffers for persons who left the frame
        active_ids = {p.id for p in persons}
        for gone_id in [pid for pid in list(self._crop_buffers) if pid not in active_ids]:
            del self._crop_buffers[gone_id]
            self._scores.pop(gone_id, None)

        n_persons = max(len(persons), 1)

        for person in persons:
            pid = person.id
            crop = person.crop

            if crop is None or crop.size == 0:
                continue
            h, w = crop.shape[:2]
            if h < _MIN_CROP_PX or w < _MIN_CROP_PX:
                continue  # too small — distant person or partial detection

            rgb = cv2.cvtColor(
                cv2.resize(crop, (self._frame_size, self._frame_size)),
                cv2.COLOR_BGR2RGB,
            )

            if pid not in self._crop_buffers:
                self._crop_buffers[pid] = deque(maxlen=self._clip_len)
            self._crop_buffers[pid].append(rgb)

            # Stagger: person A infers on frame 0, 8, 16…
            #          person B infers on frame 1, 9, 17… etc.
            slot = pid % n_persons
            if (
                len(self._crop_buffers[pid]) == self._clip_len
                and self._frame_count % self._infer_every == slot
            ):
                score = self._infer(list(self._crop_buffers[pid]))
                self._scores[pid] = score
                updated[pid] = score
                logger.debug("person %d  crop_score=%.3f", pid, score)

        return updated

    def get_score(self, track_id: int) -> float:
        """Return last violence probability for this track (0 if unseen)."""
        return self._scores.get(track_id, 0.0)

    def is_violent(self, track_id: int) -> bool:
        return self.get_score(track_id) >= self._threshold

    def all_scores(self) -> Dict[int, float]:
        return dict(self._scores)

    def reset(self) -> None:
        self._crop_buffers.clear()
        self._scores.clear()
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _infer(self, frames_rgb: list) -> float:
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in frames_rgb]
        inputs = self._processor(pil_frames, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        if self._binary:
            return torch.softmax(logits, dim=-1)[0, 1].item()
        # Kinetics fallback: label index 1 used as violence proxy
        return torch.sigmoid(logits[0, 1]).item()
