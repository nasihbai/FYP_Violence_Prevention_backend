"""
VideoMAE-based scene classifier for violence detection.

Two operating modes selected automatically at construction time:
  - binary_finetuned: loads a 2-class checkpoint (violent / neutral) produced
    by training/finetune_videomae.py.  Direct prob output — no whitelist needed.
  - kinetics_fallback: loads the pretrained Kinetics-400 checkpoint and sums
    probabilities over a manually curated violence label set.  Lower accuracy
    but works out-of-the-box before fine-tuning is complete.

Device selection: CUDA if available, else CPU.  Override with the
VIDEOMAE_DEVICE env var ("cpu" / "cuda" / "cuda:1" etc.).
"""

import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Kinetics-400 labels that carry violence signal (fallback mode only)
_KINETICS_VIOLENCE_LABELS = {
    "punching person (boxing)",
    "punching bag",
    "kicking soccer ball",
    "kicking field goal",
    "side kick",
    "drop kicking",
    "wrestling",
    "headbutting",
    "throwing axe",
    "slapping",
    "rock scissors paper",
    "sword fighting",
    "martial arts",
    "hitting baseball",
    "shooting goal (soccer)",
    "tackling",
    "javelin throw",
    "hammer throw",
}


class ClipBuffer:
    """
    Rolling frame buffer that yields fixed-length clips on demand.

    Inference fires at most once every `infer_every` frames so a single
    fast movement (one frame) cannot trigger back-to-back scores.
    """

    def __init__(self, clip_len: int = 16, stride: int = 2, infer_every: int = 8):
        # Keep stride * clip_len raw frames so we can down-sample to clip_len
        self._buf: deque = deque(maxlen=clip_len * stride)
        self._clip_len = clip_len
        self._stride = stride
        self._infer_every = infer_every  # only fire inference every N frames
        self._frames_since_infer = 0

    def push(self, frame_rgb: np.ndarray) -> bool:
        """Append one RGB frame. Returns True when a new clip is ready for inference."""
        self._buf.append(frame_rgb)
        if len(self._buf) < self._buf.maxlen:
            return False  # still warming up
        self._frames_since_infer += 1
        if self._frames_since_infer >= self._infer_every:
            self._frames_since_infer = 0
            return True
        return False

    def get_clip(self) -> list:
        """Return clip_len uniformly strided frames from the buffer."""
        frames = list(self._buf)
        return frames[:: self._stride]

    def reset(self) -> None:
        self._buf.clear()
        self._frames_since_infer = 0


class VideoMAESceneClassifier:
    """
    VideoMAE inference wrapper.

    Parameters
    ----------
    checkpoint : str | None
        Path to a fine-tuned HuggingFace checkpoint directory, OR a HuggingFace
        model hub ID.  Falls back to VideoMAEConfig hierarchy if None.
    clip_len : int
        Number of frames per clip fed to VideoMAE (must match training).
    clip_stride : int
        Sample every N raw frames to build one clip.
    threshold : float
        Violence probability above which is_violent() returns True.
    device : str | None
        "cuda" / "cpu" / "cuda:1".  Auto-detected if None.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        clip_len: int = 16,
        clip_stride: int = 2,
        threshold: float = 0.65,
        device: Optional[str] = None,
        infer_every: int = 8,
        smooth_window: int = 3,
    ):
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        env_device = os.environ.get("VIDEOMAE_DEVICE")
        if device:
            self._device = device
        elif env_device:
            self._device = env_device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if self._device.startswith("cuda") and torch.cuda.is_available():
            logger.info("VideoMAE using GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.info("VideoMAE using CPU — inference will be slower")

        ckpt = checkpoint or _resolve_checkpoint()
        logger.info("Loading VideoMAE checkpoint: %s", ckpt)

        self._processor = VideoMAEImageProcessor.from_pretrained(ckpt)
        self._model = VideoMAEForVideoClassification.from_pretrained(ckpt)
        self._model.eval()
        self._model.to(self._device)

        # Determine head mode from number of output labels
        num_labels = self._model.config.num_labels
        if num_labels == 2:
            self._mode = "binary_finetuned"
            logger.info("Head mode: binary_finetuned (2-class violent/neutral)")
        else:
            self._mode = "kinetics_fallback"
            # Build index set for violence-related Kinetics labels
            id2label = self._model.config.id2label
            self._violence_ids = {
                idx for idx, name in id2label.items() if name in _KINETICS_VIOLENCE_LABELS
            }
            logger.info(
                "Head mode: kinetics_fallback (%d violence label indices matched)",
                len(self._violence_ids),
            )

        self._threshold = threshold
        self._clip_len = clip_len
        self._buffer = ClipBuffer(clip_len=clip_len, stride=clip_stride, infer_every=infer_every)
        self._score_history: deque = deque(maxlen=smooth_window)
        self._last_score: float = 0.0  # smoothed score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_violence_prob(self) -> float:
        """Most recent violence probability (0–1)."""
        return self._last_score

    @property
    def device(self) -> str:
        return self._device

    def push_frame(self, frame_bgr: np.ndarray) -> Optional[float]:
        """
        Push one BGR frame (as returned by cv2.VideoCapture.read).

        Returns the smoothed violence probability when a new clip is scored,
        otherwise returns None. Inference fires every `infer_every` frames;
        the returned value is the rolling average of the last `smooth_window`
        clip scores — a single fast movement cannot spike it alone.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ready = self._buffer.push(frame_rgb)
        if not ready:
            return None
        raw = self._infer(self._buffer.get_clip())
        self._score_history.append(raw)
        smoothed = float(sum(self._score_history) / len(self._score_history))
        self._last_score = smoothed
        return smoothed

    def is_violent(self) -> bool:
        return self._last_score >= self._threshold

    def reset(self) -> None:
        self._buffer.reset()
        self._score_history.clear()
        self._last_score = 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _infer(self, frames_rgb: list) -> float:
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in frames_rgb]
        inputs = self._processor(pil_frames, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, num_labels)

        if self._mode == "binary_finetuned":
            probs = torch.softmax(logits, dim=-1)
            # label index 1 = "violent" as set during fine-tuning
            score = probs[0, 1].item()
        else:
            # Kinetics fallback: sum softmax probs over violence label indices
            probs = torch.softmax(logits, dim=-1)[0]
            score = float(sum(probs[i].item() for i in self._violence_ids))
            score = min(score, 1.0)

        return score


def _resolve_checkpoint() -> str:
    """Return the checkpoint path/ID to load, with fallback chain."""
    # 1. Env var
    env = os.environ.get("VIDEOMAE_CHECKPOINT")
    if env and Path(env).exists():
        return env

    # 2. Fine-tuned checkpoint saved by finetune_videomae.py
    try:
        from config.settings import VideoMAEConfig
        ckpt_path = Path(VideoMAEConfig.CHECKPOINT_PATH)
        if ckpt_path.exists() and any(ckpt_path.iterdir()):
            return str(ckpt_path)
        base = VideoMAEConfig.BASE_MODEL
    except Exception:
        base = "MCG-NJU/videomae-base-finetuned-kinetics"

    # 3. Pretrained Kinetics hub model
    logger.info("Fine-tuned checkpoint not found — using base Kinetics model (fallback mode)")
    return base


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    logger.info("Opening source: %s", source)

    clf = VideoMAESceneClassifier()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Cannot open source %s", source)
        sys.exit(1)

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            prob = clf.push_frame(frame)
            if prob is not None:
                tag = "VIOLENT" if clf.is_violent() else "neutral"
                logger.info("frame=%d  violence_prob=%.3f  [%s]", frame_count, prob, tag)
    finally:
        cap.release()
        logger.info("Done. Processed %d frames.", frame_count)
