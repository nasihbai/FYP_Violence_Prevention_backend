"""
Scene-Level Video-Clip Violence Classifier
==========================================
Replaces per-person pose-LSTM with a sliding-window scene classifier built on a
pretrained HuggingFace VideoMAE checkpoint (Kinetics-400). The classifier looks
at the whole frame — interaction between people, motion patterns, scene context
— not just one person's pose, which was the architectural ceiling of the LSTM
path. See plan: tranquil-meandering-salamander.md.

Public API:
- VideoClipClassifier (Protocol): predict(frames) -> (is_violent, confidence, top_label)
- VideoMAESceneClassifier: HF VideoMAE implementation
- ClipBuffer: thread-safe deque of recent frames
- KINETICS_VIOLENCE_LABELS: violence-adjacent Kinetics-400 class names
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Violence-adjacent Kinetics-400 action classes. Sums of these classes' softmax
# probabilities become the scene "violence probability". Class names match the
# Kinetics-400 label list used by MCG-NJU/videomae-base-finetuned-kinetics.
# Unknown labels are skipped at init with a warning — easy to tune without
# crashing.
KINETICS_VIOLENCE_LABELS: List[str] = [
    "punching person (boxing)",
    "punching bag",
    "wrestling",
    "headbutting",
    "slapping",
    "sword fighting",
    "side kick",
    "drop kicking",
    "capoeira",
    "arm wrestling",
]


@dataclass
class SceneResult:
    """Latest scene-level classification result."""
    is_violent: bool
    confidence: float
    label: str
    timestamp: float


class VideoClipClassifier(Protocol):
    """Interface every scene-level violence classifier must implement."""

    def predict(self, frames: np.ndarray) -> Tuple[bool, float, str]:
        """
        Classify a stack of frames.

        Args:
            frames: array shape (T, H, W, 3), uint8 BGR (OpenCV convention).

        Returns:
            (is_violent, violence_probability, top_label_for_debug)
        """
        ...


class ClipBuffer:
    """
    Thread-safe rolling window of the last N frames. Producers (the video
    loop) call add(); the prediction worker calls snapshot() to grab a copy.
    """

    def __init__(self, length: int = 16):
        if length < 2:
            raise ValueError(f"ClipBuffer length must be >=2, got {length}")
        self._length = length
        self._buf: deque = deque(maxlen=length)
        self._lock = threading.Lock()

    @property
    def length(self) -> int:
        return self._length

    def add(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buf.append(frame)

    def snapshot(self) -> Optional[np.ndarray]:
        """Return a (T, H, W, 3) array if the buffer is full, else None."""
        with self._lock:
            if len(self._buf) < self._length:
                return None
            # np.stack copies — safe to release the lock immediately after.
            return np.stack(list(self._buf), axis=0)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class VideoMAESceneClassifier:
    """
    Scene-level violence classifier using HuggingFace VideoMAE.

    Loads the pretrained model lazily (only when constructed; not at import
    time) so importing the module is cheap. Runs on CPU by default — for an
    FYP-scale CCTV demo this is enough since classification happens once every
    `clip_stride` frames in a worker thread.
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        device: str = "cpu",
        threshold: float = 0.6,
        violence_label_whitelist: Optional[List[str]] = None,
    ):
        # Heavy imports kept inside __init__ so that simply importing this
        # module doesn't drag in torch/transformers (used by tests etc.).
        import torch
        from transformers import (
            VideoMAEForVideoClassification,
            VideoMAEImageProcessor,
        )

        self._torch = torch
        self.model_name = model_name
        self.device = device
        self.threshold = threshold

        logger.info(f"Loading VideoMAE model: {model_name} (device={device})")
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

        # Map whitelist labels -> class indices using the model's own id2label.
        # Anything we can't find is dropped with a warning so a typo doesn't
        # silently kill the violence signal.
        id2label = self.model.config.id2label
        label2id = {v: k for k, v in id2label.items()}
        wanted = violence_label_whitelist or KINETICS_VIOLENCE_LABELS
        missing = [lbl for lbl in wanted if lbl not in label2id]
        if missing:
            logger.warning(
                f"VideoMAE: {len(missing)} whitelist labels not found in model "
                f"and were ignored: {missing}"
            )
        self.violence_class_ids: List[int] = [
            label2id[lbl] for lbl in wanted if lbl in label2id
        ]
        self.id2label = id2label
        logger.info(
            f"VideoMAE ready — {len(self.violence_class_ids)} violence-class IDs "
            f"({len(id2label)} total Kinetics classes); threshold={threshold}"
        )

        if not self.violence_class_ids:
            raise RuntimeError(
                "No violence labels matched the model's id2label — "
                "violence detection would always return 0. "
                "Check KINETICS_VIOLENCE_LABELS against the model's label list."
            )

    def predict(
        self, frames: np.ndarray
    ) -> Tuple[bool, float, str, List[Tuple[str, float]]]:
        """
        Returns (is_violent, violence_prob, top_label, top3_debug).
        top3_debug is the top-3 Kinetics (label, prob) pairs for diagnostics.
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"VideoMAE expects (T, H, W, 3) frames, got shape {frames.shape}"
            )

        # OpenCV gives BGR; the processor expects RGB.
        rgb_frames = frames[..., ::-1]
        # VideoMAEImageProcessor accepts a list of (H, W, 3) numpy arrays.
        frame_list = [f for f in rgb_frames]
        inputs = self.processor(frame_list, return_tensors="pt")
        # processor returns pixel_values: (1, T, 3, H, W) ready for the model.
        pixel_values = inputs["pixel_values"].to(self.device)

        with self._torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        probs = self._torch.softmax(outputs.logits, dim=-1)[0]
        violence_prob = float(
            probs[self.violence_class_ids].sum().item()
        )
        topk = self._torch.topk(probs, 3)
        top3: List[Tuple[str, float]] = [
            (self.id2label[int(i)], float(p))
            for p, i in zip(topk.values.tolist(), topk.indices.tolist())
        ]
        top_label = top3[0][0]
        is_violent = violence_prob > self.threshold
        return is_violent, violence_prob, top_label, top3


def _smoke_test() -> None:
    """
    Manual smoke test: build a random 16-frame clip, run the classifier,
    print the violence probability and top-5 Kinetics labels. Useful for
    confirming the install + label whitelist after pip install.

    Run with: python -m core.scene_classifier
    """
    import sys

    print("Loading VideoMAE — first run downloads ~340 MB into the HF cache.")
    clf = VideoMAESceneClassifier()
    print(f"violence_class_ids: {clf.violence_class_ids}")

    # Random uint8 frames stand in for a real CCTV clip.
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(16, 224, 224, 3), dtype=np.uint8)
    is_v, prob, top, top3 = clf.predict(frames)
    print(f"\n[random-noise input] violence_prob={prob:.4f}  top_label={top!r}  "
          f"is_violent={is_v}")
    print(f"top3: {top3}")

    # Top-5 raw labels for sanity.
    import torch
    rgb = frames[..., ::-1]
    inputs = clf.processor([f for f in rgb], return_tensors="pt")
    with torch.no_grad():
        logits = clf.model(pixel_values=inputs["pixel_values"]).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top5 = torch.topk(probs, 5)
    print("Top-5:")
    for p, i in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {p:.4f}  {clf.id2label[i]}")

    sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _smoke_test()
