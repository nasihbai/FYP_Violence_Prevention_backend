"""
PyTorch Dataset for the multi-class violence video dataset.

Expected folder layout (already present in this project):
    root/
      train/
        Fight/         *.avi  → label 1 (violent)
        HockeyFight/   *.avi  → label 1 (violent)
        MovieFights/   *.avi  → label 1 (violent)
        NonFight/      *.avi  → label 0 (neutral)
      val/
        (same structure)

Each __getitem__ decodes one video, uniformly samples `clip_len` frames,
resizes to `frame_size x frame_size`, converts BGR→RGB, and either:
  - returns a (pixel_values, label) tensor pair ready for VideoMAE, or
  - returns a raw (T, C, H, W) float32 tensor if no processor is given.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Classes that count as violent (everything else = neutral)
_VIOLENT_CLASSES = {"Fight", "HockeyFight", "MovieFights"}


class ViolenceClipDataset(Dataset):
    """
    Parameters
    ----------
    root : str | Path
        Dataset root containing train/ and val/ subdirectories.
    split : str
        "train" or "val".
    clip_len : int
        Number of frames to sample per clip (must match model expectation).
    frame_size : int
        Height and width to resize each frame to.
    processor : VideoMAEImageProcessor | None
        If provided, frames are passed through the processor and the returned
        tensor is shaped (clip_len, C, H, W) as expected by VideoMAE.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 16,
        frame_size: int = 224,
        processor=None,
    ):
        self._clip_len = clip_len
        self._frame_size = frame_size
        self._processor = processor

        split_dir = Path(root) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self._samples: list[tuple[Path, int]] = []
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = 1 if cls_dir.name in _VIOLENT_CLASSES else 0
            for ext in ("*.avi", "*.mp4", "*.mkv"):
                for vid in sorted(cls_dir.glob(ext)):
                    self._samples.append((vid, label))

        if not self._samples:
            raise ValueError(f"No video files found under {split_dir}")

        n_violent = sum(1 for _, l in self._samples if l == 1)
        n_neutral = len(self._samples) - n_violent
        print(
            f"[{split}] {len(self._samples)} clips — "
            f"violent: {n_violent}  neutral: {n_neutral}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        path, label = self._samples[idx]
        frames = self._decode_clip(path)

        if self._processor is not None:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            out = self._processor(pil_frames, return_tensors="pt")
            # processor returns shape (1, clip_len, C, H, W) — squeeze batch dim
            return out["pixel_values"].squeeze(0), label

        # Raw tensor: (clip_len, C, H, W), float32 in [0, 1]
        arr = np.stack(frames)  # (T, H, W, C)
        tensor = torch.from_numpy(arr.transpose(0, 3, 1, 2)).float() / 255.0
        return tensor, label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_clip(self, path: Path) -> list:
        """Uniformly sample clip_len frames from a video file."""
        cap = cv2.VideoCapture(str(path))
        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        # Uniform indices across the video duration
        indices = set(
            np.linspace(0, total_frames - 1, self._clip_len, dtype=int).tolist()
        )

        frames: dict[int, np.ndarray] = {}
        i = 0
        while cap.isOpened() and len(frames) < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                resized = cv2.resize(frame, (self._frame_size, self._frame_size))
                frames[i] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()

        # Build ordered list; pad with last frame if any indices were missed
        result = []
        for idx in sorted(indices):
            if idx in frames:
                result.append(frames[idx])
            elif result:
                result.append(result[-1])

        # Guarantee exactly clip_len frames
        blank = np.zeros((self._frame_size, self._frame_size, 3), dtype=np.uint8)
        while len(result) < self._clip_len:
            result.append(result[-1] if result else blank)

        return result[: self._clip_len]
