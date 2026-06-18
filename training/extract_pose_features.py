"""
RLVS Feature Extraction — GPU-accelerated.

For each clip, runs YOLOv8-pose on strided frames and calls
interaction_features.frame_feature_vector() to produce two parallel
feature matrices per clip:

  isolated_feats : (T, FEATURE_DIM_ISOLATED)   — Ablation B input
  full_feats     : (T, FEATURE_DIM_FULL)        — Ablation C input

Produces sliding-window sequences of length SEQUENCE_LENGTH from each
clip, then saves per-split .npy caches under RLVS_CACHE_DIR.

Caches:
  rlvs/{split}_X_isolated.npy  — (N_sequences, T, 10)
  rlvs/{split}_X_full.npy      — (N_sequences, T, 19)
  rlvs/{split}_y.npy           — (N_sequences,)     int32

Run once; subsequent calls load the cache and skip re-extraction.

Usage:
  python -m training.extract_pose_features
  python -m training.extract_pose_features --subset 400   # fast sanity check
  python -m training.extract_pose_features --stride 1     # higher quality
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Suppress TF startup noise (keep WARNINGS)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from config.settings import InteractionConfig as IC
from core.interaction_features import (
    frame_feature_vector,
    FEATURE_DIM_ISOLATED,
    FEATURE_DIM_FULL,
)
from training.rlvs_dataset import load_rlvs_splits, ClipList


# ─────────────────────────────────────────────────────────────────────────────
# CLIP → SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────

def _clip_to_sequences(
    video_path:   Path,
    yolo_model,
    stride:       int,
    seq_len:      int,
    max_seq:      int = 6,
    kp_conf:      float = 0.35,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Process one video clip and return lists of (T, F_iso) and (T, F_full)
    sequence arrays.  Each array is one sliding-window chunk.

    Returns:
        (iso_seqs, full_seqs) — same length; may be empty if no pose found.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Cannot open: %s", video_path)
        return [], []

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640

    iso_frames:  List[np.ndarray] = []   # per-frame isolated vectors
    full_frames: List[np.ndarray] = []   # per-frame full vectors
    prev_kps     = None
    frame_idx    = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            results = yolo_model.predict(frame, verbose=False, conf=kp_conf, iou=0.5)
            kps_list  = []
            bbox_list = []

            if results and results[0].keypoints is not None:
                r   = results[0]
                kps_data = r.keypoints.data.cpu().numpy()
                xyxy     = r.boxes.xyxy.cpu().numpy()
                for i, kps in enumerate(kps_data):
                    x1, y1, x2, y2 = xyxy[i]
                    kps_list.append(kps.astype(np.float32))
                    bbox_list.append((int(x1), int(y1), int(x2), int(y2)))

            iso_vec  = frame_feature_vector(
                kps_list, bbox_list,
                prev_kps=prev_kps,
                frame_h=frame_h, frame_w=frame_w,
                include_interaction=False,
            )
            full_vec = frame_feature_vector(
                kps_list, bbox_list,
                prev_kps=prev_kps,
                frame_h=frame_h, frame_w=frame_w,
                include_interaction=True,
            )

            iso_frames.append(iso_vec)
            full_frames.append(full_vec)
            prev_kps = kps_list if kps_list else None

            frame_idx += 1
    finally:
        cap.release()

    # ── Sliding window into fixed-length sequences ────────────────────────
    n = len(iso_frames)
    if n < seq_len:
        return [], []

    iso_arr  = np.stack(iso_frames)   # (N_frames, 10)
    full_arr = np.stack(full_frames)  # (N_frames, 19)
    slide    = max(1, seq_len // 2)
    iso_seqs, full_seqs = [], []

    for start in range(0, n - seq_len + 1, slide):
        iso_seqs.append(iso_arr[start:start+seq_len])
        full_seqs.append(full_arr[start:start+seq_len])
        if len(iso_seqs) >= max_seq:
            break

    return iso_seqs, full_seqs


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_split(
    split_name:  str,
    clips:       ClipList,
    yolo_model,
    stride:      int,
    seq_len:     int,
    cache_dir:   Path,
    subset:      Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features for all clips in a split, with .npy caching.

    Returns:
        X_iso  (N, T, 10)
        X_full (N, T, 19)
        y      (N,)  int32
    """
    suffix = f"_sub{subset}" if subset else ""
    cache_iso  = cache_dir / f"{split_name}_X_isolated{suffix}.npy"
    cache_full = cache_dir / f"{split_name}_X_full{suffix}.npy"
    cache_y    = cache_dir / f"{split_name}_y{suffix}.npy"

    if cache_iso.exists() and cache_full.exists() and cache_y.exists():
        log.info("[%s] Loading from cache: %s", split_name, cache_dir)
        X_iso  = np.load(cache_iso)
        X_full = np.load(cache_full)
        y      = np.load(cache_y)
        log.info("[%s] Loaded: iso=%s  full=%s  labels=%s",
                 split_name, X_iso.shape, X_full.shape, y.shape)
        return X_iso, X_full, y

    if subset:
        clips = clips[:subset]

    all_iso, all_full, all_y = [], [], []
    n_clips  = len(clips)
    no_pose  = 0
    t0       = time.time()

    log.info("[%s] Extracting from %d clips (stride=%d) …", split_name, n_clips, stride)

    for idx, (video_path, label) in enumerate(clips):
        iso_seqs, full_seqs = _clip_to_sequences(
            video_path, yolo_model, stride, seq_len
        )
        if iso_seqs:
            for iso, full in zip(iso_seqs, full_seqs):
                all_iso.append(iso)
                all_full.append(full)
                all_y.append(label)
        else:
            no_pose += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == n_clips:
            elapsed = time.time() - t0
            eta     = (elapsed / (idx+1)) * (n_clips - idx - 1)
            log.info("  [%s] %d/%d  seqs=%d  no_pose=%d  ETA %.1fm",
                     split_name, idx+1, n_clips,
                     len(all_iso), no_pose, eta/60)

    if not all_iso:
        log.error("[%s] Zero sequences extracted — check dataset path and YOLO conf.", split_name)
        empty = np.zeros((0, seq_len, FEATURE_DIM_ISOLATED), dtype=np.float32)
        np.save(cache_iso,  empty)
        np.save(cache_full, np.zeros((0, seq_len, FEATURE_DIM_FULL), dtype=np.float32))
        np.save(cache_y,    np.zeros(0, dtype=np.int32))
        return empty, empty[..., :FEATURE_DIM_FULL], np.zeros(0, dtype=np.int32)

    X_iso  = np.array(all_iso,  dtype=np.float32)
    X_full = np.array(all_full, dtype=np.float32)
    y      = np.array(all_y,    dtype=np.int32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_iso,  X_iso)
    np.save(cache_full, X_full)
    np.save(cache_y,    y)

    log.info("[%s] Saved cache → %s", split_name, cache_dir)
    log.info("[%s] Done: iso=%s  full=%s  violent=%d  nonviolent=%d",
             split_name, X_iso.shape, X_full.shape,
             int(y.sum()), int((y == 0).sum()))

    return X_iso, X_full, y


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract RLVS pose-interaction features")
    parser.add_argument("--stride",  type=int, default=IC.FRAME_STRIDE,
                        help="Sample every N-th frame (default: %(default)s)")
    parser.add_argument("--seq-len", type=int, default=IC.SEQUENCE_LENGTH,
                        help="Frames per LSTM sequence (default: %(default)s)")
    parser.add_argument("--subset",  type=int, default=None,
                        help="Limit to first N clips per split (quick validation)")
    parser.add_argument("--conf",    type=float, default=0.35,
                        help="YOLO keypoint confidence threshold (default: %(default)s)")
    args = parser.parse_args()

    log.info("Loading YOLOv8-pose model: %s", IC.POSE_MODEL)
    from ultralytics import YOLO
    model = YOLO(IC.POSE_MODEL)   # auto-downloads; GPU via CUDA if available

    log.info("Indexing RLVS dataset …")
    train_clips, val_clips, test_clips = load_rlvs_splits(
        dataset_root=IC.RLVS_DATASET_PATH,
        train_ratio=IC.RLVS_TRAIN_RATIO,
        val_ratio=IC.RLVS_VAL_RATIO,
        seed=IC.RLVS_SEED,
    )

    cache_dir = IC.RLVS_CACHE_DIR

    for split_name, clips in [("train", train_clips), ("val", val_clips), ("test", test_clips)]:
        _extract_split(
            split_name, clips, model,
            stride=args.stride,
            seq_len=args.seq_len,
            cache_dir=cache_dir,
            subset=args.subset,
        )

    log.info("Extraction complete.  Cache: %s", cache_dir)


if __name__ == "__main__":
    main()
