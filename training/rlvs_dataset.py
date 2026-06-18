"""
Violence dataset indexer — supports two layouts:

  Layout A (RLVS flat):
    Real Life Violence Dataset/
    ├── Violence/     V_*.mp4   → label 1
    └── NonViolence/  NV_*.mp4  → label 0

  Layout B (existing train/val):
    Complete Dataset/
    ├── train/
    │   ├── Fight/        → label 1
    │   ├── HockeyFight/  → label 1
    │   ├── MovieFights/  → label 1
    │   └── NonFight/     → label 0
    └── val/   (same structure)

Auto-detects which layout is present and returns a stratified
train / val / test split in all cases.
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

ClipList = List[Tuple[Path, int]]   # [(video_path, label), ...]

# Binary label maps per layout
_CLASS_MAP_RLVS = {
    "Violence":    1,
    "NonViolence": 0,
}
_CLASS_MAP_EXISTING = {
    "Fight":       1,
    "HockeyFight": 1,
    "MovieFights": 1,
    "NonFight":    0,
}


def _glob_videos(directory: Path) -> List[Path]:
    videos = list(directory.glob("*.mp4")) + list(directory.glob("*.avi"))
    return sorted(videos)


def _detect_layout(root: Path) -> str:
    """Return 'rlvs', 'existing', or raise if unrecognised."""
    if (root / "Violence").exists() or (root / "NonViolence").exists():
        return "rlvs"
    if (root / "train").exists() or (root / "val").exists():
        return "existing"
    raise ValueError(
        f"Cannot detect dataset layout under: {root}\n"
        "Expected either Violence/NonViolence (RLVS) or train/val subdirs."
    )


def _load_rlvs_flat(root: Path) -> ClipList:
    all_clips: ClipList = []
    for class_name, label in _CLASS_MAP_RLVS.items():
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        videos = _glob_videos(class_dir)
        logger.info("  %s: %d clips (label=%d)", class_name, len(videos), label)
        all_clips.extend((v, label) for v in videos)
    return all_clips


def _load_existing_split(root: Path, split: str) -> ClipList:
    split_dir = root / split
    clips: ClipList = []
    if not split_dir.exists():
        return clips
    for class_name, label in _CLASS_MAP_EXISTING.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        videos = _glob_videos(class_dir)
        logger.info("  [%s/%s] %d clips (label=%d)", split, class_name, len(videos), label)
        clips.extend((v, label) for v in videos)
    return clips


def load_rlvs_splits(
    dataset_root: Path = None,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    seed:         int   = 42,
) -> Tuple[ClipList, ClipList, ClipList]:
    """
    Index the dataset and return (train_clips, val_clips, test_clips).

    For Layout B (existing train/val), the existing train split is kept as-is
    and the val split is divided into val (67 %) + test (33 %) for a
    roughly 80/10/10 overall split.

    Returns:
        (train_clips, val_clips, test_clips) — each a list of (Path, label).
    """
    if dataset_root is None:
        from config.settings import InteractionConfig as _IC
        dataset_root = _IC.RLVS_DATASET_PATH

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_root}\n"
            "Set VIOLENCE_DATASET env var or RLVS_DATASET_PATH in config/settings.py"
        )

    layout = _detect_layout(dataset_root)
    logger.info("Dataset layout detected: %s  (%s)", layout, dataset_root)

    rng = random.Random(seed)

    def _stratified(clips: ClipList, ratio: float) -> Tuple[ClipList, ClipList]:
        """Split clips into (keep, remainder) preserving class balance."""
        violent     = [(p, l) for p, l in clips if l == 1]
        non_violent = [(p, l) for p, l in clips if l == 0]
        result_a, result_b = [], []
        for group in (violent, non_violent):
            g = group[:]
            rng.shuffle(g)
            n = int(len(g) * ratio)
            result_a.extend(g[:n])
            result_b.extend(g[n:])
        return result_a, result_b

    if layout == "rlvs":
        all_clips = _load_rlvs_flat(dataset_root)
        if not all_clips:
            raise RuntimeError("No video files found — check dataset_root.")
        # Stratified 3-way split
        train_clips, rest = _stratified(all_clips, train_ratio)
        val_ratio_adj = val_ratio / (1.0 - train_ratio)
        val_clips, test_clips = _stratified(rest, val_ratio_adj)

    else:  # existing train/val layout
        train_clips = _load_existing_split(dataset_root, "train")
        val_all     = _load_existing_split(dataset_root, "val")
        # Split val 67/33 → val / test
        val_clips, test_clips = _stratified(val_all, 0.67)

    rng.shuffle(train_clips)
    rng.shuffle(val_clips)
    rng.shuffle(test_clips)

    v_tr  = sum(l for _, l in train_clips)
    v_va  = sum(l for _, l in val_clips)
    v_te  = sum(l for _, l in test_clips)
    logger.info(
        "Split (seed=%d): train=%d (V=%d NV=%d)  val=%d (V=%d NV=%d)  test=%d (V=%d NV=%d)",
        seed,
        len(train_clips), v_tr, len(train_clips)-v_tr,
        len(val_clips),   v_va, len(val_clips)-v_va,
        len(test_clips),  v_te, len(test_clips)-v_te,
    )

    return train_clips, val_clips, test_clips
