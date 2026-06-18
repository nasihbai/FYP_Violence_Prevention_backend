"""
Interaction-Aware Feature Extraction — single source of truth.

Imported by BOTH training (training/extract_pose_features.py) and
live inference (core/detection_engine.py).  Do NOT duplicate this
math in any other file — train/inference skew is how the 132→309
remap bug was born.

Feature vector layout (per frame):
  ┌─ Isolated block (8) ──────────────────────────────────────────┐
  │  mean_arm_raise, max_arm_raise                                 │
  │  mean_limb_speed, max_limb_speed                               │
  │  mean_trunk_angle, max_trunk_angle  (abs, normalised to [-1,1])│
  │  mean_bbox_aspect, max_bbox_aspect  (normalised, capped at 1)  │
  ├─ Scene scalars (2) ────────────────────────────────────────────┤
  │  person_count_norm  (count / 10, capped at 1)                  │
  │  motion_energy      (same as mean_limb_speed — convenience)    │
  ├─ Interaction block (9)  ← present only in FEATURE_DIM_FULL ───┤
  │  min/max/mean  proximity    (0=far, 1=same point)              │
  │  min/max/mean  bbox_iou                                        │
  │  min/max/mean  wrist_near_opponent                             │
  └───────────────────────────────────────────────────────────────┘

  FEATURE_DIM_ISOLATED = 10   (Ablation B)
  FEATURE_DIM_FULL     = 19   (Ablation C — proposed model)
"""

import numpy as np
from typing import List, Optional, Tuple

# ── COCO 17 keypoint indices ──────────────────────────────────────────────────
KP_NOSE                              = 0
KP_L_EYE,  KP_R_EYE                 = 1, 2
KP_L_EAR,  KP_R_EAR                 = 3, 4
KP_L_SHOULDER, KP_R_SHOULDER        = 5, 6
KP_L_ELBOW,    KP_R_ELBOW           = 7, 8
KP_L_WRIST,    KP_R_WRIST           = 9, 10
KP_L_HIP,      KP_R_HIP             = 11, 12
KP_L_KNEE,     KP_R_KNEE            = 13, 14
KP_L_ANKLE,    KP_R_ANKLE           = 15, 16

# Keypoints used for speed (fast-moving limbs are most discriminative)
SPEED_KP_INDICES = [KP_L_WRIST, KP_R_WRIST, KP_L_ELBOW, KP_R_ELBOW]

CONF_THRESH = 0.30   # keypoint confidence below this → treat as missing

# ── Feature layout constants ──────────────────────────────────────────────────
FEATURE_DIM_ISOLATED = 10
FEATURE_DIM_FULL     = 19

FEATURE_LAYOUT: dict = {
    # Isolated block
    "mean_arm_raise":    0,
    "max_arm_raise":     1,
    "mean_limb_speed":   2,
    "max_limb_speed":    3,
    "mean_trunk_angle":  4,
    "max_trunk_angle":   5,
    "mean_bbox_aspect":  6,
    "max_bbox_aspect":   7,
    # Scene scalars
    "person_count_norm": 8,
    "motion_energy":     9,
    # Interaction block (only in FEATURE_DIM_FULL)
    "min_proximity":    10,
    "max_proximity":    11,
    "mean_proximity":   12,
    "min_iou":          13,
    "max_iou":          14,
    "mean_iou":         15,
    "min_wrist_opp":    16,
    "max_wrist_opp":    17,
    "mean_wrist_opp":   18,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _kp(kps: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
    """Return (x, y) if keypoint is visible, else None."""
    if kps[idx, 2] >= CONF_THRESH:
        return float(kps[idx, 0]), float(kps[idx, 1])
    return None


def _mid(kps: np.ndarray, a: int, b: int) -> Optional[Tuple[float, float]]:
    pa = _kp(kps, a)
    pb = _kp(kps, b)
    if pa and pb:
        return (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
    return None


# ── Per-person feature helpers (public — used by heuristic_classifier too) ────

def arm_raise_score(kps: np.ndarray) -> float:
    """
    0–1: fraction of wrist-above-shoulder events.
    1.0 = both wrists well above shoulder level (striking pose).
    """
    score, count = 0.0, 0
    for wrist_i, shoulder_i in [(KP_L_WRIST, KP_L_SHOULDER),
                                  (KP_R_WRIST, KP_R_SHOULDER)]:
        w = _kp(kps, wrist_i)
        s = _kp(kps, shoulder_i)
        if w and s:
            # In image coords y increases downward, so wrist above shoulder ↔ w.y < s.y
            diff  = s[1] - w[1]                      # positive when wrist is above shoulder
            score += max(0.0, min(1.0, (diff + 20) / 80.0))
            count += 1
    return score / count if count else 0.0


def trunk_angle_norm(kps: np.ndarray) -> float:
    """
    Trunk angle from vertical, normalised to [-1, 1].
    0 = upright, ±1 = lying flat horizontally.
    """
    sh = _mid(kps, KP_L_SHOULDER, KP_R_SHOULDER)
    hp = _mid(kps, KP_L_HIP,      KP_R_HIP)
    if sh is None or hp is None:
        return 0.0
    dx = hp[0] - sh[0]
    dy = hp[1] - sh[1]
    angle_deg = float(np.degrees(np.arctan2(dx, dy + 1e-6)))
    return float(np.clip(angle_deg / 90.0, -1.0, 1.0))


def bbox_aspect_norm(bbox: Tuple[int, int, int, int]) -> float:
    """
    Bounding-box width/height ratio, capped and normalised to [0, 1].
    1.0 = wide bbox (person lying flat), 0 = tall/normal.
    A person standing normally has aspect < 1; lying down gives aspect > 1.
    Cap at 3 to avoid outliers dominating.
    """
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    return min(float(bw) / bh, 3.0) / 3.0


def limb_speed_norm(kps_curr: np.ndarray,
                    kps_prev: Optional[np.ndarray],
                    frame_dim: float = 640.0) -> float:
    """
    Mean speed of fast-moving keypoints (wrists + elbows) between consecutive
    frames, normalised by frame diagonal.  Returns 0 if no previous frame.
    """
    if kps_prev is None:
        return 0.0
    speeds = []
    for idx in SPEED_KP_INDICES:
        c = _kp(kps_curr, idx)
        p = _kp(kps_prev, idx)
        if c and p:
            speeds.append(float(np.hypot(c[0] - p[0], c[1] - p[1])) / (frame_dim + 1e-6))
    return float(np.mean(speeds)) if speeds else 0.0


def per_person_features(
    kps:       np.ndarray,
    bbox:      Tuple[int, int, int, int],
    kps_prev:  Optional[np.ndarray] = None,
    frame_dim: float = 640.0,
) -> np.ndarray:
    """
    4-element vector for a single person:
      [arm_raise, limb_speed, trunk_angle_norm, bbox_aspect_norm]
    """
    return np.array([
        arm_raise_score(kps),
        limb_speed_norm(kps, kps_prev, frame_dim),
        abs(trunk_angle_norm(kps)),    # use abs so mean/max are both non-negative
        bbox_aspect_norm(bbox),
    ], dtype=np.float32)


# ── Interaction helpers ───────────────────────────────────────────────────────

def _bbox_center(bbox: Tuple) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_iou(b1: Tuple, b2: Tuple) -> float:
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def wrist_near_torso(
    kps_a:  np.ndarray,
    kps_b:  np.ndarray,
    thresh: float = 80.0,
) -> float:
    """
    0–1: how close person A's wrists are to person B's torso keypoints.
    Captures striking/grabbing interaction.
    """
    torso = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP]
    best  = 0.0
    for w in [KP_L_WRIST, KP_R_WRIST]:
        wpt = _kp(kps_a, w)
        if wpt is None:
            continue
        for t in torso:
            tpt = _kp(kps_b, t)
            if tpt is None:
                continue
            d = float(np.hypot(wpt[0] - tpt[0], wpt[1] - tpt[1]))
            if d < thresh:
                best = max(best, 1.0 - d / thresh)
    return best


def interaction_features(
    persons_kps:  List[np.ndarray],
    persons_bbox: List[Tuple],
    frame_dim:    float = 640.0,
    wrist_thresh: float = 80.0,
) -> np.ndarray:
    """
    9-element aggregated pairwise interaction vector.
    Returns zeros when fewer than 2 persons are present.

    Values: [min, max, mean] × [proximity, bbox_iou, wrist_near_opponent]
    """
    out = np.zeros(9, dtype=np.float32)
    n   = len(persons_kps)
    if n < 2:
        return out

    prox_v, iou_v, wrist_v = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            ci = _bbox_center(persons_bbox[i])
            cj = _bbox_center(persons_bbox[j])
            dist  = float(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))
            prox  = max(0.0, 1.0 - dist / (frame_dim + 1e-6))
            iou   = _bbox_iou(persons_bbox[i], persons_bbox[j])
            wi    = wrist_near_torso(persons_kps[i], persons_kps[j], wrist_thresh)
            wj    = wrist_near_torso(persons_kps[j], persons_kps[i], wrist_thresh)
            wrist = max(wi, wj)

            prox_v.append(prox)
            iou_v.append(iou)
            wrist_v.append(wrist)

    if prox_v:
        out[0], out[1], out[2] = min(prox_v),  max(prox_v),  float(np.mean(prox_v))
        out[3], out[4], out[5] = min(iou_v),   max(iou_v),   float(np.mean(iou_v))
        out[6], out[7], out[8] = min(wrist_v), max(wrist_v), float(np.mean(wrist_v))

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def frame_feature_vector(
    persons_kps:         List[np.ndarray],
    persons_bbox:        List[Tuple],
    prev_kps:            Optional[List[np.ndarray]] = None,
    frame_h:             int   = 480,
    frame_w:             int   = 640,
    include_interaction: bool  = True,
    wrist_thresh:        float = 80.0,
) -> np.ndarray:
    """
    Build one row of the feature matrix for a single frame.

    Args:
        persons_kps:         List of (17, 3) arrays — current frame keypoints.
        persons_bbox:        Matching list of (x1, y1, x2, y2) bboxes.
        prev_kps:            Same-ordered list from the previous frame (for speed).
        frame_h / frame_w:   Frame dimensions (for normalisation).
        include_interaction: False → Ablation B (isolated only, 10-dim).
                             True  → Ablation C (full, 19-dim).
        wrist_thresh:        Pixel radius for wrist-near-torso scoring.

    Returns:
        np.ndarray of shape (FEATURE_DIM_FULL,) or (FEATURE_DIM_ISOLATED,).
    """
    frame_dim = float(max(frame_h, frame_w))

    # ── Isolated block ─────────────────────────────────────────────────────
    if persons_kps:
        ppf_list = [
            per_person_features(
                kps, bbox,
                kps_prev=(prev_kps[idx] if (prev_kps and idx < len(prev_kps)) else None),
                frame_dim=frame_dim,
            )
            for idx, (kps, bbox) in enumerate(zip(persons_kps, persons_bbox))
        ]
        ppf       = np.stack(ppf_list)                          # (N, 4)
        iso_block = np.concatenate([ppf.mean(0), ppf.max(0)])   # (8,)
        mean_speed = float(ppf[:, 1].mean())
    else:
        iso_block  = np.zeros(8,  dtype=np.float32)
        mean_speed = 0.0

    scene = np.array([
        min(len(persons_kps) / 10.0, 1.0),
        mean_speed,
    ], dtype=np.float32)

    if not include_interaction:
        return np.concatenate([iso_block, scene]).astype(np.float32)

    # ── Interaction block ──────────────────────────────────────────────────
    inter = interaction_features(
        persons_kps, persons_bbox,
        frame_dim=frame_dim, wrist_thresh=wrist_thresh,
    )
    return np.concatenate([iso_block, scene, inter]).astype(np.float32)
