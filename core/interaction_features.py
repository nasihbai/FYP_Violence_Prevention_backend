"""
Interaction-Aware Feature Extraction — single source of truth.

Imported by BOTH training (training/extract_pose_features.py) and
live inference (core/detection_engine.py).  Do NOT duplicate this
math in any other file — train/inference skew is how the 132→309
remap bug was born.

Feature vector layout (per frame):
  ┌─ Isolated block (10) ─────────────────────────────────────────┐
  │  mean/max  arm_raise         (wrist above shoulder level)      │
  │  mean/max  limb_speed        (wrist+elbow speed, normalised)   │
  │  mean/max  trunk_angle       (abs, normalised to [0,1])        │
  │  mean/max  bbox_aspect       (normalised, capped at 1)         │
  │  mean/max  elbow_angle       (shoulder→elbow→wrist angle [0,1])│
  ├─ Scene scalars (2) ────────────────────────────────────────────┤
  │  person_count_norm  (count / 10, capped at 1)                  │
  │  motion_energy      (same as mean_limb_speed — convenience)    │
  ├─ Interaction block (15) ──────────────────────────────────────┤
  │  min/max/mean  proximity          (0=far, 1=same point)        │
  │  min/max/mean  bbox_iou                                        │
  │  min/max/mean  wrist_near_torso   (grabbing/striking)          │
  │  min/max/mean  head_proximity     (nose-to-nose distance)      │
  │  min/max/mean  wrist_toward_opp   (wrist velocity → opponent)  │
  └───────────────────────────────────────────────────────────────┘

  FEATURE_DIM_ISOLATED = 12   (Ablation B)
  FEATURE_DIM_FULL     = 27   (Ablation C — proposed model)
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

SPEED_KP_INDICES = [KP_L_WRIST, KP_R_WRIST, KP_L_ELBOW, KP_R_ELBOW]

CONF_THRESH = 0.30

# ── Feature layout constants ──────────────────────────────────────────────────
FEATURE_DIM_ISOLATED = 12   # isolated (10) + scene (2)
FEATURE_DIM_FULL     = 27   # isolated (10) + scene (2) + interaction (15)

FEATURE_LAYOUT: dict = {
    # Isolated block (10 values)
    "mean_arm_raise":    0,
    "max_arm_raise":     1,
    "mean_limb_speed":   2,
    "max_limb_speed":    3,
    "mean_trunk_angle":  4,
    "max_trunk_angle":   5,
    "mean_bbox_aspect":  6,
    "max_bbox_aspect":   7,
    "mean_elbow_angle":  8,   # NEW — bent elbow = striking pose
    "max_elbow_angle":   9,
    # Scene scalars (2 values)
    "person_count_norm": 10,
    "motion_energy":     11,
    # Interaction block (15 values)
    "min_proximity":     12,
    "max_proximity":     13,
    "mean_proximity":    14,
    "min_iou":           15,
    "max_iou":           16,
    "mean_iou":          17,
    "min_wrist_opp":     18,
    "max_wrist_opp":     19,
    "mean_wrist_opp":    20,
    "min_head_prox":     21,   # NEW — nose-to-nose closeness
    "max_head_prox":     22,
    "mean_head_prox":    23,
    "min_wrist_toward":  24,   # NEW — wrist velocity toward opponent
    "max_wrist_toward":  25,
    "mean_wrist_toward": 26,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _kp(kps: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
    if kps[idx, 2] >= CONF_THRESH:
        return float(kps[idx, 0]), float(kps[idx, 1])
    return None


def _mid(kps: np.ndarray, a: int, b: int) -> Optional[Tuple[float, float]]:
    pa = _kp(kps, a)
    pb = _kp(kps, b)
    if pa and pb:
        return (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
    return None


# ── Per-person feature helpers ────────────────────────────────────────────────

def arm_raise_score(kps: np.ndarray) -> float:
    """0–1: wrist above shoulder level (striking pose)."""
    score, count = 0.0, 0
    for wrist_i, shoulder_i in [(KP_L_WRIST, KP_L_SHOULDER),
                                  (KP_R_WRIST, KP_R_SHOULDER)]:
        w = _kp(kps, wrist_i)
        s = _kp(kps, shoulder_i)
        if w and s:
            diff  = s[1] - w[1]   # positive when wrist above shoulder
            score += max(0.0, min(1.0, (diff + 20) / 80.0))
            count += 1
    return score / count if count else 0.0


def trunk_angle_norm(kps: np.ndarray) -> float:
    """Trunk angle from vertical, normalised to [-1, 1]. Abs taken downstream."""
    sh = _mid(kps, KP_L_SHOULDER, KP_R_SHOULDER)
    hp = _mid(kps, KP_L_HIP,      KP_R_HIP)
    if sh is None or hp is None:
        return 0.0
    dx = hp[0] - sh[0]
    dy = hp[1] - sh[1]
    return float(np.clip(np.degrees(np.arctan2(dx, dy + 1e-6)) / 90.0, -1.0, 1.0))


def bbox_aspect_norm(bbox: Tuple[int, int, int, int]) -> float:
    """Bbox width/height ratio, capped at 3 and normalised to [0, 1]."""
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    return min(float(bw) / bh, 3.0) / 3.0


def limb_speed_norm(kps_curr: np.ndarray,
                    kps_prev: Optional[np.ndarray],
                    frame_dim: float = 640.0) -> float:
    """Mean speed of wrists+elbows between frames, normalised."""
    if kps_prev is None:
        return 0.0
    speeds = []
    for idx in SPEED_KP_INDICES:
        c = _kp(kps_curr, idx)
        p = _kp(kps_prev, idx)
        if c and p:
            speeds.append(float(np.hypot(c[0] - p[0], c[1] - p[1])) / (frame_dim + 1e-6))
    return float(np.mean(speeds)) if speeds else 0.0


def elbow_angle_norm(kps: np.ndarray) -> float:
    """
    Mean of left and right elbow angles (shoulder→elbow→wrist), normalised [0, 1].
    0 = fully bent (90° or less), 1 = fully straight arm (180°).
    A cocked punch starts near 0, a delivered punch is near 1.
    Both extremes differ from a relaxed hanging arm (~160°).
    """
    angles = []
    for sh_i, el_i, wr_i in [(KP_L_SHOULDER, KP_L_ELBOW, KP_L_WRIST),
                               (KP_R_SHOULDER, KP_R_ELBOW, KP_R_WRIST)]:
        sh = _kp(kps, sh_i)
        el = _kp(kps, el_i)
        wr = _kp(kps, wr_i)
        if sh and el and wr:
            # Vector elbow→shoulder and elbow→wrist
            v1 = np.array([sh[0] - el[0], sh[1] - el[1]])
            v2 = np.array([wr[0] - el[0], wr[1] - el[1]])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_a)))
                angles.append(angle_deg / 180.0)   # normalise to [0, 1]
    return float(np.mean(angles)) if angles else 0.5   # default = mid (unknown)


def per_person_features(
    kps:       np.ndarray,
    bbox:      Tuple[int, int, int, int],
    kps_prev:  Optional[np.ndarray] = None,
    frame_dim: float = 640.0,
) -> np.ndarray:
    """
    5-element vector for a single person:
      [arm_raise, limb_speed, trunk_angle_abs, bbox_aspect, elbow_angle]
    """
    return np.array([
        arm_raise_score(kps),
        limb_speed_norm(kps, kps_prev, frame_dim),
        abs(trunk_angle_norm(kps)),
        bbox_aspect_norm(bbox),
        elbow_angle_norm(kps),          # NEW
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
    """0–1: person A's wrist near person B's torso (grabbing/striking)."""
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


def _head_proximity(
    kps_a:     np.ndarray,
    kps_b:     np.ndarray,
    frame_dim: float = 640.0,
) -> float:
    """
    0–1: how close person A's nose is to person B's nose.
    1 = same point, 0 = opposite ends of frame.
    Falls back to shoulder midpoints when nose not visible.
    """
    def head_pt(kps):
        n = _kp(kps, KP_NOSE)
        if n:
            return n
        return _mid(kps, KP_L_SHOULDER, KP_R_SHOULDER)

    pa = head_pt(kps_a)
    pb = head_pt(kps_b)
    if pa is None or pb is None:
        return 0.0
    dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
    return max(0.0, 1.0 - dist / (frame_dim + 1e-6))


def _wrist_velocity_toward_opponent(
    kps_curr:     np.ndarray,
    kps_prev:     Optional[np.ndarray],
    bbox_opponent: Tuple,
    frame_dim:    float = 640.0,
) -> float:
    """
    0–1+: component of wrist velocity pointing toward the opponent's centre.
    Positive = wrist moving toward opponent (aggressive intent).
    Returns 0 when no previous frame is available.
    """
    if kps_prev is None:
        return 0.0
    center = _bbox_center(bbox_opponent)
    best   = 0.0
    for w in [KP_L_WRIST, KP_R_WRIST]:
        c = _kp(kps_curr, w)
        p = _kp(kps_prev, w)
        if c is None or p is None:
            continue
        # Wrist displacement vector
        vx = c[0] - p[0]
        vy = c[1] - p[1]
        # Unit vector from wrist to opponent centre
        dx = center[0] - c[0]
        dy = center[1] - c[1]
        dist = float(np.hypot(dx, dy)) + 1e-6
        dx /= dist;  dy /= dist
        # Positive projection = moving toward opponent
        toward = (vx * dx + vy * dy) / (frame_dim + 1e-6)
        best = max(best, toward)
    return max(0.0, float(best))


def interaction_features(
    persons_kps:      List[np.ndarray],
    persons_bbox:     List[Tuple],
    prev_persons_kps: Optional[List[np.ndarray]] = None,   # NEW
    frame_dim:        float = 640.0,
    wrist_thresh:     float = 80.0,
) -> np.ndarray:
    """
    15-element aggregated pairwise interaction vector.
    Returns zeros when fewer than 2 persons are present.

    Values:
      [min, max, mean] × proximity
      [min, max, mean] × bbox_iou
      [min, max, mean] × wrist_near_torso
      [min, max, mean] × head_proximity      (NEW)
      [min, max, mean] × wrist_toward_opp    (NEW)
    """
    out = np.zeros(15, dtype=np.float32)
    n   = len(persons_kps)
    if n < 2:
        return out

    prox_v, iou_v, wrist_v, head_v, toward_v = [], [], [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            ci   = _bbox_center(persons_bbox[i])
            cj   = _bbox_center(persons_bbox[j])
            dist = float(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))

            prox_v.append(max(0.0, 1.0 - dist / (frame_dim + 1e-6)))
            iou_v.append(_bbox_iou(persons_bbox[i], persons_bbox[j]))
            wi = wrist_near_torso(persons_kps[i], persons_kps[j], wrist_thresh)
            wj = wrist_near_torso(persons_kps[j], persons_kps[i], wrist_thresh)
            wrist_v.append(max(wi, wj))

            head_v.append(_head_proximity(persons_kps[i], persons_kps[j], frame_dim))

            # Wrist toward opponent (requires prev frame)
            prev_i = prev_persons_kps[i] if (prev_persons_kps and i < len(prev_persons_kps)) else None
            prev_j = prev_persons_kps[j] if (prev_persons_kps and j < len(prev_persons_kps)) else None
            ti = _wrist_velocity_toward_opponent(persons_kps[i], prev_i, persons_bbox[j], frame_dim)
            tj = _wrist_velocity_toward_opponent(persons_kps[j], prev_j, persons_bbox[i], frame_dim)
            toward_v.append(max(ti, tj))

    if prox_v:
        out[0],  out[1],  out[2]  = min(prox_v),  max(prox_v),  float(np.mean(prox_v))
        out[3],  out[4],  out[5]  = min(iou_v),   max(iou_v),   float(np.mean(iou_v))
        out[6],  out[7],  out[8]  = min(wrist_v), max(wrist_v), float(np.mean(wrist_v))
        out[9],  out[10], out[11] = min(head_v),  max(head_v),  float(np.mean(head_v))
        out[12], out[13], out[14] = min(toward_v),max(toward_v),float(np.mean(toward_v))

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
        persons_kps:         List of (17, 3) arrays — current frame.
        persons_bbox:        Matching (x1, y1, x2, y2) bboxes.
        prev_kps:            Same-ordered list from previous frame (for speed + toward).
        frame_h / frame_w:   Frame dimensions for normalisation.
        include_interaction: False → Ablation B (12-dim).
                             True  → Ablation C (27-dim).
        wrist_thresh:        Pixel radius for wrist-near-torso scoring.

    Returns:
        np.ndarray of shape (FEATURE_DIM_FULL,) or (FEATURE_DIM_ISOLATED,).
    """
    frame_dim = float(max(frame_h, frame_w))

    # ── Isolated block (5 features × mean/max = 10) ───────────────────────
    if persons_kps:
        ppf_list = [
            per_person_features(
                kps, bbox,
                kps_prev=(prev_kps[idx] if (prev_kps and idx < len(prev_kps)) else None),
                frame_dim=frame_dim,
            )
            for idx, (kps, bbox) in enumerate(zip(persons_kps, persons_bbox))
        ]
        ppf        = np.stack(ppf_list)                          # (N, 5)
        iso_block  = np.concatenate([ppf.mean(0), ppf.max(0)])   # (10,)
        mean_speed = float(ppf[:, 1].mean())
    else:
        iso_block  = np.zeros(10, dtype=np.float32)
        mean_speed = 0.0

    scene = np.array([
        min(len(persons_kps) / 10.0, 1.0),
        mean_speed,
    ], dtype=np.float32)

    if not include_interaction:
        return np.concatenate([iso_block, scene]).astype(np.float32)   # (12,)

    # ── Interaction block (15 values) ─────────────────────────────────────
    inter = interaction_features(
        persons_kps, persons_bbox,
        prev_persons_kps=prev_kps,
        frame_dim=frame_dim,
        wrist_thresh=wrist_thresh,
    )
    return np.concatenate([iso_block, scene, inter]).astype(np.float32)  # (27,)
