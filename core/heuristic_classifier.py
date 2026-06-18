"""
Rule-based fight & fall detector — Baseline A in the ablation study.

Ported from the reference YOLO-pose implementation (main.py) and adapted
to consume PersonPose objects from pose_estimator.py.

Used as:
  1. The immediate working detector in Milestone 1 (no training required).
  2. Ablation A during evaluation in train_interaction_lstm.py.
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from .pose_estimator import PersonPose
from .interaction_features import (
    arm_raise_score,
    trunk_angle_norm,
    _bbox_center,
    _bbox_iou,
    wrist_near_torso,
    KP_L_HIP, KP_R_HIP,
    CONF_THRESH,
)
from config.settings import InteractionConfig as _C

logger = logging.getLogger(__name__)


# ── Fall helpers ──────────────────────────────────────────────────────────────

_FALL_ANGLE_THRESH  = 35.0   # trunk degrees from vertical → fallen
_FALL_ASPECT_THRESH = 1.35   # bbox width/height → horizontal body
_FALL_HIP_THRESH    = 0.65   # hip Y as fraction of bbox height → near floor
_FALL_PERSIST       = 20     # frames to keep FALLEN label after vote drops


def _fall_score(kps: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    score = 0.0

    # Signal 1: trunk angle from vertical (large angle = fallen)
    angle = abs(trunk_angle_norm(kps)) * 90.0        # back to degrees
    if angle > _FALL_ANGLE_THRESH:
        score += min(1.0, (angle - _FALL_ANGLE_THRESH) / (90.0 - _FALL_ANGLE_THRESH)) * 0.45

    # Signal 2: bbox aspect ratio (wide bbox = horizontal body)
    bw = bbox[2] - bbox[0]
    bh = max(1, bbox[3] - bbox[1])
    aspect = bw / bh
    if aspect > _FALL_ASPECT_THRESH:
        score += min(1.0, (aspect - _FALL_ASPECT_THRESH) / 1.5) * 0.35

    # Signal 3: hips near the bottom of the bounding box
    hip_y = None
    hy_l = kps[KP_L_HIP, 1] if kps[KP_L_HIP, 2] >= CONF_THRESH else None
    hy_r = kps[KP_R_HIP, 1] if kps[KP_R_HIP, 2] >= CONF_THRESH else None
    if hy_l is not None and hy_r is not None:
        hip_y = (hy_l + hy_r) / 2.0
    elif hy_l is not None:
        hip_y = hy_l
    elif hy_r is not None:
        hip_y = hy_r

    if hip_y is not None and bh > 20:
        rel = (hip_y - bbox[1]) / bh
        if rel > _FALL_HIP_THRESH:
            score += min(1.0, (rel - _FALL_HIP_THRESH) / (1.0 - _FALL_HIP_THRESH)) * 0.20

    return min(1.0, score)


# ── Limb speed from history ───────────────────────────────────────────────────

def _limb_speed_from_history(history: deque) -> float:
    """Mean wrist/elbow pixel speed across the last N frames in history."""
    if len(history) < 2:
        return 0.0
    arr = list(history)
    speeds = []
    for i in range(1, len(arr)):
        prev, curr = arr[i - 1], arr[i]
        mask = (prev[:, 2] >= CONF_THRESH) & (curr[:, 2] >= CONF_THRESH)
        if mask.sum() == 0:
            continue
        d = np.linalg.norm(curr[mask, :2] - prev[mask, :2], axis=1)
        speeds.extend(d.tolist())
    return float(np.mean(speeds)) if speeds else 0.0


def _angle_variance(history: deque) -> float:
    """Body-angle variance over the history window (in degrees²)."""
    if len(history) < 3:
        return 0.0
    angles = [trunk_angle_norm(k) * 90.0 for k in history]
    return float(np.var(angles))


# ── Main classifier ───────────────────────────────────────────────────────────

class HeuristicFightClassifier:
    """
    Stateful, scene-level fight and fall detector based on geometric heuristics.

    Thresholds live in config/settings.py → InteractionConfig.
    Per-track state (keypoint history, vote queues) is maintained internally.
    """

    def __init__(self):
        self._kps_history:   Dict[int, deque] = {}
        self._fight_votes:   Dict[int, deque] = {}
        self._fall_votes:    Dict[int, deque] = {}
        self._fall_persist:  Dict[int, int]   = {}
        self._lost_count:    Dict[int, int]   = {}

    def _ensure_track(self, tid: int) -> None:
        if tid not in self._kps_history:
            self._kps_history[tid]  = deque(maxlen=_C.HEURISTIC_HISTORY)
            self._fight_votes[tid]  = deque(maxlen=_C.SMOOTH_WINDOW)
            self._fall_votes[tid]   = deque(maxlen=_C.SMOOTH_WINDOW)
            self._fall_persist[tid] = 0
            self._lost_count[tid]   = 0

    def _prune_lost(self, active: set) -> None:
        to_del = [
            tid for tid in list(self._kps_history)
            if tid not in active
            and (self._lost_count.get(tid, 0) + 1) > 10
        ]
        for tid in self._kps_history:
            if tid not in active:
                self._lost_count[tid] = self._lost_count.get(tid, 0) + 1
        for tid in to_del:
            self._kps_history.pop(tid, None)
            self._fight_votes.pop(tid, None)
            self._fall_votes.pop(tid, None)
            self._fall_persist.pop(tid, None)
            self._lost_count.pop(tid, None)

    def classify(
        self,
        persons: List[PersonPose],
    ) -> Tuple[bool, bool, float, Dict[int, bool], Dict[int, bool]]:
        """
        Args:
            persons: Current-frame PersonPose list from YOLOPoseEstimator.

        Returns:
            (any_fight, any_fall, scene_score, per_fight, per_fall)
              any_fight   — True when at least one tracked pair is fighting
              any_fall    — True when at least one person is flagged as fallen
              scene_score — 0-1 threat level (useful for the HUD bar)
              per_fight   — {track_id: bool} smoothed fight flag per person
              per_fall    — {track_id: bool} smoothed fall flag per person
        """
        active = {p.track_id for p in persons}
        self._prune_lost(active)

        per_person_score: Dict[int, float] = {}
        raw_fight:        Dict[int, bool]  = {}
        raw_fall:         Dict[int, bool]  = {}

        # ── Per-person motion + fall ───────────────────────────────────────
        for p in persons:
            tid = p.track_id
            self._ensure_track(tid)
            self._lost_count[tid] = 0
            self._kps_history[tid].append(p.keypoints)

            speed   = _limb_speed_from_history(self._kps_history[tid])
            ang_var = _angle_variance(self._kps_history[tid])
            a_raise = arm_raise_score(p.keypoints)

            score = 0.0
            if speed > _C.SPEED_FIGHT_THRESH:
                score += min(0.35, 0.35 * (speed - _C.SPEED_FIGHT_THRESH) / _C.SPEED_FIGHT_THRESH)
            if ang_var > _C.ANGLE_VARIANCE_THRESH:
                score += min(0.25, 0.25 * ang_var / (_C.ANGLE_VARIANCE_THRESH * 2))
            if a_raise > 0.3:
                score += a_raise * 0.25

            per_person_score[tid] = min(1.0, score)
            raw_fight[tid] = False
            raw_fall[tid]  = _fall_score(p.keypoints, p.bbox) >= 0.45

        # ── Pairwise fight evaluation ──────────────────────────────────────
        ids        = [p.track_id for p in persons]
        person_map = {p.track_id: p for p in persons}
        n          = len(ids)

        if n >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    pi  = person_map[ids[i]]
                    pj  = person_map[ids[j]]
                    ci  = _bbox_center(pi.bbox)
                    cj  = _bbox_center(pj.bbox)
                    dist = float(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))
                    iou  = _bbox_iou(pi.bbox, pj.bbox)

                    pair = 0.0
                    if dist < _C.PROXIMITY_THRESH:
                        pair += 0.3 * (1.0 - dist / _C.PROXIMITY_THRESH)
                    if iou > _C.OVERLAP_IOU_THRESH:
                        pair += 0.30
                    wi    = wrist_near_torso(pi.keypoints, pj.keypoints, _C.WRIST_TO_BODY_THRESH)
                    wj    = wrist_near_torso(pj.keypoints, pi.keypoints, _C.WRIST_TO_BODY_THRESH)
                    pair += (wi + wj) * 0.40
                    pair  = min(0.70, pair)

                    avg_m    = (per_person_score[ids[i]] + per_person_score[ids[j]]) / 2.0
                    combined = min(1.0, avg_m * 0.5 + pair * 0.5)

                    if combined >= _C.FIGHT_THRESHOLD:
                        raw_fight[ids[i]] = True
                        raw_fight[ids[j]] = True
                        per_person_score[ids[i]] = min(1.0, per_person_score[ids[i]] + pair * 0.5)
                        per_person_score[ids[j]] = min(1.0, per_person_score[ids[j]] + pair * 0.5)

        # ── Temporal smoothing + fall persistence ──────────────────────────
        per_fight: Dict[int, bool] = {}
        per_fall:  Dict[int, bool] = {}

        for tid in ids:
            self._fight_votes[tid].append(1 if raw_fight[tid] else 0)
            fv = self._fight_votes[tid]
            per_fight[tid] = sum(fv) > len(fv) / 2

            self._fall_votes[tid].append(1 if raw_fall[tid] else 0)
            av = self._fall_votes[tid]
            if sum(av) > len(av) / 2:
                self._fall_persist[tid] = _FALL_PERSIST
            elif self._fall_persist[tid] > 0:
                self._fall_persist[tid] -= 1
            per_fall[tid] = self._fall_persist[tid] > 0

        any_fight = any(per_fight.values())
        any_fall  = any(per_fall.values())

        alert_ids = [t for t, f in per_fight.items() if f] + \
                    [t for t, f in per_fall.items()  if f]
        alert_ids = list(set(alert_ids))
        if alert_ids:
            scene_score = float(np.mean([per_person_score[t] for t in alert_ids]))
        elif per_person_score:
            scene_score = float(np.mean(list(per_person_score.values())))
        else:
            scene_score = 0.0

        return any_fight, any_fall, scene_score, per_fight, per_fall

    def reset(self) -> None:
        self._kps_history.clear()
        self._fight_votes.clear()
        self._fall_votes.clear()
        self._fall_persist.clear()
        self._lost_count.clear()
