"""
Pose analysis module for AI Rehab Mirror.

Handles the computer-vision pipeline:
    Camera Input -> Frame Processing -> Pose Estimation -> Landmark Smoothing
    -> Joint Angle Analysis -> Exercise State Detection -> Form Evaluation
    -> Real-time Feedback -> Medical-style Visual Report

Uses MediaPipe Pose Landmarker for full-body pose landmarks, OpenCV for frame
processing and rendering, and NumPy for vector/angle math.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices (33 landmarks, COCO-style ordering)
# ---------------------------------------------------------------------------
LANDMARK_NOSE = 0
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24

# Shoulder abduction is typically measured on the dominant arm. We default to
# the right arm but the analysis is symmetric and works for either side.
ARM_SIDE = "right"


@dataclass
class PoseLandmark:
    """A single 2D/3D pose landmark with visibility confidence."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def as_xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class FramePose:
    """Pose landmarks extracted from a single frame."""

    landmarks: Dict[int, PoseLandmark]
    detected: bool = False

    def get(self, idx: int) -> Optional[PoseLandmark]:
        return self.landmarks.get(idx)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _to_np(lm: PoseLandmark) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def joint_angle(a: PoseLandmark, b: PoseLandmark, c: PoseLandmark) -> float:
    """Compute the angle (degrees) at vertex b formed by vectors b->a and b->c."""
    v1 = _to_np(a) - _to_np(b)
    v2 = _to_np(c) - _to_np(b)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


def shoulder_abduction_angle(
    shoulder: PoseLandmark, elbow: PoseLandmark, hip: PoseLandmark
) -> float:
    """
    Approximate shoulder abduction angle (degrees).

    We project the shoulder->elbow vector onto the frontal plane and measure
    the angle it makes with the vertical (torso) axis defined by shoulder->hip.
    A value near 0 means the arm is down at the side; near 90 means the arm is
    raised to shoulder height (full abduction).
    """
    arm = _to_np(elbow) - _to_np(shoulder)
    torso = _to_np(hip) - _to_np(shoulder)
    # Project arm onto the plane perpendicular to the torso to isolate the
    # abduction component (removes forward flexion).
    torso_n = torso / (np.linalg.norm(torso) + 1e-6)
    arm_proj = arm - np.dot(arm, torso_n) * torso_n
    # Vertical reference = projection of the torso onto the same plane.
    vertical = torso - np.dot(torso, torso_n) * torso_n
    n_arm = np.linalg.norm(arm_proj)
    n_vert = np.linalg.norm(vertical)
    if n_arm < 1e-6 or n_vert < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(arm_proj, vertical) / (n_arm * n_vert), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


def torso_alignment(
    shoulder_l: PoseLandmark,
    shoulder_r: PoseLandmark,
    hip_l: PoseLandmark,
    hip_r: PoseLandmark,
) -> float:
    """
    Torso lean angle (degrees from vertical). 0 = perfectly upright.
    Uses the midpoint of the shoulders vs the midpoint of the hips.
    """
    mid_shoulder = (_to_np(shoulder_l) + _to_np(shoulder_r)) / 2.0
    mid_hip = (_to_np(hip_l) + _to_np(hip_r)) / 2.0
    axis = mid_shoulder - mid_hip
    vertical = np.array([0.0, 1.0, 0.0])
    n_axis = np.linalg.norm(axis)
    if n_axis < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(axis, vertical) / n_axis, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


def left_right_symmetry(left_angle: float, right_angle: float) -> float:
    """
    Symmetry score 0..100. 100 = perfectly symmetric abduction on both sides.
    """
    diff = abs(left_angle - right_angle)
    return max(0.0, 100.0 - diff * 2.0)


# ---------------------------------------------------------------------------
# Landmark smoothing (exponential moving average)
# ---------------------------------------------------------------------------
class LandmarkSmoother:
    """Temporal smoothing of landmark positions to reduce jitter."""

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self._smoothed: Dict[int, PoseLandmark] = {}

    def update(self, landmarks: Dict[int, PoseLandmark]) -> Dict[int, PoseLandmark]:
        out: Dict[int, PoseLandmark] = {}
        for idx, lm in landmarks.items():
            prev = self._smoothed.get(idx)
            if prev is None:
                out[idx] = PoseLandmark(lm.x, lm.y, lm.z, lm.visibility)
            else:
                out[idx] = PoseLandmark(
                    x=self.alpha * prev.x + (1 - self.alpha) * lm.x,
                    y=self.alpha * prev.y + (1 - self.alpha) * lm.y,
                    z=self.alpha * prev.z + (1 - self.alpha) * lm.z,
                    visibility=lm.visibility,
                )
        self._smoothed = out
        return out

    def reset(self) -> None:
        self._smoothed = {}


# ---------------------------------------------------------------------------
# MediaPipe wrapper
# ---------------------------------------------------------------------------
class PoseEstimator:
    """Thin wrapper around MediaPipe Pose Landmarker."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path
        self._mp_pose = None
        self._pose = None
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp  # type: ignore

            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:  # pragma: no cover - env dependent
            # MediaPipe may not be installed in some environments; the rest of
            # the pipeline still works with a synthetic/fallback pose.
            print(f"[warn] MediaPipe unavailable ({exc}); using fallback pose.")
            self._mp_pose = None
            self._pose = None

    def detect(self, frame_bgr: np.ndarray) -> FramePose:
        """Detect pose landmarks in a BGR frame."""
        if self._pose is None:
            return FramePose(landmarks={}, detected=False)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return FramePose(landmarks={}, detected=False)

        landmarks: Dict[int, PoseLandmark] = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = PoseLandmark(
                x=lm.x, y=lm.y, z=lm.z, visibility=float(lm.visibility)
            )
        return FramePose(landmarks=landmarks, detected=True)

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()