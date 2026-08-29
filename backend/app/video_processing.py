"""
Video processing for AI Rehab Mirror.

Reads an input video, runs the pose-analysis pipeline frame by frame, renders
the pose skeleton + metrics overlay onto each frame, and exports the annotated
video using OpenCV (with optional FFmpeg re-encode for compatibility).
"""
from __future__ import annotations

import os
import subprocess
from typing import List

import cv2
import numpy as np

from .exercise_state_machine import ExerciseResult, ShoulderAbductionMachine
from .pose_analysis import (
    LANDMARK_LEFT_ELBOW,
    LANDMARK_LEFT_HIP,
    LANDMARK_LEFT_SHOULDER,
    LANDMARK_LEFT_WRIST,
    LANDMARK_RIGHT_ELBOW,
    LANDMARK_RIGHT_HIP,
    LANDMARK_RIGHT_SHOULDER,
    LANDMARK_RIGHT_WRIST,
    FramePose,
    LandmarkSmoother,
    PoseEstimator,
    left_right_symmetry,
    shoulder_abduction_angle,
    torso_alignment,
)

# Colors (BGR)
COLOR_SKELETON = (0, 255, 0)
COLOR_ANGLE = (255, 200, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_GOOD = (0, 200, 0)
COLOR_WARN = (0, 165, 255)


def _draw_skeleton(frame: np.ndarray, pose: FramePose) -> None:
    """Draw the pose skeleton and key joint angles on the frame."""
    h, w = frame.shape[:2]
    lm = pose.landmarks

    def pt(idx: int):
        p = lm.get(idx)
        if p is None:
            return None
        return (int(p.x * w), int(p.y * h))

    # Bone connections
    bones = [
        (LANDMARK_LEFT_SHOULDER, LANDMARK_LEFT_ELBOW),
        (LANDMARK_LEFT_ELBOW, LANDMARK_LEFT_WRIST),
        (LANDMARK_RIGHT_SHOULDER, LANDMARK_RIGHT_ELBOW),
        (LANDMARK_RIGHT_ELBOW, LANDMARK_RIGHT_WRIST),
        (LANDMARK_LEFT_SHOULDER, LANDMARK_RIGHT_SHOULDER),
        (LANDMARK_LEFT_SHOULDER, LANDMARK_LEFT_HIP),
        (LANDMARK_RIGHT_SHOULDER, LANDMARK_RIGHT_HIP),
        (LANDMARK_LEFT_HIP, LANDMARK_RIGHT_HIP),
    ]
    for a, b in bones:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, COLOR_SKELETON, 2)

    # Joints
    for idx in lm:
        p = pt(idx)
        if p:
            cv2.circle(frame, p, 4, COLOR_ANGLE, -1)


def _draw_metrics(
    frame: np.ndarray,
    angle: float,
    lean: float,
    symmetry: float,
    state: str,
    rep_count: int,
    feedback: List[str],
) -> None:
    """Draw the live metrics panel and feedback on the frame."""
    h, w = frame.shape[:2]
    panel_w = 320
    cv2.rectangle(frame, (0, 0), (panel_w, h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (panel_w, h), (60, 60, 60), 1)

    y = 30
    cv2.putText(frame, "AI REHAB MIRROR", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    y += 30
    cv2.putText(frame, f"State: {state}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ANGLE, 2)
    y += 28
    cv2.putText(frame, f"Abduction: {angle:.1f} deg", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    y += 28
    cv2.putText(frame, f"Torso lean: {lean:.1f} deg", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    y += 28
    cv2.putText(frame, f"Symmetry: {symmetry:.0f}%", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    y += 28
    cv2.putText(frame, f"Reps: {rep_count}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    y += 40

    cv2.putText(frame, "FEEDBACK", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ANGLE, 2)
    y += 26
    if not feedback:
        feedback = ["No feedback"]
    for msg in feedback[:5]:
        color = COLOR_GOOD if ("Good" in msg or "Excellent" in msg) else COLOR_ANGLE
        cv2.putText(frame, f"- {msg}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 24


def process_video(
    input_path: str,
    output_path: str,
    use_ffmpeg: bool = True,
) -> dict:
    """
    Process a video through the full pipeline and export an annotated video.

    Returns a summary dict with rep count, average score, and per-rep records.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    estimator = PoseEstimator()
    smoother = LandmarkSmoother(alpha=0.6)
    machine = ShoulderAbductionMachine()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pose = estimator.detect(frame)
        if pose.detected:
            smoothed = smoother.update(pose.landmarks)
            pose = FramePose(landmarks=smoothed, detected=True)

            shoulder = pose.get(LANDMARK_RIGHT_SHOULDER)
            elbow = pose.get(LANDMARK_RIGHT_ELBOW)
            hip = pose.get(LANDMARK_RIGHT_HIP)
            l_shoulder = pose.get(LANDMARK_LEFT_SHOULDER)
            l_elbow = pose.get(LANDMARK_LEFT_ELBOW)
            l_hip = pose.get(LANDMARK_LEFT_HIP)
            r_hip = pose.get(LANDMARK_RIGHT_HIP)

            if shoulder and elbow and hip and l_shoulder and l_elbow and l_hip and r_hip:
                angle = shoulder_abduction_angle(shoulder, elbow, hip)
                lean = torso_alignment(l_shoulder, shoulder, l_hip, r_hip)
                left_angle = shoulder_abduction_angle(l_shoulder, l_elbow, l_hip)
                symmetry = left_right_symmetry(left_angle, angle)
                result = machine.update(angle, lean, symmetry, dt=1.0 / fps)
                _draw_skeleton(frame, pose)
                _draw_metrics(
                    frame,
                    angle,
                    lean,
                    symmetry,
                    result.state.value,
                    result.rep_count,
                    result.feedback,
                )
        else:
            smoother.reset()

        writer.write(frame)

    cap.release()
    writer.release()
    estimator.close()

    # Optional FFmpeg re-encode for broad compatibility
    if use_ffmpeg and os.path.exists(output_path):
        _ffmpeg_reencode(output_path)

    return _summarize(machine.result)


def _ffmpeg_reencode(path: str) -> None:
    """Re-encode an mp4 with FFmpeg (H.264) for broad compatibility."""
    tmp = path + ".tmp.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", tmp,
            ],
            check=True,
            capture_output=True,
        )
        os.replace(tmp, path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # FFmpeg not available; keep the OpenCV-encoded file.
        if os.path.exists(tmp):
            os.remove(tmp)


def _summarize(result: ExerciseResult) -> dict:
    """Build a JSON-serializable summary from the exercise result."""
    reps = [
        {
            "rep_number": r.rep_number,
            "max_angle": r.max_angle,
            "hold_seconds": r.hold_seconds,
            "avg_torso_lean": r.avg_torso_lean,
            "symmetry": r.symmetry,
            "score": r.score,
            "quality": r.quality,
        }
        for r in result.reps
    ]
    avg_score = (
        sum(r.score for r in result.reps) / len(result.reps)
        if result.reps
        else 0.0
    )
    return {
        "rep_count": result.rep_count,
        "average_score": round(avg_score, 1),
        "reps": reps,
    }