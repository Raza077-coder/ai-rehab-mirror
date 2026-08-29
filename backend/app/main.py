"""
AI Rehab Mirror — FastAPI backend.

Serves the pose-analysis pipeline over HTTP:
  - POST /api/analyze-frame : analyze a single uploaded frame (image)
  - POST /api/analyze-video : process an uploaded video and return annotated
    video + metrics summary
  - GET  /api/health        : health check
  - GET  /api/exercise/state: current exercise state machine info
"""
from __future__ import annotations

import os
import uuid
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .exercise_state_machine import ExerciseState, ShoulderAbductionMachine
from .pose_analysis import (
    LANDMARK_LEFT_ELBOW,
    LANDMARK_LEFT_HIP,
    LANDMARK_LEFT_SHOULDER,
    LANDMARK_RIGHT_ELBOW,
    LANDMARK_RIGHT_HIP,
    LANDMARK_RIGHT_SHOULDER,
    FramePose,
    LandmarkSmoother,
    PoseEstimator,
    left_right_symmetry,
    shoulder_abduction_angle,
    torso_alignment,
)
from .video_processing import process_video

app = FastAPI(
    title="AI Rehab Mirror API",
    description="Computer-vision physiotherapy rehabilitation assistant.",
    version="1.0.0",
)

# Allow the Vite dev server (and any origin in dev) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared estimator + smoother for the live frame endpoint.
_estimator = PoseEstimator()
_smoother = LandmarkSmoother(alpha=0.6)
_machine = ShoulderAbductionMachine()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class FrameAnalysis(BaseModel):
    detected: bool
    state: str
    abduction_angle: float
    torso_lean: float
    symmetry: float
    rep_count: int
    feedback: List[str]


class RepSummary(BaseModel):
    rep_number: int
    max_angle: float
    hold_seconds: float
    avg_torso_lean: float
    symmetry: float
    score: float
    quality: List[str]


class VideoAnalysisResult(BaseModel):
    rep_count: int
    average_score: float
    reps: List[RepSummary]
    video_url: str


class HealthResponse(BaseModel):
    status: str
    mediapipe_available: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        mediapipe_available=_estimator._pose is not None,
    )


@app.get("/api/exercise/state")
def exercise_state() -> dict:
    """Return the current exercise state machine configuration."""
    return {
        "states": [s.value for s in ExerciseState],
        "thresholds": {
            "raise": _machine.raise_threshold,
            "hold": _machine.hold_threshold,
            "lower": _machine.lower_threshold,
            "hold_min_seconds": _machine.hold_min_seconds,
            "target_hold_seconds": _machine.target_hold_seconds,
            "max_torso_lean": _machine.max_torso_lean,
            "symmetry_tolerance": _machine.symmetry_tolerance,
        },
    }


@app.post("/api/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)) -> FrameAnalysis:
    """Analyze a single uploaded image frame and return live metrics."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image uploaded.")

    pose = _estimator.detect(frame)
    if not pose.detected:
        _smoother.reset()
        return FrameAnalysis(
            detected=False,
            state=_machine.result.state.value,
            abduction_angle=0.0,
            torso_lean=0.0,
            symmetry=100.0,
            rep_count=_machine.result.rep_count,
            feedback=[],
        )

    smoothed = _smoother.update(pose.landmarks)
    pose = FramePose(landmarks=smoothed, detected=True)

    shoulder = pose.get(LANDMARK_RIGHT_SHOULDER)
    elbow = pose.get(LANDMARK_RIGHT_ELBOW)
    hip = pose.get(LANDMARK_RIGHT_HIP)
    l_shoulder = pose.get(LANDMARK_LEFT_SHOULDER)
    l_elbow = pose.get(LANDMARK_LEFT_ELBOW)
    l_hip = pose.get(LANDMARK_LEFT_HIP)
    r_hip = pose.get(LANDMARK_RIGHT_HIP)

    if not (shoulder and elbow and hip and l_shoulder and l_elbow and l_hip and r_hip):
        return FrameAnalysis(
            detected=True,
            state=_machine.result.state.value,
            abduction_angle=0.0,
            torso_lean=0.0,
            symmetry=100.0,
            rep_count=_machine.result.rep_count,
            feedback=[],
        )

    angle = shoulder_abduction_angle(shoulder, elbow, hip)
    lean = torso_alignment(l_shoulder, shoulder, l_hip, r_hip)
    left_angle = shoulder_abduction_angle(l_shoulder, l_elbow, l_hip)
    symmetry = left_right_symmetry(left_angle, angle)
    result = _machine.update(angle, lean, symmetry, dt=1.0 / 30.0)

    return FrameAnalysis(
        detected=True,
        state=result.state.value,
        abduction_angle=round(angle, 1),
        torso_lean=round(lean, 1),
        symmetry=round(symmetry, 1),
        rep_count=result.rep_count,
        feedback=result.feedback,
    )


@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)) -> VideoAnalysisResult:
    """Process an uploaded video and return the annotated video + summary."""
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    input_path = os.path.join(OUTPUT_DIR, f"input_{uuid.uuid4().hex}{suffix}")
    output_path = os.path.join(OUTPUT_DIR, f"output_{uuid.uuid4().hex}.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        summary = process_video(input_path, output_path, use_ffmpeg=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    video_url = f"/api/videos/{os.path.basename(output_path)}"
    return VideoAnalysisResult(
        rep_count=summary["rep_count"],
        average_score=summary["average_score"],
        reps=[RepSummary(**r) for r in summary["reps"]],
        video_url=video_url,
    )


@app.get("/api/videos/{filename}")
def get_video(filename: str) -> FileResponse:
    """Serve a processed video file."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/reset")
def reset_session() -> dict:
    """Reset the live exercise session."""
    _machine.reset()
    _smoother.reset()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)