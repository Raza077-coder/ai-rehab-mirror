"""
Exercise state machine for shoulder abduction rehabilitation.

States: Ready -> Raising -> Holding -> Lowering -> Completed Rep

The state machine consumes per-frame metrics (shoulder abduction angle, torso
lean, symmetry) and transitions between states based on thresholds, tracking
rep quality and producing real-time feedback messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ExerciseState(str, Enum):
    READY = "Ready"
    RAISING = "Raising"
    HOLDING = "Holding"
    LOWERING = "Lowering"
    COMPLETED = "Completed"


# Thresholds (degrees) tuned for shoulder abduction
RAISE_THRESHOLD = 60.0      # arm considered "raised" above this angle
HOLD_THRESHOLD = 80.0        # target hold zone
LOWER_THRESHOLD = 30.0       # arm considered "lowered" below this
HOLD_MIN_SECONDS = 1.0       # minimum hold duration for a quality rep
TARGET_HOLD_SECONDS = 2.0    # ideal hold duration
MAX_TORSO_LEAN = 15.0        # max acceptable torso lean (degrees)
SYMMETRY_TOLERANCE = 20.0    # max acceptable left/right angle difference


@dataclass
class RepRecord:
    """A single completed repetition with quality metrics."""

    rep_number: int
    max_angle: float
    hold_seconds: float
    avg_torso_lean: float
    symmetry: float
    score: float  # 0-100
    quality: List[str] = field(default_factory=list)


@dataclass
class ExerciseResult:
    """Accumulated state for a full exercise session."""

    state: ExerciseState = ExerciseState.READY
    rep_count: int = 0
    reps: List[RepRecord] = field(default_factory=list)
    current_hold_start: Optional[float] = None
    current_hold_seconds: float = 0.0
    current_max_angle: float = 0.0
    current_lean_sum: float = 0.0
    current_lean_count: int = 0
    current_symmetry_sum: float = 0.0
    current_symmetry_count: int = 0
    feedback: List[str] = field(default_factory=list)
    last_angle: float = 0.0


class ShoulderAbductionMachine:
    """State machine for the shoulder abduction exercise."""

    def __init__(
        self,
        raise_threshold: float = RAISE_THRESHOLD,
        hold_threshold: float = HOLD_THRESHOLD,
        lower_threshold: float = LOWER_THRESHOLD,
        hold_min_seconds: float = HOLD_MIN_SECONDS,
        target_hold_seconds: float = TARGET_HOLD_SECONDS,
        max_torso_lean: float = MAX_TORSO_LEAN,
        symmetry_tolerance: float = SYMMETRY_TOLERANCE,
    ) -> None:
        self.raise_threshold = raise_threshold
        self.hold_threshold = hold_threshold
        self.lower_threshold = lower_threshold
        self.hold_min_seconds = hold_min_seconds
        self.target_hold_seconds = target_hold_seconds
        self.max_torso_lean = max_torso_lean
        self.symmetry_tolerance = symmetry_tolerance
        self.result = ExerciseResult()

    # -- public API ---------------------------------------------------------
    def reset(self) -> None:
        self.result = ExerciseResult()

    def update(
        self,
        abduction_angle: float,
        torso_lean: float,
        symmetry: float,
        dt: float = 1.0 / 30.0,
    ) -> ExerciseResult:
        """
        Advance the state machine with one frame of metrics.

        Args:
            abduction_angle: shoulder abduction angle (degrees)
            torso_lean: torso lean from vertical (degrees)
            symmetry: left/right symmetry score (0-100)
            dt: time delta in seconds since the previous frame
        """
        r = self.result
        r.last_angle = abduction_angle
        r.feedback = []

        # Track running metrics for the current rep
        r.current_max_angle = max(r.current_max_angle, abduction_angle)
        r.current_lean_sum += torso_lean
        r.current_lean_count += 1
        r.current_symmetry_sum += symmetry
        r.current_symmetry_count += 1

        # --- state transitions --------------------------------------------
        if r.state == ExerciseState.READY:
            if abduction_angle >= self.raise_threshold:
                r.state = ExerciseState.RAISING
                r.current_hold_start = None
                r.current_hold_seconds = 0.0
                r.current_max_angle = abduction_angle
                r.current_lean_sum = torso_lean
                r.current_lean_count = 1
                r.current_symmetry_sum = symmetry
                r.current_symmetry_count = 1

        elif r.state == ExerciseState.RAISING:
            if abduction_angle >= self.hold_threshold:
                r.state = ExerciseState.HOLDING
                r.current_hold_start = 0.0
                r.current_hold_seconds = 0.0
            elif abduction_angle < self.lower_threshold:
                # Gave up before reaching the hold zone
                r.state = ExerciseState.READY
                self._reset_current_rep()

        elif r.state == ExerciseState.HOLDING:
            r.current_hold_seconds += dt
            if abduction_angle < self.hold_threshold - 10.0:
                r.state = ExerciseState.LOWERING
            elif r.current_hold_seconds >= self.target_hold_seconds:
                r.state = ExerciseState.LOWERING

        elif r.state == ExerciseState.LOWERING:
            if abduction_angle <= self.lower_threshold:
                self._complete_rep()
                r.state = ExerciseState.READY

        elif r.state == ExerciseState.COMPLETED:
            r.state = ExerciseState.READY

        # --- form feedback ------------------------------------------------
        self._evaluate_form(abduction_angle, torso_lean, symmetry)

        return r

    # -- internals ---------------------------------------------------------
    def _reset_current_rep(self) -> None:
        r = self.result
        r.current_hold_start = None
        r.current_hold_seconds = 0.0
        r.current_max_angle = 0.0
        r.current_lean_sum = 0.0
        r.current_lean_count = 0
        r.current_symmetry_sum = 0.0
        r.current_symmetry_count = 0

    def _complete_rep(self) -> None:
        r = self.result
        r.rep_count += 1
        avg_lean = (
            r.current_lean_sum / r.current_lean_count
            if r.current_lean_count
            else 0.0
        )
        avg_sym = (
            r.current_symmetry_sum / r.current_symmetry_count
            if r.current_symmetry_count
            else 100.0
        )
        score = self._score_rep(
            max_angle=r.current_max_angle,
            hold_seconds=r.current_hold_seconds,
            avg_lean=avg_lean,
            symmetry=avg_sym,
        )
        quality_msgs = self._quality_feedback(
            max_angle=r.current_max_angle,
            hold_seconds=r.current_hold_seconds,
            avg_lean=avg_lean,
            symmetry=avg_sym,
        )
        r.reps.append(
            RepRecord(
                rep_number=r.rep_count,
                max_angle=round(r.current_max_angle, 1),
                hold_seconds=round(r.current_hold_seconds, 2),
                avg_torso_lean=round(avg_lean, 1),
                symmetry=round(avg_sym, 1),
                score=round(score, 1),
                quality=quality_msgs,
            )
        )
        r.feedback.append("Good repetition")
        self._reset_current_rep()

    def _evaluate_form(
        self, abduction_angle: float, torso_lean: float, symmetry: float
    ) -> None:
        r = self.result
        if r.state == ExerciseState.READY:
            return
        if r.state == ExerciseState.RAISING:
            if abduction_angle < self.raise_threshold:
                r.feedback.append("Raise your arm higher")
        if r.state == ExerciseState.HOLDING:
            if r.current_hold_seconds < self.hold_min_seconds:
                r.feedback.append("Hold position")
        if torso_lean > self.max_torso_lean:
            r.feedback.append("Keep your torso straight")
        if symmetry < (100.0 - self.symmetry_tolerance):
            r.feedback.append("Keep both sides even")

    def _score_rep(
        self, max_angle: float, hold_seconds: float, avg_lean: float, symmetry: float
    ) -> float:
        """Composite 0-100 movement score for a completed rep."""
        angle_score = min(100.0, (max_angle / self.hold_threshold) * 100.0)
        hold_score = min(100.0, (hold_seconds / self.target_hold_seconds) * 100.0)
        lean_score = max(0.0, 100.0 - (avg_lean / self.max_torso_lean) * 100.0)
        sym_score = symmetry
        return 0.4 * angle_score + 0.3 * hold_score + 0.15 * lean_score + 0.15 * sym_score

    def _quality_feedback(
        self, max_angle: float, hold_seconds: float, avg_lean: float, symmetry: float
    ) -> List[str]:
        msgs: List[str] = []
        if max_angle < self.hold_threshold:
            msgs.append("Increase range of motion")
        if hold_seconds < self.hold_min_seconds:
            msgs.append("Hold longer at the top")
        if avg_lean > self.max_torso_lean:
            msgs.append("Reduce torso lean")
        if symmetry < (100.0 - self.symmetry_tolerance):
            msgs.append("Improve left/right symmetry")
        if not msgs:
            msgs.append("Excellent form")
        return msgs