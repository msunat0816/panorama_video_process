#!/usr/bin/env python3
"""Generate panorama memory probe videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
MODE_TWO_TURN = "two_turn"
MODE_AWAY_TARGET_AWAY_TARGET = "away_target_away_target"
MODES = (MODE_TWO_TURN, MODE_AWAY_TARGET_AWAY_TARGET)


@dataclass(frozen=True)
class ProbeConfig:
    output_dir: Path
    run_timestamp: str
    mode: str
    seed: int
    variants_per_video: int
    target_yaw_deg_range: tuple[float, float]
    away_yaw_delta_deg_range: tuple[float, float]
    away_hold_sec_range: tuple[float, float]
    entry_away_hold_sec_range: tuple[float, float]
    initial_target_min_sec: float
    middle_target_hold_sec: float
    prediction_window_sec: float
    turn_speed_deg_per_sec: float
    target_fov_x_deg_range: tuple[float, float]
    away_fov_x_deg_range: tuple[float, float]
    pitch_deg: float
    roll_deg: float
    overwrite: bool
    strict: bool


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    video_id: str
    fps: float
    num_frames: int
    width: int
    height: int
    duration_sec: float


@dataclass(frozen=True)
class TwoTurnSchedule:
    target_hold_frames: int
    turn_away_frames: int
    away_hold_frames: int
    turn_back_frames: int
    final_target_hold_frames: int
    target_hold_end_frame: int
    turn_away_start_frame: int
    turn_away_end_frame: int
    away_hold_start_frame: int
    away_hold_end_frame: int
    turn_back_start_frame: int
    turn_back_end_frame: int


@dataclass(frozen=True)
class AwayTargetAwayTargetSchedule:
    entry_away_hold_frames: int
    entry_turn_to_target_frames: int
    middle_target_hold_frames: int
    turn_to_exit_away_frames: int
    exit_away_hold_frames: int
    turn_back_frames: int
    final_target_hold_frames: int
    entry_away_hold_end_frame: int
    entry_turn_to_target_start_frame: int
    entry_turn_to_target_end_frame: int
    middle_target_hold_start_frame: int
    middle_target_hold_end_frame: int
    turn_to_exit_away_start_frame: int
    turn_to_exit_away_end_frame: int
    exit_away_hold_start_frame: int
    exit_away_hold_end_frame: int
    turn_back_start_frame: int
    turn_back_end_frame: int


def parse_range(value: str, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{name} must be formatted as low,high")
    low, high = float(parts[0]), float(parts[1])
    if low > high:
        raise argparse.ArgumentTypeError(f"{name} low must be <= high")
    return low, high


def current_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


def collect_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def stable_rng(seed_text: str) -> random.Random:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return random.Random(local_seed)


def read_video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        num_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    finally:
        cap.release()

    if fps <= 0:
        raise RuntimeError(f"Video has invalid FPS: {path}")
    if num_frames <= 1 or width <= 0 or height <= 0:
        raise RuntimeError(f"Video has invalid metadata: {path}")

    duration_sec = num_frames / fps
    return VideoInfo(
        path=path,
        video_id=path.stem,
        fps=fps,
        num_frames=num_frames,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def sec_to_frames(sec: float, fps: float, *, min_frames: int = 0) -> int:
    return max(min_frames, int(round(sec * fps)))


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def lerp(start: float, end: float, weight: float) -> float:
    return start + (end - start) * weight


def normalize_yaw_deg(yaw_deg: float) -> float:
    return ((yaw_deg + 180.0) % 360.0) - 180.0


def shortest_yaw_delta_deg(start: float, end: float) -> float:
    return normalize_yaw_deg(end - start)


def lerp_yaw(start: float, end: float, weight: float) -> float:
    return normalize_yaw_deg(start + shortest_yaw_delta_deg(start, end) * weight)


def sample_away_yaw(
    target_yaw_deg: float,
    rng: random.Random,
    config: ProbeConfig,
) -> tuple[float, float]:
    delta = rng.uniform(*config.away_yaw_delta_deg_range)
    direction = rng.choice((-1.0, 1.0))
    signed_delta = direction * delta
    return normalize_yaw_deg(target_yaw_deg + signed_delta), signed_delta


def sample_two_turn_schedule(
    info: VideoInfo,
    rng: random.Random,
    config: ProbeConfig,
    away_yaw_delta_deg: float,
) -> TwoTurnSchedule:
    turn_duration_sec = abs(away_yaw_delta_deg) / config.turn_speed_deg_per_sec
    if turn_duration_sec >= config.prediction_window_sec:
        raise ValueError(
            "Turn-back duration must be shorter than --prediction-window-sec. "
            "Increase --turn-speed-deg-per-sec or --prediction-window-sec."
        )

    turn_away_frames = sec_to_frames(turn_duration_sec, info.fps, min_frames=1)
    turn_back_frames = sec_to_frames(turn_duration_sec, info.fps, min_frames=1)
    prediction_window_frames = sec_to_frames(config.prediction_window_sec, info.fps, min_frames=1)
    final_target_hold_frames = prediction_window_frames - turn_back_frames
    if final_target_hold_frames < 1:
        raise ValueError(
            "Prediction window is too short to include both turn-back and final target hold frames."
        )

    min_target_hold_frames = sec_to_frames(config.initial_target_min_sec, info.fps, min_frames=1)
    minimum_required_frames = min_target_hold_frames + turn_away_frames + prediction_window_frames
    if minimum_required_frames > info.num_frames:
        raise ValueError(
            "Video is too short for the requested initial target hold, turn-away, "
            "and prediction window durations."
        )

    max_away_hold_frames = info.num_frames - minimum_required_frames
    away_hold_low_frames = sec_to_frames(config.away_hold_sec_range[0], info.fps)
    away_hold_high_frames = sec_to_frames(config.away_hold_sec_range[1], info.fps)
    if max_away_hold_frames >= away_hold_low_frames:
        away_hold_upper = min(away_hold_high_frames, max_away_hold_frames)
        away_hold_frames = rng.randint(away_hold_low_frames, away_hold_upper)
    else:
        away_hold_frames = max_away_hold_frames

    target_hold_frames = (
        info.num_frames
        - turn_away_frames
        - away_hold_frames
        - prediction_window_frames
    )

    target_hold_end_frame = target_hold_frames
    turn_away_start_frame = target_hold_end_frame
    turn_away_end_frame = turn_away_start_frame + turn_away_frames
    away_hold_start_frame = turn_away_end_frame
    away_hold_end_frame = away_hold_start_frame + away_hold_frames
    turn_back_start_frame = away_hold_end_frame
    turn_back_end_frame = turn_back_start_frame + turn_back_frames
    final_target_hold_frames = max(0, info.num_frames - turn_back_end_frame)

    return TwoTurnSchedule(
        target_hold_frames=target_hold_frames,
        turn_away_frames=turn_away_frames,
        away_hold_frames=away_hold_frames,
        turn_back_frames=turn_back_frames,
        final_target_hold_frames=final_target_hold_frames,
        target_hold_end_frame=target_hold_end_frame,
        turn_away_start_frame=turn_away_start_frame,
        turn_away_end_frame=turn_away_end_frame,
        away_hold_start_frame=away_hold_start_frame,
        away_hold_end_frame=away_hold_end_frame,
        turn_back_start_frame=turn_back_start_frame,
        turn_back_end_frame=turn_back_end_frame,
    )


def sample_away_target_away_target_schedule(
    info: VideoInfo,
    rng: random.Random,
    config: ProbeConfig,
    away_yaw_delta_deg: float,
) -> AwayTargetAwayTargetSchedule:
    turn_duration_sec = abs(away_yaw_delta_deg) / config.turn_speed_deg_per_sec
    if turn_duration_sec >= config.prediction_window_sec:
        raise ValueError(
            "Final turn-back duration must be shorter than --prediction-window-sec. "
            "Increase --turn-speed-deg-per-sec or --prediction-window-sec."
        )

    entry_turn_to_target_frames = sec_to_frames(turn_duration_sec, info.fps, min_frames=1)
    turn_to_exit_away_frames = sec_to_frames(turn_duration_sec, info.fps, min_frames=1)
    turn_back_frames = sec_to_frames(turn_duration_sec, info.fps, min_frames=1)
    prediction_window_frames = sec_to_frames(config.prediction_window_sec, info.fps, min_frames=1)
    final_target_hold_frames = prediction_window_frames - turn_back_frames
    if final_target_hold_frames < 1:
        raise ValueError(
            "Prediction window is too short to include both turn-back and final target hold frames."
        )

    middle_target_hold_min_frames = sec_to_frames(
        config.middle_target_hold_sec, info.fps, min_frames=1
    )
    entry_away_hold_low_frames = sec_to_frames(
        config.entry_away_hold_sec_range[0], info.fps, min_frames=1
    )
    entry_away_hold_high_frames = sec_to_frames(
        config.entry_away_hold_sec_range[1], info.fps, min_frames=entry_away_hold_low_frames
    )
    exit_away_hold_low_frames = sec_to_frames(config.away_hold_sec_range[0], info.fps, min_frames=1)
    exit_away_hold_high_frames = sec_to_frames(
        config.away_hold_sec_range[1], info.fps, min_frames=exit_away_hold_low_frames
    )

    available_hold_frames = (
        info.num_frames
        - prediction_window_frames
        - entry_turn_to_target_frames
        - turn_to_exit_away_frames
    )
    minimum_hold_frames = (
        entry_away_hold_low_frames + exit_away_hold_low_frames + middle_target_hold_min_frames
    )
    if available_hold_frames < minimum_hold_frames:
        raise ValueError(
            "Video is too short for the requested entry away hold minimum, exit away "
            "hold minimum, middle target hold minimum, turns, and prediction window."
        )

    exit_away_hold_upper_frames = min(
        exit_away_hold_high_frames,
        available_hold_frames - entry_away_hold_low_frames - middle_target_hold_min_frames,
    )
    exit_away_hold_frames = rng.randint(exit_away_hold_low_frames, exit_away_hold_upper_frames)

    entry_away_hold_upper_frames = min(
        entry_away_hold_high_frames,
        available_hold_frames - exit_away_hold_frames - middle_target_hold_min_frames,
    )
    entry_away_hold_frames = rng.randint(
        entry_away_hold_low_frames, entry_away_hold_upper_frames
    )

    middle_target_hold_frames = (
        available_hold_frames - exit_away_hold_frames - entry_away_hold_frames
    )

    entry_away_hold_end_frame = entry_away_hold_frames
    entry_turn_to_target_start_frame = entry_away_hold_end_frame
    entry_turn_to_target_end_frame = (
        entry_turn_to_target_start_frame + entry_turn_to_target_frames
    )
    middle_target_hold_start_frame = entry_turn_to_target_end_frame
    middle_target_hold_end_frame = middle_target_hold_start_frame + middle_target_hold_frames
    turn_to_exit_away_start_frame = middle_target_hold_end_frame
    turn_to_exit_away_end_frame = turn_to_exit_away_start_frame + turn_to_exit_away_frames
    exit_away_hold_start_frame = turn_to_exit_away_end_frame
    exit_away_hold_end_frame = exit_away_hold_start_frame + exit_away_hold_frames
    turn_back_start_frame = exit_away_hold_end_frame
    turn_back_end_frame = turn_back_start_frame + turn_back_frames
    final_target_hold_frames = max(0, info.num_frames - turn_back_end_frame)

    return AwayTargetAwayTargetSchedule(
        entry_away_hold_frames=entry_away_hold_frames,
        entry_turn_to_target_frames=entry_turn_to_target_frames,
        middle_target_hold_frames=middle_target_hold_frames,
        turn_to_exit_away_frames=turn_to_exit_away_frames,
        exit_away_hold_frames=exit_away_hold_frames,
        turn_back_frames=turn_back_frames,
        final_target_hold_frames=final_target_hold_frames,
        entry_away_hold_end_frame=entry_away_hold_end_frame,
        entry_turn_to_target_start_frame=entry_turn_to_target_start_frame,
        entry_turn_to_target_end_frame=entry_turn_to_target_end_frame,
        middle_target_hold_start_frame=middle_target_hold_start_frame,
        middle_target_hold_end_frame=middle_target_hold_end_frame,
        turn_to_exit_away_start_frame=turn_to_exit_away_start_frame,
        turn_to_exit_away_end_frame=turn_to_exit_away_end_frame,
        exit_away_hold_start_frame=exit_away_hold_start_frame,
        exit_away_hold_end_frame=exit_away_hold_end_frame,
        turn_back_start_frame=turn_back_start_frame,
        turn_back_end_frame=turn_back_end_frame,
    )


def project_view(
    frame_bgr: np.ndarray,
    yaw_deg: float,
    fov_x_deg: float,
    config: ProbeConfig,
) -> np.ndarray:
    try:
        import equilib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'equilib'. Install it with `pip install equilib` before running."
        ) from exc

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_chw = np.transpose(frame_rgb, (2, 0, 1))
    projected = equilib.equi2pers(
        equi=frame_chw,
        rots={
            "yaw": math.radians(yaw_deg),
            "pitch": math.radians(config.pitch_deg),
            "roll": math.radians(config.roll_deg),
        },
        height=frame_bgr.shape[0],
        width=frame_bgr.shape[1],
        fov_x=fov_x_deg,
    )

    projected_hwc = np.asarray(projected)
    if projected_hwc.ndim == 3 and projected_hwc.shape[0] in (1, 3, 4):
        projected_hwc = np.transpose(projected_hwc, (1, 2, 0))
    projected_hwc = np.clip(projected_hwc, 0, 255).astype(np.uint8)
    if projected_hwc.shape[2] == 4:
        projected_hwc = projected_hwc[:, :, :3]
    return cv2.cvtColor(projected_hwc, cv2.COLOR_RGB2BGR)


def make_metadata(
    *,
    info: VideoInfo,
    input_root: Path,
    output_path: Path,
    variant_index: int,
    config: ProbeConfig,
    local_seed_text: str,
    mode: str,
    target_yaw_deg: float,
    away_yaw_deg: float,
    away_yaw_delta_deg: float,
    target_fov_x_deg: float,
    away_fov_x_deg: float,
    schedule: TwoTurnSchedule,
) -> dict:
    aspect_ratio = info.width / info.height if info.height else None
    source_video = str(info.path.relative_to(input_root)) if input_root.is_dir() else str(info.path)
    sample_id = f"{config.run_timestamp}_{info.video_id}_{mode}_v{variant_index:03d}_seed{config.seed}"
    away_hold_clamped = schedule.away_hold_frames < sec_to_frames(config.away_hold_sec_range[0], info.fps)

    metadata = {
        "sample_id": sample_id,
        "source_video": source_video,
        "output_video": str(output_path),
        "mode": mode,
        "video_id": info.video_id,
        "variant_index": variant_index,
        "seed": config.seed,
        "run_timestamp": config.run_timestamp,
        "local_seed_text": local_seed_text,
        "duration_sec": info.duration_sec,
        "fps": info.fps,
        "num_frames": info.num_frames,
        "input_width": info.width,
        "input_height": info.height,
        "output_width": info.width,
        "output_height": info.height,
        "input_aspect_ratio": aspect_ratio,
        "aspect_ratio_warning": bool(aspect_ratio and abs(aspect_ratio - 2.0) > 0.05),
        "yaw_deg": target_yaw_deg,
        "target_yaw_deg": target_yaw_deg,
        "target_yaw_deg_range": list(config.target_yaw_deg_range),
        "away_yaw_deg": away_yaw_deg,
        "away_yaw_delta_deg": away_yaw_delta_deg,
        "pitch_deg": config.pitch_deg,
        "roll_deg": config.roll_deg,
        "fov_x_deg": target_fov_x_deg,
        "target_fov_x_deg": target_fov_x_deg,
        "target_fov_x_deg_range": list(config.target_fov_x_deg_range),
        "away_fov_x_deg": away_fov_x_deg,
        "away_fov_x_deg_range": list(config.away_fov_x_deg_range),
        "transition": "two_turn_smoothstep_camera_avg_speed",
        "away_yaw_delta_deg_range": list(config.away_yaw_delta_deg_range),
        "away_hold_sec_range": list(config.away_hold_sec_range),
        "away_hold_clamped": away_hold_clamped,
        "initial_target_min_sec": config.initial_target_min_sec,
        "prediction_window_sec": config.prediction_window_sec,
        "turn_speed_deg_per_sec": config.turn_speed_deg_per_sec,
        "target_hold_frames": schedule.target_hold_frames,
        "target_hold_sec": schedule.target_hold_frames / info.fps,
        "target_hold_end_frame": schedule.target_hold_end_frame,
        "turn_away_frames": schedule.turn_away_frames,
        "turn_away_duration_sec": schedule.turn_away_frames / info.fps,
        "turn_away_start_frame": schedule.turn_away_start_frame,
        "turn_away_end_frame": schedule.turn_away_end_frame,
        "away_hold_frames": schedule.away_hold_frames,
        "away_hold_sec": schedule.away_hold_frames / info.fps,
        "away_hold_start_frame": schedule.away_hold_start_frame,
        "away_hold_end_frame": schedule.away_hold_end_frame,
        "turn_back_frames": schedule.turn_back_frames,
        "turn_back_duration_sec": schedule.turn_back_frames / info.fps,
        "turn_back_start_frame": schedule.turn_back_start_frame,
        "turn_back_end_frame": schedule.turn_back_end_frame,
        "final_target_hold_frames": schedule.final_target_hold_frames,
        "final_target_hold_sec": schedule.final_target_hold_frames / info.fps,
        "prediction_window_start_frame": schedule.turn_back_start_frame,
        "prediction_window_start_sec": schedule.turn_back_start_frame / info.fps,
        "projection_library": "equilib",
        "projection_api": "equi2pers",
    }
    return metadata


def make_away_target_away_target_metadata(
    *,
    info: VideoInfo,
    input_root: Path,
    output_path: Path,
    variant_index: int,
    config: ProbeConfig,
    local_seed_text: str,
    target_yaw_deg: float,
    away_yaw_deg: float,
    away_yaw_delta_deg: float,
    target_fov_x_deg: float,
    away_fov_x_deg: float,
    schedule: AwayTargetAwayTargetSchedule,
) -> dict:
    aspect_ratio = info.width / info.height if info.height else None
    source_video = str(info.path.relative_to(input_root)) if input_root.is_dir() else str(info.path)
    mode = MODE_AWAY_TARGET_AWAY_TARGET
    sample_id = f"{config.run_timestamp}_{info.video_id}_{mode}_v{variant_index:03d}_seed{config.seed}"
    entry_away_hold_low_frames = sec_to_frames(config.entry_away_hold_sec_range[0], info.fps)
    entry_away_hold_high_frames = sec_to_frames(config.entry_away_hold_sec_range[1], info.fps)
    exit_away_hold_high_frames = sec_to_frames(config.away_hold_sec_range[1], info.fps)
    middle_target_hold_min_frames = sec_to_frames(config.middle_target_hold_sec, info.fps)
    available_hold_frames = (
        schedule.entry_away_hold_frames
        + schedule.exit_away_hold_frames
        + schedule.middle_target_hold_frames
    )
    exit_away_hold_feasible_upper_frames = min(
        exit_away_hold_high_frames,
        available_hold_frames - entry_away_hold_low_frames - middle_target_hold_min_frames,
    )
    entry_away_hold_feasible_upper_frames = min(
        entry_away_hold_high_frames,
        available_hold_frames - schedule.exit_away_hold_frames - middle_target_hold_min_frames,
    )
    entry_away_hold_clamped = entry_away_hold_feasible_upper_frames < entry_away_hold_high_frames
    exit_away_hold_clamped = exit_away_hold_feasible_upper_frames < exit_away_hold_high_frames

    return {
        "sample_id": sample_id,
        "source_video": source_video,
        "output_video": str(output_path),
        "mode": mode,
        "away_pair_mode": "same",
        "video_id": info.video_id,
        "variant_index": variant_index,
        "seed": config.seed,
        "run_timestamp": config.run_timestamp,
        "local_seed_text": local_seed_text,
        "duration_sec": info.duration_sec,
        "fps": info.fps,
        "num_frames": info.num_frames,
        "input_width": info.width,
        "input_height": info.height,
        "output_width": info.width,
        "output_height": info.height,
        "input_aspect_ratio": aspect_ratio,
        "aspect_ratio_warning": bool(aspect_ratio and abs(aspect_ratio - 2.0) > 0.05),
        "yaw_deg": target_yaw_deg,
        "target_yaw_deg": target_yaw_deg,
        "target_yaw_deg_range": list(config.target_yaw_deg_range),
        "target_fov_x_deg": target_fov_x_deg,
        "target_fov_x_deg_range": list(config.target_fov_x_deg_range),
        "entry_away_yaw_deg": away_yaw_deg,
        "entry_away_fov_x_deg": away_fov_x_deg,
        "entry_away_delta_deg": away_yaw_delta_deg,
        "exit_away_yaw_deg": away_yaw_deg,
        "exit_away_fov_x_deg": away_fov_x_deg,
        "exit_away_delta_deg": away_yaw_delta_deg,
        "entry_exit_away_same": True,
        "entry_exit_away_distance_deg": 0.0,
        "away_yaw_deg": away_yaw_deg,
        "away_yaw_delta_deg": away_yaw_delta_deg,
        "away_fov_x_deg": away_fov_x_deg,
        "away_yaw_delta_deg_range": list(config.away_yaw_delta_deg_range),
        "away_fov_x_deg_range": list(config.away_fov_x_deg_range),
        "pitch_deg": config.pitch_deg,
        "roll_deg": config.roll_deg,
        "fov_x_deg": target_fov_x_deg,
        "transition": "away_target_away_target_same_smoothstep_camera_avg_speed",
        "entry_away_hold_sec_range": list(config.entry_away_hold_sec_range),
        "away_hold_sec_range": list(config.away_hold_sec_range),
        "entry_away_hold_clamped": entry_away_hold_clamped,
        "exit_away_hold_clamped": exit_away_hold_clamped,
        "away_hold_clamped": entry_away_hold_clamped or exit_away_hold_clamped,
        "middle_target_hold_min_sec": config.middle_target_hold_sec,
        "middle_target_hold_sec_requested": config.middle_target_hold_sec,
        "prediction_window_sec": config.prediction_window_sec,
        "turn_speed_deg_per_sec": config.turn_speed_deg_per_sec,
        "entry_away_hold_frames": schedule.entry_away_hold_frames,
        "entry_away_hold_sec": schedule.entry_away_hold_frames / info.fps,
        "entry_away_hold_end_frame": schedule.entry_away_hold_end_frame,
        "entry_turn_to_target_frames": schedule.entry_turn_to_target_frames,
        "entry_turn_to_target_duration_sec": schedule.entry_turn_to_target_frames / info.fps,
        "entry_turn_to_target_start_frame": schedule.entry_turn_to_target_start_frame,
        "entry_turn_to_target_end_frame": schedule.entry_turn_to_target_end_frame,
        "middle_target_hold_frames": schedule.middle_target_hold_frames,
        "middle_target_hold_sec": schedule.middle_target_hold_frames / info.fps,
        "middle_target_hold_start_frame": schedule.middle_target_hold_start_frame,
        "middle_target_hold_end_frame": schedule.middle_target_hold_end_frame,
        "turn_to_exit_away_frames": schedule.turn_to_exit_away_frames,
        "turn_to_exit_away_duration_sec": schedule.turn_to_exit_away_frames / info.fps,
        "turn_to_exit_away_start_frame": schedule.turn_to_exit_away_start_frame,
        "turn_to_exit_away_end_frame": schedule.turn_to_exit_away_end_frame,
        "exit_away_hold_frames": schedule.exit_away_hold_frames,
        "exit_away_hold_sec": schedule.exit_away_hold_frames / info.fps,
        "exit_away_hold_start_frame": schedule.exit_away_hold_start_frame,
        "exit_away_hold_end_frame": schedule.exit_away_hold_end_frame,
        "turn_back_frames": schedule.turn_back_frames,
        "turn_back_duration_sec": schedule.turn_back_frames / info.fps,
        "turn_back_start_frame": schedule.turn_back_start_frame,
        "turn_back_end_frame": schedule.turn_back_end_frame,
        "final_target_hold_frames": schedule.final_target_hold_frames,
        "final_target_hold_sec": schedule.final_target_hold_frames / info.fps,
        "prediction_window_start_frame": schedule.turn_back_start_frame,
        "prediction_window_start_sec": schedule.turn_back_start_frame / info.fps,
        "projection_library": "equilib",
        "projection_api": "equi2pers",
    }


def write_two_turn_probe_video(
    info: VideoInfo,
    input_root: Path,
    variant_index: int,
    config: ProbeConfig,
) -> dict:
    mode = config.mode
    local_seed_text = (
        f"{config.seed}|{info.path.relative_to(input_root) if input_root.is_dir() else info.path}|"
        f"{variant_index}|{mode}"
    )
    rng = stable_rng(local_seed_text)
    target_yaw_deg = normalize_yaw_deg(rng.uniform(*config.target_yaw_deg_range))
    away_yaw_deg, away_yaw_delta_deg = sample_away_yaw(target_yaw_deg, rng, config)
    target_fov_x_deg = rng.uniform(*config.target_fov_x_deg_range)
    away_fov_x_deg = rng.uniform(*config.away_fov_x_deg_range)
    schedule = sample_two_turn_schedule(info, rng, config, away_yaw_delta_deg)

    output_path = (
        config.output_dir
        / f"{config.run_timestamp}_{info.video_id}_{mode}_v{variant_index:03d}_seed{config.seed}.mp4"
    )
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace: {output_path}")

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {info.path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, info.fps, (info.width, info.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {output_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index < schedule.target_hold_end_frame:
                frame_yaw_deg = target_yaw_deg
                frame_fov_x_deg = target_fov_x_deg
            elif frame_index < schedule.turn_away_end_frame:
                transition_index = frame_index - schedule.turn_away_start_frame
                progress = transition_index / max(1, schedule.turn_away_frames - 1)
                weight = smoothstep(progress)
                frame_yaw_deg = lerp_yaw(target_yaw_deg, away_yaw_deg, weight)
                frame_fov_x_deg = lerp(target_fov_x_deg, away_fov_x_deg, weight)
            elif frame_index < schedule.away_hold_end_frame:
                frame_yaw_deg = away_yaw_deg
                frame_fov_x_deg = away_fov_x_deg
            elif frame_index < schedule.turn_back_end_frame:
                transition_index = frame_index - schedule.turn_back_start_frame
                progress = transition_index / max(1, schedule.turn_back_frames - 1)
                weight = smoothstep(progress)
                frame_yaw_deg = lerp_yaw(away_yaw_deg, target_yaw_deg, weight)
                frame_fov_x_deg = lerp(away_fov_x_deg, target_fov_x_deg, weight)
            else:
                frame_yaw_deg = target_yaw_deg
                frame_fov_x_deg = target_fov_x_deg
            output_frame = project_view(frame, frame_yaw_deg, frame_fov_x_deg, config)
            writer.write(output_frame)
            frame_index += 1
    finally:
        writer.release()
        cap.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames decoded from video: {info.path}")

    metadata = make_metadata(
        info=info,
        input_root=input_root,
        output_path=output_path,
        variant_index=variant_index,
        config=config,
        local_seed_text=local_seed_text,
        mode=mode,
        target_yaw_deg=target_yaw_deg,
        away_yaw_deg=away_yaw_deg,
        away_yaw_delta_deg=away_yaw_delta_deg,
        target_fov_x_deg=target_fov_x_deg,
        away_fov_x_deg=away_fov_x_deg,
        schedule=schedule,
    )
    metadata["decoded_frames"] = frame_index
    metadata["frame_count_warning"] = frame_index != info.num_frames
    return metadata


def write_away_target_away_target_probe_video(
    info: VideoInfo,
    input_root: Path,
    variant_index: int,
    config: ProbeConfig,
) -> dict:
    mode = config.mode
    local_seed_text = (
        f"{config.seed}|{info.path.relative_to(input_root) if input_root.is_dir() else info.path}|"
        f"{variant_index}|{mode}|same"
    )
    rng = stable_rng(local_seed_text)
    target_yaw_deg = normalize_yaw_deg(rng.uniform(*config.target_yaw_deg_range))
    away_yaw_deg, away_yaw_delta_deg = sample_away_yaw(target_yaw_deg, rng, config)
    target_fov_x_deg = rng.uniform(*config.target_fov_x_deg_range)
    away_fov_x_deg = rng.uniform(*config.away_fov_x_deg_range)
    schedule = sample_away_target_away_target_schedule(info, rng, config, away_yaw_delta_deg)

    output_path = (
        config.output_dir
        / f"{config.run_timestamp}_{info.video_id}_{mode}_same_v{variant_index:03d}_seed{config.seed}.mp4"
    )
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace: {output_path}")

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {info.path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, info.fps, (info.width, info.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {output_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index < schedule.entry_away_hold_end_frame:
                frame_yaw_deg = away_yaw_deg
                frame_fov_x_deg = away_fov_x_deg
            elif frame_index < schedule.entry_turn_to_target_end_frame:
                transition_index = frame_index - schedule.entry_turn_to_target_start_frame
                progress = transition_index / max(1, schedule.entry_turn_to_target_frames - 1)
                weight = smoothstep(progress)
                frame_yaw_deg = lerp_yaw(away_yaw_deg, target_yaw_deg, weight)
                frame_fov_x_deg = lerp(away_fov_x_deg, target_fov_x_deg, weight)
            elif frame_index < schedule.middle_target_hold_end_frame:
                frame_yaw_deg = target_yaw_deg
                frame_fov_x_deg = target_fov_x_deg
            elif frame_index < schedule.turn_to_exit_away_end_frame:
                transition_index = frame_index - schedule.turn_to_exit_away_start_frame
                progress = transition_index / max(1, schedule.turn_to_exit_away_frames - 1)
                weight = smoothstep(progress)
                frame_yaw_deg = lerp_yaw(target_yaw_deg, away_yaw_deg, weight)
                frame_fov_x_deg = lerp(target_fov_x_deg, away_fov_x_deg, weight)
            elif frame_index < schedule.exit_away_hold_end_frame:
                frame_yaw_deg = away_yaw_deg
                frame_fov_x_deg = away_fov_x_deg
            elif frame_index < schedule.turn_back_end_frame:
                transition_index = frame_index - schedule.turn_back_start_frame
                progress = transition_index / max(1, schedule.turn_back_frames - 1)
                weight = smoothstep(progress)
                frame_yaw_deg = lerp_yaw(away_yaw_deg, target_yaw_deg, weight)
                frame_fov_x_deg = lerp(away_fov_x_deg, target_fov_x_deg, weight)
            else:
                frame_yaw_deg = target_yaw_deg
                frame_fov_x_deg = target_fov_x_deg
            output_frame = project_view(frame, frame_yaw_deg, frame_fov_x_deg, config)
            writer.write(output_frame)
            frame_index += 1
    finally:
        writer.release()
        cap.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames decoded from video: {info.path}")

    metadata = make_away_target_away_target_metadata(
        info=info,
        input_root=input_root,
        output_path=output_path,
        variant_index=variant_index,
        config=config,
        local_seed_text=local_seed_text,
        target_yaw_deg=target_yaw_deg,
        away_yaw_deg=away_yaw_deg,
        away_yaw_delta_deg=away_yaw_delta_deg,
        target_fov_x_deg=target_fov_x_deg,
        away_fov_x_deg=away_fov_x_deg,
        schedule=schedule,
    )
    metadata["decoded_frames"] = frame_index
    metadata["frame_count_warning"] = frame_index != info.num_frames
    return metadata


def write_probe_video(
    info: VideoInfo,
    input_root: Path,
    variant_index: int,
    config: ProbeConfig,
) -> dict:
    if config.mode == MODE_TWO_TURN:
        return write_two_turn_probe_video(info, input_root, variant_index, config)
    if config.mode == MODE_AWAY_TARGET_AWAY_TARGET:
        return write_away_target_away_target_probe_video(info, input_root, variant_index, config)
    raise ValueError(f"Unsupported mode: {config.mode}")


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def make_extracted_metadata(metadata: dict) -> dict:
    if metadata["mode"] == MODE_AWAY_TARGET_AWAY_TARGET:
        return {
            "sample_id": metadata["sample_id"],
            "source_video": metadata["source_video"],
            "output_video": metadata["output_video"],
            "mode": metadata["mode"],
            "away_pair_mode": metadata["away_pair_mode"],
            "rotation_speed_deg_per_sec": metadata["turn_speed_deg_per_sec"],
            "target_yaw_deg": metadata["target_yaw_deg"],
            "target_pitch_deg": metadata["pitch_deg"],
            "target_roll_deg": metadata["roll_deg"],
            "target_fov_x_deg": metadata["target_fov_x_deg"],
            "entry_away_yaw_deg": metadata["entry_away_yaw_deg"],
            "entry_away_delta_yaw_deg": metadata["entry_away_delta_deg"],
            "entry_away_hold_duration_sec": metadata["entry_away_hold_sec"],
            "entry_turn_to_target_start_sec": (
                metadata["entry_turn_to_target_start_frame"] / metadata["fps"]
            ),
            "entry_turn_to_target_duration_sec": metadata["entry_turn_to_target_duration_sec"],
            "middle_target_start_sec": metadata["middle_target_hold_start_frame"] / metadata["fps"],
            "middle_target_hold_duration_sec": metadata["middle_target_hold_sec"],
            "turn_to_exit_away_start_sec": (
                metadata["turn_to_exit_away_start_frame"] / metadata["fps"]
            ),
            "turn_to_exit_away_duration_sec": metadata["turn_to_exit_away_duration_sec"],
            "exit_away_yaw_deg": metadata["exit_away_yaw_deg"],
            "exit_away_delta_yaw_deg": metadata["exit_away_delta_deg"],
            "exit_away_hold_duration_sec": metadata["exit_away_hold_sec"],
            "turn_back_start_sec": metadata["turn_back_start_frame"] / metadata["fps"],
            "turn_back_duration_sec": metadata["turn_back_duration_sec"],
            "target_return_sec": metadata["turn_back_end_frame"] / metadata["fps"],
            "final_target_hold_duration_sec": metadata["final_target_hold_sec"],
            "prediction_window_start_sec": metadata["prediction_window_start_sec"],
            "prediction_window_duration_sec": metadata["prediction_window_sec"],
            "entry_away_hold_clamped": metadata["entry_away_hold_clamped"],
            "exit_away_hold_clamped": metadata["exit_away_hold_clamped"],
        }

    return {
        "sample_id": metadata["sample_id"],
        "source_video": metadata["source_video"],
        "output_video": metadata["output_video"],
        "rotation_speed_deg_per_sec": metadata["turn_speed_deg_per_sec"],
        "target_yaw_deg": metadata["target_yaw_deg"],
        "target_pitch_deg": metadata["pitch_deg"],
        "target_roll_deg": metadata["roll_deg"],
        "initial_target_hold_duration_sec": metadata["target_hold_sec"],
        "turn_away_start_sec": metadata["turn_away_start_frame"] / metadata["fps"],
        "turn_away_delta_yaw_deg": metadata["away_yaw_delta_deg"],
        "away_yaw_deg": metadata["away_yaw_deg"],
        "away_pitch_deg": metadata["pitch_deg"],
        "away_roll_deg": metadata["roll_deg"],
        "turn_away_duration_sec": metadata["turn_away_duration_sec"],
        "turn_away_end_sec": metadata["turn_away_end_frame"] / metadata["fps"],
        "away_hold_duration_sec": metadata["away_hold_sec"],
        "away_leave_sec": metadata["away_hold_end_frame"] / metadata["fps"],
        "turn_back_start_sec": metadata["turn_back_start_frame"] / metadata["fps"],
        "turn_back_duration_sec": metadata["turn_back_duration_sec"],
        "target_return_sec": metadata["turn_back_end_frame"] / metadata["fps"],
        "final_target_hold_duration_sec": metadata["final_target_hold_sec"],
        "prediction_window_start_sec": metadata["prediction_window_start_sec"],
        "prediction_window_duration_sec": metadata["prediction_window_sec"],
        "away_hold_clamped": metadata["away_hold_clamped"],
    }


def run(input_path: Path, config: ProbeConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    videos = collect_videos(input_path)
    if not videos:
        raise RuntimeError(f"No supported videos found under: {input_path}")

    metadata_path = config.output_dir / f"{config.run_timestamp}_metadata.jsonl"
    extracted_metadata_path = config.output_dir / f"{config.run_timestamp}_extracted_metadata.jsonl"
    error_path = config.output_dir / f"{config.run_timestamp}_errors.jsonl"
    generated: list[dict] = []

    for video_path in videos:
        try:
            info = read_video_info(video_path)
            for variant_index in range(config.variants_per_video):
                generated.append(write_probe_video(info, input_path, variant_index, config))
        except Exception as exc:
            error_record = {
                "source_video": str(video_path),
                "mode": config.mode,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            append_jsonl(error_path, [error_record])
            print(f"ERROR {video_path}: {exc}", file=sys.stderr)
            if config.strict:
                raise

    append_jsonl(metadata_path, generated)
    append_jsonl(extracted_metadata_path, (make_extracted_metadata(record) for record in generated))
    print(f"Generated {len(generated)} {config.mode} video(s).")
    print(f"Metadata: {metadata_path}")
    print(f"Extracted metadata: {extracted_metadata_path}")
    return 0 if generated else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate processed panorama videos.")
    parser.add_argument("input", type=Path, help="Input video file or directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base output directory. Each run writes into an outputs_<run-timestamp> "
            "subfolder. Defaults to outputs/panorama_video_process."
        ),
    )
    parser.add_argument(
        "--run-timestamp",
        default=current_run_timestamp(),
        help="Timestamp prefix for output filenames, formatted like 202607211200.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=MODES, default=MODE_TWO_TURN)
    parser.add_argument("--variants-per-video", type=int, default=1)
    parser.add_argument("--target-yaw-deg-range", default="-180.0,180.0")
    parser.add_argument("--away-yaw-delta-deg-range", default=None)
    parser.add_argument("--away-hold-sec-range", default=None)
    parser.add_argument("--entry-away-hold-sec-range", default="3.0,5.0")
    parser.add_argument("--initial-target-min-sec", type=float, default=5.0)
    parser.add_argument("--middle-target-hold-sec", type=float, default=5.0)
    parser.add_argument("--prediction-window-sec", type=float, default=5.0)
    parser.add_argument("--turn-speed-deg-per-sec", type=float, default=60.0) # Reduced from 90.0
    parser.add_argument("--target-fov-x-deg-range", default=None)
    parser.add_argument("--away-fov-x-deg-range", default=None)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.variants_per_video < 1:
        parser.error("--variants-per-video must be >= 1")
    if args.initial_target_min_sec <= 0:
        parser.error("--initial-target-min-sec must be > 0")
    if args.middle_target_hold_sec <= 0:
        parser.error("--middle-target-hold-sec must be > 0")
    if args.prediction_window_sec <= 0:
        parser.error("--prediction-window-sec must be > 0")
    if args.turn_speed_deg_per_sec <= 0:
        parser.error("--turn-speed-deg-per-sec must be > 0")

    base_output_dir = args.output_dir or Path("outputs") / "panorama_video_process"
    output_dir = base_output_dir / f"outputs_{args.run_timestamp}"
    target_yaw_deg_range = parse_range(args.target_yaw_deg_range, "--target-yaw-deg-range")
    default_away_yaw_delta_deg_range = (
        "118.0,122.0" if args.mode == MODE_AWAY_TARGET_AWAY_TARGET else "130.0,150.0"
    )
    default_fov_x_deg_range = (
        "95.0,105.0" if args.mode == MODE_AWAY_TARGET_AWAY_TARGET else "100.0,110.0"
    )
    default_away_hold_sec_range = (
        "5.0,120.0" if args.mode == MODE_AWAY_TARGET_AWAY_TARGET else "60.0,120.0"
    )
    away_yaw_delta_deg_range = parse_range(
        args.away_yaw_delta_deg_range or default_away_yaw_delta_deg_range,
        "--away-yaw-delta-deg-range",
    )
    target_fov_x_deg_range = parse_range(
        args.target_fov_x_deg_range or default_fov_x_deg_range,
        "--target-fov-x-deg-range",
    )
    away_fov_x_deg_range = parse_range(
        args.away_fov_x_deg_range or default_fov_x_deg_range,
        "--away-fov-x-deg-range",
    )
    away_hold_sec_range = parse_range(
        args.away_hold_sec_range or default_away_hold_sec_range,
        "--away-hold-sec-range",
    )
    entry_away_hold_sec_range = parse_range(
        args.entry_away_hold_sec_range,
        "--entry-away-hold-sec-range",
    )
    if away_hold_sec_range[0] < 0:
        parser.error("--away-hold-sec-range must be >= 0")
    if entry_away_hold_sec_range[0] <= 0:
        parser.error("--entry-away-hold-sec-range must be > 0")
    if away_yaw_delta_deg_range[0] < 0:
        parser.error("--away-yaw-delta-deg-range must be >= 0")
    if target_fov_x_deg_range[0] <= 0:
        parser.error("--target-fov-x-deg-range must be > 0")
    if away_fov_x_deg_range[0] <= 0:
        parser.error("--away-fov-x-deg-range must be > 0")

    config = ProbeConfig(
        output_dir=output_dir,
        run_timestamp=args.run_timestamp,
        mode=args.mode,
        seed=args.seed,
        variants_per_video=args.variants_per_video,
        target_yaw_deg_range=target_yaw_deg_range,
        away_yaw_delta_deg_range=away_yaw_delta_deg_range,
        away_hold_sec_range=away_hold_sec_range,
        entry_away_hold_sec_range=entry_away_hold_sec_range,
        initial_target_min_sec=args.initial_target_min_sec,
        middle_target_hold_sec=args.middle_target_hold_sec,
        prediction_window_sec=args.prediction_window_sec,
        turn_speed_deg_per_sec=args.turn_speed_deg_per_sec,
        target_fov_x_deg_range=target_fov_x_deg_range,
        away_fov_x_deg_range=away_fov_x_deg_range,
        pitch_deg=args.pitch_deg,
        roll_deg=args.roll_deg,
        overwrite=args.overwrite,
        strict=args.strict,
    )
    return run(args.input, config)


if __name__ == "__main__":
    raise SystemExit(main())
