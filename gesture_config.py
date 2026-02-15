#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class ViewerConfig:
    camera_index: int = 0
    camera_backend: str = "auto"
    target_fps: int = 60
    capture_width: int = 960
    capture_height: int = 540
    use_multithreading: bool = True
    use_gpu_delegate: bool = False
    auto_recovery_full_res: bool = True
    inference_interval: int = 1
    inference_scale: float = 0.6
    list_cameras: bool = False
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    show_landmark_ids: bool = False
    model_path: str | None = None


def parse_args() -> ViewerConfig:
    parser = argparse.ArgumentParser(description="Real-time hand skeleton viewer.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--camera-backend",
        type=str,
        choices=("auto", "avfoundation", "any"),
        default="auto",
        help="Video backend selection (default: auto)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=60,
        help="Requested camera FPS (default: 60)",
    )
    parser.add_argument(
        "--capture-width",
        type=int,
        default=960,
        help="Requested camera width (default: 960)",
    )
    parser.add_argument(
        "--capture-height",
        type=int,
        default=540,
        help="Requested camera height (default: 540)",
    )
    parser.add_argument(
        "--single-thread",
        action="store_true",
        help="Disable threaded inference worker",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU delegate attempt for MediaPipe Tasks backend",
    )
    parser.add_argument(
        "--gpu-delegate",
        action="store_true",
        help="Enable MediaPipe Tasks GPU delegate (experimental)",
    )
    parser.add_argument(
        "--disable-recovery-full-res",
        action="store_true",
        help="Disable automatic full-resolution recovery when hands are lost",
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=1,
        help="Run hand inference every N frames (default: 1)",
    )
    parser.add_argument(
        "--inference-scale",
        type=float,
        default=0.6,
        help="Scale factor for inference frame size (default: 0.6)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe camera indices 0..5 and print available devices, then exit",
    )
    parser.add_argument("--max-hands", type=int, default=2, help="Max hands to detect")
    parser.add_argument(
        "--min-det-confidence",
        type=float,
        default=0.5,
        help="Minimum hand detection confidence",
    )
    parser.add_argument(
        "--min-track-confidence",
        type=float,
        default=0.5,
        help="Minimum hand tracking confidence",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to hand_landmarker.task (used for MediaPipe Tasks backend)",
    )
    parser.add_argument(
        "--show-landmark-ids",
        action="store_true",
        help="Draw numeric IDs for each hand landmark",
    )
    args = parser.parse_args()

    return ViewerConfig(
        camera_index=args.camera,
        camera_backend=args.camera_backend,
        target_fps=max(1, int(args.target_fps)),
        capture_width=max(160, int(args.capture_width)),
        capture_height=max(120, int(args.capture_height)),
        use_multithreading=not args.single_thread,
        use_gpu_delegate=args.gpu_delegate and not args.cpu_only,
        auto_recovery_full_res=not args.disable_recovery_full_res,
        inference_interval=max(1, int(args.inference_interval)),
        inference_scale=max(0.2, min(1.0, float(args.inference_scale))),
        list_cameras=args.list_cameras,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_det_confidence,
        min_tracking_confidence=args.min_track_confidence,
        show_landmark_ids=args.show_landmark_ids,
        model_path=args.model_path,
    )
