#!/usr/bin/env python3
"""
Entry point for the hand skeleton component.

Run:
    python hand_skeleton_component.py
"""

from __future__ import annotations

from camera_utils import list_available_cameras
from gesture_config import parse_args
from hand_skeleton_app import HandSkeletonComponent


def main() -> None:
    config = parse_args()
    if config.list_cameras:
        cameras = list_available_cameras(
            config.camera_backend,
            config.capture_width,
            config.capture_height,
            config.target_fps,
        )
        if cameras:
            print("Available cameras:")
            for camera_index, backend_name in cameras:
                print(f"  - index {camera_index} via {backend_name}")
        else:
            print("No camera was detected.")
        return
    app = HandSkeletonComponent(config)
    app.run()


if __name__ == "__main__":
    main()
