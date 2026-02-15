#!/usr/bin/env python3
from __future__ import annotations

import sys
import time

import cv2


def camera_backend_candidates(backend_choice: str):
    if backend_choice == "any":
        return [("any", cv2.CAP_ANY)]
    if backend_choice == "avfoundation":
        return [("avfoundation", cv2.CAP_AVFOUNDATION)]
    if sys.platform == "darwin":
        return [("avfoundation", cv2.CAP_AVFOUNDATION), ("any", cv2.CAP_ANY)]
    return [("any", cv2.CAP_ANY)]


def try_open_capture(
    camera_index: int,
    backend_choice: str,
    capture_width: int,
    capture_height: int,
    target_fps: int,
):
    for backend_name, backend_code in camera_backend_candidates(backend_choice):
        capture = cv2.VideoCapture(camera_index, backend_code)
        # Low-latency + high-frame-rate request. The camera/driver can ignore
        # unsupported values, so we read back the real FPS later.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Request hardware-assisted decode path when backend supports it.
        if hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
            capture.set(
                cv2.CAP_PROP_HW_ACCELERATION,
                getattr(cv2, "VIDEO_ACCELERATION_ANY", 1),
            )
        if hasattr(cv2, "CAP_PROP_HW_DEVICE"):
            capture.set(cv2.CAP_PROP_HW_DEVICE, 0)
        # Prefer MJPG camera stream where available to reduce USB bandwidth
        # bottlenecks at high FPS.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        capture.set(cv2.CAP_PROP_FPS, target_fps)
        if not capture.isOpened():
            capture.release()
            continue

        read_ok = False
        for _ in range(5):
            ok, _ = capture.read()
            if ok:
                read_ok = True
                break
            time.sleep(0.03)

        if read_ok:
            return capture, backend_name

        capture.release()
    return None, None


def list_available_cameras(
    backend_choice: str,
    capture_width: int,
    capture_height: int,
    target_fps: int,
    max_index: int = 6,
):
    found = []
    for camera_index in range(max_index):
        capture, backend_name = try_open_capture(
            camera_index, backend_choice, capture_width, capture_height, target_fps
        )
        if capture is None:
            continue
        found.append((camera_index, backend_name))
        capture.release()
    return found
