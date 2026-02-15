#!/usr/bin/env python3
"""
Real-time hand skeleton viewer using OpenCV + MediaPipe.

This component detects up to two hands and draws hand landmarks/connections
so you can inspect tracking quality visually.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _configure_local_cache_dirs() -> None:
    """
    Keep cache/config writes local to this project so imports don't fail or
    spam warnings when default home cache paths are not writable.
    """
    project_dir = Path(__file__).resolve().parent
    cache_dir = project_dir / ".cache"
    mpl_dir = project_dir / ".mplconfig"
    cache_dir.mkdir(exist_ok=True)
    mpl_dir.mkdir(exist_ok=True)
    (cache_dir / "fontconfig").mkdir(exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


_configure_local_cache_dirs()

import cv2
import mediapipe as mp


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


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


def _camera_backend_candidates(backend_choice: str):
    if backend_choice == "any":
        return [("any", cv2.CAP_ANY)]
    if backend_choice == "avfoundation":
        return [("avfoundation", cv2.CAP_AVFOUNDATION)]
    if sys.platform == "darwin":
        return [("avfoundation", cv2.CAP_AVFOUNDATION), ("any", cv2.CAP_ANY)]
    return [("any", cv2.CAP_ANY)]


def _try_open_capture(
    camera_index: int,
    backend_choice: str,
    capture_width: int,
    capture_height: int,
    target_fps: int,
):
    for backend_name, backend_code in _camera_backend_candidates(backend_choice):
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


def _list_available_cameras(
    backend_choice: str,
    capture_width: int,
    capture_height: int,
    target_fps: int,
    max_index: int = 6,
):
    found = []
    for camera_index in range(max_index):
        capture, backend_name = _try_open_capture(
            camera_index, backend_choice, capture_width, capture_height, target_fps
        )
        if capture is None:
            continue
        found.append((camera_index, backend_name))
        capture.release()
    return found


class HandSkeletonComponent:
    def __init__(self, config: ViewerConfig) -> None:
        self.config = config
        self.backend = self._detect_backend()
        self.acceleration_mode = "CPU"
        self._last_timestamp_ms = 0
        self._frame_index = 0
        self._last_inference_time = None
        self.inference_fps = 0.0
        self.cached_solution_results = None
        self.cached_tasks_results = None
        self.missing_hands_streak = 0
        self.pinch_pairs = [
            ("index-thumb", 8),
            ("middle-thumb", 12),
            ("ring-thumb", 16),
            ("pinky-thumb", 20),
        ]
        self.pinch_press_ratio = 0.35
        self.pinch_release_ratio = 0.50
        self.knob_press_ratio = 0.35
        self.knob_release_ratio = 0.50
        self.knob_step_degrees = 8.0
        self.pinch_counters = {}
        self.pinch_active = {}
        self.knob_state = {}

        if self.backend == "solutions":
            self._init_solutions_backend()
        else:
            self._init_tasks_backend()

    @staticmethod
    def _detect_backend() -> str:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            return "solutions"
        return "tasks"

    def _init_solutions_backend(self) -> None:
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                self.acceleration_mode = "OpenCL"
            else:
                self.acceleration_mode = "CPU"
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            model_complexity=1,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def _init_tasks_backend(self) -> None:
        model_path = self._ensure_tasks_model()
        vision = mp.tasks.vision

        def _create_with_delegate(delegate):
            base_options = mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=delegate,
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.config.max_num_hands,
                min_hand_detection_confidence=self.config.min_detection_confidence,
                min_hand_presence_confidence=self.config.min_tracking_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            return vision.HandLandmarker.create_from_options(options)

        self.hand_landmarker = None
        can_try_gpu_delegate = self.config.use_gpu_delegate and hasattr(
            mp.tasks.BaseOptions, "Delegate"
        )
        if can_try_gpu_delegate and sys.platform == "darwin" and self.config.use_multithreading:
            print(
                "GPU delegate disabled: macOS + threaded inference is unstable in "
                "MediaPipe Tasks. Falling back to CPU delegate."
            )
            can_try_gpu_delegate = False

        if can_try_gpu_delegate:
            try:
                self.hand_landmarker = _create_with_delegate(
                    mp.tasks.BaseOptions.Delegate.GPU
                )
                self.acceleration_mode = "GPU delegate"
            except Exception:
                self.hand_landmarker = None

        if self.hand_landmarker is None:
            self.hand_landmarker = _create_with_delegate(
                mp.tasks.BaseOptions.Delegate.CPU
                if hasattr(mp.tasks.BaseOptions, "Delegate")
                else None
            )
            self.acceleration_mode = "CPU delegate"

        self.hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def _ensure_tasks_model(self) -> Path:
        if self.config.model_path:
            path = Path(self.config.model_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {path}. "
                    "Pass a valid --model-path to a hand_landmarker.task file."
                )
            return path

        model_dir = Path(__file__).resolve().parent / "models"
        model_path = model_dir / "hand_landmarker.task"
        if model_path.exists():
            return model_path

        model_dir.mkdir(exist_ok=True)
        tmp_path = model_path.with_suffix(".task.tmp")
        try:
            with urllib.request.urlopen(DEFAULT_MODEL_URL, timeout=60) as response:
                with open(tmp_path, "wb") as output_file:
                    shutil.copyfileobj(response, output_file)
            tmp_path.replace(model_path)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                "Failed to download the hand landmarker model automatically. "
                "Download it manually from "
                f"{DEFAULT_MODEL_URL} and run with --model-path."
            ) from exc

        return model_path

    def _next_timestamp_ms(self) -> int:
        now_ms = int(time.time() * 1000)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms
        return now_ms

    def _draw_tasks_landmarks(
        self,
        image,
        landmarks: Sequence,
    ) -> None:
        frame_h, frame_w, _ = image.shape

        for connection in self.hand_connections:
            start = landmarks[connection.start]
            end = landmarks[connection.end]
            sx = int(start.x * frame_w)
            sy = int(start.y * frame_h)
            ex = int(end.x * frame_w)
            ey = int(end.y * frame_h)
            cv2.line(image, (sx, sy), (ex, ey), (0, 180, 255), 2, cv2.LINE_AA)

        for landmark in landmarks:
            lx = int(landmark.x * frame_w)
            ly = int(landmark.y * frame_h)
            cv2.circle(image, (lx, ly), 3, (0, 255, 255), -1, cv2.LINE_AA)

    def _try_open_capture(self, camera_index: int):
        return _try_open_capture(
            camera_index,
            self.config.camera_backend,
            self.config.capture_width,
            self.config.capture_height,
            self.config.target_fps,
        )

    def list_available_cameras(self, max_index: int = 6):
        return _list_available_cameras(
            self.config.camera_backend,
            self.config.capture_width,
            self.config.capture_height,
            self.config.target_fps,
            max_index,
        )

    def _should_run_inference(self) -> bool:
        return (
            self._frame_index % self.config.inference_interval == 0
            or (
                self.backend == "solutions"
                and self.cached_solution_results is None
            )
            or (
                self.backend == "tasks"
                and self.cached_tasks_results is None
            )
        )

    def _prepare_inference_frame(self, frame):
        if self.config.auto_recovery_full_res and self.missing_hands_streak >= 4:
            return frame
        if self.config.inference_scale >= 0.999:
            return frame
        scale = max(0.2, min(1.0, self.config.inference_scale))
        inf_w = max(160, int(frame.shape[1] * scale))
        inf_h = max(120, int(frame.shape[0] * scale))
        return cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_LINEAR)

    def _tick_inference_fps(self) -> None:
        now = time.time()
        if self._last_inference_time is not None:
            instant = 1.0 / max(now - self._last_inference_time, 1e-6)
            if self.inference_fps == 0.0:
                self.inference_fps = instant
            else:
                self.inference_fps = 0.85 * self.inference_fps + 0.15 * instant
        self._last_inference_time = now

    def _update_pinch_counters_from_solution_results(self, results) -> None:
        seen_labels = set()
        if results and results.multi_hand_landmarks:
            self.missing_hands_streak = 0
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_label = f"Hand {i + 1}"
                if results.multi_handedness and i < len(results.multi_handedness):
                    classification = results.multi_handedness[i].classification[0]
                    handedness_label = self._normalize_handedness_label(
                        classification.label
                    )
                seen_labels.add(handedness_label)
                self._update_pinch_counters(handedness_label, hand_landmarks.landmark)
        else:
            self.missing_hands_streak += 1
        self._reset_missing_hand_states(seen_labels)

    def _update_pinch_counters_from_tasks_results(self, results) -> None:
        seen_labels = set()
        if results and results.hand_landmarks:
            self.missing_hands_streak = 0
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                handedness_label = f"Hand {i + 1}"
                if i < len(results.handedness) and results.handedness[i]:
                    top_handedness = results.handedness[i][0]
                    handedness_label = self._normalize_handedness_label(
                        top_handedness.category_name or handedness_label
                    )
                seen_labels.add(handedness_label)
                self._update_pinch_counters(handedness_label, hand_landmarks)
        else:
            self.missing_hands_streak += 1
        self._reset_missing_hand_states(seen_labels)

    def _run_inference_once(self, frame):
        inference_frame = self._prepare_inference_frame(frame)
        if self.backend == "solutions":
            rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
            return self.hands.process(rgb_frame)

        rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.hand_landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

    def _store_results(self, results) -> None:
        if self.backend == "solutions":
            self.cached_solution_results = results
            self._update_pinch_counters_from_solution_results(results)
            return
        self.cached_tasks_results = results
        self._update_pinch_counters_from_tasks_results(results)

    def _current_cached_results(self):
        if self.backend == "solutions":
            return self.cached_solution_results
        return self.cached_tasks_results

    @staticmethod
    def _normalize_handedness_label(label: str) -> str:
        cleaned = label.strip().lower()
        if cleaned == "left":
            return "Right Hand"
        if cleaned == "right":
            return "Left Hand"
        return label

    @staticmethod
    def _distance(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _ensure_hand_counter(self, hand_label: str) -> None:
        if hand_label not in self.pinch_counters:
            self.pinch_counters[hand_label] = {
                pinch_name: 0 for pinch_name, _ in self.pinch_pairs
            }
            self.pinch_active[hand_label] = {
                pinch_name: False for pinch_name, _ in self.pinch_pairs
            }
        if hand_label not in self.knob_state:
            self.knob_state[hand_label] = {
                "active": False,
                "last_angle": None,
                "residual": 0.0,
                "cw_steps": 0,
                "ccw_steps": 0,
                "value": 0,
                "direction": "Idle",
            }

    def _hand_scale(self, landmarks: Sequence) -> float:
        # Use a palm-size proxy so pinch thresholds are hand-size and distance
        # invariant enough for webcam usage.
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        pinky_mcp = landmarks[17]
        palm_width = self._distance(index_mcp, pinky_mcp)
        palm_length = self._distance(wrist, middle_mcp)
        return max(palm_width, palm_length, 1e-4)

    def _update_pinch_counters(self, hand_label: str, landmarks: Sequence) -> None:
        self._ensure_hand_counter(hand_label)
        hand_scale = self._hand_scale(landmarks)
        thumb_tip = landmarks[4]
        for pinch_name, tip_index in self.pinch_pairs:
            finger_tip = landmarks[tip_index]
            pinch_ratio = self._distance(thumb_tip, finger_tip) / hand_scale
            currently_active = self.pinch_active[hand_label][pinch_name]
            if not currently_active and pinch_ratio <= self.pinch_press_ratio:
                self.pinch_counters[hand_label][pinch_name] += 1
                self.pinch_active[hand_label][pinch_name] = True
            elif currently_active and pinch_ratio >= self.pinch_release_ratio:
                self.pinch_active[hand_label][pinch_name] = False
        self._update_knob_state(hand_label, landmarks, hand_scale)

    @staticmethod
    def _normalize_angle_delta(delta_degrees: float) -> float:
        while delta_degrees > 180.0:
            delta_degrees -= 360.0
        while delta_degrees < -180.0:
            delta_degrees += 360.0
        return delta_degrees

    @staticmethod
    def _compute_knob_angle(landmarks: Sequence) -> float:
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        return math.degrees(
            math.atan2(middle_mcp.y - index_mcp.y, middle_mcp.x - index_mcp.x)
        )

    def _update_knob_state(
        self,
        hand_label: str,
        landmarks: Sequence,
        hand_scale: float,
    ) -> None:
        state = self.knob_state[hand_label]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        index_ratio = self._distance(thumb_tip, index_tip) / hand_scale
        middle_ratio = self._distance(thumb_tip, middle_tip) / hand_scale
        tri_pinch_active = (
            index_ratio <= self.knob_press_ratio and middle_ratio <= self.knob_press_ratio
        )
        tri_pinch_released = (
            index_ratio >= self.knob_release_ratio or middle_ratio >= self.knob_release_ratio
        )

        current_angle = self._compute_knob_angle(landmarks)
        if tri_pinch_active:
            if not state["active"]:
                state["active"] = True
                state["last_angle"] = current_angle
                state["residual"] = 0.0
                state["direction"] = "Hold"
                return

            prev_angle = state["last_angle"]
            if prev_angle is None:
                state["last_angle"] = current_angle
                state["direction"] = "Hold"
                return

            delta = self._normalize_angle_delta(current_angle - prev_angle)
            state["last_angle"] = current_angle
            state["residual"] += delta

            rotated = False
            while state["residual"] >= self.knob_step_degrees:
                state["cw_steps"] += 1
                state["value"] += 1
                state["residual"] -= self.knob_step_degrees
                state["direction"] = "CW"
                rotated = True
            while state["residual"] <= -self.knob_step_degrees:
                state["ccw_steps"] += 1
                state["value"] -= 1
                state["residual"] += self.knob_step_degrees
                state["direction"] = "CCW"
                rotated = True
            if not rotated:
                state["direction"] = "Hold"
            return

        if state["active"] and tri_pinch_released:
            state["active"] = False
            state["last_angle"] = None
            state["residual"] = 0.0
            state["direction"] = "Idle"

    def _reset_missing_hand_states(self, seen_labels: set[str]) -> None:
        for hand_label in list(self.pinch_active.keys()):
            if hand_label in seen_labels:
                continue
            for pinch_name in self.pinch_active[hand_label]:
                self.pinch_active[hand_label][pinch_name] = False
            if hand_label in self.knob_state:
                self.knob_state[hand_label]["active"] = False
                self.knob_state[hand_label]["last_angle"] = None
                self.knob_state[hand_label]["residual"] = 0.0
                self.knob_state[hand_label]["direction"] = "Idle"

    def _draw_pinch_counters(self, output) -> int:
        y = 214
        ordered_labels = [
            label for label in ("Left Hand", "Right Hand") if label in self.pinch_counters
        ]
        ordered_labels.extend(
            [label for label in self.pinch_counters if label not in ordered_labels]
        )
        if not ordered_labels:
            cv2.putText(
                output,
                "Pinch counters: waiting for hands...",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return y + 24

        for hand_label in ordered_labels:
            counts = self.pinch_counters[hand_label]
            cv2.putText(
                output,
                (
                    f"{hand_label} | Index:{counts['index-thumb']} "
                    f"Middle:{counts['middle-thumb']} "
                    f"Ring:{counts['ring-thumb']} "
                    f"Pinky:{counts['pinky-thumb']}"
                ),
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 24
        return y

    def _draw_knob_states(self, output, start_y: int) -> None:
        y = start_y + 4
        ordered_labels = [
            label for label in ("Left Hand", "Right Hand") if label in self.knob_state
        ]
        ordered_labels.extend([label for label in self.knob_state if label not in ordered_labels])

        if not ordered_labels:
            cv2.putText(
                output,
                "Knob: waiting for hands...",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return

        for hand_label in ordered_labels:
            state = self.knob_state[hand_label]
            cv2.putText(
                output,
                (
                    f"{hand_label} Knob | Value:{state['value']} "
                    f"CW:{state['cw_steps']} CCW:{state['ccw_steps']} "
                    f"State:{state['direction']}"
                ),
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 22

    def _draw_from_results(self, frame, results):
        output = frame.copy()
        hand_count = 0

        if self.backend == "solutions":
            if results and results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(
                        output,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    handedness_label = f"Hand {i + 1}"
                    handedness_score = None
                    if results.multi_handedness and i < len(results.multi_handedness):
                        classification = results.multi_handedness[i].classification[0]
                        handedness_label = self._normalize_handedness_label(
                            classification.label
                        )
                        handedness_score = classification.score

                    frame_h, frame_w, _ = output.shape
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    x = int(wrist.x * frame_w)
                    y = int(wrist.y * frame_h)
                    text = handedness_label
                    if handedness_score is not None:
                        text = f"{handedness_label} ({handedness_score:.2f})"
                    cv2.putText(
                        output,
                        text,
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    if self.config.show_landmark_ids:
                        for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                            lx = int(landmark.x * frame_w)
                            ly = int(landmark.y * frame_h)
                            cv2.putText(
                                output,
                                str(landmark_id),
                                (lx + 2, ly + 2),
                                cv2.FONT_HERSHEY_PLAIN,
                                0.7,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
            return output, hand_count

        if results and results.hand_landmarks:
            hand_count = len(results.hand_landmarks)
            frame_h, frame_w, _ = output.shape
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                self._draw_tasks_landmarks(output, hand_landmarks)

                handedness_label = f"Hand {i + 1}"
                handedness_score = None
                if i < len(results.handedness) and results.handedness[i]:
                    top_handedness = results.handedness[i][0]
                    handedness_label = self._normalize_handedness_label(
                        top_handedness.category_name or handedness_label
                    )
                    handedness_score = top_handedness.score

                wrist = hand_landmarks[0]
                x = int(wrist.x * frame_w)
                y = int(wrist.y * frame_h)
                text = handedness_label
                if handedness_score is not None:
                    text = f"{handedness_label} ({handedness_score:.2f})"
                cv2.putText(
                    output,
                    text,
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if self.config.show_landmark_ids:
                    for landmark_id, landmark in enumerate(hand_landmarks):
                        lx = int(landmark.x * frame_w)
                        ly = int(landmark.y * frame_h)
                        cv2.putText(
                            output,
                            str(landmark_id),
                            (lx + 2, ly + 2),
                            cv2.FONT_HERSHEY_PLAIN,
                            0.7,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
        return output, hand_count

    def process_frame(self, frame, run_inference: bool = True):
        # Fixed selfie-like view: raising left hand appears on the left side.
        frame = cv2.flip(frame, 1)
        self._frame_index += 1

        if run_inference and self._should_run_inference():
            results = self._run_inference_once(frame)
            self._store_results(results)
            self._tick_inference_fps()

        return self._draw_from_results(frame, self._current_cached_results())

    def close(self) -> None:
        if self.backend == "solutions":
            self.hands.close()
            return
        self.hand_landmarker.close()

    def run(self) -> None:
        capture, camera_backend = self._try_open_capture(self.config.camera_index)

        if capture is None:
            discovered = self.list_available_cameras()
            discovered_text = (
                ", ".join(
                    f"index {camera_index} ({backend_name})"
                    for camera_index, backend_name in discovered
                )
                if discovered
                else "none detected"
            )
            permission_hint = ""
            if sys.platform == "darwin":
                permission_hint = (
                    "\nmacOS fix: System Settings -> Privacy & Security -> Camera, "
                    "then enable access for your Terminal app and restart Terminal."
                )
            raise RuntimeError(
                f"Unable to open camera index {self.config.camera_index} "
                f"using backend='{self.config.camera_backend}'. "
                f"Requested {self.config.capture_width}x{self.config.capture_height}"
                f" @ {self.config.target_fps} FPS. "
                f"Detected cameras: {discovered_text}.\n"
                "Try --camera 1 (or another index), close other camera apps, "
                "or use --camera-backend avfoundation." + permission_hint
            )

        camera_reported_fps = capture.get(cv2.CAP_PROP_FPS)
        inference_pool = (
            ThreadPoolExecutor(max_workers=1) if self.config.use_multithreading else None
        )
        pending_inference = None

        prev_time = time.time()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if self.config.use_multithreading:
                    # Render path stays responsive while inference runs in worker.
                    display_frame = cv2.flip(frame, 1)
                    self._frame_index += 1
                    if pending_inference and pending_inference.done():
                        try:
                            results = pending_inference.result()
                            self._store_results(results)
                            self._tick_inference_fps()
                        except Exception:
                            pass
                        pending_inference = None

                    if pending_inference is None and self._should_run_inference():
                        pending_inference = inference_pool.submit(
                            self._run_inference_once,
                            display_frame.copy(),
                        )
                    output, hand_count = self._draw_from_results(
                        display_frame, self._current_cached_results()
                    )
                else:
                    output, hand_count = self.process_frame(frame)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                cv2.putText(
                    output,
                    f"Hands: {hand_count}/{self.config.max_num_hands}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    f"FPS: {fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    f"Backend: {self.backend}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    f"Acceleration: {self.acceleration_mode}",
                    (10, 114),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    f"Camera: index {self.config.camera_index} via {camera_backend}",
                    (10, 136),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    (
                        f"Target FPS: {self.config.target_fps} | "
                        + (
                            f"Camera FPS: {camera_reported_fps:.1f}"
                            if camera_reported_fps and camera_reported_fps > 1.0
                            else "Camera FPS: n/a"
                        )
                    ),
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    (
                        f"Inference FPS: {self.inference_fps:.1f} | "
                        f"every {self.config.inference_interval} frame(s) "
                        f"@ {self.config.inference_scale:.2f} scale "
                        f"| threaded: {'on' if self.config.use_multithreading else 'off'}"
                    ),
                    (10, 182),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    "View: fixed selfie mirror",
                    (10, 204),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                next_y = self._draw_pinch_counters(output)
                self._draw_knob_states(output, next_y)
                cv2.putText(
                    output,
                    "Keys: q/esc=quit, n=landmark IDs | Knob: pinch thumb+index+middle",
                    (10, output.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow("Hand Skeleton Component", output)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break
                if key == ord("n"):
                    self.config.show_landmark_ids = not self.config.show_landmark_ids
        finally:
            if pending_inference is not None:
                try:
                    pending_inference.cancel()
                except Exception:
                    pass
            if inference_pool is not None:
                inference_pool.shutdown(wait=False)
            capture.release()
            cv2.destroyAllWindows()
            self.close()


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


def main() -> None:
    config = parse_args()
    if config.list_cameras:
        cameras = _list_available_cameras(
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
