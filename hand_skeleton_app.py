#!/usr/bin/env python3
"""
Real-time hand skeleton viewer using OpenCV + MediaPipe.

This component detects up to two hands and draws hand landmarks/connections
so you can inspect tracking quality visually.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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

from camera_utils import list_available_cameras, try_open_capture
from gesture_config import ViewerConfig
from mini_game import PingPongGame
from sign_mode import SignLanguageMode


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


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
        self.show_counters_overlay = True
        self.active_mode = "plain"
        self.game = PingPongGame()
        self._window_name = "Hand Skeleton Component"
        self._pending_mouse_click: tuple[int, int] | None = None
        self._last_output_size: tuple[int, int] = (0, 0)
        self.sign_mode: SignLanguageMode | None = None
        self.template_hands = None
        self.template_landmarker = None
        self._tasks_model_path: Path | None = None

        if self.backend == "solutions":
            self._init_solutions_backend()
        else:
            self._init_tasks_backend()

        self._init_template_extractors()
        self.sign_mode = SignLanguageMode(self._extract_template_landmarks_from_image)

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
        self._tasks_model_path = model_path
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

    def _init_template_extractors(self) -> None:
        """Create static-image detectors to improve template landmark extraction."""
        if self.backend == "solutions":
            try:
                self.template_hands = self.mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=max(
                        0.25, min(0.95, self.config.min_detection_confidence * 0.6)
                    ),
                    min_tracking_confidence=0.25,
                )
            except Exception as exc:
                print(f"[sign-mode] warning: failed to init static solutions extractor: {exc}")
                self.template_hands = None
            return

        if self._tasks_model_path is None:
            return

        try:
            vision = mp.tasks.vision
            base_options = mp.tasks.BaseOptions(model_asset_path=str(self._tasks_model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=max(
                    0.25, min(0.95, self.config.min_detection_confidence * 0.6)
                ),
                min_hand_presence_confidence=max(
                    0.25, min(0.95, self.config.min_tracking_confidence * 0.6)
                ),
                min_tracking_confidence=0.25,
            )
            self.template_landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            print(f"[sign-mode] warning: failed to init IMAGE extractor: {exc}")
            self.template_landmarker = None

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

    def _on_mouse_event(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            click_x = int(x)
            click_y = int(y)

            # Map window-space coordinates into frame-space coordinates when
            # OpenCV window scaling (retina / resize) is active.
            frame_w, frame_h = self._last_output_size
            if frame_w > 0 and frame_h > 0:
                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect(self._window_name)
                except Exception:
                    win_w, win_h = 0, 0
                if win_w > 0 and win_h > 0:
                    click_x = int(click_x * (frame_w / float(win_w)))
                    click_y = int(click_y * (frame_h / float(win_h)))

            self._pending_mouse_click = (click_x, click_y)

    def _extract_template_landmarks_from_image(self, image_path: str) -> dict | None:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None

        image_bgr = self._prepare_template_image(image)
        for variant in self._build_template_image_variants(image_bgr):
            detected = self._detect_template_landmarks_on_bgr(variant)
            if detected is not None:
                landmarks_norm, handedness = detected
                return {
                    "landmarks_norm": landmarks_norm,
                    "handedness": handedness,
                    "image": image_bgr,
                }

            # Fallback: mirrored image can help with left/right-heavy templates.
            detected = self._detect_template_landmarks_on_bgr(cv2.flip(variant, 1))
            if detected is not None:
                landmarks_norm, handedness = detected
                return {
                    "landmarks_norm": landmarks_norm,
                    "handedness": handedness,
                    "image": image_bgr,
                }
        return None

    def _prepare_template_image(self, image):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 3:
            return image
        if image.shape[2] == 4:
            # Composite RGBA over white to avoid transparent/black artifacts.
            bgr = image[:, :, :3].astype("float32")
            alpha = (image[:, :, 3].astype("float32") / 255.0)[:, :, None]
            out = (bgr * alpha) + (255.0 * (1.0 - alpha))
            return out.astype("uint8")
        return image[:, :, :3]

    def _build_template_image_variants(self, image_bgr) -> list:
        variants = [image_bgr]
        h, w = image_bgr.shape[:2]
        max_dim = max(h, w)
        if max_dim < 320:
            scale = 320.0 / max_dim
            resized = cv2.resize(
                image_bgr,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
            variants.append(resized)

        for base in list(variants):
            padded = cv2.copyMakeBorder(
                base,
                24,
                24,
                24,
                24,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            variants.append(padded)
            boosted = cv2.convertScaleAbs(base, alpha=1.15, beta=18)
            variants.append(boosted)

        return variants

    def _detect_template_landmarks_on_bgr(
        self,
        image_bgr,
    ) -> tuple[list[tuple[float, float]], str] | None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions":
            detectors = [self.template_hands, self.hands]
            for detector in detectors:
                if detector is None:
                    continue
                try:
                    results = detector.process(rgb)
                except Exception:
                    continue
                if not results or not results.multi_hand_landmarks:
                    continue

                landmarks = results.multi_hand_landmarks[0].landmark
                landmarks_norm = [(float(lm.x), float(lm.y)) for lm in landmarks]
                handedness = "Right Hand"
                if results.multi_handedness and results.multi_handedness[0].classification:
                    handedness = self._normalize_handedness_label(
                        results.multi_handedness[0].classification[0].label
                    )
                return landmarks_norm, handedness
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.template_landmarker is not None:
            try:
                results = self.template_landmarker.detect(mp_image)
            except Exception:
                results = None
            if results and results.hand_landmarks:
                landmarks = results.hand_landmarks[0]
                landmarks_norm = [(float(lm.x), float(lm.y)) for lm in landmarks]
                handedness = "Right Hand"
                if results.handedness and results.handedness[0]:
                    handedness = self._normalize_handedness_label(
                        results.handedness[0][0].category_name or handedness
                    )
                return landmarks_norm, handedness

        # Last-resort fallback to VIDEO mode detector.
        try:
            results = self.hand_landmarker.detect_for_video(mp_image, self._next_timestamp_ms())
        except Exception:
            results = None
        if not results or not results.hand_landmarks:
            return None

        landmarks = results.hand_landmarks[0]
        landmarks_norm = [(float(lm.x), float(lm.y)) for lm in landmarks]
        handedness = "Right Hand"
        if results.handedness and results.handedness[0]:
            handedness = self._normalize_handedness_label(
                results.handedness[0][0].category_name or handedness
            )
        return landmarks_norm, handedness

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
        return try_open_capture(
            camera_index,
            self.config.camera_backend,
            self.config.capture_width,
            self.config.capture_height,
            self.config.target_fps,
        )

    def list_available_cameras(self, max_index: int = 6):
        return list_available_cameras(
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

    def _extract_hand_interaction_points(self, results, frame_shape) -> dict:
        frame_h, frame_w = frame_shape[:2]
        hand_points: dict[str, dict] = {}
        label_counts: dict[str, int] = {}

        if not results:
            return hand_points

        if self.backend == "solutions":
            if not results.multi_hand_landmarks:
                return hand_points
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_label = f"Hand {i + 1}"
                if results.multi_handedness and i < len(results.multi_handedness):
                    classification = results.multi_handedness[i].classification[0]
                    handedness_label = self._normalize_handedness_label(
                        classification.label
                    )
                count = label_counts.get(handedness_label, 0) + 1
                label_counts[handedness_label] = count
                hand_id = (
                    handedness_label if count == 1 else f"{handedness_label} #{count}"
                )

                lm = hand_landmarks.landmark
                thumb = (int(lm[4].x * frame_w), int(lm[4].y * frame_h))
                index = (int(lm[8].x * frame_w), int(lm[8].y * frame_h))
                middle = (int(lm[12].x * frame_w), int(lm[12].y * frame_h))
                landmarks_norm = [(float(p.x), float(p.y)) for p in lm]
                pinch_center = (
                    int((thumb[0] + index[0] + middle[0]) / 3),
                    int((thumb[1] + index[1] + middle[1]) / 3),
                )
                hand_points[hand_id] = {
                    "handedness": handedness_label,
                    "index_tip": index,
                    "pinch_center": pinch_center,
                    "landmarks_norm": landmarks_norm,
                    "index_pinch_active": bool(
                        self.pinch_active.get(handedness_label, {}).get("index-thumb", False)
                    ),
                    "tri_pinch_active": bool(
                        self.knob_state.get(handedness_label, {}).get("active", False)
                    ),
                }
            return hand_points

        if not results.hand_landmarks:
            return hand_points
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            handedness_label = f"Hand {i + 1}"
            if i < len(results.handedness) and results.handedness[i]:
                top_handedness = results.handedness[i][0]
                handedness_label = self._normalize_handedness_label(
                    top_handedness.category_name or handedness_label
                )
            count = label_counts.get(handedness_label, 0) + 1
            label_counts[handedness_label] = count
            hand_id = handedness_label if count == 1 else f"{handedness_label} #{count}"

            lm = hand_landmarks
            thumb = (int(lm[4].x * frame_w), int(lm[4].y * frame_h))
            index = (int(lm[8].x * frame_w), int(lm[8].y * frame_h))
            middle = (int(lm[12].x * frame_w), int(lm[12].y * frame_h))
            landmarks_norm = [(float(p.x), float(p.y)) for p in lm]
            pinch_center = (
                int((thumb[0] + index[0] + middle[0]) / 3),
                int((thumb[1] + index[1] + middle[1]) / 3),
            )
            hand_points[hand_id] = {
                "handedness": handedness_label,
                "index_tip": index,
                "pinch_center": pinch_center,
                "landmarks_norm": landmarks_norm,
                "index_pinch_active": bool(
                    self.pinch_active.get(handedness_label, {}).get("index-thumb", False)
                ),
                "tri_pinch_active": bool(
                    self.knob_state.get(handedness_label, {}).get("active", False)
                ),
            }
        return hand_points

    def _toggle_game_mode(self) -> None:
        if self.active_mode == "game":
            self.active_mode = "plain"
        else:
            self.active_mode = "game"

    def _toggle_sign_mode(self) -> None:
        if self.active_mode == "sign":
            self.active_mode = "plain"
            if self.sign_mode is not None:
                self.sign_mode.reset_to_menu()
        else:
            if self.sign_mode is not None:
                self.sign_mode.reset_to_menu()
            self.active_mode = "sign"

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

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._on_mouse_event)

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
                    current_results = self._current_cached_results()
                    output, hand_count = self._draw_from_results(display_frame, current_results)
                else:
                    output, hand_count = self.process_frame(frame)
                    current_results = self._current_cached_results()

                now = time.time()
                dt = max(1e-4, min(0.05, now - prev_time))
                fps = 1.0 / dt
                prev_time = now
                hand_points = self._extract_hand_interaction_points(
                    current_results,
                    output.shape,
                )

                if self.active_mode == "sign" and self.sign_mode is not None:
                    if self._pending_mouse_click is not None:
                        self.sign_mode.register_mouse_click(*self._pending_mouse_click)
                        self._pending_mouse_click = None
                else:
                    self._pending_mouse_click = None

                if self.active_mode == "game":
                    self.game.update_and_draw(output, hand_points, dt)
                elif self.active_mode == "sign" and self.sign_mode is not None:
                    self.sign_mode.update_and_draw(output, hand_points, dt)

                if self.show_counters_overlay:
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

                # Keep latest frame dimensions for robust mouse click mapping.
                self._last_output_size = (output.shape[1], output.shape[0])
                cv2.putText(
                    output,
                    (
                        f"Mode: {self.active_mode.upper()} | "
                        "1=toggle game 2=two-player 3=bot 4=sign r=reset"
                    ),
                    (10, output.shape[0] - 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    (
                        "Keys: q/esc=quit, n=landmark IDs, c=counters | "
                        "Knob: pinch thumb+index+middle | Sign: pinch/mouse click"
                    ),
                    (10, output.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(self._window_name, output)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break
                if key == ord("n"):
                    self.config.show_landmark_ids = not self.config.show_landmark_ids
                if key in (ord("c"), ord("C")):
                    self.show_counters_overlay = not self.show_counters_overlay
                if key == ord("1"):
                    self._toggle_game_mode()
                if key == ord("2"):
                    self.active_mode = "game"
                    self.game.set_mode("two_player")
                if key == ord("3"):
                    self.active_mode = "game"
                    self.game.set_mode("bot")
                if key == ord("4"):
                    self._toggle_sign_mode()
                if key in (ord("r"), ord("R")) and self.active_mode == "game":
                    self.game.reset_match()
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
