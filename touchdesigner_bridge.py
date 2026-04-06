#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from typing import Mapping


class TouchDesignerOSCBridge:
    """Best-effort OSC bridge for streaming gesture state to TouchDesigner."""

    def __init__(
        self,
        enabled: bool = False,
        host: str = "127.0.0.1",
        port: int = 9000,
        send_landmarks: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.host = host
        self.port = int(port)
        self.send_landmarks = bool(send_landmarks)
        self._client = None
        self._state: dict[str, dict[str, float]] = {}

        if not self.enabled:
            return

        try:
            from pythonosc.udp_client import SimpleUDPClient

            self._client = SimpleUDPClient(self.host, self.port)
            print(
                "[touchdesigner] OSC enabled -> "
                f"udp://{self.host}:{self.port} "
                f"(landmarks={'on' if self.send_landmarks else 'off'})"
            )
        except Exception as exc:
            self.enabled = False
            self._client = None
            print(
                "[touchdesigner] OSC disabled: could not initialize python-osc "
                f"({exc}). Install with: pip install python-osc"
            )

    @staticmethod
    def _mode_id(mode: str) -> int:
        return {
            "plain": 0,
            "game": 1,
            "sign": 2,
            "touchdesigner": 3,
            "td": 3,
        }.get(mode, 0)

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _landmark_xy(landmarks_norm, idx: int) -> tuple[float, float] | None:
        if not isinstance(landmarks_norm, list):
            return None
        if idx < 0 or idx >= len(landmarks_norm):
            return None
        lm = landmarks_norm[idx]
        if not isinstance(lm, (tuple, list)) or len(lm) < 2:
            return None
        return float(lm[0]), float(lm[1])

    @classmethod
    def _center_of_landmarks(cls, landmarks_norm) -> tuple[float, float] | None:
        if not isinstance(landmarks_norm, list) or not landmarks_norm:
            return None
        sx = 0.0
        sy = 0.0
        count = 0
        for lm in landmarks_norm[:21]:
            if isinstance(lm, (tuple, list)) and len(lm) >= 2:
                sx += float(lm[0])
                sy += float(lm[1])
                count += 1
        if count == 0:
            return None
        return sx / count, sy / count

    @classmethod
    def _compute_shape_features(cls, landmarks_norm) -> tuple[float, float, float]:
        """Return pinch_strength, openness, depth_proxy (all normalized)."""
        wrist = cls._landmark_xy(landmarks_norm, 0)
        thumb = cls._landmark_xy(landmarks_norm, 4)
        index = cls._landmark_xy(landmarks_norm, 8)
        middle_mcp = cls._landmark_xy(landmarks_norm, 9)
        index_mcp = cls._landmark_xy(landmarks_norm, 5)
        pinky_mcp = cls._landmark_xy(landmarks_norm, 17)

        if not all([wrist, thumb, index, middle_mcp, index_mcp, pinky_mcp]):
            return 0.0, 0.0, 0.0

        palm_size = max(
            cls._distance(wrist, middle_mcp),
            cls._distance(index_mcp, pinky_mcp),
            1e-5,
        )

        pinch_distance_norm = cls._distance(thumb, index) / palm_size
        pinch_strength = cls._clamp(1.0 - pinch_distance_norm / 0.45, 0.0, 1.0)

        fingertip_indices = (8, 12, 16, 20)
        tip_dists = []
        for tip_idx in fingertip_indices:
            tip = cls._landmark_xy(landmarks_norm, tip_idx)
            if tip is not None:
                tip_dists.append(cls._distance(wrist, tip) / palm_size)
        openness = cls._clamp(sum(tip_dists) / max(1, len(tip_dists)), 0.0, 3.0)

        depth_proxy = cls._clamp(palm_size, 0.0, 1.0)
        return pinch_strength, openness, depth_proxy

    def _velocity_for(self, hand_key: str, x: float, y: float, now: float) -> tuple[float, float, float]:
        prev = self._state.get(hand_key)
        if not prev:
            self._state[hand_key] = {
                "x": x,
                "y": y,
                "t": now,
                "vx": 0.0,
                "vy": 0.0,
            }
            return 0.0, 0.0, 0.0

        dt = max(1e-4, now - float(prev.get("t", now)))
        raw_vx = (x - float(prev.get("x", x))) / dt
        raw_vy = (y - float(prev.get("y", y))) / dt

        old_vx = float(prev.get("vx", 0.0))
        old_vy = float(prev.get("vy", 0.0))
        vx = 0.75 * old_vx + 0.25 * raw_vx
        vy = 0.75 * old_vy + 0.25 * raw_vy
        speed = math.hypot(vx, vy)

        self._state[hand_key] = {
            "x": x,
            "y": y,
            "t": now,
            "vx": vx,
            "vy": vy,
        }
        return vx, vy, speed

    def send_frame(
        self,
        hand_points: Mapping[str, dict],
        fps: float,
        mode: str,
        frame_size: tuple[int, int],
    ) -> None:
        if not self.enabled or self._client is None:
            return

        frame_w, frame_h = frame_size
        if frame_w <= 0 or frame_h <= 0:
            return

        client = self._client
        now = time.monotonic()

        left_present = 0
        right_present = 0
        pinch_any = 0.0
        tri_pinch_any = 0.0
        max_speed = 0.0

        try:
            client.send_message("/app/fps", float(fps))
            client.send_message("/app/mode", int(self._mode_id(mode)))
            client.send_message("/hands/count", int(len(hand_points)))

            sorted_hands = sorted(hand_points.items(), key=lambda kv: kv[0])
            for idx, (hand_id, data) in enumerate(sorted_hands):
                handedness = str(data.get("handedness", f"Hand {idx + 1}"))
                htext = handedness.lower()
                if "left" in htext:
                    handedness_id = 1
                    left_present = 1
                elif "right" in htext:
                    handedness_id = 2
                    right_present = 1
                else:
                    handedness_id = 0

                landmarks_norm = data.get("landmarks_norm")
                index_tip = data.get("index_tip")

                if isinstance(landmarks_norm, list) and len(landmarks_norm) > 8:
                    ix = float(landmarks_norm[8][0])
                    iy = float(landmarks_norm[8][1])
                    px = int(ix * frame_w)
                    py = int(iy * frame_h)
                elif index_tip:
                    px = int(index_tip[0])
                    py = int(index_tip[1])
                    ix = px / float(frame_w)
                    iy = py / float(frame_h)
                else:
                    px, py = 0, 0
                    ix, iy = 0.0, 0.0

                pinch = 1.0 if bool(data.get("index_pinch_active", False)) else 0.0
                tri_pinch = 1.0 if bool(data.get("tri_pinch_active", False)) else 0.0
                pinch_any = max(pinch_any, pinch)
                tri_pinch_any = max(tri_pinch_any, tri_pinch)

                pinch_strength, openness, depth_proxy = self._compute_shape_features(
                    landmarks_norm
                )
                center = self._center_of_landmarks(landmarks_norm)
                wrist = self._landmark_xy(landmarks_norm, 0)

                vx, vy, speed = self._velocity_for(str(hand_id), ix, iy, now)
                max_speed = max(max_speed, speed)

                base = f"/hands/{idx}"
                client.send_message(f"{base}/x", ix)
                client.send_message(f"{base}/y", iy)
                client.send_message(f"{base}/x_px", px)
                client.send_message(f"{base}/y_px", py)
                client.send_message(f"{base}/vx", vx)
                client.send_message(f"{base}/vy", vy)
                client.send_message(f"{base}/speed", speed)
                client.send_message(f"{base}/pinch", pinch)
                client.send_message(f"{base}/tri_pinch", tri_pinch)
                client.send_message(f"{base}/pinch_strength", pinch_strength)
                client.send_message(f"{base}/openness", openness)
                client.send_message(f"{base}/depth_proxy", depth_proxy)
                client.send_message(f"{base}/handedness_id", handedness_id)

                if center is not None:
                    client.send_message(f"{base}/center/x", float(center[0]))
                    client.send_message(f"{base}/center/y", float(center[1]))
                if wrist is not None:
                    client.send_message(f"{base}/wrist/x", float(wrist[0]))
                    client.send_message(f"{base}/wrist/y", float(wrist[1]))

                if self.send_landmarks and isinstance(landmarks_norm, list):
                    for lm_idx, lm in enumerate(landmarks_norm[:21]):
                        lx = float(lm[0])
                        ly = float(lm[1])
                        client.send_message(f"{base}/lm/{lm_idx}/x", lx)
                        client.send_message(f"{base}/lm/{lm_idx}/y", ly)

            client.send_message("/hands/left_present", int(left_present))
            client.send_message("/hands/right_present", int(right_present))
            client.send_message("/hands/pinch_any", float(pinch_any))
            client.send_message("/hands/tri_pinch_any", float(tri_pinch_any))
            client.send_message("/hands/max_speed", float(max_speed))
        except Exception:
            # Never let OSC issues break the camera/gesture loop.
            return
