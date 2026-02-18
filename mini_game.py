#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import cv2


POWERUP_SPEED_BALL = "speed_ball"
POWERUP_PADDLE_SIZE = "paddle_size"
POWERUP_SHIELD = "shield"
POWERUP_CHAOS_ORB = "chaos_orb"


@dataclass
class PowerupDrop:
    kind: str
    target_side: str  # left | right
    x: float
    y: float
    vx: float
    radius: float


@dataclass
class PlayerEffects:
    paddle_size_remaining: float = 0.0
    shield_charges: int = 0
    queued_attack: str | None = None  # speed_ball | chaos_orb


class PingPongGame:
    """Rudimentary pong variant controlled by detected hands + power-ups."""

    def __init__(self) -> None:
        self.mode = "two_player"  # two_player | bot
        self.started = False
        self.ready_hold_seconds = 0.0

        self.left_score = 0
        self.right_score = 0

        self.frame_size = (0, 0)
        self.court_rect = (0, 0, 0, 0)  # x0, y0, x1, y1

        self.paddle_width = 18
        self.paddle_height = 120  # base paddle height
        self.left_paddle_height = 120.0
        self.right_paddle_height = 120.0
        self.left_paddle_x = 0.0
        self.left_paddle_y = 0.0
        self.right_paddle_x = 0.0
        self.right_paddle_y = 0.0

        self.ball_radius = 14.0
        self.ball_pos = [0.0, 0.0]
        self.ball_vel = [0.0, 0.0]

        self.base_ball_speed = 450.0
        self.max_ball_speed = 2200.0
        self.round_pause = 0.0

        self.paddle_follow_speed = 1400.0
        self.bot_follow_speed = 980.0

        # Power-up config + state
        self.active_powerups: list[PowerupDrop] = []
        self.powerup_spawn_timer = random.uniform(2.8, 4.6)
        self.max_active_powerups = 3
        self.powerup_move_speed = 260.0
        self.powerup_radius = 11.0
        self.spawn_counts_left = 0
        self.spawn_counts_right = 0
        self.last_spawn_side: str | None = None

        # Per-side effects
        self.left_effects = PlayerEffects()
        self.right_effects = PlayerEffects()

    def set_mode(self, mode: str) -> None:
        normalized = "bot" if mode == "bot" else "two_player"
        if normalized != self.mode:
            self.mode = normalized
            self.started = False
            self.ready_hold_seconds = 0.0
            self._reset_match_state()
            self._reset_round(serve_direction=random.choice((-1.0, 1.0)))

    def reset_match(self) -> None:
        self.left_score = 0
        self.right_score = 0
        self.started = False
        self.ready_hold_seconds = 0.0
        self._reset_match_state()
        self._reset_round(serve_direction=random.choice((-1.0, 1.0)))

    def _reset_match_state(self) -> None:
        self.round_pause = 0.0
        self.active_powerups.clear()
        self.powerup_spawn_timer = random.uniform(2.8, 4.6)
        self.spawn_counts_left = 0
        self.spawn_counts_right = 0
        self.last_spawn_side = None
        self.left_effects = PlayerEffects()
        self.right_effects = PlayerEffects()

    def update_and_draw(self, output, hand_points: dict, dt: float) -> None:
        frame_h, frame_w = output.shape[:2]
        self._ensure_layout(frame_w, frame_h)

        dt = max(1e-4, min(0.05, float(dt)))
        self._tick_effect_timers(dt)

        left_input, right_input, ready, detection_summary = self._pick_inputs(hand_points)
        self._update_paddles(left_input, right_input, dt)

        if not self.started:
            if ready:
                self.ready_hold_seconds += dt
            else:
                self.ready_hold_seconds = 0.0

            if ready and self.ready_hold_seconds >= 0.55:
                self.started = True
                self._reset_round(serve_direction=random.choice((-1.0, 1.0)))
                self.round_pause = 0.45
        else:
            self._update_powerups(dt)
            self._step_ball(dt)

        self._draw_scene(output, detection_summary, ready)

    def _ensure_layout(self, frame_w: int, frame_h: int) -> None:
        if self.frame_size == (frame_w, frame_h):
            return

        self.frame_size = (frame_w, frame_h)
        x_margin = int(frame_w * 0.08)
        y_margin = int(frame_h * 0.14)
        self.court_rect = (x_margin, y_margin, frame_w - x_margin, frame_h - y_margin)

        court_h = self.court_rect[3] - self.court_rect[1]
        min_dim = min(frame_w, frame_h)

        self.paddle_width = max(12, int(min_dim * 0.018))
        self.paddle_height = max(80, int(court_h * 0.22))
        self.left_paddle_height = float(self.paddle_height)
        self.right_paddle_height = float(self.paddle_height)
        self.ball_radius = float(max(8, int(min_dim * 0.015)))

        self.left_paddle_x = float(self.court_rect[0] + 14)
        self.right_paddle_x = float(self.court_rect[2] - self.paddle_width - 14)

        center_y = (self.court_rect[1] + self.court_rect[3]) * 0.5
        self.left_paddle_y = center_y - self.left_paddle_height * 0.5
        self.right_paddle_y = center_y - self.right_paddle_height * 0.5

        self._reset_round(serve_direction=random.choice((-1.0, 1.0)))

    def _pick_inputs(self, hand_points: dict):
        entries = []
        right_count = 0
        left_count = 0

        for hand_id, data in hand_points.items():
            index_tip = data.get("index_tip")
            if not index_tip:
                continue
            x, y = float(index_tip[0]), float(index_tip[1])
            handedness = str(data.get("handedness", "Unknown"))
            if handedness == "Right Hand":
                right_count += 1
            elif handedness == "Left Hand":
                left_count += 1
            entries.append((x, y, hand_id, handedness))

        entries.sort(key=lambda item: item[0])

        detection_summary = (
            f"Detected hands -> Right:{right_count} Left:{left_count} Total:{len(entries)}"
        )

        if self.mode == "two_player":
            if len(entries) < 2:
                return None, None, False, detection_summary
            left_input = entries[0]
            right_input = entries[-1]
            return left_input, right_input, True, detection_summary

        # Bot mode: one hand controls left paddle; right paddle is AI.
        if not entries:
            return None, None, False, detection_summary
        left_input = entries[0]
        return left_input, None, True, detection_summary

    def _tick_effect_timers(self, dt: float) -> None:
        self.left_effects.paddle_size_remaining = max(
            0.0, self.left_effects.paddle_size_remaining - dt
        )
        self.right_effects.paddle_size_remaining = max(
            0.0, self.right_effects.paddle_size_remaining - dt
        )

        self._sync_paddle_size(is_left=True)
        self._sync_paddle_size(is_left=False)

    def _sync_paddle_size(self, is_left: bool) -> None:
        effects = self.left_effects if is_left else self.right_effects
        prev_h = self.left_paddle_height if is_left else self.right_paddle_height
        target_h = self._effective_paddle_height(effects)
        if abs(prev_h - target_h) < 1e-3:
            return

        center_y = (self.left_paddle_y + prev_h * 0.5) if is_left else (self.right_paddle_y + prev_h * 0.5)
        new_y = center_y - target_h * 0.5

        if is_left:
            self.left_paddle_height = target_h
            self.left_paddle_y = new_y
        else:
            self.right_paddle_height = target_h
            self.right_paddle_y = new_y

    def _effective_paddle_height(self, effects: PlayerEffects) -> float:
        multiplier = 1.45 if effects.paddle_size_remaining > 0.0 else 1.0
        return float(self.paddle_height) * multiplier

    def _update_paddles(self, left_input, right_input, dt: float) -> None:
        max_delta = self.paddle_follow_speed * max(0.0, dt)

        if left_input is not None:
            target_y = left_input[1] - self.left_paddle_height * 0.5
            self.left_paddle_y = self._approach(self.left_paddle_y, target_y, max_delta)

        if self.mode == "bot":
            lead = self.ball_vel[1] * 0.08
            target_y = (self.ball_pos[1] + lead) - self.right_paddle_height * 0.5
            bot_delta = self.bot_follow_speed * max(0.0, dt)
            self.right_paddle_y = self._approach(self.right_paddle_y, target_y, bot_delta)
        elif right_input is not None:
            target_y = right_input[1] - self.right_paddle_height * 0.5
            self.right_paddle_y = self._approach(self.right_paddle_y, target_y, max_delta)

        self._clamp_paddles()

    def _clamp_paddles(self) -> None:
        y_min = float(self.court_rect[1])
        left_max = float(self.court_rect[3] - self.left_paddle_height)
        right_max = float(self.court_rect[3] - self.right_paddle_height)
        self.left_paddle_y = max(y_min, min(left_max, self.left_paddle_y))
        self.right_paddle_y = max(y_min, min(right_max, self.right_paddle_y))

    @staticmethod
    def _approach(current: float, target: float, max_delta: float) -> float:
        delta = target - current
        if delta > max_delta:
            return current + max_delta
        if delta < -max_delta:
            return current - max_delta
        return target

    def _update_powerups(self, dt: float) -> None:
        self._spawn_powerups_if_needed(dt)
        self._move_and_collect_powerups(dt)

    def _spawn_powerups_if_needed(self, dt: float) -> None:
        if not self.started or self.round_pause > 0.35:
            return

        self.powerup_spawn_timer -= dt
        if self.powerup_spawn_timer > 0.0:
            return

        if len(self.active_powerups) >= self.max_active_powerups:
            self.powerup_spawn_timer = random.uniform(0.8, 1.4)
            return

        side = self._choose_spawn_side()
        kind = self._choose_powerup_kind()
        x0, y0, x1, y1 = self.court_rect
        spawn_x = float((x0 + x1) * 0.5)
        spawn_y = float(random.uniform(y0 + 26, y1 - 26))
        vx = -self.powerup_move_speed if side == "left" else self.powerup_move_speed

        self.active_powerups.append(
            PowerupDrop(
                kind=kind,
                target_side=side,
                x=spawn_x,
                y=spawn_y,
                vx=vx,
                radius=self.powerup_radius,
            )
        )

        if side == "left":
            self.spawn_counts_left += 1
        else:
            self.spawn_counts_right += 1
        self.last_spawn_side = side
        self.powerup_spawn_timer = random.uniform(2.8, 4.6)

    def _choose_spawn_side(self) -> str:
        left_w = 1.0
        right_w = 1.0

        diff = self.spawn_counts_left - self.spawn_counts_right
        if diff > 0:
            right_w += 0.75 * abs(diff)
        elif diff < 0:
            left_w += 0.75 * abs(diff)

        if self.last_spawn_side == "left":
            right_w += 0.35
        elif self.last_spawn_side == "right":
            left_w += 0.35

        total = left_w + right_w
        if total <= 0.0:
            return "left"
        return "left" if random.random() < (left_w / total) else "right"

    def _choose_powerup_kind(self) -> str:
        roll = random.random()
        if roll < 0.32:
            return POWERUP_SPEED_BALL
        if roll < 0.62:
            return POWERUP_PADDLE_SIZE
        if roll < 0.85:
            return POWERUP_SHIELD
        return POWERUP_CHAOS_ORB

    def _move_and_collect_powerups(self, dt: float) -> None:
        x0, _, x1, _ = self.court_rect
        remaining: list[PowerupDrop] = []

        for drop in self.active_powerups:
            drop.x += drop.vx * dt

            if drop.x < x0 - 40 or drop.x > x1 + 40:
                continue

            if drop.target_side == "left":
                rx, ry, rw, rh = self._paddle_rect(is_left=True)
                if self._circle_rect_overlap(drop.x, drop.y, drop.radius, rx, ry, rw, rh):
                    self._apply_powerup_to_side("left", drop.kind)
                    continue
            else:
                rx, ry, rw, rh = self._paddle_rect(is_left=False)
                if self._circle_rect_overlap(drop.x, drop.y, drop.radius, rx, ry, rw, rh):
                    self._apply_powerup_to_side("right", drop.kind)
                    continue

            remaining.append(drop)

        self.active_powerups = remaining

    def _apply_powerup_to_side(self, side: str, kind: str) -> None:
        effects = self.left_effects if side == "left" else self.right_effects

        if kind == POWERUP_PADDLE_SIZE:
            effects.paddle_size_remaining = max(effects.paddle_size_remaining, 5.0)
        elif kind == POWERUP_SHIELD:
            effects.shield_charges = min(1, effects.shield_charges + 1)
        elif kind in (POWERUP_SPEED_BALL, POWERUP_CHAOS_ORB):
            # Limited stack: single queued attack slot.
            effects.queued_attack = kind

    def _step_ball(self, dt: float) -> None:
        if self.round_pause > 0.0:
            self.round_pause = max(0.0, self.round_pause - dt)
            return

        self.ball_pos[0] += self.ball_vel[0] * dt
        self.ball_pos[1] += self.ball_vel[1] * dt

        # Safety cap in case of stacked speed effects / collisions.
        self._cap_ball_speed(self.max_ball_speed)

        x0, y0, x1, y1 = self.court_rect

        if self.ball_pos[1] - self.ball_radius <= y0:
            self.ball_pos[1] = y0 + self.ball_radius
            self.ball_vel[1] = abs(self.ball_vel[1])
        elif self.ball_pos[1] + self.ball_radius >= y1:
            self.ball_pos[1] = y1 - self.ball_radius
            self.ball_vel[1] = -abs(self.ball_vel[1])

        self._handle_paddle_bounce(is_left=True)
        self._handle_paddle_bounce(is_left=False)

        if self.ball_pos[0] + self.ball_radius < x0:
            if self._consume_shield_and_reflect(side="left"):
                return
            self.right_score += 1
            self._reset_round(serve_direction=-1.0)
            self.round_pause = 0.75
            return

        if self.ball_pos[0] - self.ball_radius > x1:
            if self._consume_shield_and_reflect(side="right"):
                return
            self.left_score += 1
            self._reset_round(serve_direction=1.0)
            self.round_pause = 0.75

    def _consume_shield_and_reflect(self, side: str) -> bool:
        effects = self.left_effects if side == "left" else self.right_effects
        if effects.shield_charges <= 0:
            return False

        x0, _, x1, _ = self.court_rect
        effects.shield_charges -= 1
        if side == "left":
            self.ball_pos[0] = x0 + self.ball_radius + 2.0
            self.ball_vel[0] = abs(self.ball_vel[0])
        else:
            self.ball_pos[0] = x1 - self.ball_radius - 2.0
            self.ball_vel[0] = -abs(self.ball_vel[0])

        self._cap_ball_speed(self.max_ball_speed)
        return True

    def _handle_paddle_bounce(self, is_left: bool) -> None:
        px = self.left_paddle_x if is_left else self.right_paddle_x
        py = self.left_paddle_y if is_left else self.right_paddle_y
        pw = self.paddle_width
        ph = self.left_paddle_height if is_left else self.right_paddle_height

        ball_x, ball_y = self.ball_pos
        r = self.ball_radius

        if is_left:
            if self.ball_vel[0] >= 0.0:
                return
            if ball_x - r > px + pw or ball_x + r < px:
                return
        else:
            if self.ball_vel[0] <= 0.0:
                return
            if ball_x + r < px or ball_x - r > px + pw:
                return

        if ball_y + r < py or ball_y - r > py + ph:
            return

        relative = (ball_y - (py + ph * 0.5)) / max(1.0, ph * 0.5)
        relative = max(-1.0, min(1.0, relative))

        speed = min(self.max_ball_speed, math.hypot(self.ball_vel[0], self.ball_vel[1]) * 1.04)
        angle = relative * 0.95

        effects = self.left_effects if is_left else self.right_effects
        if effects.queued_attack == POWERUP_SPEED_BALL:
            speed = min(2200.0, speed * 1.85)
            effects.queued_attack = None
        elif effects.queued_attack == POWERUP_CHAOS_ORB:
            angle += math.radians(random.uniform(-75.0, 75.0))
            speed = min(self.max_ball_speed, speed * 1.08)
            effects.queued_attack = None

        angle = max(-1.35, min(1.35, angle))
        vx_mag = speed * max(0.25, math.cos(angle))
        vy = speed * math.sin(angle)

        if is_left:
            self.ball_vel[0] = abs(vx_mag)
            self.ball_pos[0] = px + pw + r + 1.0
        else:
            self.ball_vel[0] = -abs(vx_mag)
            self.ball_pos[0] = px - r - 1.0
        self.ball_vel[1] = vy

    def _cap_ball_speed(self, max_speed: float) -> None:
        speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
        if speed <= max_speed or speed <= 1e-6:
            return
        scale = max_speed / speed
        self.ball_vel[0] *= scale
        self.ball_vel[1] *= scale

    def _reset_round(self, serve_direction: float) -> None:
        x0, y0, x1, y1 = self.court_rect
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        self.ball_pos = [float(cx), float(cy)]

        theta = random.uniform(-0.32, 0.32)
        self.ball_vel = [
            serve_direction * self.base_ball_speed * math.cos(theta),
            self.base_ball_speed * math.sin(theta),
        ]

        self.active_powerups.clear()
        self.powerup_spawn_timer = random.uniform(2.8, 4.6)

    def _paddle_rect(self, is_left: bool) -> tuple[float, float, float, float]:
        if is_left:
            return (
                self.left_paddle_x,
                self.left_paddle_y,
                float(self.paddle_width),
                self.left_paddle_height,
            )
        return (
            self.right_paddle_x,
            self.right_paddle_y,
            float(self.paddle_width),
            self.right_paddle_height,
        )

    @staticmethod
    def _circle_rect_overlap(
        cx: float,
        cy: float,
        radius: float,
        rx: float,
        ry: float,
        rw: float,
        rh: float,
    ) -> bool:
        closest_x = min(max(cx, rx), rx + rw)
        closest_y = min(max(cy, ry), ry + rh)
        dx = cx - closest_x
        dy = cy - closest_y
        return dx * dx + dy * dy <= radius * radius

    def _draw_scene(self, output, detection_summary: str, ready: bool) -> None:
        x0, y0, x1, y1 = self.court_rect

        tint = output.copy()
        cv2.rectangle(tint, (x0, y0), (x1, y1), (18, 22, 28), -1)
        cv2.addWeighted(tint, 0.45, output, 0.55, 0.0, output)

        cv2.rectangle(output, (x0, y0), (x1, y1), (200, 200, 200), 2, cv2.LINE_AA)

        mid_x = (x0 + x1) // 2
        for yy in range(y0 + 6, y1 - 6, 26):
            cv2.line(output, (mid_x, yy), (mid_x, yy + 12), (160, 160, 160), 2, cv2.LINE_AA)

        self._draw_powerups(output)

        l0 = (int(self.left_paddle_x), int(self.left_paddle_y))
        l1 = (
            int(self.left_paddle_x + self.paddle_width),
            int(self.left_paddle_y + self.left_paddle_height),
        )
        r0 = (int(self.right_paddle_x), int(self.right_paddle_y))
        r1 = (
            int(self.right_paddle_x + self.paddle_width),
            int(self.right_paddle_y + self.right_paddle_height),
        )
        cv2.rectangle(output, l0, l1, (80, 220, 255), -1, cv2.LINE_AA)
        cv2.rectangle(output, r0, r1, (255, 190, 80), -1, cv2.LINE_AA)

        bx, by = int(self.ball_pos[0]), int(self.ball_pos[1])
        cv2.circle(output, (bx, by), int(self.ball_radius), (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(output, (bx, by), int(self.ball_radius), (55, 55, 55), 1, cv2.LINE_AA)

        score_text = f"{self.left_score}   :   {self.right_score}"
        cv2.putText(
            output,
            score_text,
            (mid_x - 58, y0 - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

        mode_text = "Two People" if self.mode == "two_player" else "Vs Bot"
        cv2.putText(
            output,
            f"Ping Pong | Mode: {mode_text}",
            (x0, y0 - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

        speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
        cv2.putText(
            output,
            f"Ball speed: {speed:.0f}",
            (x1 - 190, y0 - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            f"PU spawns L:{self.spawn_counts_left} R:{self.spawn_counts_right}",
            (x1 - 220, y0 - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        if not self.started:
            requirement = (
                "Need 2 hands for two-player (ideally both right hands)."
                if self.mode == "two_player"
                else "Need 1 hand for bot mode."
            )
            readiness = "READY - starting..." if ready else "WAITING FOR HANDS"
            cv2.putText(
                output,
                requirement,
                (x0 + 16, y1 - 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                readiness,
                (x0 + 16, y1 - 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (120, 255, 120) if ready else (120, 180, 255),
                2,
                cv2.LINE_AA,
            )

        self._draw_side_effects(output, side="left")
        self._draw_side_effects(output, side="right")

        cv2.putText(
            output,
            detection_summary,
            (x0 + 16, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def _draw_powerups(self, output) -> None:
        for drop in self.active_powerups:
            color, label = self._powerup_style(drop.kind)
            center = (int(drop.x), int(drop.y))
            cv2.circle(output, center, int(drop.radius), color, -1, cv2.LINE_AA)
            cv2.circle(output, center, int(drop.radius), (24, 24, 24), 1, cv2.LINE_AA)
            cv2.putText(
                output,
                label,
                (center[0] - 12, center[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )

    def _draw_side_effects(self, output, side: str) -> None:
        effects = self.left_effects if side == "left" else self.right_effects
        x0, y0, x1, _ = self.court_rect
        lines: list[str] = []

        if effects.paddle_size_remaining > 0.0:
            lines.append(f"SIZE {effects.paddle_size_remaining:.1f}s")
        if effects.shield_charges > 0:
            lines.append(f"SHIELD x{effects.shield_charges}")
        if effects.queued_attack == POWERUP_SPEED_BALL:
            lines.append("QUEUE: SPEED")
        elif effects.queued_attack == POWERUP_CHAOS_ORB:
            lines.append("QUEUE: CHAOS")

        if not lines:
            lines.append("No powerups")

        base_y = y0 + 20
        if side == "left":
            x = x0 + 12
            for idx, line in enumerate(lines):
                cv2.putText(
                    output,
                    line,
                    (x, base_y + idx * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (228, 228, 228),
                    1,
                    cv2.LINE_AA,
                )
            return

        # Right side text aligned inward.
        for idx, line in enumerate(lines):
            text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            x = x1 - 12 - text_size[0]
            cv2.putText(
                output,
                line,
                (x, base_y + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (228, 228, 228),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _powerup_style(kind: str) -> tuple[tuple[int, int, int], str]:
        if kind == POWERUP_SPEED_BALL:
            return (70, 70, 255), "SPD"
        if kind == POWERUP_PADDLE_SIZE:
            return (90, 220, 120), "SIZE"
        if kind == POWERUP_SHIELD:
            return (70, 220, 255), "SHD"
        return (220, 110, 250), "CHS"


# Backward-compat alias used by older app code paths.
BallBasketGame = PingPongGame
