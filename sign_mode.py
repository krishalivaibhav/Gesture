from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import cv2

from sign_curriculum import (
    LEARNER_LEVEL_2_WORDS,
    LEARNER_LEVEL_3_SENTENCES,
    PRACTICE_SENTENCES,
    STATIC_LETTERS,
)

LoaderResult = dict[str, object]
TemplateLoader = Callable[[str], LoaderResult | None]


@dataclass
class PromptSession:
    prompts: list[str]
    prompt_index: int
    char_index: int
    char_states: list[str]
    score: int


class SignLanguageMode:
    """Interactive sign-language learner/practice controller."""

    STATE_MENU = "menu"
    STATE_LEARNER_LEVEL_SELECT = "learner_level_select"
    STATE_LEARNER_RUN = "learner_run"
    STATE_PRACTICE_RUN = "practice_run"

    def __init__(self, template_loader: TemplateLoader) -> None:
        self.template_loader = template_loader

        self.state = self.STATE_MENU
        self.active_level = 0
        self.session: PromptSession | None = None

        self.static_letters = STATIC_LETTERS[:]
        self.templates: dict[str, list[float]] = {}
        self.reference_images: dict[str, object] = {}
        self.available_letters: set[str] = set()

        self.assets_dir = Path(__file__).resolve().parent / "assets" / "signs" / "alphabet"
        self.custom_template_path = self.assets_dir / "custom_templates.json"

        self.menu_buttons: dict[str, tuple[int, int, int, int]] = {}
        self.level_buttons: dict[int, tuple[int, int, int, int]] = {}

        self._last_pinch_active: dict[str, bool] = {}
        self._pinch_click_cooldown: dict[str, int] = {}
        self._pending_mouse_clicks: list[tuple[int, int]] = []

        self._candidate_letter: str | None = None
        self._candidate_frames = 0
        self._hold_min_frames = 10
        self._confidence_threshold_score = 5.5
        self._cooldown_frames = 0

        self._last_prediction = "-"
        self._last_confidence = 0.0
        self._status_message = "Load sign templates to begin."

        self._feature_weights = self._build_feature_weights()
        self._load_templates()

    def reset_to_menu(self) -> None:
        self.state = self.STATE_MENU
        self.active_level = 0
        self.session = None
        self._candidate_letter = None
        self._candidate_frames = 0
        self._cooldown_frames = 0
        self._status_message = "Select Learner or Practice mode."

    def register_mouse_click(self, x: int, y: int) -> None:
        self._pending_mouse_clicks.append((int(x), int(y)))

    def update_and_draw(self, frame, hand_points: Mapping[str, dict], dt: float) -> None:
        _ = dt
        self._draw_background(frame)

        if self.state == self.STATE_MENU:
            self._draw_menu(frame)
            self._draw_input_cursors(frame, hand_points)
            self._handle_menu_clicks(hand_points)
            return

        if self.state == self.STATE_LEARNER_LEVEL_SELECT:
            self._draw_level_select(frame)
            self._draw_input_cursors(frame, hand_points)
            self._handle_level_select_clicks(hand_points)
            return

        if self.state in (self.STATE_LEARNER_RUN, self.STATE_PRACTICE_RUN):
            click_points = self._collect_click_points(hand_points)
            if self._handle_back_to_menu_click(frame, click_points):
                return
            self._handle_capture_template_click(frame, hand_points, click_points)
            self._update_sign_session(hand_points)
            self._draw_session(frame)
            self._draw_input_cursors(frame, hand_points)
            return

    def _load_templates(self) -> None:
        self.templates.clear()
        self.reference_images.clear()
        self.available_letters.clear()

        if not self.assets_dir.exists():
            self._status_message = (
                "Template folder missing: assets/signs/alphabet. "
                "Add sign images to enable matching."
            )
            return

        extensions = (".png", ".jpg", ".jpeg")
        for letter in self.static_letters:
            image_path: Path | None = None
            for ext in extensions:
                candidate = self.assets_dir / f"{letter}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                continue

            loaded = self.template_loader(str(image_path))
            if not loaded:
                print(f"[sign-mode] warning: landmarks not found in {image_path}")
                continue

            landmarks_norm = loaded.get("landmarks_norm")
            if not isinstance(landmarks_norm, list) or len(landmarks_norm) != 21:
                print(f"[sign-mode] warning: invalid landmark payload for {image_path}")
                continue

            handedness = str(loaded.get("handedness", "Right Hand"))
            encoded = self._encode_landmarks(landmarks_norm, handedness)
            if encoded is None:
                print(f"[sign-mode] warning: failed to encode landmarks for {image_path}")
                continue

            self.templates[letter] = encoded
            self.available_letters.add(letter)

            image = loaded.get("image")
            if image is not None:
                self.reference_images[letter] = image

        self._load_custom_templates()

        if not self.templates:
            self._status_message = (
                "No usable templates found. Add A.png..Y.png (without J/Z) in assets/signs/alphabet."
            )
        else:
            self._status_message = (
                f"Loaded {len(self.templates)} sign templates. "
                "Press 4 for Sign mode menu."
            )
            print(f"[sign-mode] loaded templates: {sorted(self.available_letters)}")

    def _load_custom_templates(self) -> None:
        if not self.custom_template_path.exists():
            return
        try:
            payload = json.loads(self.custom_template_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[sign-mode] warning: failed to read custom templates: {exc}")
            return

        if not isinstance(payload, dict):
            return

        loaded_count = 0
        for letter, vector in payload.items():
            if not isinstance(letter, str) or len(letter) != 1:
                continue
            letter = letter.upper()
            if letter not in self.static_letters:
                continue
            if not isinstance(vector, list) or len(vector) != len(self._feature_weights):
                continue
            try:
                normalized_vector = [float(v) for v in vector]
            except Exception:
                continue
            self.templates[letter] = normalized_vector
            self.available_letters.add(letter)
            loaded_count += 1

        if loaded_count > 0:
            print(f"[sign-mode] loaded {loaded_count} custom templates")

    def _save_custom_template(self, letter: str, encoded: list[float]) -> None:
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        payload: dict[str, list[float]] = {}
        if self.custom_template_path.exists():
            try:
                existing = json.loads(self.custom_template_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    for k, v in existing.items():
                        if isinstance(k, str) and isinstance(v, list):
                            try:
                                payload[k.upper()] = [float(x) for x in v]
                            except Exception:
                                continue
            except Exception:
                payload = {}

        payload[letter.upper()] = [float(v) for v in encoded]
        self.custom_template_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _draw_background(self, frame) -> None:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, h), (26, 20, 40), -1)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0.0, frame)

    def _draw_menu(self, frame) -> None:
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            "Sign Language Mode",
            (max(18, w // 2 - 190), 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (240, 240, 250),
            2,
            cv2.LINE_AA,
        )

        button_w = 260
        button_h = 64
        gap = 26
        start_x = (w - (button_w * 2 + gap)) // 2
        y = 150

        learner_rect = (start_x, y, button_w, button_h)
        practice_rect = (start_x + button_w + gap, y, button_w, button_h)
        self.menu_buttons = {
            "learner": learner_rect,
            "practice": practice_rect,
        }

        self._draw_button(frame, learner_rect, "Learner Mode", (84, 160, 245))
        self._draw_button(frame, practice_rect, "Practice Mode", (248, 176, 72))

        self._draw_template_status(frame)
        self._draw_hint_footer(frame)

    def _draw_level_select(self, frame) -> None:
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            "Learner Levels",
            (max(18, w // 2 - 130), 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (240, 240, 250),
            2,
            cv2.LINE_AA,
        )

        button_w = 210
        button_h = 60
        gap = 24
        total_w = button_w * 3 + gap * 2
        start_x = (w - total_w) // 2
        y = 148

        self.level_buttons = {}
        for idx, label in ((1, "Level 1: Letters"), (2, "Level 2: Words"), (3, "Level 3: Sentences")):
            x = start_x + (idx - 1) * (button_w + gap)
            rect = (x, y, button_w, button_h)
            self.level_buttons[idx] = rect
            self._draw_button(frame, rect, label, (130, 198, 102), font_scale=0.58)

        self._draw_template_status(frame)
        self._draw_hint_footer(frame)

    def _draw_template_status(self, frame) -> None:
        h, _ = frame.shape[:2]
        cv2.putText(
            frame,
            f"Templates: {len(self.templates)} loaded | supported letters: A-I, K-Y",
            (20, h - 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self._status_message,
            (20, h - 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 226),
            1,
            cv2.LINE_AA,
        )

    def _draw_hint_footer(self, frame) -> None:
        h, _ = frame.shape[:2]
        cv2.putText(
            frame,
            "Use pinch or mouse click to select. Press 4 to exit Sign mode.",
            (20, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (232, 232, 242),
            1,
            cv2.LINE_AA,
        )

    def _draw_button(
        self,
        frame,
        rect: tuple[int, int, int, int],
        label: str,
        color: tuple[int, int, int],
        font_scale: float = 0.70,
    ) -> None:
        x, y, w, h = rect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.24, frame, 0.76, 0.0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        tx = x + (w - text_size[0]) // 2
        ty = y + (h + text_size[1]) // 2
        cv2.putText(
            frame,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (250, 250, 252),
            1,
            cv2.LINE_AA,
        )

    def _handle_menu_clicks(self, hand_points: Mapping[str, dict]) -> None:
        click_points = self._collect_click_points(hand_points)
        for x, y in click_points:
            if self._point_in_rect((x, y), self.menu_buttons.get("learner")):
                self.state = self.STATE_LEARNER_LEVEL_SELECT
                self._status_message = "Choose a learner level."
                return
            if self._point_in_rect((x, y), self.menu_buttons.get("practice")):
                self._start_practice_mode()
                return

    def _handle_level_select_clicks(self, hand_points: Mapping[str, dict]) -> None:
        click_points = self._collect_click_points(hand_points)
        for x, y in click_points:
            for level, rect in self.level_buttons.items():
                if self._point_in_rect((x, y), rect):
                    self._start_learner_level(level)
                    return

    def _handle_back_to_menu_click(
        self,
        frame,
        click_points: list[tuple[int, int]],
    ) -> bool:
        back_rect = self._back_button_rect(frame.shape[1])
        for x, y in click_points:
            if self._point_in_rect((x, y), back_rect):
                self.reset_to_menu()
                return True
        return False

    def _handle_capture_template_click(
        self,
        frame,
        hand_points: Mapping[str, dict],
        click_points: list[tuple[int, int]],
    ) -> None:
        capture_rect = self._capture_button_rect(frame.shape[1])
        for x, y in click_points:
            if self._point_in_rect((x, y), capture_rect):
                self._capture_current_target_template(hand_points)
                return

    def _start_learner_level(self, level: int) -> None:
        self.active_level = int(level)
        prompts = self._build_learner_prompts(level)
        prompts = self._filter_prompts_by_templates(prompts)
        if not prompts:
            self._status_message = "No valid prompts for this level with current templates."
            self.state = self.STATE_LEARNER_LEVEL_SELECT
            return

        self.session = self._create_session(prompts)
        self.state = self.STATE_LEARNER_RUN
        self._status_message = f"Learner level {level} started."

    def _start_practice_mode(self) -> None:
        prompts = self._filter_prompts_by_templates(PRACTICE_SENTENCES)
        if not prompts:
            self._status_message = "No valid practice sentences with current templates."
            self.state = self.STATE_MENU
            return

        self.session = self._create_session(prompts)
        self.state = self.STATE_PRACTICE_RUN
        self.active_level = 0
        self._status_message = "Practice mode started."

    def _build_learner_prompts(self, level: int) -> list[str]:
        if level == 1:
            return self.static_letters[:]
        if level == 2:
            return LEARNER_LEVEL_2_WORDS[:]
        return LEARNER_LEVEL_3_SENTENCES[:]

    def _filter_prompts_by_templates(self, prompts: list[str]) -> list[str]:
        filtered: list[str] = []
        for prompt in prompts:
            normalized = prompt.upper()
            valid = True
            for ch in normalized:
                if ch == " ":
                    continue
                if ch not in self.available_letters:
                    valid = False
                    break
            if valid:
                filtered.append(normalized)
        return filtered

    def _create_session(self, prompts: list[str]) -> PromptSession:
        first = prompts[0]
        states = ["tick" if ch == " " else "pending" for ch in first]
        char_index = self._first_pending_index(states)
        return PromptSession(
            prompts=prompts,
            prompt_index=0,
            char_index=char_index,
            char_states=states,
            score=0,
        )

    def _update_sign_session(self, hand_points: Mapping[str, dict]) -> None:
        if self.session is None:
            return

        target = self._current_target_char()
        if target is None:
            self._advance_prompt()
            target = self._current_target_char()
            if target is None:
                return

        predicted, confidence, qualified = self._predict_current_sign(hand_points)
        self._last_prediction = predicted if predicted else "-"
        self._last_confidence = confidence

        # If the recognizer's top prediction already matches the active target
        # letter, allow progression even at low confidence. This keeps learner
        # flow smooth for visually similar handshapes.
        if predicted is not None and predicted == target:
            qualified = True

        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1
            return

        stable_letter = self._stable_letter_event(predicted, qualified)
        if stable_letter is None:
            return

        if stable_letter == target:
            self.session.char_states[self.session.char_index] = "tick"
            self._advance_char()
            self._status_message = f"Correct: {stable_letter}"
            self._cooldown_frames = 5
            if self._current_target_char() is None:
                self._advance_prompt()
        elif self.state == self.STATE_PRACTICE_RUN:
            self.session.char_states[self.session.char_index] = "cross"
            self._status_message = f"Wrong sign: {stable_letter}, expected {target}"
            self._cooldown_frames = 5
        else:
            self._status_message = f"Try again: expected {target}"

    def _predict_current_sign(
        self,
        hand_points: Mapping[str, dict],
    ) -> tuple[str | None, float, bool]:
        sample = self._pick_signing_sample(hand_points)
        if sample is None:
            self._candidate_letter = None
            self._candidate_frames = 0
            return None, 0.0, False

        landmarks_norm, handedness = sample
        encoded = self._encode_landmarks(landmarks_norm, handedness)
        if encoded is None or not self.templates:
            return None, 0.0, False

        best_letter = None
        best_distance = 1e9
        for letter, template in self.templates.items():
            dist = self._weighted_l2(encoded, template)
            if dist < best_distance:
                best_distance = dist
                best_letter = letter

        if best_letter is None:
            return None, 0.0, False

        confidence = math.exp(-3.0 * best_distance)
        confidence_score = max(0.0, min(10.0, confidence * 10.0))
        qualified = confidence_score >= self._confidence_threshold_score
        return best_letter, confidence_score, qualified

    def _pick_signing_sample(
        self,
        hand_points: Mapping[str, dict],
    ) -> tuple[list[tuple[float, float]], str] | None:
        entries: list[tuple[int, list[tuple[float, float]], str]] = []
        for hand_data in hand_points.values():
            landmarks_norm = hand_data.get("landmarks_norm")
            if not isinstance(landmarks_norm, list) or len(landmarks_norm) != 21:
                continue
            handedness = str(hand_data.get("handedness", "Right Hand"))
            priority = 2
            if handedness == "Right Hand":
                priority = 0
            elif handedness == "Left Hand":
                priority = 1
            entries.append((priority, landmarks_norm, handedness))

        if not entries:
            return None

        entries.sort(key=lambda item: item[0])
        _, landmarks_norm, handedness = entries[0]
        return landmarks_norm, handedness

    def _stable_letter_event(self, predicted: str | None, qualified: bool) -> str | None:
        if not qualified or predicted is None:
            self._candidate_letter = None
            self._candidate_frames = 0
            return None

        if predicted != self._candidate_letter:
            self._candidate_letter = predicted
            self._candidate_frames = 1
            return None

        self._candidate_frames += 1
        if self._candidate_frames >= self._hold_min_frames:
            self._candidate_letter = None
            self._candidate_frames = 0
            return predicted
        return None

    def _advance_char(self) -> None:
        if self.session is None:
            return
        self.session.char_index = self._first_pending_index(self.session.char_states)

    def _advance_prompt(self) -> None:
        if self.session is None:
            return
        self.session.score += 1
        self.session.prompt_index = (self.session.prompt_index + 1) % len(self.session.prompts)
        prompt = self.session.prompts[self.session.prompt_index]
        self.session.char_states = ["tick" if ch == " " else "pending" for ch in prompt]
        self.session.char_index = self._first_pending_index(self.session.char_states)

    def _current_prompt(self) -> str:
        if self.session is None:
            return ""
        return self.session.prompts[self.session.prompt_index]

    def _current_target_char(self) -> str | None:
        if self.session is None:
            return None
        if self.session.char_index < 0 or self.session.char_index >= len(self.session.char_states):
            return None
        prompt = self._current_prompt()
        if self.session.char_index >= len(prompt):
            return None
        ch = prompt[self.session.char_index]
        if ch == " ":
            return None
        return ch

    def _draw_session(self, frame) -> None:
        if self.session is None:
            return

        h, w = frame.shape[:2]
        prompt = self._current_prompt()
        target = self._current_target_char()

        title = "Learner" if self.state == self.STATE_LEARNER_RUN else "Practice"
        subtitle = f"{title} Mode"
        if self.state == self.STATE_LEARNER_RUN:
            subtitle = f"Learner Mode | Level {self.active_level}"

        cv2.putText(
            frame,
            subtitle,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (244, 244, 252),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Score: {self.session.score}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (236, 236, 245),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"Prediction: {self._last_prediction} "
                f"({self._last_confidence:.2f}/10 | pass >= {self._confidence_threshold_score:.1f})"
            ),
            (20, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (226, 226, 236),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            self._status_message,
            (20, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (214, 214, 224),
            1,
            cv2.LINE_AA,
        )

        self._draw_prompt_grid(frame, prompt, self.session.char_states, self.session.char_index)

        if self.state == self.STATE_LEARNER_RUN:
            self._draw_reference_panel(frame, target)
        else:
            self._draw_practice_panel(frame)

        capture_rect = self._capture_button_rect(w)
        self._draw_button(frame, capture_rect, "Use This Gesture", (76, 140, 210), font_scale=0.50)
        back_rect = self._back_button_rect(w)
        self._draw_button(frame, back_rect, "Back", (200, 92, 92), font_scale=0.58)

    def _draw_prompt_grid(
        self,
        frame,
        prompt: str,
        states: list[str],
        active_index: int,
    ) -> None:
        h, w = frame.shape[:2]
        start_x = 22
        start_y = 174
        cell_w = 44
        cell_h = 52
        gap = 8
        max_cols = max(6, (w - 44) // (cell_w + gap))

        for idx, ch in enumerate(prompt):
            row = idx // max_cols
            col = idx % max_cols
            x = start_x + col * (cell_w + gap)
            y = start_y + row * (cell_h + 26)

            blank_rect = (x + 7, y, cell_w - 14, 20)
            char_rect = (x, y + 24, cell_w, cell_h)

            cv2.rectangle(
                frame,
                (char_rect[0], char_rect[1]),
                (char_rect[0] + char_rect[2], char_rect[1] + char_rect[3]),
                (88, 88, 110),
                1,
                cv2.LINE_AA,
            )

            if idx == active_index:
                cv2.rectangle(
                    frame,
                    (char_rect[0] - 2, char_rect[1] - 2),
                    (char_rect[0] + char_rect[2] + 2, char_rect[1] + char_rect[3] + 2),
                    (255, 212, 108),
                    2,
                    cv2.LINE_AA,
                )

            cv2.rectangle(
                frame,
                (blank_rect[0], blank_rect[1]),
                (blank_rect[0] + blank_rect[2], blank_rect[1] + blank_rect[3]),
                (152, 152, 176),
                1,
                cv2.LINE_AA,
            )

            state = states[idx] if idx < len(states) else "pending"
            if state == "tick":
                self._draw_tick(
                    frame,
                    blank_rect[0] + 4,
                    blank_rect[1] + 5,
                    blank_rect[2] - 8,
                    blank_rect[3] - 8,
                    (92, 228, 132),
                )
            elif state == "cross":
                self._draw_cross(
                    frame,
                    blank_rect[0] + 4,
                    blank_rect[1] + 5,
                    blank_rect[2] - 8,
                    blank_rect[3] - 8,
                    (246, 114, 114),
                )

            draw_ch = ch if ch != " " else "_"
            cv2.putText(
                frame,
                draw_ch,
                (x + 13, y + 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (240, 240, 248),
                2,
                cv2.LINE_AA,
            )

    def _draw_reference_panel(self, frame, target_letter: str | None) -> None:
        h, w = frame.shape[:2]
        panel_w = 260
        panel_h = 300
        x0 = w - panel_w - 18
        y0 = 120

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (22, 24, 36), -1)
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (176, 176, 196), 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            "Reference",
            (x0 + 12, y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (236, 236, 246),
            1,
            cv2.LINE_AA,
        )

        target_text = target_letter if target_letter else "-"
        cv2.putText(
            frame,
            f"Target: {target_text}",
            (x0 + 12, y0 + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (226, 226, 236),
            1,
            cv2.LINE_AA,
        )

        if target_letter and target_letter in self.reference_images:
            image = self.reference_images[target_letter]
            if image is not None:
                self._blit_image(frame, image, x0 + 12, y0 + 70, panel_w - 24, panel_h - 84)
        else:
            cv2.putText(
                frame,
                "No image",
                (x0 + 72, y0 + 168),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (212, 212, 222),
                1,
                cv2.LINE_AA,
            )

    def _draw_practice_panel(self, frame) -> None:
        h, w = frame.shape[:2]
        x0 = w - 278
        y0 = 120
        cv2.rectangle(frame, (x0, y0), (w - 18, y0 + 112), (72, 56, 34), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Practice: no hint images",
            (x0 + 12, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (234, 226, 212),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Tick = correct, Cross = wrong",
            (x0 + 12, y0 + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (226, 218, 206),
            1,
            cv2.LINE_AA,
        )

    def _blit_image(self, frame, image, x: int, y: int, w: int, h: int) -> None:
        if image is None:
            return
        ih, iw = image.shape[:2]
        if ih <= 0 or iw <= 0 or w <= 0 or h <= 0:
            return

        scale = min(w / float(iw), h / float(ih))
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

        px = x + (w - nw) // 2
        py = y + (h - nh) // 2

        frame_h, frame_w = frame.shape[:2]
        if px < 0 or py < 0 or px + nw > frame_w or py + nh > frame_h:
            return

        frame[py : py + nh, px : px + nw] = resized

    def _collect_click_points(self, hand_points: Mapping[str, dict]) -> list[tuple[int, int]]:
        click_points: list[tuple[int, int]] = []

        current_ids = set(hand_points.keys())
        for hand_id in list(self._last_pinch_active.keys()):
            if hand_id not in current_ids:
                self._last_pinch_active.pop(hand_id, None)
                self._pinch_click_cooldown.pop(hand_id, None)

        for hand_id, hand_data in hand_points.items():
            pinch_center = hand_data.get("pinch_center")
            index_tip = hand_data.get("index_tip")
            click_anchor = index_tip if index_tip else pinch_center
            pinch_active = bool(hand_data.get("index_pinch_active", False))
            if not pinch_active:
                pinch_active = self._estimate_index_pinch_from_landmarks(
                    hand_data.get("landmarks_norm")
                )

            was_active = self._last_pinch_active.get(hand_id, False)
            cooldown = self._pinch_click_cooldown.get(hand_id, 0)

            if pinch_active and click_anchor:
                if not was_active or cooldown <= 0:
                    click_points.append((int(click_anchor[0]), int(click_anchor[1])))
                    self._pinch_click_cooldown[hand_id] = 8
                else:
                    self._pinch_click_cooldown[hand_id] = max(0, cooldown - 1)
            else:
                self._pinch_click_cooldown[hand_id] = 0

            self._last_pinch_active[hand_id] = pinch_active

        if self._pending_mouse_clicks:
            click_points.extend(self._pending_mouse_clicks)
            self._pending_mouse_clicks.clear()

        return click_points

    @staticmethod
    def _estimate_index_pinch_from_landmarks(landmarks_norm) -> bool:
        if not isinstance(landmarks_norm, list) or len(landmarks_norm) != 21:
            return False
        try:
            thumb = landmarks_norm[4]
            index = landmarks_norm[8]
            wrist = landmarks_norm[0]
            middle_mcp = landmarks_norm[9]
            index_mcp = landmarks_norm[5]
            pinky_mcp = landmarks_norm[17]

            pinch_dist = math.hypot(float(thumb[0]) - float(index[0]), float(thumb[1]) - float(index[1]))
            palm_len = math.hypot(float(wrist[0]) - float(middle_mcp[0]), float(wrist[1]) - float(middle_mcp[1]))
            palm_wid = math.hypot(float(index_mcp[0]) - float(pinky_mcp[0]), float(index_mcp[1]) - float(pinky_mcp[1]))
            hand_scale = max(1e-5, palm_len, palm_wid)
            ratio = pinch_dist / hand_scale
            return ratio <= 0.42
        except Exception:
            return False

    def _draw_input_cursors(self, frame, hand_points: Mapping[str, dict]) -> None:
        for hand_data in hand_points.values():
            cursor = hand_data.get("index_tip") or hand_data.get("pinch_center")
            if not cursor:
                continue
            x = int(cursor[0])
            y = int(cursor[1])

            pinch_active = bool(hand_data.get("index_pinch_active", False))
            if not pinch_active:
                pinch_active = self._estimate_index_pinch_from_landmarks(
                    hand_data.get("landmarks_norm")
                )

            color = (110, 240, 138) if pinch_active else (255, 210, 118)
            cv2.circle(frame, (x, y), 10 if pinch_active else 8, color, 2, cv2.LINE_AA)
            cv2.line(frame, (x - 14, y), (x + 14, y), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - 14), (x, y + 14), color, 1, cv2.LINE_AA)

    def _capture_current_target_template(self, hand_points: Mapping[str, dict]) -> None:
        target = self._current_target_char()
        if target is None:
            self._status_message = "No active alphabet to capture right now."
            return

        sample = self._pick_signing_sample(hand_points)
        if sample is None:
            self._status_message = "No hand detected for capture."
            return

        landmarks_norm, handedness = sample
        encoded = self._encode_landmarks(landmarks_norm, handedness)
        if encoded is None:
            self._status_message = "Could not encode hand gesture. Try again."
            return

        self.templates[target] = encoded
        self.available_letters.add(target)
        try:
            self._save_custom_template(target, encoded)
            self._status_message = f"Captured template for '{target}' and saved for future runs."
        except Exception as exc:
            self._status_message = f"Captured '{target}' for this run only (save failed: {exc})"

        self._candidate_letter = None
        self._candidate_frames = 0
        self._cooldown_frames = 4

    def _capture_button_rect(self, frame_w: int) -> tuple[int, int, int, int]:
        return (frame_w - 318, 18, 198, 40)

    def _back_button_rect(self, frame_w: int) -> tuple[int, int, int, int]:
        return (frame_w - 108, 18, 90, 40)

    @staticmethod
    def _point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int] | None) -> bool:
        if rect is None:
            return False
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    @staticmethod
    def _first_pending_index(states: list[str]) -> int:
        for idx, state in enumerate(states):
            if state == "pending":
                return idx
        return len(states)

    def _encode_landmarks(
        self,
        landmarks_norm: list[tuple[float, float]],
        handedness: str,
    ) -> list[float] | None:
        if len(landmarks_norm) != 21:
            return None

        pts = [[float(x), float(y)] for x, y in landmarks_norm]
        wrist_x, wrist_y = pts[0]

        for p in pts:
            p[0] -= wrist_x
            p[1] -= wrist_y

        # Canonical right-hand orientation.
        if handedness == "Left Hand":
            for p in pts:
                p[0] = -p[0]

        palm_width = math.hypot(pts[5][0] - pts[17][0], pts[5][1] - pts[17][1])
        palm_len = math.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1])
        scale = max(palm_width, palm_len, 1e-5)

        for p in pts:
            p[0] /= scale
            p[1] /= scale

        anchor_dx = pts[9][0] - pts[0][0]
        anchor_dy = pts[9][1] - pts[0][1]
        angle = math.atan2(anchor_dy, anchor_dx)
        rotate_by = (-math.pi / 2.0) - angle
        cos_r = math.cos(rotate_by)
        sin_r = math.sin(rotate_by)

        rotated: list[float] = []
        for x, y in pts:
            rx = x * cos_r - y * sin_r
            ry = x * sin_r + y * cos_r
            rotated.extend((rx, ry))
        return rotated

    @staticmethod
    def _draw_tick(frame, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        p1 = (x + int(0.12 * w), y + int(0.56 * h))
        p2 = (x + int(0.42 * w), y + int(0.86 * h))
        p3 = (x + int(0.90 * w), y + int(0.18 * h))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        cv2.line(frame, p2, p3, color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_cross(frame, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        cv2.line(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x, y + h), color, 2, cv2.LINE_AA)

    def _weighted_l2(self, a: list[float], b: list[float]) -> float:
        total_w = 0.0
        acc = 0.0
        for i, (av, bv) in enumerate(zip(a, b)):
            w = self._feature_weights[i]
            d = av - bv
            acc += w * d * d
            total_w += w
        if total_w <= 0.0:
            return 1e9
        return math.sqrt(acc / total_w)

    @staticmethod
    def _build_feature_weights() -> list[float]:
        # 21 landmarks * 2 coords. Emphasize fingertips and key joints.
        landmark_weights = [1.0] * 21
        for idx in (4, 8, 12, 16, 20):
            landmark_weights[idx] = 1.9
        for idx in (5, 9, 13, 17):
            landmark_weights[idx] = 1.3

        feature_weights: list[float] = []
        for lw in landmark_weights:
            feature_weights.extend((lw, lw))
        return feature_weights
