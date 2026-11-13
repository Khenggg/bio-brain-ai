"""Procedural room simulator with a leaping cat-like creature and swaying background distractor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class SimMetadata:
    """Ground-truth metadata exposed ONLY to external evaluation, never to the AI Brain."""

    cat_norm_xy: tuple[float, float]
    behavior: str  # idle | crouch | leap | walk
    is_occluded: bool


class CatMotionSimulator:
    """Renders raw BGR frames of a leaping cat in a cluttered room with repetitive distractors."""

    def __init__(self, width: int = 640, height: int = 480, seed: int = 42) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

        self._bg_base = self._build_room_background()
        self._step = 0

        # Creature state in pixel coordinates
        self._x = width * 0.22
        self._y = height * 0.72
        self._vx = 0.0
        self._vy = 0.0
        self._facing = 1.0
        self._behavior = "idle"
        self._phase_timer = 20

    def _build_room_background(self) -> np.ndarray:
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Soft wall gradient
        wall_h = int(self.height * 0.72)
        for y in range(wall_h):
            val = int(195 - 35 * (y / max(1, wall_h)))
            bg[y, :] = (val - 12, val - 4, val + 6)  # warm off-white BGR
        # Wooden floor planks
        bg[wall_h:, :] = (78, 108, 142)
        for py in range(wall_h, self.height, 26):
            cv2.line(bg, (0, py), (self.width, py), (62, 86, 118), 1)
        # Subtle wall frame painting (static clutter to verify predictive suppression)
        cv2.rectangle(bg, (230, 75), (410, 195), (90, 75, 65), 4)
        cv2.rectangle(bg, (242, 87), (398, 183), (145, 162, 175), -1)
        cv2.circle(bg, (320, 130), 28, (95, 140, 195), -1)
        # Static bookshelf on right
        cv2.rectangle(bg, (505, 110), (610, 345), (55, 72, 95), -1)
        for sy in (165, 225, 285):
            cv2.line(bg, (510, sy), (605, sy), (38, 50, 68), 3)
        return bg

    def _draw_swaying_plant(self, img: np.ndarray, t: int) -> None:
        """Periodic distractor testing dorsal predictive habituation."""
        pot_x, pot_y = 75, int(self.height * 0.73)
        cv2.rectangle(img, (pot_x - 22, pot_y), (pot_x + 22, pot_y + 42), (65, 85, 135), -1)
        sway = math.sin(t * 0.22) * 10.0
        for idx, angle in enumerate((-0.45, -0.15, 0.2, 0.5)):
            tip_x = int(pot_x + math.sin(angle) * 58 + sway * (0.7 + idx * 0.1))
            tip_y = int(pot_y - math.cos(angle) * 68)
            cv2.line(img, (pot_x, pot_y), (tip_x, tip_y), (55, 135, 65), 5)
            cv2.circle(img, (tip_x, tip_y), 7, (65, 155, 75), -1)

    def _draw_cat(self, img: np.ndarray, t: int) -> None:
        cx, cy = int(self._x), int(self._y)
        d = 1 if self._facing >= 0 else -1

        # Stretch & arch pose according to leap / crouch / walk
        stretch = 1.28 if self._behavior == "leap" else (0.82 if self._behavior == "crouch" else 1.0)
        crouch_drop = 8 if self._behavior == "crouch" else 0
        body_rx = int(28 * stretch)
        body_ry = int(15 / math.sqrt(stretch))

        # Tail curving dynamically
        tail_pts = []
        for i in range(6):
            frac = i / 5.0
            tx = cx - d * int(body_rx + 18 * frac)
            ty = cy + crouch_drop - int(12 * math.sin(frac * 2.5 + t * 0.18) + 18 * frac)
            tail_pts.append([tx, ty])
        cv2.polylines(img, [np.array(tail_pts, dtype=np.int32)], False, (42, 48, 62), 6, cv2.LINE_AA)

        # Animated legs
        stride = math.sin(t * 0.45) * 14 if self._behavior == "walk" else (18 if self._behavior == "leap" else 3)
        for lx_off, phase in ((-15, 1.0), (15, -1.0)):
            foot_x = cx + lx_off + int(stride * phase * d)
            foot_y = cy + crouch_drop + (14 if self._behavior == "leap" else 24)
            cv2.line(img, (cx + lx_off, cy + crouch_drop + 5), (foot_x, foot_y), (42, 48, 62), 5)

        # Torso (dark tabby/charcoal contrast)
        cv2.ellipse(img, (cx, cy + crouch_drop), (body_rx, body_ry), 0, 0, 360, (48, 56, 72), -1)
        cv2.ellipse(img, (cx, cy + crouch_drop + 3), (body_rx - 8, body_ry - 6), 0, 0, 360, (92, 108, 128), -1)

        # Head + pointy ears
        hx = cx + d * (body_rx + 6)
        hy = cy + crouch_drop - (12 if self._behavior != "crouch" else 5)
        cv2.circle(img, (hx, hy), 13, (48, 56, 72), -1)
        ear_l = np.array([[hx - 10, hy - 6], [hx - 12, hy - 22], [hx - 1, hy - 12]], dtype=np.int32)
        ear_r = np.array([[hx + 1, hy - 12], [hx + 11, hy - 22], [hx + 10, hy - 6]], dtype=np.int32)
        cv2.fillPoly(img, [ear_l, ear_r], (48, 56, 72))
        # Bright eyes
        cv2.circle(img, (hx + d * 4, hy - 2), 2, (110, 225, 240), -1)

    def _draw_foreground_occluder(self, img: np.ndarray) -> bool:
        """Draw a foreground ottoman/box around center-bottom so cat can pass behind briefly."""
        box_x1, box_y1, box_x2, box_y2 = 275, 295, 365, 390
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (82, 92, 110), -1)
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (55, 62, 75), 2)
        return box_x1 <= self._x <= box_x2 and box_y1 <= self._y <= box_y2

    def step(self) -> tuple[np.ndarray, SimMetadata]:
        self._step += 1
        ground_y = self.height * 0.72
        self._phase_timer -= 1

        if self._behavior == "idle":
            if self._phase_timer <= 0:
                self._behavior = "crouch"
                self._phase_timer = 14
        elif self._behavior == "crouch":
            if self._phase_timer <= 0:
                self._behavior = "leap"
                target_right = self._x < self.width * 0.5
                self._facing = 1.0 if target_right else -1.0
                self._vx = self._facing * float(self.rng.uniform(7.5, 11.0))
                self._vy = -float(self.rng.uniform(11.5, 15.0))
                self._phase_timer = 45
        elif self._behavior == "leap":
            self._x += self._vx
            self._y += self._vy
            self._vy += 0.65  # gravity
            if self._x < 90 or self._x > self.width - 90:
                self._vx *= -0.8
                self._x = float(np.clip(self._x, 90, self.width - 90))
            if self._y >= ground_y:
                self._y = ground_y
                self._vx = 0.0
                self._vy = 0.0
                self._behavior = "walk"
                self._phase_timer = 28
                self._vx = self._facing * 2.8
        elif self._behavior == "walk":
            self._x += self._vx
            if self._x < 110 or self._x > self.width - 110:
                self._facing *= -1.0
                self._vx *= -1.0
            if self._phase_timer <= 0:
                self._behavior = "crouch" if self.rng.random() < 0.7 else "idle"
                self._vx = 0.0
                self._phase_timer = 16

        frame = self._bg_base.copy()
        self._draw_swaying_plant(frame, self._step)
        self._draw_cat(frame, self._step)
        is_occluded = self._draw_foreground_occluder(frame)

        # Tiny realistic sensor grain
        noise = self.rng.integers(-2, 3, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        norm_x = (self._x / self.width) * 2.0 - 1.0
        norm_y = (self._y / self.height) * 2.0 - 1.0
        return frame, SimMetadata(cat_norm_xy=(norm_x, norm_y), behavior=self._behavior, is_occluded=is_occluded)
