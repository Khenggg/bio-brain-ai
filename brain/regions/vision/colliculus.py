"""Superior Colliculus: continuous neural-field gaze control (Saccade, Pursuit, IOR) without bboxes."""

from __future__ import annotations

import math

import numpy as np
import torch

from brain.core.types import GazeState


class SuperiorColliculusGaze:
    """Drives unconscious eye movements from continuous surprise energy and Inhibition of Return."""

    def __init__(
        self,
        periphery_size: int = 96,
        device: torch.device | str = "cpu",
        attention_threshold: float = 0.012,
        saccade_distance: float = 0.30,
        ior_decay: float = 0.92,
        ior_strength: float = 0.70,
        saccade_cooldown_ticks: int = 4,
    ) -> None:
        self.periphery_size = periphery_size
        self.device = torch.device(device)
        self.attention_threshold = attention_threshold
        self.saccade_distance = saccade_distance
        self.ior_decay = ior_decay
        self.ior_strength = ior_strength
        self.saccade_cooldown_ticks = max(0, int(saccade_cooldown_ticks))

        coords = torch.linspace(-1.0, 1.0, periphery_size, device=self.device)
        self._yy, self._xx = torch.meshgrid(coords, coords, indexing="ij")
        self._ior_map = torch.zeros(periphery_size, periphery_size, device=self.device)
        self.gaze = GazeState()
        self._saccade_cooldown = 0
        self._cand_xy: tuple[float, float] | None = None
        self._cand_hits = 0

    def step(
        self,
        surprise_map: torch.Tensor,
        flow_xy: tuple[float, float] = (0.0, 0.0),
        topdown_boredom: float = 0.0,
    ) -> tuple[GazeState, tuple[float, float], torch.Tensor]:
        """Update eye position unconsciously toward dominant continuous surprise density."""
        with torch.no_grad():
            if self._saccade_cooldown > 0:
                self._saccade_cooldown -= 1

            boredom = float(min(max(topdown_boredom, 0.0), 1.0))

            # Decay Inhibition of Return map over time
            self._ior_map.mul_(self.ior_decay)

            # Attentional hysteresis: keeps smooth lock onto active moving target vs edge flicker
            dist2 = (self._xx - self.gaze.x) ** 2 + (self._yy - self.gaze.y) ** 2
            local_spot = torch.exp(-dist2 / (2.0 * 0.25 * 0.25))

            # Top-down familiarity fatigue inhibits the already-known central spot
            if boredom > 0.25:
                central_fatigue = torch.exp(-dist2 / (2.0 * 0.20 * 0.20))
                self._ior_map = torch.clamp(
                    self._ior_map + (0.06 * boredom) * central_fatigue, 0.0, 1.0
                )

            # Effective salience suppresses recently boring or over-familiar gaze locations
            effective = surprise_map * (1.0 - self.ior_strength * self._ior_map)
            # Boredom loosens central lock and increases peripheral exploratory curiosity
            local_boost = 0.85 * (1.0 - 0.85 * boredom)
            periph_boost = 0.35 * boredom * (1.0 - local_spot)
            effective = effective * (1.0 + local_boost * local_spot + periph_boost)

            peak_val = float(effective.max().item())
            peak_flat = int(torch.argmax(effective).item())
            py_idx, px_idx = divmod(peak_flat, self.periphery_size)

            # Local soft-centroid around salience peak (smooth body center of energy, NO bbox!)
            tx = float(self._xx[py_idx, px_idx].item())
            ty = float(self._yy[py_idx, px_idx].item())
            peak_dist2 = (self._xx - tx) ** 2 + (self._yy - ty) ** 2
            local_mask = torch.exp(-peak_dist2 / (2.0 * 0.22 * 0.22))
            weights = effective * local_mask
            w_sum = float(weights.sum().item())
            if w_sum > 1e-6:
                tx = float((weights * self._xx).sum().item() / w_sum)
                ty = float((weights * self._yy).sum().item() / w_sum)

            # Check local surprise under current fovea: if boring, build IOR so eye looks elsewhere
            local_surprise = float((surprise_map * torch.exp(-dist2 / (2.0 * 0.14 * 0.14))).max().item())
            if local_surprise < self.attention_threshold * 0.8:
                ior_pulse = torch.exp(-dist2 / (2.0 * 0.15 * 0.15))
                self._ior_map = torch.clamp(self._ior_map + 0.18 * ior_pulse, 0.0, 1.0)

            if peak_val < self.attention_threshold:
                self._cand_xy = None
                self._cand_hits = 0
                self.gaze.vx *= 0.6
                self.gaze.vy *= 0.6
                self.gaze.x = float(np.clip(self.gaze.x + self.gaze.vx, -0.95, 0.95))
                self.gaze.y = float(np.clip(self.gaze.y + self.gaze.vy, -0.95, 0.95))
                self.gaze.mode = "fixate"
                return self.gaze, (tx, ty), effective

            dx = tx - self.gaze.x
            dy = ty - self.gaze.y
            dist = math.hypot(dx, dy)

            allow_saccade = False
            if dist > self.saccade_distance and self._saccade_cooldown == 0:
                if local_surprise < self.attention_threshold:
                    allow_saccade = True
                elif peak_val >= local_surprise * 2.5:
                    allow_saccade = True
                else:
                    if (
                        self._cand_xy is not None
                        and math.hypot(tx - self._cand_xy[0], ty - self._cand_xy[1]) < 0.25
                    ):
                        self._cand_hits += 1
                    else:
                        self._cand_xy = (tx, ty)
                        self._cand_hits = 1
                    required_hits = 1 if boredom > 0.50 else 2
                    allow_saccade = self._cand_hits >= required_hits
            else:
                self._cand_xy = None
                self._cand_hits = 0

            if allow_saccade:
                # Ballistic unconscious saccade toward confirmed peripheral surprise
                gain = 0.65
                self.gaze.vx = dx * gain
                self.gaze.vy = dy * gain
                self.gaze.mode = "saccade"
                self._saccade_cooldown = self.saccade_cooldown_ticks
                self._cand_xy = None
                self._cand_hits = 0
            else:
                # Smooth pursuit: dampen stray distant pull if peripheral distractor is unconfirmed
                pull = 0.42 if dist <= self.saccade_distance else 0.18
                fx, fy = flow_xy
                target_vx = dx * pull + fx * 0.50
                target_vy = dy * pull + fy * 0.50
                self.gaze.vx = 0.40 * self.gaze.vx + 0.60 * target_vx
                self.gaze.vy = 0.40 * self.gaze.vy + 0.60 * target_vy
                self.gaze.mode = "pursuit"

            self.gaze.x = float(np.clip(self.gaze.x + self.gaze.vx, -0.95, 0.95))
            self.gaze.y = float(np.clip(self.gaze.y + self.gaze.vy, -0.95, 0.95))
            return self.gaze, (tx, ty), effective

    def reset(self) -> None:
        self._ior_map.zero_()
        self.gaze = GazeState()
        self._saccade_cooldown = 0
        self._cand_xy = None
        self._cand_hits = 0
