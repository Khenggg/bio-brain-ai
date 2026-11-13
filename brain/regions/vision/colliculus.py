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
    ) -> None:
        self.periphery_size = periphery_size
        self.device = torch.device(device)
        self.attention_threshold = attention_threshold
        self.saccade_distance = saccade_distance
        self.ior_decay = ior_decay
        self.ior_strength = ior_strength

        coords = torch.linspace(-1.0, 1.0, periphery_size, device=self.device)
        self._yy, self._xx = torch.meshgrid(coords, coords, indexing="ij")
        self._ior_map = torch.zeros(periphery_size, periphery_size, device=self.device)
        self.gaze = GazeState()

    def step(
        self,
        surprise_map: torch.Tensor,
        flow_xy: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[GazeState, tuple[float, float], torch.Tensor]:
        """Update eye position unconsciously toward dominant continuous surprise density."""
        with torch.no_grad():
            # Decay Inhibition of Return map over time
            self._ior_map.mul_(self.ior_decay)

            # Attentional hysteresis: mild Gaussian bias near current gaze prevents jitter
            dist2 = (self._xx - self.gaze.x) ** 2 + (self._yy - self.gaze.y) ** 2
            local_spot = torch.exp(-dist2 / (2.0 * 0.25 * 0.25))

            # Effective salience suppresses recently boring gaze locations
            effective = surprise_map * (1.0 - self.ior_strength * self._ior_map)
            # Boost ongoing surprise near current gaze slightly to maintain smooth pursuit
            effective = effective * (1.0 + 0.45 * local_spot)

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
            local_surprise = float((surprise_map * torch.exp(-dist2 / (2.0 * 0.12 * 0.12))).max().item())
            if local_surprise < self.attention_threshold * 0.8:
                ior_pulse = torch.exp(-dist2 / (2.0 * 0.15 * 0.15))
                self._ior_map = torch.clamp(self._ior_map + 0.18 * ior_pulse, 0.0, 1.0)

            if peak_val < self.attention_threshold:
                # Quiet scene: relax velocity and stabilize fixation
                self.gaze.vx *= 0.6
                self.gaze.vy *= 0.6
                self.gaze.x = float(np.clip(self.gaze.x + self.gaze.vx, -0.95, 0.95))
                self.gaze.y = float(np.clip(self.gaze.y + self.gaze.vy, -0.95, 0.95))
                self.gaze.mode = "fixate"
                return self.gaze, (tx, ty), effective

            dx = tx - self.gaze.x
            dy = ty - self.gaze.y
            dist = math.hypot(dx, dy)

            if dist > self.saccade_distance:
                # Ballistic unconscious saccade toward sudden peripheral surprise
                gain = 0.65
                self.gaze.vx = dx * gain
                self.gaze.vy = dy * gain
                self.gaze.mode = "saccade"
            else:
                # Smooth pursuit: spring pull toward target + predictive flow lead
                fx, fy = flow_xy
                target_vx = dx * 0.42 + fx * 0.50
                target_vy = dy * 0.42 + fy * 0.50
                self.gaze.vx = 0.35 * self.gaze.vx + 0.65 * target_vx
                self.gaze.vy = 0.35 * self.gaze.vy + 0.65 * target_vy
                self.gaze.mode = "pursuit"

            self.gaze.x = float(np.clip(self.gaze.x + self.gaze.vx, -0.95, 0.95))
            self.gaze.y = float(np.clip(self.gaze.y + self.gaze.vy, -0.95, 0.95))
            return self.gaze, (tx, ty), effective

    def reset(self) -> None:
        self._ior_map.zero_()
        self.gaze = GazeState()
