"""Unified Visual Cortex implementing silent perception + unconscious active gaze."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from brain.core.types import SensoryPacket, WorkspaceState
from brain.core.workspace import BrainRegion
from brain.regions.vision.colliculus import SuperiorColliculusGaze
from brain.regions.vision.dorsal import DorsalMotionStream
from brain.regions.vision.retina import DualResolutionRetina
from brain.regions.vision.ventral import VentralSemanticStream


class VisualCortex(BrainRegion):
    """Bio-inspired Visual Brain: Retina -> Dorsal Prediction -> Gaze Reflex -> Ventral Latent."""

    name = "vision"

    def __init__(
        self,
        periphery_size: int = 96,
        fovea_size: int = 140,
        fovea_span: float = 0.44,
        ventral_backend: str = "auto",
        model_dir: Path | str | None = None,
        device: str | torch.device | None = None,
        bgr_input: bool = True,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.bgr_input = bgr_input

        self.retina = DualResolutionRetina(
            periphery_size=periphery_size,
            fovea_size=fovea_size,
            fovea_span=fovea_span,
            device=self.device,
        )
        self.dorsal = DorsalMotionStream(
            periphery_size=periphery_size,
            device=self.device,
        )
        self.colliculus = SuperiorColliculusGaze(
            periphery_size=periphery_size,
            device=self.device,
        )
        self.ventral = VentralSemanticStream(
            latent_dim=384,
            device=self.device,
            backend=ventral_backend,
            model_dir=model_dir,
        )

    def subconscious_step(
        self, raw_input: Any, feedback: WorkspaceState | None = None
    ) -> SensoryPacket:
        """Run one silent perception-action tick on a raw camera/video frame."""
        t0 = time.perf_counter()

        # 1. Sample periphery + sharp fovea at current gaze position
        periphery, fovea, span_xy = self.retina.sample(
            raw_input, self.colliculus.gaze, bgr_input=self.bgr_input
        )

        # 2. Dorsal stream predicts periphery dynamics & cancels static/repetitive clutter
        surprise_map, surprise_peak, motion_energy, flow_xy = self.dorsal.step(periphery)

        # 3. Superior Colliculus shifts gaze unconsciously (saccade / smooth pursuit)
        gaze, peak_xy, effective_salience = self.colliculus.step(surprise_map, flow_xy=flow_xy)

        # 4. Resample fovea at updated gaze when an abrupt saccade occurred for crisp perception
        if gaze.mode == "saccade":
            _, fovea, span_xy = self.retina.sample(raw_input, gaze, bgr_input=self.bgr_input)

        # 5. Ventral stream encodes sharp foveal content into continuous latent attractor
        latent_state = self.ventral.encode(
            fovea=fovea,
            surprise_energy=surprise_peak,
            gaze_vx=gaze.vx,
            gaze_vy=gaze.vy,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        vram_mb = (
            torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            if self.device.type == "cuda"
            else 0.0
        )

        return SensoryPacket(
            region_name=self.name,
            timestamp=time.time(),
            latent_state=latent_state,
            surprise_energy=surprise_peak,
            motion_energy=motion_energy,
            salience_peak_xy=peak_xy,
            gaze=gaze,
            diagnostics={
                "periphery": periphery.detach().to("cpu"),
                "fovea": fovea.detach().to("cpu"),
                "surprise_map": surprise_map.detach().to("cpu"),
                "effective_salience": effective_salience.detach().to("cpu"),
                "ior_map": self.colliculus._ior_map.detach().to("cpu"),
                "span_xy": span_xy,
                "latency_ms": latency_ms,
                "vram_mb": vram_mb,
                "backend": self.ventral.backend_name,
                "device": str(self.device),
            },
        )

    def reset(self) -> None:
        self.dorsal.reset()
        self.colliculus.reset()
        self.ventral.reset()
