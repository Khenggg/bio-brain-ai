"""Thalamus & Global Workspace: routes silent subconscious regions into conscious ignition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from brain.core.memory import EpisodicAttractorMemory
from brain.core.types import SensoryPacket, WorkspaceState


class BrainRegion(ABC):
    """Base contract for any autonomous sensory or cognitive lobe."""

    name: str

    @abstractmethod
    def subconscious_step(
        self, raw_input: Any, feedback: WorkspaceState | None = None
    ) -> SensoryPacket:
        """Process sensory/internal streams silently without forced symbolic labels."""

    def reset(self) -> None:
        """Reset transient recurrent states."""


class ThalamusWorkspace:
    """Coordinates multiple silent brain regions and gates Global Workspace ignition."""

    def __init__(
        self,
        memory: EpisodicAttractorMemory | None = None,
        base_ignition_floor: float = 0.08,
        habituation_rate: float = 0.05,
        surprise_gain: float = 1.65,
    ) -> None:
        self.regions: dict[str, BrainRegion] = {}
        self.memory = memory or EpisodicAttractorMemory()
        self.base_ignition_floor = base_ignition_floor
        self.habituation_rate = habituation_rate
        self.surprise_gain = surprise_gain

        self._step_id = 0
        self._surprise_ema = 0.04
        self._surprise_var = 0.001
        self._last_state: WorkspaceState | None = None

    def register_region(self, region: BrainRegion) -> None:
        self.regions[region.name] = region

    def step(self, inputs: dict[str, Any]) -> tuple[WorkspaceState, dict[str, SensoryPacket]]:
        """Run all active regions silently, then decide conscious Workspace ignition."""
        self._step_id += 1
        packets: dict[str, SensoryPacket] = {}
        subconscious_activity: dict[str, float] = {}

        for name, region in self.regions.items():
            if name not in inputs:
                continue
            packet = region.subconscious_step(inputs[name], feedback=self._last_state)
            packets[name] = packet
            subconscious_activity[name] = packet.surprise_energy

        if not packets:
            state = WorkspaceState(
                step_id=self._step_id,
                conscious_ignition=False,
                ignition_strength=0.0,
                active_latent=None,
                memory_resonance=0.0,
                memory_episode_count=self.memory.count,
                adaptive_threshold=self.base_ignition_floor,
                subconscious_regions=subconscious_activity,
            )
            self._last_state = state
            return state, packets

        # Competition among sensory regions: highest surprise + novelty wins Workspace access
        dominant = max(packets.values(), key=lambda p: p.surprise_energy)
        resonance, _ = self.memory.resonate(dominant.latent_state)
        novelty_bonus = (1.0 - resonance) * 0.06 if self.memory.count > 0 else 0.08
        salience_score = dominant.surprise_energy + 0.35 * dominant.motion_energy + novelty_bonus

        adaptive_threshold = max(
            self.base_ignition_floor,
            self._surprise_ema + self.surprise_gain * (self._surprise_var**0.5),
        )
        conscious_ignition = salience_score >= adaptive_threshold

        # Update habituation statistics slowly so steady clutter raises tolerance
        delta = dominant.surprise_energy - self._surprise_ema
        self._surprise_ema += self.habituation_rate * delta
        self._surprise_var = (1.0 - self.habituation_rate) * self._surprise_var + (
            self.habituation_rate * (delta * delta)
        )

        active_latent: torch.Tensor | None = None
        if conscious_ignition:
            resonance, _ = self.memory.consolidate(dominant.latent_state)
            active_latent = dominant.latent_state

        state = WorkspaceState(
            step_id=self._step_id,
            conscious_ignition=conscious_ignition,
            ignition_strength=float(salience_score),
            active_latent=active_latent,
            memory_resonance=float(resonance),
            memory_episode_count=self.memory.count,
            adaptive_threshold=float(adaptive_threshold),
            subconscious_regions=subconscious_activity,
        )
        self._last_state = state
        return state, packets

    def reset(self) -> None:
        self._step_id = 0
        self._surprise_ema = 0.04
        self._surprise_var = 0.001
        self._last_state = None
        self.memory.reset()
        for region in self.regions.values():
            region.reset()
