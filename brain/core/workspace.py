"""Thalamus & Global Workspace: routes silent subconscious regions into conscious ignition."""

from __future__ import annotations

from abc import ABC, abstractmethod
import threading
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
        self._boredom = 0.0
        self._recent_episode_idx = -1
        self._same_episode_streak = 0
        self._last_state: WorkspaceState | None = None
        self._active_packets: dict[str, SensoryPacket] = {}
        self._lock = threading.RLock()

    def register_region(self, region: BrainRegion) -> None:
        with self._lock:
            self.regions[region.name] = region

    @property
    def latest_state(self) -> WorkspaceState | None:
        with self._lock:
            return self._last_state

    def step(self, inputs: dict[str, Any]) -> tuple[WorkspaceState, dict[str, SensoryPacket]]:
        """Run all active regions silently, then decide conscious Workspace ignition."""
        with self._lock:
            regions_snapshot = list(self.regions.items())
            feedback = self._last_state

        packets: dict[str, SensoryPacket] = {}
        for name, region in regions_snapshot:
            if name not in inputs:
                continue
            packet = region.subconscious_step(inputs[name], feedback=feedback)
            packets[name] = packet

        with self._lock:
            self._step_id += 1
            self._active_packets.update(packets)
            state = self._evaluate_packets(
                packets,
                source_region=",".join(packets.keys()),
                update_habituation=True,
            )
            return state, packets

    def process_packet(
        self,
        packet: SensoryPacket,
        coincidence_window_s: float = 0.35,
        region_rates_hz: dict[str, float] | None = None,
    ) -> WorkspaceState:
        """Asynchronously integrate one arriving SensoryPacket into Global Workspace."""
        with self._lock:
            self._step_id += 1
            self._active_packets[packet.region_name] = packet

            newest_ts = max(p.timestamp for p in self._active_packets.values())
            stale = [
                name
                for name, pkt in self._active_packets.items()
                if (newest_ts - pkt.timestamp) > coincidence_window_s
            ]
            for name in stale:
                del self._active_packets[name]

            update_habituation = packet.modality_event != "semantic"
            return self._evaluate_packets(
                dict(self._active_packets),
                source_region=packet.region_name,
                update_habituation=update_habituation,
                region_rates_hz=region_rates_hz,
            )

    def _evaluate_packets(
        self,
        packets: dict[str, SensoryPacket],
        source_region: str = "",
        update_habituation: bool = True,
        region_rates_hz: dict[str, float] | None = None,
    ) -> WorkspaceState:
        subconscious_activity = {name: pkt.surprise_energy for name, pkt in packets.items()}
        rates = dict(region_rates_hz) if region_rates_hz is not None else {}

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
                source_region=source_region,
                region_rates_hz=rates,
                familiarity_boredom=float(self._boredom),
                forgotten_count=self.memory.forgotten_count,
                long_term_episode_count=self.memory.long_term_count,
                active_episode_idx=-1,
                active_episode_strength=0.0,
            )
            self._last_state = state
            return state

        # Competition among sensory regions: highest surprise + novelty wins Workspace access
        dominant = max(packets.values(), key=lambda p: p.surprise_energy)
        resonance, best_idx = self.memory.resonate(dominant.latent_state)
        novelty_bonus = (1.0 - resonance) * 0.06 if self.memory.count > 0 else 0.08
        salience_score = dominant.surprise_energy + 0.35 * dominant.motion_energy + novelty_bonus

        adaptive_threshold = max(
            self.base_ignition_floor,
            self._surprise_ema + self.surprise_gain * (self._surprise_var**0.5),
        ) + (0.03 * self._boredom)
        conscious_ignition = salience_score >= adaptive_threshold

        # Update habituation statistics slowly so steady clutter raises tolerance
        if update_habituation:
            delta = dominant.surprise_energy - self._surprise_ema
            self._surprise_ema += self.habituation_rate * delta
            self._surprise_var = (1.0 - self.habituation_rate) * self._surprise_var + (
                self.habituation_rate * (delta * delta)
            )

        active_latent: torch.Tensor | None = None
        created_new = False
        if conscious_ignition:
            # Reflex ticks can ignite immediately while memory consolidates on fresh semantic updates
            if dominant.is_fresh_semantic or self.memory.count == 0:
                resonance, created_new = self.memory.consolidate(
                    dominant.latent_state, step_id=self._step_id
                )
                _, best_idx = self.memory.resonate(dominant.latent_state)
            active_latent = dominant.latent_state

        if created_new:
            self._boredom *= 0.50
            self._same_episode_streak = 0
            self._recent_episode_idx = best_idx

        if update_habituation:
            exclude_idx = best_idx if best_idx >= 0 and resonance >= 0.80 else -1
            if self.memory.step_decay(step_id=self._step_id, exclude_idx=exclude_idx) > 0:
                resonance, best_idx = self.memory.resonate(dominant.latent_state)
                if self._recent_episode_idx >= self.memory.count:
                    self._recent_episode_idx = best_idx

            if not created_new:
                if best_idx >= 0 and resonance >= 0.86:
                    is_new_visit = best_idx != self._recent_episode_idx
                    self.memory.note_recognition(
                        best_idx, step_id=self._step_id, bump_hit=is_new_visit
                    )
                    if is_new_visit:
                        self._recent_episode_idx = best_idx
                        self._same_episode_streak = 1
                    else:
                        self._same_episode_streak += 1

                    _, ep_hits = self.memory.exposure_of(best_idx)
                    familiarity_drive = min(
                        1.0,
                        0.12 * max(0, ep_hits - 1) + 0.012 * max(0, self._same_episode_streak - 45),
                    )
                    self._boredom += 0.06 * (familiarity_drive - self._boredom)
                elif best_idx < 0 or resonance < 0.80:
                    self._boredom *= 0.70
                    self._same_episode_streak = 0
                    self._recent_episode_idx = -1
                else:
                    self._boredom *= 0.95
                    self._same_episode_streak = max(0, self._same_episode_streak - 1)

        matched_idx = best_idx if best_idx >= 0 and resonance >= 0.80 else -1
        ep_strength = self.memory.exposure_of(matched_idx)[0] if matched_idx >= 0 else 0.0

        state = WorkspaceState(
            step_id=self._step_id,
            conscious_ignition=conscious_ignition,
            ignition_strength=float(salience_score),
            active_latent=active_latent,
            memory_resonance=float(resonance),
            memory_episode_count=self.memory.count,
            adaptive_threshold=float(adaptive_threshold),
            subconscious_regions=subconscious_activity,
            source_region=source_region,
            region_rates_hz=rates,
            familiarity_boredom=float(self._boredom),
            forgotten_count=self.memory.forgotten_count,
            long_term_episode_count=self.memory.long_term_count,
            active_episode_idx=matched_idx,
            active_episode_strength=float(ep_strength),
        )
        self._last_state = state
        return state

    def reset(self) -> None:
        with self._lock:
            self._step_id = 0
            self._surprise_ema = 0.04
            self._surprise_var = 0.001
            self._boredom = 0.0
            self._recent_episode_idx = -1
            self._same_episode_streak = 0
            self._last_state = None
            self._active_packets.clear()
            self.memory.reset()
            for region in self.regions.values():
                region.reset()
